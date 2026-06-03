"""LLM provider integrations: OpenAI, Anthropic, Ollama, Bedrock, demo."""
import hashlib, os, sys, time
from datetime import datetime

try:
    import httpx
except ImportError:
    httpx = None


def _require_httpx():
    if httpx is None:
        print("Error: httpx not installed. Run: pip install httpx", file=sys.stderr)
        sys.exit(1)


def _retry(fn, retries=3, retry_codes=(429, 500, 502, 503)):
    """Execute fn with exponential backoff on transient failures."""
    for attempt in range(retries):
        try:
            return fn()
        except (httpx.TimeoutException, httpx.ConnectError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
        except httpx.HTTPStatusError as e:
            if e.response.status_code in retry_codes and attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def call(config, model, messages, temperature=0.0):
    """Route to appropriate provider. Returns (content, latency_ms, tokens_in, tokens_out)."""
    provider = _detect_provider(config, model)
    api_key = os.environ.get(config.get("api_key_env", "OPENAI_API_KEY"), "")

    if provider == "anthropic":
        return _anthropic(os.environ.get("ANTHROPIC_API_KEY", api_key), model, messages, temperature)
    elif provider == "ollama":
        return _ollama(config.get("ollama_url", "http://localhost:11434"), model, messages, temperature)
    elif provider == "bedrock":
        return _bedrock(model, messages, temperature, config.get("bedrock_region", "us-west-2"))
    else:
        return _openai(config.get("base_url", "https://api.openai.com/v1"), api_key, model, messages, temperature)


def demo(model, messages, temperature=0.0):
    """Simulated LLM with model-specific quality baselines and time-based drift."""
    import random
    prompt = messages[-1]["content"]
    hour_key = datetime.now().strftime("%Y%m%d%H")
    seed = int(hashlib.md5(f"{model}{prompt}{hour_key}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    quality = {"gpt-4o": 0.85, "gpt-4o-mini": 0.70, "gpt-4": 0.80,
               "claude-3-5-sonnet": 0.82, "claude-3-opus": 0.88, "claude-3-haiku": 0.65}
    base = next((v for k, v in quality.items() if model.startswith(k)), 0.7)
    drift_factor = 1.0 - (datetime.now().weekday() * 0.02)

    responses = [
        "The sky is blue due to Rayleigh scattering. Shorter blue wavelengths scatter more than red. This makes the sky appear blue to observers on Earth.",
        "Here's a concise answer: the key insight is that systems evolve over time and require monitoring.",
        'def is_palindrome(s):\n    """Check if string is palindrome."""\n    return s.lower() == s.lower()[::-1]',
        '{"name": "Alice", "age": 30, "hobbies": ["reading", "hiking"]}',
        "I can't provide instructions on that topic. It's important to note the legal and ethical concerns.",
        "Let me calculate: 17 * 24 = 17 * 20 + 17 * 4 = 340 + 68 = 408. The answer is 408.",
    ]
    idx = int(hashlib.md5(prompt.encode()).hexdigest()[:4], 16) % len(responses)
    response = responses[idx]
    if rng.random() > (base * drift_factor):
        response = "I'll need more context to provide a complete answer. Could you clarify?"

    latency = rng.randint(300, 800) if "mini" in model or "haiku" in model else rng.randint(600, 2500)
    return response, latency, rng.randint(20, 80), rng.randint(50, 300)


# ─── Private provider implementations ────────────────────────────────────────

def _detect_provider(config, model):
    provider = config.get("provider", "auto")
    if provider not in ("auto",):
        return provider
    if "claude" in model and not config.get("base_url", "").rstrip("/").endswith("/v1"):
        return "anthropic"
    if model.startswith(("us.", "anthropic.", "amazon.", "meta.")):
        return "bedrock"
    return "openai"


def _openai(base_url, api_key, model, messages, temperature):
    _require_httpx()
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}

    def do():
        t0 = time.time()
        r = httpx.post(url, json=payload, headers=headers, timeout=120)
        latency = int((time.time() - t0) * 1000)
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        return data["choices"][0]["message"]["content"], latency, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    return _retry(do)


def _anthropic(api_key, model, messages, temperature):
    _require_httpx()
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    system, msgs = None, []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            msgs.append({"role": m["role"], "content": m["content"]})
    payload = {"model": model, "messages": msgs, "max_tokens": 4096, "temperature": temperature}
    if system:
        payload["system"] = system

    def do():
        t0 = time.time()
        r = httpx.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers, timeout=120)
        latency = int((time.time() - t0) * 1000)
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        return data["content"][0]["text"], latency, usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    return _retry(do, retry_codes=(429, 500, 502, 529))


def _ollama(base_url, model, messages, temperature):
    _require_httpx()
    payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}}
    t0 = time.time()
    r = httpx.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=300)
    latency = int((time.time() - t0) * 1000)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"], latency, data.get("prompt_eval_count", 0), data.get("eval_count", 0)


def _bedrock(model, messages, temperature, region):
    try:
        import boto3
    except ImportError:
        print("Error: boto3 not installed. Run: pip install boto3", file=sys.stderr)
        sys.exit(1)
    client = boto3.client("bedrock-runtime", region_name=region)
    system, msgs = [], []
    for m in messages:
        if m["role"] == "system":
            system.append({"text": m["content"]})
        else:
            msgs.append({"role": m["role"], "content": [{"text": m["content"]}]})
    kwargs = {"modelId": model, "messages": msgs, "inferenceConfig": {"temperature": temperature, "maxTokens": 4096}}
    if system:
        kwargs["system"] = system
    t0 = time.time()
    resp = client.converse(**kwargs)
    latency = int((time.time() - t0) * 1000)
    usage = resp.get("usage", {})
    return resp["output"]["message"]["content"][0]["text"], latency, usage.get("inputTokens", 0), usage.get("outputTokens", 0)
