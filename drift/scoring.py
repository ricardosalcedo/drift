"""Scoring: LLM-as-judge, heuristic checks, and cost estimation."""
import hashlib, json, re
from . import providers

# Per-1M token pricing
PRICING = {
    "gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00), "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50), "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00), "claude-3-haiku": (0.25, 1.25),
    "claude-3-opus": (15.00, 75.00), "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (0.80, 4.00),
}


def estimate_cost(model, tokens_in, tokens_out):
    """Estimate USD cost from token counts. Returns None if model unknown."""
    pricing = next((v for k, v in PRICING.items() if model.startswith(k)), None)
    if not pricing:
        return None
    return (tokens_in * pricing[0] + tokens_out * pricing[1]) / 1_000_000


def judge(config, judge_model, prompt, response, criteria, demo=False):
    """LLM-as-judge scoring. Returns (score: 0-10, reason: str)."""
    judge_prompt = (
        f"Score the following LLM response on a scale of 0-10.\n\n"
        f"ORIGINAL PROMPT: {prompt}\n\nRESPONSE TO JUDGE: {response}\n\n"
        f"CRITERIA: {criteria}\n\n"
        f'Reply with ONLY a JSON object: {{"score": <0-10>, "reason": "<one sentence>"}}'
    )
    if demo:
        import random
        seed = int(hashlib.md5(response[:50].encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        return round(rng.uniform(5.0, 9.5), 1), rng.choice([
            "Good structure and clarity", "Adequate but could be more specific",
            "Strong analytical response", "Somewhat generic", "Excellent depth"])

    content, _, _, _ = providers.call(config, judge_model, [{"role": "user", "content": judge_prompt}])
    try:
        match = re.search(r'\{[^}]+\}', content)
        if match:
            result = json.loads(match.group())
            return float(result["score"]), result.get("reason", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None, f"Judge parse failed: {content[:100]}"


def heuristics(response, checks):
    """Run heuristic checks. Returns (score: 0-10, reason: str)."""
    if not checks:
        return None, "No checks"
    passed, failures = 0, []
    for ck in checks:
        ct, val = ck.get("type", "contains"), ck.get("value", "")
        ok = False
        if ct == "contains":    ok = val.lower() in response.lower()
        elif ct == "not_contains": ok = val.lower() not in response.lower()
        elif ct == "regex":     ok = bool(re.search(val, response))
        elif ct == "min_length": ok = len(response) >= int(val)
        elif ct == "max_length": ok = len(response) <= int(val)
        if ok:
            passed += 1
        else:
            failures.append(f"{ct}:{val[:20]}")
    score = round((passed / len(checks)) * 10, 1)
    reason = f"{passed}/{len(checks)} checks" + (f" (fail: {', '.join(failures[:2])})" if failures else "")
    return score, reason


def evaluate(config, model, prompt_text, response, checks, criteria, judge_model, demo=False):
    """Combined scoring: heuristics + judge. Returns (score, reason)."""
    h_score, h_reason = heuristics(response, checks)
    j_score, j_reason = judge(config, judge_model, prompt_text, response, criteria, demo=demo)

    if h_score is not None and j_score is not None:
        score = round((h_score + j_score) / 2, 1)
        reason = f"H:{h_score}({h_reason}) J:{j_score}({j_reason})"
    elif j_score is not None:
        score, reason = j_score, j_reason
    else:
        score, reason = h_score, h_reason
    return score, reason
