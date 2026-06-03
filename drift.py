#!/usr/bin/env python3
"""drift — Track LLM quality over time with golden prompts."""

import argparse, json, os, re, sqlite3, sys, time
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
except ImportError:
    httpx = None

__version__ = "0.1.0"

DB_PATH = Path("drift.db")
CONFIG_PATH = Path("drift.json")

# ─── Database ────────────────────────────────────────────────────────────────

def get_db():
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            score REAL,
            judge_reason TEXT,
            latency_ms INTEGER,
            tokens_used INTEGER,
            cost_usd REAL
        );
        CREATE INDEX IF NOT EXISTS idx_runs_model_prompt ON runs(model, prompt_id);
        CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(timestamp);
    """)
    return db

# ─── LLM Client ──────────────────────────────────────────────────────────────

def call_llm(base_url, api_key, model, messages, temperature=0.0):
    """Call OpenAI-compatible API. Returns (content, latency_ms, tokens)."""
    if httpx is None:
        print("Error: httpx not installed. Run: pip install httpx", file=sys.stderr)
        sys.exit(1)
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    t0 = time.time()
    r = httpx.post(url, json=payload, headers=headers, timeout=120)
    latency = int((time.time() - t0) * 1000)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("total_tokens", 0)
    return content, latency, tokens

def call_llm_demo(model, messages, temperature=0.0):
    """Simulated LLM for demo mode."""
    import random, hashlib
    prompt = messages[-1]["content"]
    seed = int(hashlib.md5(f"{model}{prompt}{datetime.now().isoformat()[:13]}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    responses = [
        "The answer involves careful consideration of multiple factors. First, we need to understand the context.",
        "Here's a concise answer: the key insight is that systems evolve over time and require monitoring.",
        "Let me break this down step by step:\n1. Identify the core problem\n2. Apply systematic analysis\n3. Validate results",
        "Based on my analysis, the most important factor is consistency in evaluation methodology.",
        "This is a complex topic. The short answer is: it depends on your specific requirements and constraints.",
    ]
    return rng.choice(responses), rng.randint(200, 2000), rng.randint(50, 500)

# ─── Scoring ─────────────────────────────────────────────────────────────────

def score_with_judge(base_url, api_key, judge_model, prompt, response, criteria, demo=False):
    """Use LLM-as-judge to score a response. Returns (score 0-10, reason)."""
    judge_prompt = f"""Score the following LLM response on a scale of 0-10.

ORIGINAL PROMPT: {prompt}

RESPONSE TO JUDGE: {response}

CRITERIA: {criteria}

Reply with ONLY a JSON object: {{"score": <0-10>, "reason": "<one sentence>"}}"""

    messages = [{"role": "user", "content": judge_prompt}]
    if demo:
        import random, hashlib
        seed = int(hashlib.md5(response[:50].encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        score = round(rng.uniform(5.0, 9.5), 1)
        reasons = ["Good structure and clarity", "Adequate but could be more specific",
                   "Strong analytical response", "Somewhat generic", "Excellent depth"]
        return score, rng.choice(reasons)
    
    content, _, _ = call_llm(base_url, api_key, judge_model, messages, temperature=0.0)
    try:
        # Extract JSON from response
        match = re.search(r'\{[^}]+\}', content)
        if match:
            result = json.loads(match.group())
            return float(result["score"]), result.get("reason", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None, f"Judge parse failed: {content[:100]}"

def score_with_heuristics(response, checks):
    """Score using regex/heuristic checks. Returns (score 0-10, reason)."""
    if not checks:
        return None, "No heuristic checks defined"
    passed = 0
    failures = []
    for check in checks:
        check_type = check.get("type", "contains")
        value = check.get("value", "")
        if check_type == "contains" and value.lower() in response.lower():
            passed += 1
        elif check_type == "not_contains" and value.lower() not in response.lower():
            passed += 1
        elif check_type == "regex" and re.search(value, response):
            passed += 1
        elif check_type == "min_length" and len(response) >= int(value):
            passed += 1
        elif check_type == "max_length" and len(response) <= int(value):
            passed += 1
        else:
            failures.append(f"{check_type}:{value}")
    score = round((passed / len(checks)) * 10, 1)
    reason = f"{passed}/{len(checks)} checks passed" + (f" (failed: {', '.join(failures[:3])})" if failures else "")
    return score, reason

# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_init(args):
    """Create a drift.json config with sample golden prompts."""
    if CONFIG_PATH.exists() and not args.force:
        print(f"Config already exists: {CONFIG_PATH}. Use --force to overwrite.")
        return
    config = {
        "version": 1,
        "models": ["gpt-4o-mini"],
        "judge_model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "alert_threshold": 15,
        "prompts": [
            {
                "id": "reasoning-basic",
                "prompt": "Explain why the sky is blue in exactly 3 sentences.",
                "criteria": "Must be exactly 3 sentences, scientifically accurate, mention Rayleigh scattering",
                "checks": [
                    {"type": "regex", "value": r"\. .+\. .+\."},
                    {"type": "contains", "value": "scatter"}
                ]
            },
            {
                "id": "code-python",
                "prompt": "Write a Python function that checks if a string is a palindrome. Include a docstring.",
                "criteria": "Working Python code, has docstring, handles edge cases, is concise",
                "checks": [
                    {"type": "contains", "value": "def "},
                    {"type": "regex", "value": r'""".*"""|\'\'\'.*\'\'\''},
                    {"type": "contains", "value": "return"}
                ]
            },
            {
                "id": "format-json",
                "prompt": "Return a JSON object with keys 'name', 'age', 'hobbies' for a fictional person. Return ONLY the JSON.",
                "criteria": "Valid JSON, has all required keys, hobbies is an array, no extra text",
                "checks": [
                    {"type": "contains", "value": '"name"'},
                    {"type": "contains", "value": '"age"'},
                    {"type": "contains", "value": '"hobbies"'},
                    {"type": "regex", "value": r'\[.*\]'}
                ]
            }
        ]
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print(f"✓ Created {CONFIG_PATH} with {len(config['prompts'])} golden prompts")
    print(f"  Set {config['api_key_env']} env var, or use --demo to test without API keys")

def cmd_run(args):
    """Execute golden prompts against configured models."""
    config = load_config()
    db = get_db()
    demo = args.demo
    prompts = config["prompts"]
    models = args.models.split(",") if args.models else config["models"]
    ts = datetime.now(timezone.utc).isoformat()
    
    base_url = config.get("base_url", "https://api.openai.com/v1")
    api_key = os.environ.get(config.get("api_key_env", "OPENAI_API_KEY"), "")
    judge_model = config.get("judge_model", models[0])
    
    if not demo and not api_key:
        print(f"Error: Set {config.get('api_key_env', 'OPENAI_API_KEY')} or use --demo", file=sys.stderr)
        sys.exit(1)

    total = len(prompts) * len(models)
    print(f"Running {total} evaluations ({len(prompts)} prompts × {len(models)} models)...")
    
    for model in models:
        for p in prompts:
            prompt_id = p["id"]
            prompt_text = p["prompt"]
            sys.stdout.write(f"  {model} / {prompt_id}...")
            sys.stdout.flush()
            
            messages = [{"role": "user", "content": prompt_text}]
            if demo:
                response, latency, tokens = call_llm_demo(model, messages)
            else:
                response, latency, tokens = call_llm(base_url, api_key, model, messages)
            
            # Score: try heuristics first, then judge
            h_score, h_reason = score_with_heuristics(response, p.get("checks", []))
            j_score, j_reason = score_with_judge(
                base_url, api_key, judge_model, prompt_text, response, p.get("criteria", ""), demo=demo
            )
            
            # Combined score: average of heuristic and judge (if both available)
            if h_score is not None and j_score is not None:
                score = round((h_score + j_score) / 2, 1)
                reason = f"H:{h_score}({h_reason}) J:{j_score}({j_reason})"
            elif j_score is not None:
                score, reason = j_score, j_reason
            else:
                score, reason = h_score, h_reason

            db.execute(
                "INSERT INTO runs (timestamp, model, prompt_id, prompt, response, score, judge_reason, latency_ms, tokens_used) VALUES (?,?,?,?,?,?,?,?,?)",
                (ts, model, prompt_id, prompt_text, response, score, reason, latency, tokens)
            )
            print(f" score={score}/10 ({latency}ms)")
    
    db.commit()
    print(f"\n✓ Run complete. Results saved to {DB_PATH}")

def cmd_report(args):
    """Show quality trends over time."""
    db = get_db()
    model_filter = f"AND model = '{args.model}'" if args.model else ""
    limit = args.last or 10
    
    # Get distinct models
    models = [r[0] for r in db.execute(f"SELECT DISTINCT model FROM runs WHERE 1=1 {model_filter}").fetchall()]
    if not models:
        print("No data yet. Run 'drift run' first.")
        return
    
    for model in models:
        print(f"\n{'═'*60}")
        print(f"  Model: {model}")
        print(f"{'═'*60}")
        
        # Per-prompt trends
        prompts = [r[0] for r in db.execute("SELECT DISTINCT prompt_id FROM runs WHERE model = ?", (model,)).fetchall()]
        for pid in prompts:
            rows = db.execute(
                "SELECT timestamp, score, latency_ms FROM runs WHERE model = ? AND prompt_id = ? ORDER BY timestamp DESC LIMIT ?",
                (model, pid, limit)
            ).fetchall()
            if not rows:
                continue
            scores = [r["score"] for r in rows if r["score"] is not None]
            if not scores:
                continue
            avg = sum(scores) / len(scores)
            latest = scores[0]
            trend = "→"
            if len(scores) >= 2:
                trend = "↑" if scores[0] > scores[-1] else "↓" if scores[0] < scores[-1] else "→"
            bar = "█" * int(latest) + "░" * (10 - int(latest))
            print(f"  {pid:<20} {bar} {latest:>4}/10 (avg:{avg:.1f}) {trend}")
        
        # Overall stats
        stats = db.execute(
            "SELECT AVG(score) as avg_score, AVG(latency_ms) as avg_latency, COUNT(*) as n FROM runs WHERE model = ?",
            (model,)
        ).fetchone()
        print(f"\n  Overall: avg={stats['avg_score']:.1f}/10  latency={stats['avg_latency']:.0f}ms  runs={stats['n']}")

def cmd_alert(args):
    """Check for quality regressions."""
    config = load_config()
    db = get_db()
    threshold = args.threshold or config.get("alert_threshold", 15)
    window = args.window or 5
    
    alerts = []
    models = [r[0] for r in db.execute("SELECT DISTINCT model FROM runs").fetchall()]
    
    for model in models:
        prompts = [r[0] for r in db.execute("SELECT DISTINCT prompt_id FROM runs WHERE model = ?", (model,)).fetchall()]
        for pid in prompts:
            rows = db.execute(
                "SELECT score FROM runs WHERE model = ? AND prompt_id = ? AND score IS NOT NULL ORDER BY timestamp DESC LIMIT ?",
                (model, pid, window * 2)
            ).fetchall()
            scores = [r["score"] for r in rows]
            if len(scores) < window + 1:
                continue
            recent = sum(scores[:window]) / window
            older = sum(scores[window:]) / len(scores[window:])
            if older > 0:
                drop_pct = ((older - recent) / older) * 100
                if drop_pct >= threshold:
                    alerts.append({
                        "model": model, "prompt_id": pid,
                        "drop": drop_pct, "recent_avg": recent, "baseline_avg": older
                    })
    
    if not alerts:
        print("✓ No quality regressions detected.")
        return
    
    print(f"⚠ {len(alerts)} REGRESSION(S) DETECTED:\n")
    for a in alerts:
        print(f"  🔴 {a['model']} / {a['prompt_id']}")
        print(f"     Drop: {a['drop']:.1f}% (baseline: {a['baseline_avg']:.1f} → recent: {a['recent_avg']:.1f})")
        print()
    sys.exit(1)  # Non-zero exit for CI/cron integration

def cmd_history(args):
    """Show raw run history."""
    db = get_db()
    limit = args.last or 20
    model_filter = f"AND model = '{args.model}'" if args.model else ""
    rows = db.execute(
        f"SELECT timestamp, model, prompt_id, score, latency_ms, judge_reason FROM runs WHERE 1=1 {model_filter} ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    ).fetchall()
    if not rows:
        print("No history. Run 'drift run' first.")
        return
    print(f"{'Timestamp':<22} {'Model':<16} {'Prompt':<20} {'Score':<7} {'Latency':<8} Reason")
    print("─" * 100)
    for r in rows:
        ts = r['timestamp'][:19].replace('T', ' ')
        print(f"{ts:<22} {r['model']:<16} {r['prompt_id']:<20} {r['score'] or 'N/A':<7} {r['latency_ms']}ms{'':<4} {(r['judge_reason'] or '')[:30]}")

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_config():
    if not CONFIG_PATH.exists():
        print(f"No config found. Run 'drift init' first.", file=sys.stderr)
        sys.exit(1)
    return json.loads(CONFIG_PATH.read_text())

# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="drift", description="Track LLM quality over time")
    parser.add_argument("--version", action="version", version=f"drift {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="Create config with golden prompts")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing config")

    p_run = sub.add_parser("run", help="Execute golden prompts against models")
    p_run.add_argument("--demo", action="store_true", help="Use simulated LLM (no API key needed)")
    p_run.add_argument("--models", help="Comma-separated model list (overrides config)")

    p_report = sub.add_parser("report", help="Show quality trends")
    p_report.add_argument("--model", help="Filter to specific model")
    p_report.add_argument("--last", type=int, help="Number of recent runs to show")

    p_alert = sub.add_parser("alert", help="Check for quality regressions")
    p_alert.add_argument("--threshold", type=float, help="Drop %% to trigger alert (default: 15)")
    p_alert.add_argument("--window", type=int, help="Number of recent runs to compare (default: 5)")

    p_hist = sub.add_parser("history", help="Show raw run history")
    p_hist.add_argument("--model", help="Filter to specific model")
    p_hist.add_argument("--last", type=int, help="Number of entries to show")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    {"init": cmd_init, "run": cmd_run, "report": cmd_report, "alert": cmd_alert, "history": cmd_history}[args.command](args)

if __name__ == "__main__":
    main()
