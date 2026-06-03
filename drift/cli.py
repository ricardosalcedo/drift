"""CLI commands and argument parsing."""
import argparse, concurrent.futures, csv, io, json, os, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, config, db, display, providers, scoring, stats


def cmd_init(args):
    """Create drift.json with sample golden prompts."""
    if config.CONFIG_PATH.exists() and not args.force:
        print(f"Config exists: {config.CONFIG_PATH}. Use --force to overwrite.")
        return
    config.save(config.DEFAULT_CONFIG)
    prompts = config.DEFAULT_CONFIG["prompts"]
    tags = sorted({t for p in prompts for t in p.get("tags", [])})
    print(f"{display.green('✓')} Created {config.CONFIG_PATH} with {len(prompts)} golden prompts")
    print(f"  Tags: {', '.join(tags)} | Provider: auto-detect | Cache: 24h")


def cmd_run(args):
    """Execute golden prompts against models."""
    cfg = config.load()
    database = db.connect()
    models = args.models.split(",") if args.models else cfg["models"]
    prompts = cfg["prompts"]
    if args.tag:
        prompts = [p for p in prompts if args.tag in p.get("tags", [])]
        if not prompts:
            print(f"No prompts with tag '{args.tag}'")
            return

    judge_model = cfg.get("judge_model", models[0])
    ts = datetime.now(timezone.utc).isoformat()
    cache_hours = cfg.get("cache_hours", 24)

    if not args.demo:
        api_key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"), "")
        if not api_key and cfg.get("provider") not in ("ollama", "bedrock"):
            print("Error: Set API key or use --demo", file=sys.stderr)
            sys.exit(1)

    total = len(prompts) * len(models)
    print(f"{display.bold(f'Running {total} evaluations')} ({len(prompts)} prompts × {len(models)} models)"
          f"{' [parallel]' if args.parallel else ''}\n")

    def _eval(model, p):
        """Evaluate single prompt. Thread-safe."""
        pid, text, criteria = p["id"], p["prompt"], p.get("criteria", "")
        tags_str = ",".join(p.get("tags", []))
        local_db = sqlite3.connect(str(db.DB_PATH))
        local_db.row_factory = sqlite3.Row
        version = db.get_prompt_version(local_db, pid, text, criteria)
        local_db.close()

        msgs = [{"role": "user", "content": text}]
        if args.demo:
            response, latency, tok_in, tok_out = providers.demo(model, msgs)
        else:
            response, latency, tok_in, tok_out = providers.call(cfg, model, msgs)

        score, reason = scoring.evaluate(cfg, model, text, response, p.get("checks", []), criteria, judge_model, demo=args.demo)
        cost = scoring.estimate_cost(model, tok_in, tok_out)
        return {"model": model, "prompt_id": pid, "prompt_text": text, "response": response,
                "score": score, "reason": reason, "latency": latency, "tok_in": tok_in,
                "tok_out": tok_out, "cost": cost, "tags": tags_str, "version": version}

    # Run evaluations
    results, total_cost, cached_count = [], 0.0, 0

    if args.parallel and total > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, total)) as pool:
            futures = {pool.submit(_eval, m, p): (m, p) for m in models for p in prompts}
            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                lat = r["latency"]
                print(f"  {display.dim(r['model'])} / {r['prompt_id']}... {display.score_color(r['score'])} {display.dim(f'({lat}ms)')}")
                if r["cost"]:
                    total_cost += r["cost"]
                results.append(r)
    else:
        for model in models:
            for p in prompts:
                pid, text = p["id"], p["prompt"]
                sys.stdout.write(f"  {display.dim(model)} / {pid}...")
                sys.stdout.flush()

                # Cache check
                if not args.force and not args.demo:
                    cached = db.cache_get(database, model, text, cache_hours)
                    if cached:
                        cached_count += 1
                        print(f" {display.cyan('cached')} {cached['score']}/10")
                        version = db.get_prompt_version(database, pid, text, p.get("criteria", ""))
                        database.execute(
                            "INSERT INTO runs (timestamp,model,prompt_id,prompt_version,prompt,response,score,judge_reason,latency_ms,tokens_in,tokens_out,cost_usd,tags,cached) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (ts, model, pid, version, text, cached["response"], cached["score"], cached["judge_reason"], cached["latency_ms"], cached["tokens_in"], cached["tokens_out"], 0, ",".join(p.get("tags", [])), 1))
                        results.append({"model": model, "prompt_id": pid, "score": cached["score"], "cached": True})
                        continue

                r = _eval(model, p)
                if r["cost"]:
                    total_cost += r["cost"]
                if not args.demo:
                    db.cache_set(database, model, r["prompt_text"], r["response"], r["score"], r["reason"], r["latency"], r["tok_in"], r["tok_out"])
                lat = r["latency"]
                print(f" {display.score_color(r['score'])} {display.dim(f'({lat}ms)')}")
                results.append(r)

    # Persist all results
    for r in results:
        if r.get("cached"):
            continue
        database.execute(
            "INSERT INTO runs (timestamp,model,prompt_id,prompt_version,prompt,response,score,judge_reason,latency_ms,tokens_in,tokens_out,cost_usd,tags,cached) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts, r["model"], r["prompt_id"], r.get("version", ""), r.get("prompt_text", ""), r.get("response", ""),
             r["score"], r.get("reason", ""), r.get("latency", 0), r.get("tok_in", 0), r.get("tok_out", 0),
             r.get("cost"), r.get("tags", ""), 0))
    database.commit()

    cache_msg = f" ({cached_count} cached)" if cached_count else ""
    cost_msg = display.dim(f"Cost: ${total_cost:.4f}") if total_cost else ""
    print(f"\n{display.green('✓')} Run complete.{cache_msg} {cost_msg}")
    if args.json:
        print(json.dumps({"timestamp": ts, "results": [{"model": r["model"], "prompt_id": r["prompt_id"], "score": r["score"]} for r in results], "total_cost_usd": total_cost}, indent=2))


def cmd_report(args):
    """Quality trends with statistics."""
    database = db.connect()
    model_filter = "AND model = ?" if args.model else ""
    params = [args.model] if args.model else []
    limit = args.last or 10

    models = [r[0] for r in database.execute(f"SELECT DISTINCT model FROM runs WHERE 1=1 {model_filter}", params).fetchall()]
    if not models:
        print("No data. Run 'drift run' first.")
        return

    for model in models:
        print(f"\n{display.bold('═' * 70)}\n  {display.bold(model)}\n{'═' * 70}")
        prompts = [r[0] for r in database.execute("SELECT DISTINCT prompt_id FROM runs WHERE model=?", (model,)).fetchall()]
        for pid in prompts:
            scores = [r["score"] for r in database.execute(
                "SELECT score FROM runs WHERE model=? AND prompt_id=? AND score IS NOT NULL ORDER BY timestamp DESC LIMIT ?",
                (model, pid, limit)).fetchall()]
            if not scores:
                continue
            mean, ci_lo, ci_hi = stats.confidence_interval(scores)
            sd = stats.std_dev(scores)
            trend = "→" if len(scores) < 2 else ("↑" if scores[0] > scores[-1] else "↓" if scores[0] < scores[-1] else "→")
            trend_c = display.green(trend) if trend == "↑" else display.red(trend) if trend == "↓" else display.dim(trend)
            spark = display.sparkline(list(reversed(scores)))
            bar = display.bar_chart(scores[0])
            print(f"  {pid:<20} {bar} {scores[0]:>4}/10 {display.dim(f'μ={mean:.1f} σ={sd:.1f} CI=[{ci_lo:.1f},{ci_hi:.1f}]')} {spark} {trend_c}")

        st = database.execute("SELECT AVG(score) as s, AVG(latency_ms) as l, COUNT(*) as n, SUM(cost_usd) as c FROM runs WHERE model=?", (model,)).fetchone()
        cost_str = f"  cost=${st['c']:.4f}" if st['c'] else ""
        avg_s, avg_l, n = st['s'], st['l'], st['n']
        print(f"\n  {display.dim(f'Overall: avg={avg_s:.1f}/10  latency={avg_l:.0f}ms  runs={n}{cost_str}')}")

    if len(models) > 1:
        _print_comparison(database, models)


def cmd_compare(args):
    """Head-to-head model comparison."""
    database = db.connect()
    models = args.models.split(",")
    if len(models) < 2:
        print("Need at least 2 models")
        sys.exit(1)
    _print_comparison(database, models, head_to_head=True)


def cmd_versions(args):
    """Prompt version history and cross-version comparison."""
    database = db.connect()
    pid = args.prompt_id
    query = "SELECT * FROM prompt_versions WHERE prompt_id=? ORDER BY created_at" if pid else "SELECT * FROM prompt_versions ORDER BY prompt_id, created_at"
    versions = database.execute(query, (pid,) if pid else ()).fetchall()
    if not versions:
        print("No versions tracked yet.")
        return

    current_pid = None
    for v in versions:
        if v["prompt_id"] != current_pid:
            current_pid = v["prompt_id"]
            print(f"\n  {display.bold(current_pid)}")
        scores = [r["score"] for r in database.execute("SELECT score FROM runs WHERE prompt_id=? AND prompt_version=? AND score IS NOT NULL", (v["prompt_id"], v["version"])).fetchall()]
        stat_str = ""
        if scores:
            mean, lo, hi = stats.confidence_interval(scores)
            stat_str = f"  n={len(scores)} μ={mean:.1f} CI=[{lo:.1f},{hi:.1f}]"
        text_preview = v["prompt_text"][:60] + ("..." if len(v["prompt_text"]) > 60 else "")
        print(f"    {v['version']}: {display.dim(text_preview)}{stat_str}")

    if pid:
        ver_list = [v["version"] for v in versions]
        if len(ver_list) >= 2:
            print(f"\n  {display.bold('Version Comparison:')}")
            for i in range(len(ver_list) - 1):
                old_scores = [r["score"] for r in database.execute("SELECT score FROM runs WHERE prompt_id=? AND prompt_version=? AND score IS NOT NULL", (pid, ver_list[i])).fetchall()]
                new_scores = [r["score"] for r in database.execute("SELECT score FROM runs WHERE prompt_id=? AND prompt_version=? AND score IS NOT NULL", (pid, ver_list[i+1])).fetchall()]
                if old_scores and new_scores:
                    t_stat, sig = stats.welch_t_test(new_scores, old_scores)
                    direction = "improved" if t_stat > 0 else "regressed"
                    sig_mark = display.bold(" *SIGNIFICANT*") if sig else ""
                    print(f"    {ver_list[i]}→{ver_list[i+1]}: {direction} (t={t_stat:.2f}){sig_mark}")


def cmd_test(args):
    """A/B test two prompt variants."""
    cfg = config.load()
    model = args.model or cfg["models"][0]
    judge_model = cfg.get("judge_model", model)
    n = args.n or 5
    criteria = args.criteria or "Quality, accuracy, helpfulness, and clarity"

    print(f"\n{display.bold('A/B PROMPT TEST')}")
    print(f"  Model: {model} | Runs: {n}\n")
    print(f"  {display.bold('A:')} {args.a[:60]}")
    print(f"  {display.bold('B:')} {args.b[:60]}\n")

    scores_a, scores_b = [], []
    for i in range(n):
        sys.stdout.write(f"  Round {i+1}/{n}...")
        sys.stdout.flush()
        for prompt, scores in [(args.a, scores_a), (args.b, scores_b)]:
            msgs = [{"role": "user", "content": prompt}]
            resp = providers.demo(model, msgs) if args.demo else providers.call(cfg, model, msgs)
            s, _ = scoring.judge(cfg, judge_model, prompt, resp[0], criteria, demo=args.demo)
            if s:
                scores.append(s)
        print(f" A={scores_a[-1] if scores_a else '?'} B={scores_b[-1] if scores_b else '?'}")

    if not scores_a or not scores_b:
        print("\n  No valid scores.")
        return

    mean_a, lo_a, hi_a = stats.confidence_interval(scores_a)
    mean_b, lo_b, hi_b = stats.confidence_interval(scores_b)
    t_stat, sig = stats.welch_t_test(scores_a, scores_b)

    print(f"\n  {display.bold('RESULTS')}")
    print(f"  A: μ={mean_a:.2f} CI=[{lo_a:.1f},{hi_a:.1f}] σ={stats.std_dev(scores_a):.2f}")
    print(f"  B: μ={mean_b:.2f} CI=[{lo_b:.1f},{hi_b:.1f}] σ={stats.std_dev(scores_b):.2f}")
    print(f"  t={t_stat:.3f} | {'SIGNIFICANT' if sig else 'not significant'}")
    if sig:
        winner = "A" if mean_a > mean_b else "B"
        print(f"\n  {display.green(f'★ WINNER: Variant {winner}')} (+{abs(mean_a-mean_b):.1f})")
    else:
        print(f"\n  {display.yellow('≈ No significant difference')}")


def cmd_alert(args):
    """Check for regressions with statistical significance."""
    cfg = config.load()
    database = db.connect()
    threshold = args.threshold or cfg.get("alert_threshold", 15)
    window = args.window or 5
    channels = cfg.get("alert_channels", ["stdout"])

    alerts = []
    for model in [r[0] for r in database.execute("SELECT DISTINCT model FROM runs").fetchall()]:
        for pid in [r[0] for r in database.execute("SELECT DISTINCT prompt_id FROM runs WHERE model=?", (model,)).fetchall()]:
            scores = [r["score"] for r in database.execute(
                "SELECT score FROM runs WHERE model=? AND prompt_id=? AND score IS NOT NULL ORDER BY timestamp DESC LIMIT ?",
                (model, pid, window * 2)).fetchall()]
            if len(scores) < window + 1:
                continue
            recent, older = scores[:window], scores[window:]
            recent_avg, older_avg = sum(recent) / len(recent), sum(older) / len(older)
            if older_avg > 0:
                drop = ((older_avg - recent_avg) / older_avg) * 100
                if drop >= threshold:
                    t_stat, sig = stats.welch_t_test(older, recent)
                    alerts.append({"model": model, "prompt_id": pid, "drop": drop,
                                   "recent_avg": recent_avg, "baseline_avg": older_avg,
                                   "significant": sig, "t_stat": t_stat})

    if not alerts:
        print(f"{display.green('✓')} No quality regressions detected.")
        if args.json:
            print(json.dumps({"alerts": [], "status": "ok"}))
        return

    sig_count = sum(1 for a in alerts if a["significant"])
    msg = f"⚠ {len(alerts)} REGRESSION(S) ({sig_count} significant):\n"
    for a in alerts:
        sig = display.bold(" ★") if a["significant"] else ""
        msg += f"\n  🔴 {a['model']} / {a['prompt_id']}{sig}\n     Drop: {a['drop']:.1f}% ({a['baseline_avg']:.1f} → {a['recent_avg']:.1f})\n"

    for ch in channels:
        if ch == "stdout":
            print(display.red(msg) if sig_count else display.yellow(msg))
        elif ch.startswith("file:"):
            Path(ch[5:]).write_text(msg)

    if args.json:
        print(json.dumps({"alerts": alerts, "status": "regression"}, default=str))
    if sig_count:
        sys.exit(1)


def cmd_check(args):
    """Validate drift.json."""
    if not config.CONFIG_PATH.exists():
        print(f"{display.red('✗')} No config.")
        sys.exit(1)
    cfg = config.load()
    errors, warnings = config.validate(cfg)
    n_prompts = len(cfg.get("prompts", []))
    n_models = len(cfg.get("models", []))
    print(f"\n  {display.bold('Config Validation:')} {config.CONFIG_PATH}")
    print(f"  Prompts: {n_prompts} | Models: {n_models} | Provider: {cfg.get('provider', 'auto')}")
    if errors:
        print(f"\n  {display.red(f'✗ {len(errors)} error(s):')}")
        for e in errors:
            print(f"    • {e}")
    if warnings:
        print(f"\n  {display.yellow(f'⚠ {len(warnings)} warning(s):')}")
        for w in warnings:
            print(f"    • {w}")
    if not errors and not warnings:
        print(f"\n  {display.green('✓ Valid. No issues.')}")
    if errors:
        sys.exit(1)


def cmd_clean(args):
    """Purge old runs, cache, or reset."""
    if args.all:
        os.remove(str(db.DB_PATH))
        print(f"{display.green('✓')} Deleted {db.DB_PATH}")
        return
    database = db.connect()
    if args.cache:
        database.execute("DELETE FROM cache")
        database.commit()
        print(f"{display.green('✓')} Cache cleared")
    elif args.keep:
        total_deleted = 0
        for combo in database.execute("SELECT DISTINCT model, prompt_id FROM runs").fetchall():
            ids = [r["id"] for r in database.execute(
                "SELECT id FROM runs WHERE model=? AND prompt_id=? ORDER BY timestamp DESC LIMIT ?",
                (combo["model"], combo["prompt_id"], args.keep)).fetchall()]
            if ids:
                ph = ",".join("?" * len(ids))
                total_deleted += database.execute(
                    f"DELETE FROM runs WHERE model=? AND prompt_id=? AND id NOT IN ({ph})",
                    [combo["model"], combo["prompt_id"]] + ids).rowcount
        database.commit()
        print(f"{display.green('✓')} Kept last {args.keep} per prompt. Deleted {total_deleted}.")
    elif args.before:
        n = database.execute("DELETE FROM runs WHERE timestamp < ?", (args.before,)).rowcount
        database.commit()
        print(f"{display.green('✓')} Deleted {n} runs before {args.before}")
    else:
        runs = database.execute("SELECT COUNT(*) as n FROM runs").fetchone()["n"]
        cache = database.execute("SELECT COUNT(*) as n FROM cache").fetchone()["n"]
        size = db.DB_PATH.stat().st_size if db.DB_PATH.exists() else 0
        print(f"  Runs: {runs} | Cache: {cache} | Size: {size/1024:.1f}KB")
        print(f"  Options: --cache | --keep N | --before DATE | --all")


def cmd_dashboard(args):
    """TUI dashboard."""
    import time as _time
    database = db.connect()

    def render():
        models = [r[0] for r in database.execute("SELECT DISTINCT model FROM runs").fetchall()]
        if not models:
            return "No data."
        lines = ["\033[2J\033[H",
                 f"  {display.bold('DRIFT DASHBOARD')}  —  {display.dim(datetime.now().strftime('%H:%M:%S'))}\n",
                 f"  {'Model':<20} {'Avg':<7} {'Trend':<14} {'Runs':<6} {'Latency'}"]
        for model in models:
            st = database.execute("SELECT AVG(score) as s, COUNT(*) as n, AVG(latency_ms) as l FROM runs WHERE model=?", (model,)).fetchone()
            scores = [r["score"] for r in database.execute("SELECT score FROM runs WHERE model=? AND score IS NOT NULL ORDER BY timestamp DESC LIMIT 12", (model,)).fetchall()]
            spark = display.sparkline(list(reversed(scores)))
            lines.append(f"  {model:<20} {display.score_color(round(st['s'],1)):<7} {spark:<14} {st['n']:<6} {st['l']:.0f}ms")
        lines.append(f"\n  {display.dim('Ctrl+C to exit')}")
        return "\n".join(lines)

    if args.once:
        print(render())
        return
    try:
        while True:
            print(render(), flush=True)
            _time.sleep(args.refresh)
    except KeyboardInterrupt:
        print()


def cmd_export(args):
    """Export run data as CSV or JSON."""
    database = db.connect()
    rows = database.execute("SELECT * FROM runs ORDER BY timestamp DESC").fetchall()
    if not rows:
        print("No data.")
        return
    cols = ["id", "timestamp", "model", "prompt_id", "prompt_version", "score", "judge_reason", "latency_ms", "tokens_in", "tokens_out", "cost_usd", "tags", "cached"]
    if args.format == "json":
        output = json.dumps([{k: r[k] for k in cols if k in r.keys()} for r in rows], indent=2)
    else:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols if k in r.keys()})
        output = buf.getvalue()
    if args.output:
        Path(args.output).write_text(output)
        print(f"{display.green('✓')} Exported {len(rows)} records to {args.output}")
    else:
        print(output)


def cmd_history(args):
    """Raw run history."""
    database = db.connect()
    limit = args.last or 20
    model_filter = "AND model = ?" if args.model else ""
    params = ([args.model] if args.model else []) + [limit]
    rows = database.execute(f"SELECT timestamp, model, prompt_id, prompt_version, score, latency_ms, cached FROM runs WHERE 1=1 {model_filter} ORDER BY timestamp DESC LIMIT ?", params).fetchall()
    if not rows:
        print("No history.")
        return
    print(f"  {'Timestamp':<22} {'Model':<16} {'Prompt':<20} {'Ver':<5} {'Score':<7} {'Lat':<7} {'Cache'}")
    print(f"  {'─'*90}")
    for r in rows:
        ts = r["timestamp"][:19].replace("T", " ")
        cached = display.cyan("⚡") if r["cached"] else " "
        print(f"  {display.dim(ts):<22} {r['model']:<16} {r['prompt_id']:<20} {r['prompt_version'] or '—':<5} {display.score_color(r['score']):<7} {r['latency_ms']}ms  {cached}")


def cmd_add(args):
    """Add a golden prompt interactively."""
    cfg = config.load()
    pid = args.id or input("  Prompt ID: ").strip()
    if not pid:
        return
    prompt = args.prompt or input("  Prompt text: ").strip()
    if not prompt:
        return
    criteria = args.criteria or input("  Criteria: ").strip()
    tags_str = args.tags or input("  Tags (comma-sep): ").strip()
    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
    checks = []
    if not args.no_checks:
        print(f"  {display.dim('Heuristic checks (empty to finish):')}")
        while True:
            ct = input("    Type: ").strip()
            if not ct:
                break
            val = input("    Value: ").strip()
            if val:
                checks.append({"type": ct, "value": val})
    cfg["prompts"].append({"id": pid, "tags": tags, "prompt": prompt, "criteria": criteria, "checks": checks})
    config.save(cfg)
    print(f"\n{display.green('✓')} Added '{pid}'")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _print_comparison(database, models, head_to_head=False):
    """Print side-by-side model comparison table."""
    placeholders = ",".join("?" * len(models))
    prompts = sorted({r[0] for r in database.execute(f"SELECT DISTINCT prompt_id FROM runs WHERE model IN ({placeholders})", models).fetchall()})
    if not prompts:
        print("No data.")
        return

    if head_to_head:
        print(f"\n{display.bold('HEAD-TO-HEAD COMPARISON')}\n")
    else:
        print(f"\n{display.bold('═' * 70)}\n  {display.bold('COMPARISON')}\n{'═' * 70}")

    wins = {m: 0 for m in models}
    header = f"  {'Prompt':<20}" + "".join(f" {m:<14}" for m in models) + (" Winner" if head_to_head else "")
    print(header)
    print(f"  {'─'*20}" + "".join(f" {'─'*14}" for _ in models))

    for pid in prompts:
        row = f"  {pid:<20}"
        scores = {}
        for m in models:
            r = database.execute("SELECT AVG(score) as s FROM runs WHERE model=? AND prompt_id=?", (m, pid)).fetchone()
            s = round(r["s"], 1) if r["s"] else None
            scores[m] = s
            row += f" {display.score_color(s) if s else display.dim('N/A'):<14}"
        valid = {m: s for m, s in scores.items() if s is not None}
        if valid and head_to_head:
            winner = max(valid, key=valid.get)
            wins[winner] += 1
            row += f" {display.green(winner)}"
        print(row)

    if head_to_head:
        print(f"\n  {display.bold('Wins:')} " + " | ".join(f"{m}: {w}" for m, w in sorted(wins.items(), key=lambda x: -x[1])))


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="drift", description="Track LLM quality over time")
    parser.add_argument("--version", action="version", version=f"drift {__version__}")
    parser.add_argument("--no-color", action="store_true")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("init", help="Create config"); p.add_argument("--force", action="store_true")
    p = sub.add_parser("run", help="Execute golden prompts")
    p.add_argument("--demo", action="store_true"); p.add_argument("--models"); p.add_argument("--tag")
    p.add_argument("--json", action="store_true"); p.add_argument("--force", action="store_true")
    p.add_argument("--parallel", action="store_true")
    p = sub.add_parser("report", help="Quality trends"); p.add_argument("--model"); p.add_argument("--last", type=int); p.add_argument("--json", action="store_true")
    p = sub.add_parser("compare", help="Head-to-head"); p.add_argument("models"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("versions", help="Prompt versions"); p.add_argument("prompt_id", nargs="?")
    p = sub.add_parser("test", help="A/B test prompts")
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--criteria"); p.add_argument("--model"); p.add_argument("--n", type=int)
    p.add_argument("--demo", action="store_true"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("add", help="Add golden prompt")
    p.add_argument("--id"); p.add_argument("--prompt"); p.add_argument("--criteria")
    p.add_argument("--tags"); p.add_argument("--no-checks", action="store_true")
    p = sub.add_parser("alert", help="Check regressions"); p.add_argument("--threshold", type=float); p.add_argument("--window", type=int); p.add_argument("--json", action="store_true")
    p = sub.add_parser("check", help="Validate config")
    p = sub.add_parser("clean", help="Purge data")
    p.add_argument("--cache", action="store_true"); p.add_argument("--keep", type=int)
    p.add_argument("--before"); p.add_argument("--all", action="store_true")
    p = sub.add_parser("dashboard", help="TUI dashboard"); p.add_argument("--refresh", type=int, default=5); p.add_argument("--once", action="store_true")
    p = sub.add_parser("export", help="Export data"); p.add_argument("--format", choices=["csv", "json"], default="csv"); p.add_argument("--output", "-o")
    p = sub.add_parser("history", help="Run history"); p.add_argument("--model"); p.add_argument("--last", type=int); p.add_argument("--json", action="store_true")

    args = parser.parse_args()
    if args.no_color:
        display.NO_COLOR = True
    if not args.command:
        parser.print_help()
        sys.exit(1)

    {"init": cmd_init, "run": cmd_run, "report": cmd_report, "compare": cmd_compare,
     "versions": cmd_versions, "test": cmd_test, "add": cmd_add, "alert": cmd_alert,
     "check": cmd_check, "clean": cmd_clean, "dashboard": cmd_dashboard,
     "export": cmd_export, "history": cmd_history}[args.command](args)
