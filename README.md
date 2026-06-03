# drift 🔍

**Is your LLM getting worse?** Track quality over time with golden prompts.

Define golden prompts with grading criteria. Run them against models on a schedule. Get alerted when quality drops.

## Install

```bash
pip install httpx  # only runtime dependency
```

## Quick Start

```bash
python3 drift.py init                    # Create config with 5 sample prompts
python3 drift.py run --demo              # Test without API keys
python3 drift.py run --demo --models "gpt-4o-mini,gpt-4o,claude-3-5-sonnet"
python3 drift.py report                  # Quality trends + model comparison
python3 drift.py compare "gpt-4o,gpt-4o-mini"  # Head-to-head
python3 drift.py alert                   # Check for regressions
```

## Commands

| Command | Description |
|---------|-------------|
| `drift init` | Create `drift.json` with sample golden prompts |
| `drift run` | Execute prompts against models |
| `drift run --tag reasoning` | Run only prompts with specific tag |
| `drift report` | Quality trends with model comparison |
| `drift compare "m1,m2"` | Head-to-head with win tracking |
| `drift add` | Interactively add a golden prompt |
| `drift alert` | Regression detection (non-zero exit = alert) |
| `drift export` | Export as CSV or JSON |
| `drift history` | Raw run log |

## Features

- **Multi-model comparison** — side-by-side scores, trend arrows, winner tracking
- **Cost tracking** — per-run and cumulative cost with token-level pricing
- **Prompt tags** — group by category (`reasoning`, `code`, `format`, `safety`)
- **Dual scoring** — LLM-as-judge + heuristic regex/contains checks
- **Retry logic** — exponential backoff for rate limits and transient errors
- **Colored output** — respects `NO_COLOR` env, `--no-color` flag, and non-tty
- **JSON output** — `--json` flag on every command for CI/scripting
- **Alert channels** — stdout, file, or webhook URL
- **Export** — CSV or JSON dump of all run data
- **Demo mode** — full functionality without API keys

## Configuration

```json
{
  "version": 2,
  "models": ["gpt-4o-mini", "gpt-4o"],
  "judge_model": "gpt-4o-mini",
  "base_url": "https://api.openai.com/v1",
  "api_key_env": "OPENAI_API_KEY",
  "alert_threshold": 15,
  "alert_channels": ["stdout", "file:drift-alerts.log", "https://hooks.slack.com/..."],
  "prompts": [
    {
      "id": "reasoning-basic",
      "tags": ["reasoning"],
      "prompt": "Explain why the sky is blue in 3 sentences.",
      "criteria": "Scientifically accurate, exactly 3 sentences",
      "checks": [
        {"type": "contains", "value": "scatter"},
        {"type": "regex", "value": "\\. .+\\. .+\\."}
      ]
    }
  ]
}
```

### Heuristic Checks

| Type | Description |
|------|-------------|
| `contains` | Case-insensitive substring match |
| `not_contains` | Must NOT contain string |
| `regex` | Match regex pattern |
| `min_length` | Minimum character count |
| `max_length` | Maximum character count |

### Supported Pricing Models

gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, claude-3-5-sonnet, claude-3-haiku, claude-3-opus (prefix-matched, so `gpt-4o-2024-08-06` works too)

## Automation

```bash
# Cron: daily quality check
0 6 * * * cd ~/drift && python3 drift.py run && python3 drift.py alert

# CI gate: fail if quality drops
python3 drift.py run && python3 drift.py alert --threshold 10 --json

# Export for dashboards
python3 drift.py export --format json -o /tmp/drift-data.json
```

## Design

- **Single file** — `drift.py`, zero package structure
- **SQLite** — zero-setup local storage with auto-migration
- **OpenAI-compatible** — works with any provider (OpenAI, Anthropic, Ollama, vLLM)
- **No daemon** — use cron/systemd for scheduling
- **Exit codes** — `alert` returns 1 on regression (CI-friendly)
- **Minimal deps** — only `httpx` (stdlib otherwise)

## License

MIT
