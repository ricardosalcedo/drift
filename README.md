# drift 🔍

**Is your LLM getting worse?** Track quality over time with golden prompts.

Define golden prompts with grading criteria. Run them against models on a schedule. Get alerted when quality drops — with statistical significance testing.

## Install

```bash
pip install httpx  # only required dependency
# Optional: pip install boto3 (for Bedrock provider)
```

## Quick Start

```bash
python3 drift.py init                              # 5 golden prompts, 4 categories
python3 drift.py run --demo                        # no API key needed
python3 drift.py run --demo --models "gpt-4o-mini,gpt-4o,claude-3-5-sonnet"
python3 drift.py report                            # trends + statistics + comparison
python3 drift.py dashboard --once                  # TUI overview
python3 drift.py alert                             # regression check with significance
```

## Commands

| Command | Description |
|---------|-------------|
| `drift init` | Create config with sample golden prompts |
| `drift run [--demo] [--tag X] [--force]` | Execute prompts (--force ignores cache) |
| `drift report` | Quality trends with CI, std dev, sparklines |
| `drift compare "m1,m2"` | Head-to-head with win tracking |
| `drift versions [prompt_id]` | Prompt version history + significance test |
| `drift dashboard [--once]` | TUI with sparklines, model table, live refresh |
| `drift add` | Interactively add a golden prompt |
| `drift alert` | Regression detection with Welch's t-test |
| `drift export [--format csv\|json]` | Export all data |
| `drift history` | Raw run log |

## Features

### Core
- **Dual scoring** — LLM-as-judge + heuristic regex/contains checks
- **Multi-model comparison** — side-by-side, trend arrows, win tracking
- **Prompt tags** — group by category (`reasoning`, `code`, `format`, `safety`)

### v0.3
- **Prompt versioning** — auto-detects text changes, tracks scores per version
- **Statistical confidence** — std dev, confidence intervals, Welch's t-test
- **Response caching** — skip unchanged prompts (configurable TTL, `--force` to override)
- **TUI dashboard** — sparklines, model table, live refresh
- **Native providers** — Anthropic, Ollama, Bedrock (no proxy needed)
- **Provider auto-detection** — routes based on model name

### Infrastructure
- **Cost tracking** — per-run and cumulative with token-level pricing
- **Retry logic** — exponential backoff for rate limits and transient errors
- **Colored output** — respects `NO_COLOR` env / `--no-color` flag
- **JSON output** — `--json` on every command for CI/scripting
- **Alert channels** — stdout, `file:path`, or webhook URL
- **Export** — CSV or JSON dump of all run data

## Providers

Drift auto-detects the provider from model name, or you can set it explicitly:

| Provider | Config | Models |
|----------|--------|--------|
| OpenAI (default) | `base_url` + `api_key_env` | gpt-4o, gpt-4o-mini, etc. |
| Anthropic | `ANTHROPIC_API_KEY` env | claude-3-5-sonnet, claude-3-opus, etc. |
| Ollama | `ollama_url` (default: localhost:11434) | llama3, mistral, etc. |
| Bedrock | `bedrock_region` + AWS creds | us.anthropic.*, amazon.*, meta.* |
| Any OpenAI-compatible | Set `base_url` | vLLM, LiteLLM, etc. |

## Configuration

```json
{
  "version": 3,
  "provider": "auto",
  "models": ["gpt-4o-mini", "claude-3-5-sonnet"],
  "judge_model": "gpt-4o-mini",
  "base_url": "https://api.openai.com/v1",
  "api_key_env": "OPENAI_API_KEY",
  "ollama_url": "http://localhost:11434",
  "bedrock_region": "us-west-2",
  "alert_threshold": 15,
  "alert_channels": ["stdout", "file:drift-alerts.log"],
  "cache_hours": 24,
  "prompts": [...]
}
```

## Statistics

Report shows per-prompt:
- **μ** — mean score
- **σ** — standard deviation
- **CI** — 95% confidence interval (t-distribution)
- **Sparkline** — visual trend of recent scores

Alert uses **Welch's t-test** to determine if a regression is statistically significant (p<0.05). Only significant regressions cause non-zero exit.

## Prompt Versioning

When you change a prompt's text or criteria in `drift.json`, drift auto-detects the change and creates a new version. View history with:

```bash
drift versions                    # all prompts
drift versions reasoning-basic    # specific prompt + cross-version comparison
```

## Automation

```bash
# Daily cron with caching (skip unchanged prompts)
0 6 * * * cd ~/drift && python3 drift.py run && python3 drift.py alert

# Force fresh evaluation (ignore cache)
python3 drift.py run --force

# CI gate
python3 drift.py run && python3 drift.py alert --threshold 10 --json
```

## Design

- **Single file** — `drift.py` (894 lines), zero package structure
- **SQLite** — local storage with auto-migration
- **Minimal deps** — only `httpx` required (boto3 optional for Bedrock)
- **No daemon** — use cron/systemd for scheduling
- **Exit codes** — `alert` returns 1 only on *statistically significant* regression

## License

MIT
