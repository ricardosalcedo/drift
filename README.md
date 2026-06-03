# drift 🔍

**Is your LLM getting worse?** Track quality over time with golden prompts.

Define golden prompts with grading criteria. Run them against models on a schedule. Get alerted when quality drops — with statistical significance testing.

## Install

```bash
# From source
pip install httpx
git clone https://github.com/ricardosalcedo/drift.git
cd drift && python3 drift.py --help

# As package (pip install from PyPI — coming soon)
pip install drift-llm
drift --help

# Optional providers
pip install boto3  # for AWS Bedrock
```

## Quick Start

```bash
drift init                                          # 5 golden prompts, 4 categories
drift run --demo                                    # no API key needed
drift run --demo --models "gpt-4o-mini,gpt-4o,claude-3-5-sonnet"
drift report                                        # trends + CI + comparison
drift compare "gpt-4o,gpt-4o-mini"                  # head-to-head
drift test --a "Explain X" --b "What is X?" --demo  # A/B test
drift dashboard --once                              # TUI overview
drift alert                                         # regression check
```

## Commands

| Command | Description |
|---------|-------------|
| `drift init [--force]` | Create `drift.json` with sample golden prompts |
| `drift run [--demo] [--models M] [--tag T] [--parallel] [--force]` | Execute prompts against models |
| `drift report [--model M] [--last N]` | Quality trends with CI, sparklines, comparison |
| `drift compare "m1,m2,..."` | Head-to-head with win tracking |
| `drift test --a "..." --b "..." [--n 5] [--model M]` | A/B test two prompt variants |
| `drift versions [prompt_id]` | Prompt version history + significance test |
| `drift add` | Interactively add a golden prompt |
| `drift alert [--threshold N] [--window N]` | Regression detection with Welch's t-test |
| `drift check` | Validate `drift.json` (schema, regexes, missing fields) |
| `drift clean [--cache\|--keep N\|--before DATE\|--all]` | Purge old data |
| `drift dashboard [--once] [--refresh N]` | TUI with sparklines and live refresh |
| `drift export [--format csv\|json] [-o file]` | Export all run data |
| `drift history [--model M] [--last N]` | Raw run log with versions and cache indicator |

All commands support `--json` for machine-readable output.

## Features

- **Dual scoring** — LLM-as-judge + heuristic regex/contains checks, combined
- **Multi-model comparison** — side-by-side scores, trend arrows, win tracking
- **A/B prompt testing** — test two prompt variants with statistical significance
- **Prompt versioning** — auto-detects text changes, tracks scores per version
- **Statistical confidence** — μ, σ, 95% CI (t-distribution), Welch's t-test
- **Response caching** — skip unchanged prompts (configurable TTL, `--force` to override)
- **Parallel execution** — `--parallel` runs evaluations concurrently via ThreadPool
- **Native providers** — OpenAI, Anthropic, Ollama, Bedrock (auto-detected from model name)
- **TUI dashboard** — sparklines, model table, live refresh
- **Cost tracking** — per-run and cumulative with token-level pricing
- **Retry logic** — exponential backoff for rate limits and transient errors
- **Config validation** — `drift check` catches schema errors before you waste API calls
- **Alert channels** — stdout, `file:path`, or webhook URL
- **Colored output** — respects `NO_COLOR` env / `--no-color` flag
- **Prompt tags** — filter runs by category (`reasoning`, `code`, `format`, `safety`)

## Providers

Auto-detected from model name, or set `"provider"` explicitly in config:

| Provider | Config | Models |
|----------|--------|--------|
| OpenAI (default) | `base_url` + `api_key_env` | gpt-4o, gpt-4o-mini, etc. |
| Anthropic | `ANTHROPIC_API_KEY` env | claude-3-5-sonnet, claude-3-opus, etc. |
| Ollama | `ollama_url` (default: localhost:11434) | llama3, mistral, etc. |
| Bedrock | `bedrock_region` + AWS creds | us.anthropic.*, amazon.*, meta.* |
| Any OpenAI-compatible | Set `base_url` | vLLM, LiteLLM, Together, etc. |

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

## Statistics

Report shows per-prompt:
- **μ** — mean score | **σ** — standard deviation
- **CI** — 95% confidence interval (t-distribution)
- **Sparkline** — visual trend of recent scores

Alert uses **Welch's t-test** — only *statistically significant* regressions (p<0.05) cause non-zero exit code.

## A/B Testing

Compare two prompt variants head-to-head:

```bash
drift test \
  --a "Explain photosynthesis in 2 sentences" \
  --b "How do plants make food? Answer in 2 sentences" \
  --model gpt-4o-mini --n 10
```

Reports mean, CI, and Welch's t-test significance for each variant.

## Prompt Versioning

When you change a prompt's text or criteria in `drift.json`, drift auto-detects the change and creates a new version:

```bash
drift versions                    # all prompts
drift versions reasoning-basic    # specific prompt + cross-version t-test
```

## Automation

```bash
# Daily cron
0 6 * * * cd ~/drift && drift run && drift alert

# Parallel (faster for many models)
drift run --parallel --models "gpt-4o-mini,gpt-4o,claude-3-5-sonnet"

# CI gate
drift run && drift alert --threshold 10 --json

# GitHub Actions — see .github/workflows/drift.yml
```

## Project Structure

```
drift/
├── __init__.py      # Version
├── __main__.py      # python -m drift
├── cli.py           # 13 commands + argument parsing
├── config.py        # Load, save, validate, defaults
├── db.py            # Schema, migrations, versioning, cache
├── display.py       # Colors, sparkline, bar chart
├── providers.py     # OpenAI, Anthropic, Ollama, Bedrock, demo
├── scoring.py       # LLM-as-judge, heuristics, combined evaluate()
└── stats.py         # std_dev, confidence_interval, welch_t_test
```

## Design Principles

- **Modular** — each module has one responsibility, no circular deps
- **DRY** — shared scoring, display, and comparison logic
- **Minimal deps** — only `httpx` required (boto3 optional)
- **SQLite** — zero-setup local storage with auto-migration
- **No daemon** — use cron/systemd for scheduling
- **CI-friendly** — JSON output, non-zero exit on significant regression

## License

MIT
