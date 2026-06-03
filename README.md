# drift 🔍

**Is your LLM getting worse?** Track quality over time with golden prompts.

Define golden prompts with grading criteria. Run them against models on a schedule. Get alerted when quality drops.

## Install

```bash
pip install httpx  # only runtime dependency
```

## Quick Start

```bash
# Initialize with sample golden prompts
python3 drift.py init

# Run in demo mode (no API key needed)
python3 drift.py run --demo

# View report
python3 drift.py report

# Check for regressions
python3 drift.py alert
```

## Commands

| Command | Description |
|---------|-------------|
| `drift init` | Create `drift.json` config with sample golden prompts |
| `drift run` | Execute prompts against configured models |
| `drift report` | Show quality trends per model/prompt |
| `drift alert` | Check for regressions (non-zero exit = regression) |
| `drift history` | Show raw run log |

## How It Works

1. **Define golden prompts** in `drift.json` with quality criteria
2. **Run evaluations** — each prompt is sent to each model
3. **Score responses** using:
   - **LLM-as-judge**: another model rates the response 0-10
   - **Heuristic checks**: regex, contains, length constraints
   - Combined score = average of both
4. **Track over time** — scores stored in SQLite (`drift.db`)
5. **Alert on regression** — detects drops exceeding threshold

## Configuration

```json
{
  "models": ["gpt-4o-mini", "gpt-4o"],
  "judge_model": "gpt-4o-mini",
  "base_url": "https://api.openai.com/v1",
  "api_key_env": "OPENAI_API_KEY",
  "alert_threshold": 15,
  "prompts": [
    {
      "id": "reasoning-basic",
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

### Heuristic Check Types

| Type | Description |
|------|-------------|
| `contains` | Response contains string (case-insensitive) |
| `not_contains` | Response does NOT contain string |
| `regex` | Response matches regex pattern |
| `min_length` | Response is at least N chars |
| `max_length` | Response is at most N chars |

## Automation

```bash
# Cron: run daily at 6am, alert on regression
0 6 * * * cd /path/to/drift && python3 drift.py run && python3 drift.py alert

# CI: gate deployments on quality
python3 drift.py run --models gpt-4o-mini && python3 drift.py alert --threshold 10
```

## Demo Mode

Test without API keys:

```bash
python3 drift.py init
python3 drift.py run --demo
python3 drift.py run --demo  # run again to build history
python3 drift.py report
python3 drift.py alert
```

## Design

- **Single file** — `drift.py`, no package structure needed
- **SQLite** — zero-setup local storage
- **OpenAI-compatible** — works with any provider (OpenAI, Anthropic via proxy, Ollama, vLLM)
- **No daemon** — use cron/systemd for scheduling
- **Exit codes** — `alert` returns non-zero on regression (CI-friendly)

## License

MIT
