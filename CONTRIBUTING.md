# Contributing

## Branch Strategy

- `main` is protected — no direct pushes
- All changes go through pull requests
- CI must pass before merge

## Workflow

```bash
# 1. Create feature branch
git checkout main && git pull
git checkout -b dev/my-feature

# 2. Make changes, commit
git add -A && git commit -m "feat: description"

# 3. Push and create PR
git push -u origin dev/my-feature
# Then open PR on GitHub (or use gh CLI)

# 4. After merge, clean up
git checkout main && git pull
git branch -d dev/my-feature
```

## Branch Naming

| Prefix | Use |
|--------|-----|
| `dev/` | Feature development |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |

## Commit Messages

Format: `type: short description`

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

## Testing

Before pushing, verify all commands work:

```bash
rm -f drift.db drift.json
python3 drift.py init
python3 drift.py check
python3 drift.py run --demo
python3 drift.py alert
```
