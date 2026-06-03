"""Terminal colors, formatting, and display helpers."""
import os, sys

NO_COLOR = os.environ.get("NO_COLOR") or not sys.stdout.isatty()


def _c(text, code):
    return str(text) if NO_COLOR else f"\033[{code}m{text}\033[0m"


def red(t): return _c(t, "31")
def green(t): return _c(t, "32")
def yellow(t): return _c(t, "33")
def blue(t): return _c(t, "34")
def cyan(t): return _c(t, "36")
def dim(t): return _c(t, "2")
def bold(t): return _c(t, "1")


def score_color(score):
    """Color a score value based on quality threshold."""
    if (score or 0) >= 7: return green(f"{score}/10")
    if (score or 0) >= 5: return yellow(f"{score}/10")
    return red(f"{score}/10")


def sparkline(values, width=12):
    """Unicode sparkline from numeric values."""
    if not values:
        return " " * width
    chars = "▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn or 1
    return "".join(
        chars[min(len(chars) - 1, int((v - mn) / rng * (len(chars) - 1)))]
        for v in values[-width:]
    )


def bar_chart(score):
    """Simple 10-char bar chart."""
    n = int(score or 0)
    return green("█") * n + dim("░") * (10 - n)
