"""Statistical analysis: confidence intervals, t-tests, standard deviation."""
import math

# t-values for 95% CI (small sample sizes)
_T_VALUES = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def std_dev(scores):
    """Sample standard deviation."""
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    return math.sqrt(sum((x - mean) ** 2 for x in scores) / (len(scores) - 1))


def confidence_interval(scores, confidence=0.95):
    """Return (mean, lower, upper) 95% CI using t-distribution."""
    n = len(scores)
    if n < 2:
        return sum(scores) / max(n, 1), 0, 10
    mean = sum(scores) / n
    se = std_dev(scores) / math.sqrt(n)
    t = _T_VALUES.get(n, 1.96)
    margin = t * se
    return mean, max(0, mean - margin), min(10, mean + margin)


def welch_t_test(group1, group2):
    """Welch's t-test. Returns (t_stat, significant_at_p05)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0, False
    m1, m2 = sum(group1) / n1, sum(group2) / n2
    s1, s2 = std_dev(group1), std_dev(group2)
    se = math.sqrt(s1**2 / n1 + s2**2 / n2)
    if se == 0:
        return 0.0, False
    t_stat = (m1 - m2) / se
    return t_stat, abs(t_stat) > 2.0
