"""0-100 reporting scale.

The judge still scores on each case's native rubric points (the rubric texts it reads embed them); the
rescale happens here, at scoring/reporting time, from the stored per-item points:
- factor = 100 / M, M = sum of the case's rubric item points;
- points_100[k] = points[k] * factor; trap and overuse penalties are scaled by the same factor;
- total_100 = sum of the unrounded scaled items and penalties, rounded half-up to one decimal;
- derived_100 = 0 if any required_full item is below its full points, else total_100.
Pass = derived != 0, unchanged; there is no min_total or percent threshold.
"""
from decimal import ROUND_HALF_UP, Decimal

SCALE_MAX = 100


def round1(x: float) -> float:
    """Half-up rounding to one decimal; the 9-decimal string drops binary noise such as 6.2499999999."""
    return float(Decimal(f"{x:.9f}").quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def rescale(points: dict, total: float, rubric: dict, core: list) -> dict:
    """points: per-item native points (keys = rubric keys); total: native total incl. penalties;
    rubric: case.yaml scoring.rubric; core: pass_rule.required_full."""
    assert set(points) == set(rubric), (sorted(points), sorted(rubric))
    max_native = sum(v["points"] for v in rubric.values())
    factor = SCALE_MAX / max_native
    scaled = {k: points[k] * factor for k in rubric}
    penalty = total - sum(points.values())          # trap + overuse penalties, native (<= 0)
    total_100 = round1(sum(scaled.values()) + penalty * factor)
    core_full = all(points[k] == rubric[k]["points"] for k in core)
    derived_100 = total_100 if core_full else 0
    # pass = derived != 0 must not change with the scale
    assert (derived_100 != 0) == ((total if core_full else 0) != 0), (total, total_100, core_full)
    return {"max_native": max_native, "points_100": {k: round1(v) for k, v in scaled.items()},
            "total_100": total_100, "derived_100": derived_100}
