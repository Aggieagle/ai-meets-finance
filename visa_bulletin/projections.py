"""Visa bulletin projection utilities.

This module contains a small, dependency-free projection routine that improves
on naive month-to-month averages by:

- Sorting historical bulletins so irregular input order does not skew results.
- Using a sliding window of recent bulletins instead of the full history.
- Comparing several estimators (trimmed median/mean, arithmetic mean, EWMA)
  against the recent history and choosing the one that best backtests.
- Applying caps tied to recent distributional behavior to dampen the effect of
  one-off retrogressions or big jumps.
- Allowing category-level projections via :func:`project_categories`.

The functions are intentionally conservative: if the most recent cutoff date is
``"C"`` (current) or if there is not enough movement data, the projection will
return ``None`` rather than guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from statistics import mean, median, quantiles
from typing import Callable, Dict, Iterable, List, Optional, Sequence


@dataclass
class BulletinEntry:
    """A single category cutoff inside a monthly bulletin.

    Attributes
    ----------
    bulletin_month:
        The month of the bulletin (as ``YYYY-MM``) used for sorting.
    cutoff:
        The category cutoff date (``YYYY-MM-DD``) or ``"C"`` for current.
    """

    bulletin_month: str
    cutoff: str

    def bulletin_date(self) -> date:
        return datetime.strptime(self.bulletin_month + "-01", "%Y-%m-%d").date()

    def cutoff_date(self) -> Optional[date]:
        value = self.cutoff.strip().upper()
        if value == "C":
            return None
        return datetime.strptime(value, "%Y-%m-%d").date()


def _sorted_history(history: Iterable[BulletinEntry]) -> List[BulletinEntry]:
    return sorted(history, key=lambda entry: entry.bulletin_date())


def _recent_movements(history: Sequence[BulletinEntry], window: int = 6) -> List[int]:
    movements: List[int] = []
    for prev, current in zip(history, history[1:]):
        prev_cutoff = prev.cutoff_date()
        current_cutoff = current.cutoff_date()
        if prev_cutoff is None or current_cutoff is None:
            # Cannot derive movement if either month is current/unknown.
            continue
        movement_days = (current_cutoff - prev_cutoff).days
        movements.append(movement_days)
    # Focus on the most recent window to avoid stale, misleading data.
    return movements[-window:]


def _trimmed_median(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) > 2:
        ordered = ordered[1:-1]
    return float(median(ordered))


def _trimmed_mean(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) > 2:
        ordered = ordered[1:-1]
    return float(mean(ordered))


def _ewma(values: Sequence[int], alpha: float = 0.65) -> float:
    """Exponentially weighted moving average with heavier emphasis on recency."""

    if not values:
        return 0.0
    estimate = float(values[0])
    for value in values[1:]:
        estimate = alpha * float(value) + (1 - alpha) * estimate
    return estimate


def _select_estimator(movements: Sequence[int]) -> Callable[[Sequence[int]], float]:
    """Choose the estimator that best backtests against the observed movements."""

    candidates: List[tuple[str, Callable[[Sequence[int]], float]]] = [
        ("trimmed_median", _trimmed_median),
        ("trimmed_mean", _trimmed_mean),
        ("median", lambda vals: float(median(vals)) if vals else 0.0),
        ("mean", lambda vals: float(mean(vals)) if vals else 0.0),
        ("ewma", _ewma),
    ]

    if len(movements) < 2:
        return _trimmed_median

    scores: Dict[str, float] = {name: 0.0 for name, _ in candidates}
    for idx in range(1, len(movements)):
        training = movements[:idx]
        actual = movements[idx]
        for name, estimator in candidates:
            predicted = estimator(training)
            scores[name] += abs(predicted - actual)

    preference = ["trimmed_median", "trimmed_mean", "median", "mean", "ewma"]
    best_name = min(scores, key=lambda name: (scores[name], preference.index(name)))
    return dict(candidates)[best_name]


def _movement_cap(movements: Sequence[int]) -> int:
    if not movements:
        return 30
    if len(movements) < 2:
        return max(30, movements[-1])
    q3 = quantiles(movements, n=4, method="inclusive")[2]
    return max(30, int(round(q3)))


def project_cutoff_date(
    history: Iterable[BulletinEntry],
    months_ahead: int = 1,
    window: int = 6,
) -> Optional[date]:
    """Project a future cutoff date for a single category.

    Parameters
    ----------
    history:
        Iterable of :class:`BulletinEntry` objects. They can be in any order; the
        function sorts them chronologically.
    months_ahead:
        Number of bulletins to project into the future (defaults to the next
        bulletin).
    window:
        Number of most recent month-to-month movements to use when calculating
        the trimmed median.

    Returns
    -------
    ``datetime.date`` if a projection can be made, otherwise ``None`` (for
    example, if the category is current or no recent movement exists).
    """

    sorted_history = _sorted_history(list(history))
    if len(sorted_history) < 2:
        return None

    last_cutoff = sorted_history[-1].cutoff_date()
    if last_cutoff is None:
        return None

    movements = _recent_movements(sorted_history, window=window)
    if not movements:
        return None

    estimator = _select_estimator(movements)
    daily_change = estimator(movements)

    # Keep projections conservative by capping to the recent distribution and
    # the latest observed movement. This prevents early-year jumps from
    # overstating forward progress in later projections while allowing
    # accelerating categories to break past the fixed 30-day ceiling when the
    # data supports it.
    last_movement = movements[-1]
    bounded_change = min(daily_change, last_movement, _movement_cap(movements))
    projected_days = int(round(bounded_change * months_ahead))
    return last_cutoff + timedelta(days=projected_days)


def project_categories(
    histories: Dict[str, Iterable[BulletinEntry]],
    months_ahead: int = 1,
    window: int = 6,
) -> Dict[str, Optional[date]]:
    """Project cutoff dates for multiple categories.

    Parameters
    ----------
    histories:
        Mapping of category name to an iterable of bulletin entries for that
        category.
    months_ahead:
        Number of bulletins to look ahead.
    window:
        Window size for the trimmed median calculation.

    Returns
    -------
    Dictionary mapping category name to a projected ``date`` or ``None``.
    """

    projections: Dict[str, Optional[date]] = {}
    for category, category_history in histories.items():
        projections[category] = project_cutoff_date(
            category_history, months_ahead=months_ahead, window=window
        )
    return projections
