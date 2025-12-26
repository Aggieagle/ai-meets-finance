from datetime import date

import pytest

from visa_bulletin.projections import BulletinEntry, project_categories, project_cutoff_date


def test_projection_uses_recent_trimmed_median():
    history = [
        BulletinEntry("2024-01", "2023-09-01"),
        BulletinEntry("2024-02", "2023-10-01"),
        BulletinEntry("2024-03", "2023-11-15"),
        BulletinEntry("2024-04", "2024-01-01"),
        # Large jump that should be trimmed out
        BulletinEntry("2024-05", "2024-03-15"),
        BulletinEntry("2024-06", "2024-04-15"),
    ]

    projected = project_cutoff_date(history, months_ahead=1, window=4)
    assert projected == date(2024, 5, 16)


def test_projection_respects_current_cutoff():
    history = [
        BulletinEntry("2024-01", "2023-09-01"),
        BulletinEntry("2024-02", "C"),
    ]
    assert project_cutoff_date(history) is None


def test_category_projection_handles_multiple_categories():
    histories = {
        "F1": [
            BulletinEntry("2024-01", "2020-01-01"),
            BulletinEntry("2024-02", "2020-02-01"),
        ],
        "F2A": [
            BulletinEntry("2024-01", "C"),
            BulletinEntry("2024-02", "C"),
        ],
    }

    projections = project_categories(histories)
    assert projections["F1"] == date(2020, 3, 3)
    assert projections["F2A"] is None


@pytest.mark.parametrize(
    "months_ahead, expected",
    [
        (1, date(2024, 3, 31)),
        (2, date(2024, 5, 15)),
    ],
)
def test_projection_multiple_months(months_ahead, expected):
    history = [
        BulletinEntry("2024-01", "2023-09-01"),
        BulletinEntry("2024-02", "2023-10-01"),
        BulletinEntry("2024-03", "2023-11-15"),
        BulletinEntry("2024-04", "2024-01-01"),
        BulletinEntry("2024-05", "2024-02-15"),
    ]

    assert project_cutoff_date(history, months_ahead=months_ahead, window=3) == expected


def test_projection_selects_best_estimator_for_trend():
    history = [
        BulletinEntry("2024-01", "2024-01-01"),
        BulletinEntry("2024-02", "2024-01-11"),
        BulletinEntry("2024-03", "2024-01-31"),
        BulletinEntry("2024-04", "2024-03-11"),
        BulletinEntry("2024-05", "2024-05-30"),
    ]

    # Exponentially weighted averages should win the backtest here and allow a
    # larger capped projection that follows the accelerating trend.
    projected = project_cutoff_date(history, months_ahead=1, window=5)
    assert projected == date(2024, 7, 19)
