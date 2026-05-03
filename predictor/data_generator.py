"""
Synthetic training-data generator.

We model each international match using a Poisson goal model whose
expected-goal rates are derived from each team's ELO rating.  This
produces a realistic distribution of wins, draws and losses that an
ML classifier can learn from.

Reference
---------
Maher, M.J. (1982). Modelling association football scores.
Statistica Neerlandica, 36(3), 109-118.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np

# Average goals per team per match (calibrated to international football)
_BASE_GOALS = 1.25


def _expected_goals(elo_a: float, elo_b: float) -> tuple[float, float]:
    """Return (λ_a, λ_b) – Poisson goal-rate parameters."""
    # Win probability for team A given their ELO difference
    p_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    p_b = 1.0 - p_a
    # Scale so that equal teams both expect _BASE_GOALS
    lambda_a = _BASE_GOALS * 2.0 * p_a
    lambda_b = _BASE_GOALS * 2.0 * p_b
    return lambda_a, lambda_b


def generate_training_data(
    n_samples: int = 10_000,
    elo_range: tuple[int, int] = (1300, 2100),
    elo_step: int = 25,
    knockout_fraction: float = 0.35,
    seed: int | None = 42,
) -> list[tuple[float, float, int, int]]:
    """
    Generate synthetic match data for training the ML model.

    Parameters
    ----------
    n_samples:
        Number of matches to simulate.
    elo_range:
        (min, max) for randomly sampled ELO ratings.
    elo_step:
        Granularity of ELO ratings in the pool.
    knockout_fraction:
        Fraction of generated matches that are knockout matches.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    List of ``(elo_a, elo_b, result, is_knockout)`` tuples where
    *result* is ``2`` (win for A), ``1`` (draw) or ``0`` (loss for A).
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    lo, hi = elo_range
    elo_pool = list(range(lo, hi + 1, elo_step))

    data: list[tuple[float, float, int, int]] = []

    for _ in range(n_samples):
        elo_a = float(random.choice(elo_pool) + rng.integers(-25, 26))
        elo_b = float(random.choice(elo_pool) + rng.integers(-25, 26))

        lambda_a, lambda_b = _expected_goals(elo_a, elo_b)
        goals_a = int(rng.poisson(lambda_a))
        goals_b = int(rng.poisson(lambda_b))
        is_knockout = int(random.random() < knockout_fraction)
        if is_knockout:
            lambda_a *= 0.9  # fewer goals in knockout matches
            lambda_b *= 0.9 # (calibration based on historical data)

        if goals_a > goals_b:
            result = 2  # win for A
        elif goals_a < goals_b:
            result = 0  # loss for A
        else:
            result = 1  # draw

        data.append((elo_a, elo_b, result, is_knockout))

    return data
