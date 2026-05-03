"""
Machine-learning model for predicting World Cup match outcomes.

Architecture
------------
A scikit-learn Pipeline composed of:
  1. ``StandardScaler``  – normalises features to zero mean / unit variance.
  2. ``LogisticRegression`` (multinomial) – predicts win / draw / loss
     probabilities.

Features
--------
For each match (team A vs team B):

  * ``elo_diff``       – signed ELO difference (A minus B).
  * ``elo_diff_sq``    – squared ELO difference (captures non-linearity).
  * ``elo_avg``        – mean of both ratings (proxy for match quality).
  * ``is_knockout``    – 1 if the match is a knockout tie, 0 otherwise.
  

Class labels
------------
  0 → loss for team A
  1 → draw
  2 → win  for team A
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_LOSS = 0
_DRAW = 1
_WIN = 2


class MatchPredictor:
    """ML-based match-outcome predictor.

    Usage
    -----
    >>> from predictor.data_generator import generate_training_data
    >>> predictor = MatchPredictor()
    >>> predictor.fit(generate_training_data())
    >>> p_win, p_draw, p_loss = predictor.predict_proba(1900, 1700)
    """

    LOSS = _LOSS
    DRAW = _DRAW
    WIN = _WIN

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        solver="lbfgs",
                        max_iter=max_iter,
                    ),
                ),
            ]
        )
        self._trained = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_features(
        elo_a: float, elo_b: float, is_knockout: bool = False
    ) -> list[float]:
        diff = elo_a - elo_b
        return [diff, diff**2, abs(diff), (elo_a + elo_b) / 2.0, float(is_knockout)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        training_data: Sequence[tuple[float, float, int, int]],
    ) -> "MatchPredictor":
        """Train the model.

        Parameters
        ----------
        training_data:
            Iterable of ``(elo_a, elo_b, result, is_knockout)`` tuples
            as produced by :func:`~predictor.data_generator.generate_training_data`.
        """
        X = [self._build_features(ea, eb, bool(ko)) for ea, eb, _, ko in training_data]
        y = [r for _, _, r, _ in training_data]
        self._pipeline.fit(X, y)
        self._trained = True
        return self

    def predict_proba(
        self, elo_a: float, elo_b: float, is_knockout: bool = False
    ) -> tuple[float, float, float]:
        """Return ``(p_win, p_draw, p_loss)`` for team A.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before predicting. Call fit() first.")

        features = [self._build_features(elo_a, elo_b, is_knockout)]
        proba = self._pipeline.predict_proba(features)[0]
        classes: list[int] = list(self._pipeline.classes_)

        proba_map = {cls: p for cls, p in zip(classes, proba)}
        return (
            proba_map.get(_WIN, 0.0),
            proba_map.get(_DRAW, 0.0),
            proba_map.get(_LOSS, 0.0),
        )

    def simulate_match(
        self, elo_a: float, elo_b: float, is_knockout: bool = False
    ) -> str:
        """Stochastically simulate a single match.

        Returns
        -------
        ``'W'`` – team A wins, ``'D'`` – draw, ``'L'`` – team A loses.

        In a *knockout* match a draw is impossible: if the ML model
        assigns a draw probability the tie is resolved via a
        simulated penalty shootout (coin-flip on the remaining
        probability mass).
        """
        p_win, p_draw, p_loss = self.predict_proba(elo_a, elo_b, is_knockout)

        if is_knockout and p_draw > 0:
            # Re-distribute draw probability as a 50/50 penalties outcome
            p_win += p_draw / 2.0
            p_loss += p_draw / 2.0
            p_draw = 0.0

        r = random.random()
        if r < p_win:
            return "W"
        if r < p_win + p_draw:
            return "D"
        return "L"
