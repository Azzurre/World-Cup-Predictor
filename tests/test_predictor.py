"""Unit tests for the World Cup Predictor."""

from __future__ import annotations

import math
import random

import pytest

from predictor.data_generator import generate_training_data
from predictor.model import MatchPredictor
from predictor.simulation import run_monte_carlo, simulate_tournament
from predictor.teams import TEAM_RATINGS
from predictor.tournament import GROUPS_2026


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_model() -> MatchPredictor:
    """Return a model trained on a small synthetic dataset."""
    data = generate_training_data(n_samples=3_000, seed=0)
    model = MatchPredictor()
    model.fit(data)
    return model


# ---------------------------------------------------------------------------
# Teams / ratings
# ---------------------------------------------------------------------------

class TestTeamRatings:
    def test_correct_team_count(self):
        assert len(TEAM_RATINGS) == 48

    def test_all_ratings_positive(self):
        for team, rating in TEAM_RATINGS.items():
            assert rating > 0, f"{team} has non-positive rating"

    def test_argentina_is_strongest(self):
        max_team = max(TEAM_RATINGS, key=TEAM_RATINGS.__getitem__)
        assert max_team == "Argentina"

    def test_ratings_realistic_range(self):
        """All ELO ratings should fall between 1300 and 2200."""
        for team, rating in TEAM_RATINGS.items():
            assert 1300 <= rating <= 2200, f"{team}: rating {rating} out of range"


# ---------------------------------------------------------------------------
# Tournament groups
# ---------------------------------------------------------------------------

class TestGroups:
    def test_twelve_groups(self):
        assert len(GROUPS_2026) == 12

    def test_four_teams_per_group(self):
        for group, teams in GROUPS_2026.items():
            assert len(teams) == 4, f"Group {group} has {len(teams)} teams"

    def test_exactly_48_unique_teams(self):
        all_teams = [t for teams in GROUPS_2026.values() for t in teams]
        assert len(all_teams) == 48
        assert len(set(all_teams)) == 48, "Duplicate team in groups"

    def test_all_group_teams_have_ratings(self):
        for group, teams in GROUPS_2026.items():
            for team in teams:
                assert team in TEAM_RATINGS, f"{team} in group {group} has no rating"


# ---------------------------------------------------------------------------
# Training data generator
# ---------------------------------------------------------------------------

class TestDataGenerator:
    def test_correct_sample_count(self):
        data = generate_training_data(n_samples=500, seed=1)
        assert len(data) == 500

    def test_result_labels_valid(self):
        data = generate_training_data(n_samples=200, seed=2)
        for _, _, result, _ in data:
            assert result in {0, 1, 2}

    def test_is_knockout_binary(self):
        data = generate_training_data(n_samples=200, seed=3)
        for _, _, _, ko in data:
            assert ko in {0, 1}

    def test_elo_values_in_expected_range(self):
        data = generate_training_data(n_samples=200, seed=4)
        for elo_a, elo_b, _, _ in data:
            assert 1250 < elo_a < 2200
            assert 1250 < elo_b < 2200

    def test_all_outcomes_present(self):
        """With enough samples, all three outcomes should appear."""
        data = generate_training_data(n_samples=2_000, seed=5)
        outcomes = {r for _, _, r, _ in data}
        assert outcomes == {0, 1, 2}

    def test_reproducibility(self):
        d1 = generate_training_data(n_samples=100, seed=99)
        d2 = generate_training_data(n_samples=100, seed=99)
        assert d1 == d2

    def test_different_seeds_differ(self):
        d1 = generate_training_data(n_samples=100, seed=1)
        d2 = generate_training_data(n_samples=100, seed=2)
        assert d1 != d2


# ---------------------------------------------------------------------------
# ML model
# ---------------------------------------------------------------------------

class TestMatchPredictor:
    def test_untrained_model_raises(self):
        model = MatchPredictor()
        with pytest.raises(RuntimeError):
            model.predict_proba(1800, 1700)

    def test_probabilities_sum_to_one(self, trained_model):
        p_win, p_draw, p_loss = trained_model.predict_proba(1900, 1700)
        assert math.isclose(p_win + p_draw + p_loss, 1.0, abs_tol=1e-6)

    def test_stronger_team_favoured(self, trained_model):
        p_win_strong, _, _ = trained_model.predict_proba(2000, 1500)
        p_win_weak, _, _ = trained_model.predict_proba(1500, 2000)
        assert p_win_strong > p_win_weak

    def test_equal_teams_roughly_equal(self, trained_model):
        p_win, _, p_loss = trained_model.predict_proba(1700, 1700)
        assert abs(p_win - p_loss) < 0.05

    def test_no_draw_in_knockout(self, trained_model):
        results = {
            trained_model.simulate_match(1700, 1700, is_knockout=True)
            for _ in range(200)
        }
        assert "D" not in results

    def test_simulate_match_returns_valid_result(self, trained_model):
        for _ in range(50):
            result = trained_model.simulate_match(1800, 1700)
            assert result in {"W", "D", "L"}

    def test_fit_returns_self(self):
        model = MatchPredictor()
        data = generate_training_data(n_samples=500, seed=7)
        returned = model.fit(data)
        assert returned is model


# ---------------------------------------------------------------------------
# Tournament simulation
# ---------------------------------------------------------------------------

class TestSimulateTournament:
    def test_returns_valid_team(self, trained_model):
        champion = simulate_tournament(trained_model)
        all_teams = set(TEAM_RATINGS.keys())
        assert champion in all_teams

    def test_multiple_runs_produce_different_results(self, trained_model):
        """With enough runs, at least two different champions should appear."""
        champions = {simulate_tournament(trained_model) for _ in range(20)}
        assert len(champions) > 1


class TestMonteCarlo:
    def test_probabilities_sum_to_one(self, trained_model):
        probs = run_monte_carlo(trained_model, n_simulations=200, seed=0)
        total = sum(probs.values())
        assert math.isclose(total, 1.0, abs_tol=1e-6)

    def test_all_winners_are_valid_teams(self, trained_model):
        all_teams = set(TEAM_RATINGS.keys())
        probs = run_monte_carlo(trained_model, n_simulations=200, seed=1)
        for team in probs:
            assert team in all_teams

    def test_strongest_team_is_top_favourite(self, trained_model):
        """Argentina (highest ELO) should be the most-predicted champion."""
        probs = run_monte_carlo(trained_model, n_simulations=2_000, seed=42)
        top_team = max(probs, key=probs.__getitem__)
        assert top_team == "Argentina"

    def test_reproducibility_with_seed(self, trained_model):
        p1 = run_monte_carlo(trained_model, n_simulations=100, seed=77)
        p2 = run_monte_carlo(trained_model, n_simulations=100, seed=77)
        assert p1 == p2
