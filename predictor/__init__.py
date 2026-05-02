"""FIFA World Cup Predictor package."""

from .teams import TEAM_RATINGS
from .tournament import GROUPS_2026
from .model import MatchPredictor
from .data_generator import generate_training_data
from .simulation import simulate_tournament, run_monte_carlo

__all__ = [
    "TEAM_RATINGS",
    "GROUPS_2026",
    "MatchPredictor",
    "generate_training_data",
    "simulate_tournament",
    "run_monte_carlo",
]
