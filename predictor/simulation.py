"""
Tournament simulation engine.

Runs a complete 2026 FIFA World Cup simulation using the ML-based
:class:`~predictor.model.MatchPredictor`:

1. **Group stage** – round-robin within each 4-team group.  Points
   (3W / 1D / 0L) determine standings; ELO rating breaks ties.
2. **Third-place selection** – best 8 third-placed teams (by points
   then ELO) advance alongside the 24 group qualifiers (2 per group).
3. **Knockout rounds** – R32 → R16 → QF → SF → Final.
   No draws: a draw is resolved as a penalty shootout.
4. **Monte Carlo** – run the full simulation *n* times and aggregate
   the championship-win count for every team.
"""

from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations

from .model import MatchPredictor
from .teams import TEAM_RATINGS
from .tournament import GROUPS_2026


# ---------------------------------------------------------------------------
# Group-stage helpers
# ---------------------------------------------------------------------------

def _simulate_group(
    teams: list[str],
    predictor: MatchPredictor,
    ratings: dict[str, float],
) -> list[dict]:
    """
    Simulate a single group (round-robin) and return standings sorted by
    (points DESC, elo DESC).

    Each entry in the returned list is a dict with keys:
    ``team``, ``points``, ``elo``.
    """
    pts: dict[str, int] = {t: 0 for t in teams}

    for team_a, team_b in combinations(teams, 2):
        result = predictor.simulate_match(
            ratings[team_a], ratings[team_b], is_knockout=False
        )
        if result == "W":
            pts[team_a] += 3
        elif result == "D":
            pts[team_a] += 1
            pts[team_b] += 1
        else:
            pts[team_b] += 3

    standings = [
        {"team": t, "points": pts[t], "elo": ratings[t]} for t in teams
    ]
    standings.sort(key=lambda s: (s["points"], s["elo"]), reverse=True)
    return standings


def _run_group_stage(
    groups: dict[str, list[str]],
    predictor: MatchPredictor,
    ratings: dict[str, float],
) -> tuple[list[str], list[str], list[dict]]:
    """
    Simulate all 12 groups.

    Returns
    -------
    winners : list[str]
        12 group winners.
    runners_up : list[str]
        12 group runners-up.
    third_place_pool : list[dict]
        12 third-place finishers with their points / ELO, ready for
        best-8 selection.
    """
    winners: list[str] = []
    runners_up: list[str] = []
    third_place_pool: list[dict] = []

    for group_name, teams in groups.items():
        standing = _simulate_group(teams, predictor, ratings)
        winners.append(standing[0]["team"])
        runners_up.append(standing[1]["team"])
        third_place_pool.append(standing[2])

    return winners, runners_up, third_place_pool


# ---------------------------------------------------------------------------
# Knockout helpers
# ---------------------------------------------------------------------------

def _build_r32_bracket(
    winners: list[str],
    runners_up: list[str],
    third_place_pool: list[dict],
) -> list[tuple[str, str]]:
    """
    Pair 32 teams for the Round of 32.

    The first 12 matches follow the pre-determined bracket seeding
    (1A vs 2F, 1B vs 2E, …).  The remaining 4 matches fill in the
    8 best third-place qualifiers.
    """
    group_letters = list(GROUPS_2026.keys())  # A … L

    # Map "1X" → winner of group X, "2X" → runner-up of group X
    winner_map = {f"1{g}": w for g, w in zip(group_letters, winners)}
    runner_map = {f"2{g}": r for g, r in zip(group_letters, runners_up)}
    slot_map = {**winner_map, **runner_map}

    # Best 8 third-place teams (points desc, ELO desc)
    best_thirds = sorted(
        third_place_pool, key=lambda s: (s["points"], s["elo"]), reverse=True
    )[:8]
    third_teams = [s["team"] for s in best_thirds]

    matchups: list[tuple[str, str]] = []

    # Fixed bracket slots (group winners vs runners-up)
    fixed_pairs = [
        ("1A", "2F"), ("1B", "2E"), ("1C", "2D"), ("1D", "2C"),
        ("1E", "2B"), ("1F", "2A"), ("1G", "2L"), ("1H", "2K"),
        ("1I", "2J"), ("1J", "2I"), ("1K", "2H"), ("1L", "2G"),
    ]
    for slot_a, slot_b in fixed_pairs:
        matchups.append((slot_map[slot_a], slot_map[slot_b]))

    # Third-place slots: pair them in seeded order
    for i in range(0, 8, 2):
        matchups.append((third_teams[i], third_teams[i + 1]))

    return matchups  # 16 matchups, 32 teams


def _simulate_knockout_round(
    matchups: list[tuple[str, str]],
    predictor: MatchPredictor,
    ratings: dict[str, float],
) -> list[str]:
    """Simulate a single knockout round and return the list of winners."""
    round_winners: list[str] = []
    for team_a, team_b in matchups:
        result = predictor.simulate_match(
            ratings[team_a], ratings[team_b], is_knockout=True
        )
        round_winners.append(team_a if result == "W" else team_b)
    return round_winners


def _pair_for_next_round(teams: list[str]) -> list[tuple[str, str]]:
    """Pair teams sequentially: 1st vs 2nd, 3rd vs 4th, …"""
    return [(teams[i], teams[i + 1]) for i in range(0, len(teams), 2)]


# ---------------------------------------------------------------------------
# Public simulation API
# ---------------------------------------------------------------------------

def simulate_tournament(
    predictor: MatchPredictor,
    groups: dict[str, list[str]] | None = None,
    ratings: dict[str, float] | None = None,
) -> str:
    """
    Simulate a complete World Cup and return the name of the winner.

    Parameters
    ----------
    predictor:
        A *trained* :class:`~predictor.model.MatchPredictor`.
    groups:
        Group-stage draw.  Defaults to :data:`~predictor.tournament.GROUPS_2026`.
    ratings:
        ELO ratings.  Defaults to :data:`~predictor.teams.TEAM_RATINGS`.
    """
    if groups is None:
        groups = GROUPS_2026
    if ratings is None:
        ratings = TEAM_RATINGS

    # --- Group stage ---
    winners, runners_up, third_pool = _run_group_stage(groups, predictor, ratings)

    # --- Round of 32 ---
    r32_matchups = _build_r32_bracket(winners, runners_up, third_pool)
    r16_teams = _simulate_knockout_round(r32_matchups, predictor, ratings)

    # --- Round of 16 ---
    qf_teams = _simulate_knockout_round(
        _pair_for_next_round(r16_teams), predictor, ratings
    )

    # --- Quarter-finals ---
    sf_teams = _simulate_knockout_round(
        _pair_for_next_round(qf_teams), predictor, ratings
    )

    # --- Semi-finals ---
    finalists = _simulate_knockout_round(
        _pair_for_next_round(sf_teams), predictor, ratings
    )

    # --- Final ---
    champion = _simulate_knockout_round(
        _pair_for_next_round(finalists), predictor, ratings
    )
    return champion[0]


def run_monte_carlo(
    predictor: MatchPredictor,
    n_simulations: int = 10_000,
    groups: dict[str, list[str]] | None = None,
    ratings: dict[str, float] | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    """
    Run *n_simulations* complete World Cup simulations and return each
    team's estimated probability of winning the tournament.

    Parameters
    ----------
    predictor:
        A *trained* :class:`~predictor.model.MatchPredictor`.
    n_simulations:
        Number of Monte Carlo iterations.
    groups:
        Group draw.  Defaults to :data:`~predictor.tournament.GROUPS_2026`.
    ratings:
        ELO ratings.  Defaults to :data:`~predictor.teams.TEAM_RATINGS`.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict mapping team name → win probability (values sum to 1.0).
    """
    if seed is not None:
        random.seed(seed)
    if groups is None:
        groups = GROUPS_2026
    if ratings is None:
        ratings = TEAM_RATINGS

    win_counts: dict[str, int] = defaultdict(int)

    for _ in range(n_simulations):
        champion = simulate_tournament(predictor, groups, ratings)
        win_counts[champion] += 1

    return {team: count / n_simulations for team, count in win_counts.items()}
