#!/usr/bin/env python3
"""
FIFA World Cup 2026 Predictor – command-line interface.

Examples
--------
# Default: 10 000-simulation Monte Carlo, print top-10 favourites
python main.py

# Faster run with 1 000 simulations
python main.py --simulations 1000

# Simulate a single tournament and print every round
python main.py --single

# Show win probabilities for all 48 teams
python main.py --all-teams
"""

from __future__ import annotations

import argparse
import sys
import time

from predictor.data_generator import generate_training_data
from predictor.model import MatchPredictor
from predictor.simulation import run_monte_carlo, simulate_tournament
from predictor.teams import TEAM_RATINGS
from predictor.tournament import GROUPS_2026


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_and_train_model(verbose: bool = True) -> MatchPredictor:
    """Generate training data, train the ML model, and return it."""
    if verbose:
        print("Generating training data …", end=" ", flush=True)
    data = generate_training_data(n_samples=10_000, seed=42)
    if verbose:
        print(f"{len(data):,} matches generated.")

    model = MatchPredictor()
    if verbose:
        print("Training ML model (Logistic Regression) …", end=" ", flush=True)
    t0 = time.perf_counter()
    model.fit(data)
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"done in {elapsed:.2f}s.")
    return model


def _print_probabilities(
    probs: dict[str, float],
    top_n: int | None = None,
) -> None:
    """Pretty-print win-probability table."""
    all_teams = sorted(TEAM_RATINGS.keys())
    # Make sure every team is present (teams that never won get 0 %)
    full = {t: probs.get(t, 0.0) for t in all_teams}
    ranked = sorted(full.items(), key=lambda x: x[1], reverse=True)

    if top_n is not None:
        ranked = ranked[:top_n]

    print(f"\n{'Rank':<5} {'Team':<20} {'Win probability':>16}")
    print("-" * 44)
    for rank, (team, prob) in enumerate(ranked, start=1):
        bar = "█" * int(prob * 50)
        print(f"{rank:<5} {team:<20} {prob * 100:>8.2f} %  {bar}")
    print()


def _simulate_single(model: MatchPredictor) -> None:
    """Simulate one full tournament and narrate the rounds."""
    from predictor.simulation import (
        _build_r32_bracket,
        _pair_for_next_round,
        _run_group_stage,
        _simulate_knockout_round,
    )

    ratings = TEAM_RATINGS
    groups = GROUPS_2026

    print("\n" + "=" * 60)
    print("  2026 FIFA WORLD CUP – SINGLE TOURNAMENT SIMULATION")
    print("=" * 60)

    # --- Group stage ---
    print("\n── GROUP STAGE ──────────────────────────────────────────")
    winners, runners_up, third_pool = _run_group_stage(groups, model, ratings)
    for i, (g, w, r) in enumerate(zip(groups.keys(), winners, runners_up)):
        print(f"  Group {g}: 1st {w:<20}  2nd {r}")

    # Best 8 thirds
    best_thirds = sorted(
        third_pool, key=lambda s: (s["points"], s["elo"]), reverse=True
    )[:8]
    print("\n  Best third-place qualifiers:")
    for entry in best_thirds:
        print(f"    {entry['team']}  ({entry['points']} pts, ELO {entry['elo']:.0f})")

    # --- Round of 32 ---
    r32 = _build_r32_bracket(winners, runners_up, third_pool)
    r16 = _simulate_knockout_round(r32, model, ratings)
    print("\n── ROUND OF 32 ──────────────────────────────────────────")
    for (a, b), winner in zip(r32, r16):
        print(f"  {a:<20} vs {b:<20} → {winner}")

    # --- Round of 16 ---
    qf = _simulate_knockout_round(_pair_for_next_round(r16), model, ratings)
    print("\n── ROUND OF 16 ──────────────────────────────────────────")
    for (a, b), winner in zip(_pair_for_next_round(r16), qf):
        print(f"  {a:<20} vs {b:<20} → {winner}")

    # --- Quarter-finals ---
    sf = _simulate_knockout_round(_pair_for_next_round(qf), model, ratings)
    print("\n── QUARTER-FINALS ───────────────────────────────────────")
    for (a, b), winner in zip(_pair_for_next_round(qf), sf):
        print(f"  {a:<20} vs {b:<20} → {winner}")

    # --- Semi-finals ---
    finalists = _simulate_knockout_round(_pair_for_next_round(sf), model, ratings)
    print("\n── SEMI-FINALS ──────────────────────────────────────────")
    for (a, b), winner in zip(_pair_for_next_round(sf), finalists):
        print(f"  {a:<20} vs {b:<20} → {winner}")

    # --- Final ---
    champion = _simulate_knockout_round(_pair_for_next_round(finalists), model, ratings)
    final_a, final_b = finalists
    print("\n── FINAL ────────────────────────────────────────────────")
    print(f"  {final_a:<20} vs {final_b:<20} → {champion[0]}")

    print("\n" + "=" * 60)
    print(f"  🏆  WORLD CUP CHAMPION: {champion[0]}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="FIFA World Cup 2026 Predictor (ML-powered)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=10_000,
        metavar="N",
        help="Number of Monte Carlo simulations (default: 10 000).",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Simulate a single tournament and print the bracket.",
    )
    parser.add_argument(
        "--all-teams",
        action="store_true",
        help="Print win probabilities for all 48 teams.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        metavar="N",
        help="Number of top teams to display (default: 10; ignored with --all-teams).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="Random seed for reproducible simulations.",
    )
    args = parser.parse_args()

    model = _build_and_train_model(verbose=True)

    if args.single:
        _simulate_single(model)
        return 0

    top_n = None if args.all_teams else args.top
    label = "all 48 teams" if args.all_teams else f"top {args.top} teams"

    print(f"\nRunning {args.simulations:,} Monte Carlo simulations …", end=" ", flush=True)
    t0 = time.perf_counter()
    probs = run_monte_carlo(
        model,
        n_simulations=args.simulations,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.1f}s.")

    print(f"\n2026 FIFA World Cup predicted win probabilities ({label}):")
    _print_probabilities(probs, top_n=top_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
