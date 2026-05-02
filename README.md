# World-Cup-Predictor

A **machine-learning–powered** FIFA World Cup 2026 predictor written in Python.

## How it works

| Layer | What it does |
|---|---|
| **Data generation** (`predictor/data_generator.py`) | Generates realistic synthetic match data using a **Poisson goal model** calibrated to international-football statistics. |
| **ML model** (`predictor/model.py`) | Trains a **Logistic Regression** classifier (via scikit-learn) on the generated data.  Features: ELO difference, squared ELO difference, mean ELO, and a knockout-match flag. |
| **Simulation** (`predictor/simulation.py`) | Uses the trained model's per-match probabilities to simulate the complete 2026 World Cup (group stage → Round of 32 → R16 → QF → SF → Final). |
| **Monte Carlo** (`predictor/simulation.py`) | Runs the simulation thousands of times to produce stable **win-probability estimates** for all 48 teams. |

## Project structure

```
World-Cup-Predictor/
├── main.py                    # CLI entry point
├── requirements.txt
├── predictor/
│   ├── __init__.py
│   ├── teams.py               # 48 teams + ELO ratings (2026 WC)
│   ├── tournament.py          # Group draw & R32 bracket
│   ├── data_generator.py      # Poisson-based training-data generator
│   ├── model.py               # MatchPredictor (scikit-learn pipeline)
│   └── simulation.py          # Group stage, knockout rounds, Monte Carlo
└── tests/
    └── test_predictor.py      # 28 pytest unit tests
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Monte Carlo win-probability table (default: 10 000 simulations)

```bash
python main.py
```

### Faster run with fewer simulations

```bash
python main.py --simulations 1000
```

### Simulate a single tournament and print every round

```bash
python main.py --single
```

### Show win probabilities for all 48 teams

```bash
python main.py --all-teams
```

### Reproducible results with a fixed seed

```bash
python main.py --simulations 5000 --seed 42
```

## Example output

```
Generating training data … 10,000 matches generated.
Training ML model (Logistic Regression) … done in 0.03s.

Running 10,000 Monte Carlo simulations … done in 178.4s.

2026 FIFA World Cup predicted win probabilities (top 10 teams):

Rank  Team                  Win probability
--------------------------------------------
1     Argentina               28.50 %  ██████████████
2     France                  14.80 %  ███████
3     England                  9.10 %  ████
4     Brazil                   7.30 %  ███
5     Spain                    6.60 %  ███
...
```

## Running the tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Architecture notes

* **No external data required** – training data is generated on-the-fly from ELO-derived Poisson distributions.
* **Easily extensible** – swap in a different `MatchPredictor` (e.g. a Random Forest or Gradient Boosting model) without touching the simulation layer.
* **2026 format** – supports the expanded 48-team / 12-group / Round-of-32 format.
