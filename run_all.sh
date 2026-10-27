#!/usr/bin/env bash
# Reproduce every output from the raw data and the dated input files.
set -e
cd "$(dirname "$0")"
export PYTHONPATH=src
python src/parse_candidates.py && python src/candidates.py    > output/candidates_log.txt          # nominee names for all 435 districts
python src/fundamentals.py   | tee output/fundamentals_log.txt   # Model 1: fit + leave-one-cycle-out backtest
python src/house_model.py    > output/house_log.txt              # 435 districts
python src/senate_model.py   > output/senate_log.txt             # 35 Senate races (Model 2)
python src/picks.py          | tee output/picks_log.txt          # posterior, simulation, pick rule, CSVs
python src/submission.py   | tee output/submission_log.txt     # submission CSVs (required columns only)
