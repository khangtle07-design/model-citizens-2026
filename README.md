# Model Citizens 2026 — Congress forecast and pick model

Predicts the winner of all 470 races (435 House + 35 Senate, including the Ohio and Florida
specials) and chooses picks for the contest's scoring rule: a correct pick earns
`-log2(p)`, where `p` is Kalshi's Nov 1 price on the side picked (floored at 1%); a wrong pick earns 0.

Outputs:

| File | Use |
|---|---|
| `predictions/full_congress_submission.csv` | **The submission** (Full Congress track: 470 rows, required columns only) |
| `predictions/full_congress_predictions.csv` | Detailed version: Kalshi price, points if correct, expected points, against-market flag, name source |

Submission columns: `state, district, chamber, predicted_winner, party, model_probability`.
House districts are written `CA-11` (at-large seats `AK-AL`); Senate rows have a blank district
except the Ohio and Florida specials (`SPECIAL`). `model_probability` is the model's chance that
the predicted winner wins.

## How to run

```bash
pip install -r requirements.txt
bash run_all.sh
```

**Before submitting, refresh the market prices to the Nov 1 pull** (the benchmark the contest scores
against): update `kalshi_dem_side_prob` in `data/inputs/senate_2026.csv` and `kalshi_p_D` in
`data/inputs/kalshi_house_2026-10-26.csv`, then rerun. The pick rule depends on those prices.

## Pipeline

| Step | File | What it does |
|---|---|---|
| 0 | `src/parse_candidates.py`, `src/candidates.py` | Nominee names for all 435 districts, with a per-row `name_source` |
| 1 | `src/fundamentals.py` | Model 1. Fits House margins 2004–2024 on lagged margin, incumbency and national swing; leave-one-cycle-out backtest |
| 2 | `src/house_model.py` | 435 districts: fundamentals (lagged-result and presidential-lean versions averaged) blended with expert ratings; attaches Kalshi district prices |
| 3 | `src/senate_model.py` | Model 2. Fundamentals prior + nonpartisan polling average, precision-weighted, with a shared national polling error |
| 5 | `src/submission.py` | Writes the submission CSV with only the required columns |
| 4 | `src/picks.py` | Shrinks the model toward the market, simulates 20,000 correlated elections, picks under the scoring rule with a reasonableness floor |

## Data

| Source | Used for | File |
|---|---|---|
| MIT Election Data + Science Lab | House results 1976–2024, Senate 1976–2018 | `data/raw/` |
| 2024 presidential result by district | district lean | `data/raw/pres2024_by_cd.csv` |
| Silver Bulletin, VoteHub, RealClearPolling, Politix (only where nothing else publishes an average) | Senate polling averages, Oct 21–26 | `data/inputs/senate_2026.csv` |
| RealClearPolling, Poliwave, NBC poll of polls | generic ballot (D+8.4) | `data/inputs/national_environment.json` |
| Cook Political Report, Sabato's Crystal Ball, Inside Elections | House ratings (Sep 25–Oct 23) | `data/inputs/house_ratings_2026.csv` |
| Kalshi | race prices, Oct 17–26 snapshot | `senate_2026.csv`, `kalshi_house_2026-10-26.csv` |
| Ballotpedia / news | 2026 redistricting, retirements, same-party races | `data/inputs/house_overrides_2026.csv` |
| The Ballot Brief (built from state certified candidate lists), Wikipedia/news race pages | House nominee names, 311 districts | `data/raw/house_candidates_snippets_2026-10-26.txt` |
| unitedstates/congress-legislators (Sep 2026) | sitting members, used as the name where the incumbent is running | `data/raw/legislators-current_2026-10-26.yaml` |

**Source hygiene.** Polling inputs use nonpartisan averages, and where averages disagree
(RealClearPolling's pollster mix tends to run more Republican than Silver Bulletin or VoteHub)
the model takes their mean rather than picking one. Campaign internals and partisan-sponsored polls
are excluded from the averages (for example Impact Research (D) in South Carolina, and
InsiderAdvantage/Trafalgar results are not used as standalone inputs). Prediction-market promotional
sites ("polls lie, markets don't", "why X will win") were used only as mirrors of Kalshi prices,
never for analysis.

## Known limitations

- **Candidate names.** 415 of 435 House picks carry a real name: 311 from certified
  general-election lists, 104 from the sitting incumbent where nothing shows they left the race.
  18 safe or open seats still say "nominee (name not collected)" (CA-38, LA-05, LA-06, ME-02,
  NV-02, NY-12, SD, TN-06, TX-19/21/22/30/32/38, UT-01/02/04, WA-04), and MI-01 and MI-04 use the
  incumbent's name without confirming he is on the ballot. Check `name_source` before submitting.
  Names never affect which party is picked or the points.

- **No historical market data**, so the model-vs-market blend weight (0.5 Senate, 0.35 House) is a
  judgment, not fitted. The slides' Model 3 (an upset classifier trained on past ratings and prices)
  was not built for the same reason; its edge/expected-points logic is used in `picks.py`.
- **Redistricted seats** (TX, CA, FL, NC, OH, UT, TN, LA, AL) use hand-estimated presidential margins
  under the new lines (`house_overrides_2026.csv`, marked "estimate").
- The Senate polling-error parameters are standard values from the forecasting literature, not
  re-estimated on a 2026-like polling archive.
- About 200 safe House seats were not visible in the Kalshi snapshot; their prices are imputed at
  90–99.5¢. This changes points only marginally (a correct safe-seat pick is worth ≤ 0.15).
- Eight Kalshi "Likely D" districts are known to be in that 75–90¢ bucket but their exact price was
  not visible; they are set at 82¢.
- The simulation scores picks against the model's own probabilities, so the expected scores are
  what the model believes, not an independent estimate.
