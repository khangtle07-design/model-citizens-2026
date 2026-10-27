"""
Turn probabilities into picks under the contest scoring rule.

Scoring: correct pick on side i earns -log2(max(p_K(i), 0.01)); wrong pick earns 0.

Why not just "maximize expected points"?  If you believe the market exactly (q = p_K),
EV(underdog) = (1-p)*(-log2(1-p)) is ALWAYS larger than EV(favorite) = p*(-log2 p)
for p > 0.5 (because -x log x peaks at x = 1/e).  A naive EV maximizer therefore picks
every underdog even with zero information -- the "just favor upsets" trap.

Our rule:
  1. q = posterior P(win) = logit-average of our model and the market
     (w_model = 0.5 Senate, 0.35 House, where our model is cruder):
     the market is a strong, well-calibrated forecaster, so we shrink toward it.
  2. Default pick = the side with q > 0.5.
  3. Switch to the other side only if ALL hold:
        q_other >= TAU      (reasonable: never back someone we give < TAU)
        q_other - p_K(other) >= DELTA   (a real disagreement with the market, not
                                         just the convexity of the scoring rule)
        EV(other) > EV(default)
  4. TAU >= 0.40 is fixed by judgment (reasonableness). Within that, TAU/DELTA are chosen on a
     grid by maximizing mean - 0.5*SD of the simulated total
     score (simulated with a shared national error so races miss together).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

FLOOR = 0.01
TAU_FLOOR = 0.40
W_MODEL = {"Senate": 0.5, "House": 0.35}   # model weight vs market in logit space
logit = lambda p: np.log(p / (1 - p))
expit = lambda x: 1 / (1 + np.exp(-x))


def pts(p):
    return -np.log2(np.maximum(p, FLOOR))


def posterior(df):
    pm = np.clip(df.p_model_D.values, 0.002, 0.998)
    pk = np.clip(df.p_market_D.values, 0.002, 0.998)
    w = df.chamber.map(W_MODEL).values
    return expit(w * logit(pm) + (1 - w) * logit(pk))


def choose(df, tau, delta):
    q = posterior(df)
    pk = df.p_market_D.values
    default_D = q >= 0.5
    q_def = np.where(default_D, q, 1 - q)
    q_oth = 1 - q_def
    pk_def = np.where(default_D, pk, 1 - pk)
    pk_oth = 1 - pk_def
    ev_def, ev_oth = q_def * pts(pk_def), q_oth * pts(pk_oth)
    switch = (q_oth >= tau) & (q_oth - pk_oth >= delta) & (ev_oth > ev_def)
    pick_D = np.where(switch, ~default_D, default_D)
    return pick_D, q


def simulate(df, n=20000, seed=7):
    """Gaussian copula: each race keeps its posterior marginal; a shared national factor
    correlates the misses (rho_i = shared variance / total variance)."""
    rng = np.random.default_rng(seed)
    q = posterior(df)
    rho = df.rho.values
    z = rng.standard_normal((n, 1))
    e = rng.standard_normal((n, len(df)))
    latent = np.sqrt(rho) * z + np.sqrt(1 - rho) * e
    return latent < norm.ppf(q)          # True = D-side wins


def score(df, pick_D, wins_D):
    pk = df.p_market_D.values
    val = np.where(pick_D, pts(pk), pts(1 - pk))
    correct = wins_D == pick_D
    return (correct * val).sum(axis=1)


def grid(df, sims):
    res = []
    # TAU_FLOOR is the "reasonable guess" constraint: never pick a side we give < 40%.
    # It is a modelling choice, not something the optimiser can find: with this scoring rule no
    # level of risk aversion rejects "pick every underdog" (see strategy comparison).
    for tau in [TAU_FLOOR, 0.45, 0.50]:
        for delta in [0.0, 0.02, 0.03, 0.05, 0.08, 1.0]:
            pk, _ = choose(df, tau, delta)
            s = score(df, pk, sims)
            res.append(dict(tau=tau, delta=delta, n_upsets=int((pk != (df.p_market_D.values >= .5)).sum()),
                            mean=s.mean(), sd=s.std(), p10=np.percentile(s, 10),
                            p50=np.median(s), objective=s.mean() - 0.5 * s.std()))
    return pd.DataFrame(res)


def strategies(df, sims, tau, delta):
    """Score distributions of alternative pick rules, for the write-up."""
    pk = df.p_market_D.values
    q = posterior(df)
    ev_D, ev_R = q * pts(pk), (1 - q) * pts(1 - pk)
    out = {}
    for name, pick in {
        "chalk (always Kalshi favourite)": pk >= 0.5,
        "model favourite (q > .5, no EV switching)": q >= 0.5,
        "pure EV max (no reasonableness limits)": ev_D >= ev_R,
        "aggressive variant (tau=0.30, delta=0)": choose(df, 0.30, 0.0)[0],
        f"chosen rule (tau={tau}, delta={delta})": choose(df, tau, delta)[0],
    }.items():
        s = score(df, pick, sims)
        out[name] = dict(n_against_market=int((pick != (pk >= .5)).sum()), mean=s.mean(), sd=s.std(),
                         p10=np.percentile(s, 10), p50=np.median(s), p90=np.percentile(s, 90),
                         exp_correct=float(np.where(pick, q, 1 - q).sum()))
    return pd.DataFrame(out).T


def write_outputs(df, pick_D, q, tag):
    pk = df.p_market_D.values
    p_pick_mkt = np.where(pick_D, pk, 1 - pk)
    q_pick = np.where(pick_D, q, 1 - q)
    party = np.where(pick_D, df.dem_side_party.values, "R")
    cand = np.where(pick_D, df.dem_side_candidate.values, df.rep_candidate.values)
    res = pd.DataFrame(dict(
        race_id=df.race_id, chamber=df.chamber, state=df.state, district=df.district,
        predicted_winner=cand, predicted_party=party,
        model_prob=np.round(q_pick, 4), kalshi_prob_pick=np.round(p_pick_mkt, 4),
        points_if_correct=np.round(pts(p_pick_mkt), 3),
        expected_points=np.round(q_pick * pts(p_pick_mkt), 3),
        against_market=(pick_D != (pk >= .5)).astype(int),
        market_source=df.market_source,
        name_source=np.where(pick_D, df.get("name_source_D", pd.Series([""]*len(df))).values,
                             df.get("name_source_R", pd.Series([""]*len(df))).values)))
    res.to_csv(f"predictions/{tag}.csv", index=False)
    return res


if __name__ == "__main__":
    import json, os, sys
    sys.path.insert(0, "src")
    import senate_model, house_model
    env = json.load(open("data/inputs/national_environment.json"))
    sen = senate_model.build(env)
    hou, _ = house_model.build(env)
    hou["dem_side_candidate"] = "Democratic nominee"
    hou["dem_side_party"] = "D"
    hou["rep_candidate"] = "Republican nominee"
    cand = pd.read_csv("data/inputs/house_candidates_2026.csv").set_index("cd") \
        if os.path.exists("data/inputs/house_candidates_2026.csv") else None
    if cand is not None:
        for c in ["dem_side_candidate", "rep_candidate", "dem_side_party"]:
            m = hou.district.map(cand[c]) if c in cand else None
            if m is not None:
                hou[c] = m.fillna(hou[c])
        for c in ["name_source_D", "name_source_R"]:
            hou[c] = hou.district.map(cand[c]).fillna("placeholder")
    cols = ["race_id", "chamber", "state", "district", "dem_side_candidate", "dem_side_party",
            "rep_candidate", "p_model_D", "p_market_D", "rho", "market_source"]
    sen["name_source_D"] = sen["name_source_R"] = "Senate inputs file"
    cols += ["name_source_D", "name_source_R"]
    full = pd.concat([sen[cols], hou[cols]], ignore_index=True)
    # same-party seats: outcome is certain at the party level -> never a switch candidate
    full["same_party"] = full.race_id.isin(hou.loc[hou.same_party != "", "race_id"])
    os.makedirs("predictions", exist_ok=True)

    sims = simulate(full)               # one national factor shared by all 470 races
    g = grid(full, sims).sort_values("objective", ascending=False)
    g.to_csv("output/pick_rule_grid.csv", index=False)
    best = g.iloc[0]
    tau, delta = float(best.tau), float(best.delta)
    print("Pick-rule grid (top 8, full congress):"); print(g.head(8).round(3).to_string(index=False))

    pick_D, q = choose(full, tau, delta)
    pick_D = np.where(full.same_party, full.p_market_D.values >= .5, pick_D)
    full_res = write_outputs(full, pick_D, q, "full_congress_predictions")
    s_idx = (full.chamber == "Senate").values
    sen_df = full[s_idx].reset_index(drop=True)

    st_full = strategies(full, sims, tau, delta)
    st_sen = strategies(sen_df, sims[:, s_idx], tau, delta)
    st_full.to_csv("output/strategy_comparison_full.csv"); st_sen.to_csv("output/strategy_comparison_senate.csv")
    print("\nStrategy comparison, full congress:"); print(st_full.round(2).to_string())
    print("\nStrategy comparison, senate track:"); print(st_sen.round(2).to_string())
    print("\nAgainst-market picks:")
    print(full_res[full_res.against_market == 1][["race_id", "predicted_winner", "predicted_party",
          "model_prob", "kalshi_prob_pick", "points_if_correct"]].to_string(index=False))
    json.dump(dict(tau=tau, delta=delta, w_model=W_MODEL), open("output/chosen_rule.json", "w"), indent=2)
