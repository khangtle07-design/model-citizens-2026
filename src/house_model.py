"""
2026 House: one row per district (435).

  mu_A  lagged-House-result model (Model 1, fit on 2004-2024)   -- unchanged lines only
  mu_B  presidential-lean model (2024 Harris-Trump margin mapped to House margin
        with a 2024 cross-section fit), then shifted by the 2026 national swing
  P_fund = Phi(mu / sigma), sigma from historical residuals (incumbent vs open)
  P_rating = historical hit rate of the consensus rating category
  P_model  = logit-average of P_fund and P_rating (rating only where we have one)
  P_market = Kalshi district prices (Oct 26 snapshot); safe seats not captured are imputed
             at 90-99.5c. Replace with the Nov 1 pull.
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

import fundamentals as F

REDRAWN_2026 = {"AL", "CA", "FL", "LA", "NC", "OH", "TN", "TX", "UT"}
RATING_P = {"Safe": 0.99, "Likely": 0.92, "Lean": 0.78, "Toss-up": 0.50}
KALSHI_EXPECTED_D_SEATS = 238.3   # Kalshi seat-count market mean, Oct 26 (sanity check only)
logit = lambda p: np.log(p / (1 - p))
expit = lambda x: 1 / (1 + np.exp(-x))


def rating_to_pD(r):
    if r == "Toss-up":
        return 0.5
    cat, side = r.split()
    p = RATING_P[cat]
    return p if side == "D" else 1 - p


def load_pres():
    p = pd.read_csv("data/raw/pres2024_by_cd.csv")
    p["cd"] = p.CD.str.replace("-AL", "-01", regex=False)
    num = lambda s: pd.to_numeric(s.astype(str).str.rstrip("%"), errors="coerce")
    p["pres24"] = num(p["Harris %"]) - num(p["Trump %"])
    # NY-21 is blank in the source sheet; Trump carried it by ~21 points (public results).
    p.loc[p.cd == "NY-21", "pres24"] = -21.0
    return p[["cd", "pres24"]]


def build(env):
    fit = json.load(open("output/fundamentals_fit.json"))
    c = fit["coefs"]
    swing26 = (env["generic_ballot_dem_margin"] - env["house_popular_vote_2024_dem_margin"]) / 100

    allp = F.build_all()
    h24 = allp[allp.year == 2024].copy()
    h24["cd"] = h24.st + "-" + h24.cd.astype(int).astype(str).str.zfill(2)
    h24 = h24.drop_duplicates("cd")
    df = load_pres().merge(
        h24[["cd", "st", "margin", "contested", "win_party", "inc_t"]].rename(
            columns={"margin": "m24", "inc_t": "inc24"}), on="cd", how="left")
    df["st"] = df.cd.str[:2]

    # Model B mapping fit on the 2024 cross-section (contested D-vs-R races)
    cs = df[df.contested.fillna(False).astype(bool)].dropna(subset=["m24", "pres24"])
    lr = LinearRegression().fit(np.c_[cs.pres24 / 100, cs.inc24], cs.m24)
    a, bp, bi = lr.intercept_, lr.coef_[0], lr.coef_[1]

    ov = pd.read_csv("data/inputs/house_overrides_2026.csv").set_index("cd")
    rt = pd.read_csv("data/inputs/house_ratings_2026.csv").set_index("cd")
    incmap = {"D": 1, "R": -1, "I": -0.5, "open": 0}

    rows = []
    for _, r in df.iterrows():
        o = ov.loc[r.cd] if r.cd in ov.index else None
        changed = (r.st in REDRAWN_2026) and not (o is not None and o.lines_changed == 0)
        pres = r.pres24
        if o is not None and not pd.isna(o.pres_margin_new):
            pres = float(o.pres_margin_new)
        if o is not None and isinstance(o.inc_2026, str):
            inc = incmap[o.inc_2026]
        else:
            inc = {"D": 1, "R": -1}.get(r.win_party, 0)
        same = o.same_party if (o is not None and isinstance(o.same_party, str)) else ""

        muB = a + bp * pres / 100 + bi * inc + c["swing"] * swing26
        muA = np.nan
        if (not changed) and bool(r.contested) and not pd.isna(r.m24):
            muA = (c["intercept"] + c["lag_margin"] * r.m24 + c["inc_t"] * inc
                   + c["inc_prev"] * (r.inc24 if not pd.isna(r.inc24) else 0)
                   + c["swing"] * swing26)
        mu = muB if np.isnan(muA) else 0.5 * muA + 0.5 * muB
        sd = fit["sd_inc"] if inc != 0 else fit["sd_open"]
        if np.isnan(muA):
            sd *= 1.10          # lean-only estimate is noisier (esp. hand-estimated new lines)
        sd_nat = c["swing"] * env["national_error_sd_pts"] / 100   # generic-ballot uncertainty
        sd = float(np.sqrt(sd ** 2 + sd_nat ** 2))
        p_fund = norm.cdf(mu / sd)
        rating = rt.rating[r.cd] if r.cd in rt.index else ""
        if same:
            p_fund = 1.0 if same == "D" else 0.0
        if rating and not same:
            p_rate = rating_to_pD(rating)
            p_model = expit(0.5 * logit(np.clip(p_fund, .005, .995)) + 0.5 * logit(p_rate))
        else:
            p_rate, p_model = np.nan, p_fund
        rows.append(dict(race_id=f"HOUSE-{r.cd}", chamber="House", state=r.st, district=r.cd,
                         special=0, pres_margin_used=pres, lines_changed=int(changed),
                         inc_2026={1: "D", -1: "R", -0.5: "I", 0: "open"}[inc],
                         same_party=same, mu_A=muA, mu_B=muB, mu=mu, sigma=sd,
                         rho=sd_nat ** 2 / sd ** 2,
                         p_fund=p_fund, rating=rating, p_rating=p_rate, p_model_D=p_model))
    out = pd.DataFrame(rows)

    # --- market: Kalshi district prices (Oct 26 snapshot) ---
    # Every non-safe Kalshi bucket list was captured in full, so a district missing from the file is
    # in a Safe bucket (favourite >= 90c). For those we keep the side our fundamentals favour and put
    # the price at the sharpened fundamentals odds clipped to 90-99.5c.
    k = pd.read_csv("data/inputs/kalshi_house_2026-10-26.csv").set_index("cd")
    sharp = expit(2.0 * logit(np.clip(out.p_model_D, 1e-4, 1 - 1e-4)))
    safe_fill = np.where(sharp >= 0.5, np.clip(sharp, 0.90, 0.995), np.clip(sharp, 0.005, 0.10))
    pm, src, bucket = [], [], []
    for i, cd in enumerate(out.district):
        if out.same_party.iloc[i]:
            pm.append(0.995 if out.same_party.iloc[i] == "D" else 0.005)
            src.append("same-party race (seat outcome certain)"); bucket.append("same-party")
        elif cd in k.index:
            pm.append(float(k.kalshi_p_D[cd])); src.append("Kalshi " + k.value_type[cd]); bucket.append(k.kalshi_bucket[cd])
        else:
            pm.append(float(safe_fill[i])); src.append("Kalshi Safe bucket (>=90c), value imputed")
            bucket.append("Safe D" if safe_fill[i] >= 0.5 else "Safe R")
    out["p_market_D"] = pm
    out["market_source"] = src
    out["kalshi_bucket"] = bucket
    out["review_flag"] = ((out.rating == "") & (out.mu.abs() < 0.07) & (out.same_party == "")).astype(int)
    pr = out.p_market_D.values
    s = float("nan")
    meta = dict(model_B=dict(intercept=a, pres=bp, inc=bi), swing26=swing26,
                kalshi_seat_mean=KALSHI_EXPECTED_D_SEATS, expected_D_seats_model=float(out.p_model_D.sum()),
                expected_D_seats_market_proxy=float(pr.sum()))
    return out, meta


if __name__ == "__main__":
    env = json.load(open("data/inputs/national_environment.json"))
    out, meta = build(env)
    print(json.dumps(meta, indent=2))
    print("Flagged (unrated but fundamentals within 7 pts):")
    print(out[out.review_flag == 1][["district", "pres_margin_used", "inc_2026", "mu_A", "mu_B", "p_fund"]]
          .round(3).to_string(index=False))
    out.to_csv("output/house_model_detail.csv", index=False)
