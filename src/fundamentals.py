"""
Model 1 -- Fundamentals regression, trained on MIT MEDSL U.S. House results.

Builds a district-year panel, fits
    margin_t = b0 + b1*lag_margin + b2*inc_t + b3*inc_prev + b4*national_swing + e
(margin = Democratic two-party margin, as a fraction), plus the slide's logistic
P(D win) version, and backtests both with leave-one-cycle-out CV.
"""
import json
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

RAW = "data/raw/house_1976_2024.tab"

# National House popular vote margin (D minus R, percentage points). Public record.
NATIONAL_HOUSE_MARGIN = {
    2002: -4.6, 2004: -2.6, 2006: 8.0, 2008: 10.6, 2010: -6.8, 2012: 1.2,
    2014: -5.7, 2016: -1.1, 2018: 8.6, 2020: 3.1, 2022: -2.8, 2024: -2.6,
}
# Cycles whose lagged result is not comparable (new lines everywhere or in a state).
CENSUS_YEARS = {1982, 1992, 2002, 2012, 2022}
MIDDECADE = {  # (year, state_po) where lines changed vs. the prior cycle
    (2004, "TX"), (2006, "TX"), (2006, "GA"), (2016, "FL"), (2016, "VA"),
    (2016, "NC"), (2018, "PA"), (2020, "NC"), (2024, "AL"), (2024, "LA"),
    (2024, "NY"), (2024, "NC"), (2024, "GA"),
}


def _major(party):
    p = str(party).upper()
    if "REPUBLICAN" in p:
        return "R"
    if "DEMOCRAT" in p and "SOCIAL" not in p:
        return "D"
    return "O"


def _lastname(name):
    toks = re.sub(r"[^A-Z ]", " ", str(name).upper()).split()
    toks = [t for t in toks if t not in {"JR", "SR", "II", "III", "IV"}]
    return toks[-1] if toks else ""


def build_all():
    """District-year table for every cycle (no training filters)."""
    h = pd.read_csv(RAW, sep=",", low_memory=False)
    tf = lambda s: s.astype(str).str.upper().eq("TRUE")
    h = h[(h.stage == "GEN") & (~tf(h.special)) & (~tf(h.writein))].copy()
    h["district"] = h["district"].astype(int).replace(0, 1)
    h["maj"] = h["party"].map(_major)
    # collapse fusion lines / vote modes onto the candidate
    c = (h.groupby(["year", "state_po", "district", "candidate"])
           .agg(votes=("candidatevotes", "sum"),
                maj=("maj", lambda s: "D" if "D" in set(s) else ("R" if "R" in set(s) else "O")))
           .reset_index())
    rows = []
    for (yr, st, d), g in c.groupby(["year", "state_po", "district"]):
        g = g.sort_values("votes", ascending=False)
        dv = g[g.maj == "D"].votes.max() if (g.maj == "D").any() else 0
        rv = g[g.maj == "R"].votes.max() if (g.maj == "R").any() else 0
        winner = g.iloc[0]
        rows.append(dict(
            year=yr, st=st, cd=d, dv=dv, rv=rv,
            contested=bool(dv > 0 and rv > 0),
            margin=(dv - rv) / (dv + rv) if dv + rv > 0 else np.nan,
            win_party=winner.maj, win_last=_lastname(winner.candidate),
            d_last=_lastname(g[g.maj == "D"].iloc[0].candidate) if dv > 0 else "",
            r_last=_lastname(g[g.maj == "R"].iloc[0].candidate) if rv > 0 else "",
        ))
    p = pd.DataFrame(rows).sort_values(["st", "cd", "year"])
    prev = p[["year", "st", "cd", "margin", "contested", "win_party", "win_last"]].copy()
    prev["year"] = prev["year"] + 2
    prev.columns = ["year", "st", "cd", "lag_margin", "lag_contested", "lag_win_party", "lag_win_last"]
    p = p.merge(prev, on=["year", "st", "cd"], how="left")

    def inc(r, wp, wl):
        if r[wp] == "D" and r[wl] and r[wl] == r["d_last"]:
            return 1
        if r[wp] == "R" and r[wl] and r[wl] == r["r_last"]:
            return -1
        return 0
    p["inc_t"] = p.apply(lambda r: inc(r, "lag_win_party", "lag_win_last"), axis=1)
    lag_inc = p[["year", "st", "cd", "inc_t"]].copy()
    lag_inc["year"] += 2
    p = p.merge(lag_inc.rename(columns={"inc_t": "inc_prev"}), on=["year", "st", "cd"], how="left")
    p["swing"] = p.year.map(lambda y: (NATIONAL_HOUSE_MARGIN.get(y, np.nan)
                                       - NATIONAL_HOUSE_MARGIN.get(y - 2, np.nan)) / 100)
    return p


def build_panel():
    p = build_all()
    ok = ~p.year.isin(CENSUS_YEARS)
    ok &= ~p.apply(lambda r: (r.year, r.st) in MIDDECADE, axis=1)
    ok &= p.contested & p.lag_contested.fillna(False).astype(bool)
    ok &= p.swing.notna() & p.inc_prev.notna()
    ok &= p.year >= 2004
    return p[ok].copy()


FEATS = ["lag_margin", "inc_t", "inc_prev", "swing"]


def fit(panel):
    X, y = panel[FEATS].values, panel.margin.values
    ols = LinearRegression().fit(X, y)
    resid = y - ols.predict(X)
    comp = panel.lag_margin.abs() < 0.30
    sd_inc = resid[comp & (panel.inc_t != 0)].std()
    sd_open = resid[comp & (panel.inc_t == 0)].std()
    # Year-level shared error (how much do residuals move together within a cycle?)
    yr_mean = pd.Series(resid).groupby(panel.year.values).mean()
    logit = LogisticRegression(C=1e4, max_iter=2000).fit(X, (y > 0).astype(int))
    return ols, logit, dict(sd_inc=sd_inc, sd_open=sd_open, sd_year=yr_mean.std(),
                            n=len(panel), years=sorted(panel.year.unique().tolist()))


def backtest(panel):
    """Leave-one-cycle-out: train on all other cycles, predict the held-out one."""
    from scipy.stats import norm
    out = []
    for yr in sorted(panel.year.unique()):
        tr, te = panel[panel.year != yr], panel[panel.year == yr]
        ols, logit, s = fit(tr)
        mu = ols.predict(te[FEATS].values)
        sd = np.where(te.inc_t != 0, s["sd_inc"], s["sd_open"])
        p_ols = norm.cdf(mu / sd)
        p_log = logit.predict_proba(te[FEATS].values)[:, 1]
        win = (te.margin.values > 0).astype(int)
        comp = (te.lag_margin.abs() < 0.15).values
        out.append(dict(
            year=int(yr), n=len(te), n_competitive=int(comp.sum()),
            acc_ols=float(((p_ols > .5) == win).mean()),
            acc_ols_competitive=float(((p_ols > .5) == win)[comp].mean()) if comp.any() else np.nan,
            brier_ols=float(((p_ols - win) ** 2).mean()),
            brier_logit=float(((p_log - win) ** 2).mean()),
            mae_margin_pts=float(np.abs(mu - te.margin.values).mean() * 100),
        ))
    return pd.DataFrame(out)


if __name__ == "__main__":
    panel = build_panel()
    ols, logit, s = fit(panel)
    coefs = dict(zip(["intercept"] + FEATS, [ols.intercept_] + ols.coef_.tolist()))
    print("OLS margin model:", {k: round(v, 4) for k, v in coefs.items()})
    print("Logit P(D win) coefs:", dict(zip(FEATS, np.round(logit.coef_[0], 3))))
    print("Residual SDs:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
    bt = backtest(panel)
    print(bt.round(3).to_string(index=False))
    json.dump(dict(coefs=coefs, logit=dict(zip(FEATS, logit.coef_[0].tolist()),
                   intercept=float(logit.intercept_[0])), **{k: v for k, v in s.items()}),
              open("output/fundamentals_fit.json", "w"), indent=2, default=float)
    bt.to_csv("output/backtest_house_fundamentals.csv", index=False)
