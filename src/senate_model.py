"""
2026 Senate (Model 2: poll aggregate blended with a fundamentals prior).

fundamentals mu_F = pres2024 margin
                    + b_swing * national swing since 2024            (b from Model 1)
                    + incumbency (2.0 pts elected / 1.0 appointed, toward incumbent party)
                    + 0.5 * candidate effect  (candidate's last-race margin minus the
                                               same-cycle presidential margin; halved for
                                               regression to the mean)
polls       mu_P = nonpartisan-weighted average from the inputs file
posterior   precision-weighted:  1/sd_F^2 and 1/sd_P(n)^2
P(D-side)   = Phi(mu_post / sd_total), sd_total includes a shared national poll error.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

SD_FUND = 8.0      # pts; fundamentals-only Senate error (no polls)


def poll_sd(n):
    if n >= 10:
        return 4.5     # ~historical final-week Senate polling-average error
    if n >= 3:
        return 5.5
    return 7.0


def build(env):
    fit = json.load(open("output/fundamentals_fit.json"))
    b_swing = fit["coefs"]["swing"]
    swing = env["generic_ballot_dem_margin"] - env["house_popular_vote_2024_dem_margin"]
    s = pd.read_csv("data/inputs/senate_2026.csv")
    rows = []
    for _, r in s.iterrows():
        inc_pts = {"elected": 2.0, "appointed": 1.0, "open": 0.0}[r.incumbent_type]
        inc_pts *= 1 if r.incumbent_party == "D" else -1
        if r.incumbent_type == "open":
            inc_pts = 0.0
        indep = r.dem_side_party == "I"
        # independents running in deep-red states get the swing only partially
        sw = b_swing * swing * (0.8 if indep else 1.0)
        mu_f = r.pres2024_margin + sw + inc_pts + 0.5 * r.cand_effect_raw
        poll, n = r.poll_margin, r.n_polls
        if not pd.isna(r.extra_signal_margin):
            # non-poll vote signal (e.g. Alaska's August top-four primary), precision-weighted in
            sp0 = poll_sd(n)
            wa, wb = 1 / sp0 ** 2, 1 / r.extra_signal_sd ** 2
            poll = (wa * poll + wb * r.extra_signal_margin) / (wa + wb)
        if pd.isna(poll):
            mu, sd_post = mu_f, SD_FUND
        else:
            sp = poll_sd(n)
            w_p, w_f = 1 / sp ** 2, 1 / SD_FUND ** 2
            mu = (w_p * poll + w_f * mu_f) / (w_p + w_f)
            sd_post = (w_p + w_f) ** -0.5
        sd_tot = np.sqrt(sd_post ** 2 + env["senate_poll_shared_error_sd_pts"] ** 2)
        p = norm.cdf(mu / sd_tot)
        rows.append(dict(race_id=r.race_id, chamber="Senate", state=r.state, district="",
                         special=int(r.special), dem_side_candidate=r.dem_side_candidate,
                         dem_side_party=r.dem_side_party, rep_candidate=r.rep_candidate,
                         mu_fund=mu_f, poll_margin=poll, mu=mu, sigma=sd_tot,
                         rho=env["senate_poll_shared_error_sd_pts"] ** 2 / sd_tot ** 2,
                         p_model_D=p, p_market_D=r.kalshi_dem_side_prob,
                         market_source="Kalshi snapshot Oct 21-26 (replace with Nov 1)"))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    env = json.load(open("data/inputs/national_environment.json"))
    out = build(env)
    print(out[["race_id", "dem_side_candidate", "rep_candidate", "mu_fund", "poll_margin", "mu",
               "sigma", "p_model_D", "p_market_D"]].round(2).to_string(index=False))
    print("Expected D-side Senate wins (of 35):", round(out.p_model_D.sum(), 2))
    out.to_csv("output/senate_model_detail.csv", index=False)
