"""
Build data/inputs/house_candidates_2026.csv: the Democratic-side and Republican nominee name for
all 435 districts, so predictions name a person instead of "Republican nominee".

Sources, in priority order (recorded per row in name_source_D / name_source_R):
  1. "certified list": general-election fields collected from The Ballot Brief's U.S. House page
     (built from state certified candidate lists) and news/Wikipedia race pages, Oct 2026.
     Raw text kept in data/raw/house_candidates_snippets_2026-10-26.txt.
  2. "incumbent": the sitting member from unitedstates/congress-legislators (updated Sep 2026),
     used only when the member is not known to be retiring, defeated in a primary, or running in
     a different district.
  3. otherwise a labelled placeholder ("Republican nominee (open seat, name not collected)").
"""
import json
import pandas as pd
import yaml

TERR = {"AS", "DC", "GU", "MP", "PR", "VI"}
# Incumbents not on the 2026 ballot in their own district (retiring, other office, lost primary,
# resigned/died, or moved district). Source: Wikipedia / Ballotpedia retirement lists, Oct 2026.
NOT_RUNNING = set("""
AL-01 AZ-01 AZ-05 CA-48 FL-02 FL-11 FL-16 FL-19 GA-01 GA-10 GA-11 IA-02 IA-04 KY-06 LA-05 MI-10
MT-01 NE-02 NV-02 NY-21 OK-01 SC-01 SC-05 SD-01 TN-06 TX-08 TX-10 TX-19 TX-21 TX-22 TX-38 UT-04
WA-04 WI-07 WY-01 NC-11 CA-11 CA-26 FL-24 GA-13 IL-02 IL-04 IL-07 IL-08 IL-09 LA-06 ME-02 MD-05
MA-06 MI-11 MN-02 NH-01 NJ-12 NY-07 NY-12 PA-03 TN-09 TX-30 TX-33 TX-37 CO-01 CT-01 MI-13 NY-10
NY-13 TX-18 FL-07 KY-04 TN-05 TX-02 FL-20 TX-23 CA-14 TX-09 TX-32 TX-35 CA-38 UT-01
""".split())
# Incumbent's 2026 candidacy could not be confirmed (gap in the retirement list we could read).
UNVERIFIED = {"MI-01", "MI-04"}
# same-party top-two races: which candidate we pick (incumbent where there is one)
SAME_PARTY_PICK = {"CA-04": "Mike Thompson", "CA-07": "Doris Matsui", "CA-12": "Lateefah Simon",
                   "CA-14": "Aisha Wahab", "CA-29": "Luz Maria Rivas",
                   "CA-11": "Scott Wiener", "CA-40": "Young Kim"}
SAME_PARTY_NOTE = {"CA-11": "open D-vs-D seat; favourite not verified",
                   "CA-40": "R-vs-R between two incumbents; favourite not verified"}


def build():
    col = json.load(open("data/raw/house_candidates_collected_2026-10-26.json"))
    leg = yaml.safe_load(open("data/raw/legislators-current_2026-10-26.yaml"))
    inc = {}
    for p in leg:
        t = p["terms"][-1]
        if t["type"] != "rep" or t["state"] in TERR:
            continue
        cd = f'{t["state"]}-{(t.get("district") or 1):02d}'
        inc[cd] = (p["name"].get("official_full") or f'{p["name"]["first"]} {p["name"]["last"]}',
                   t["party"][0])
    on_ballot_elsewhere = {n for cd, v in col.items() for n in v["D"] + v["R"]}
    rows = []
    for cd in sorted(set(inc) | set(col) | {"CA-14", "FL-20", "TX-23"}):
        c = col.get(cd, {"D": [], "R": [], "other": []})
        out = dict(cd=cd, dem_side_party="D", note="")
        for side, label in (("D", "Democratic"), ("R", "Republican")):
            key = "dem_side_candidate" if side == "D" else "rep_candidate"
            if cd in SAME_PARTY_PICK and c[side]:
                out[key], src = SAME_PARTY_PICK[cd], "certified list"
                out["note"] = SAME_PARTY_NOTE.get(cd, "same-party race; incumbent picked")
            elif c[side]:
                out[key], src = c[side][0], "certified list"
            elif (cd in inc and inc[cd][1] == side and cd not in NOT_RUNNING
                  and inc[cd][0] not in on_ballot_elsewhere):
                out[key] = inc[cd][0]
                src = "incumbent (candidacy not verified)" if cd in UNVERIFIED else "incumbent"
            else:
                open_ = cd in NOT_RUNNING or (cd in inc and inc[cd][1] != side) is False
                out[key] = f"{label} nominee (name not collected)"
                src = "placeholder"
            out[f"name_source_{side}"] = src
        rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv("data/inputs/house_candidates_2026.csv", index=False)
    return df


if __name__ == "__main__":
    df = build()
    print(len(df), "districts")
    print(df.name_source_D.value_counts().to_dict(), df.name_source_R.value_counts().to_dict())
