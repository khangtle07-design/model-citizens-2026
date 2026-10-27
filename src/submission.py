"""
Write the Full Congress submission file with exactly the required columns:
    state, district, chamber, predicted_winner, party, model_probability
House rows come first, with full district IDs (CA-11, AK-AL). Senate rows have a blank district,
except the two specials, marked SPECIAL. model_probability is the model's (posterior) chance that
predicted_winner wins.
"""
import pandas as pd

AT_LARGE = {"AK", "DE", "ND", "SD", "VT", "WY"}
SPECIALS = {"SEN-OH", "SEN-FL"}
COLS = ["state", "district", "chamber", "predicted_winner", "party", "model_probability"]


def slim(src, dst):
    f = pd.read_csv(src)

    def dist(r):
        if r.chamber == "Senate":
            return "SPECIAL" if r.race_id in SPECIALS else ""
        st, n = r.district.split("-")
        return f"{st}-AL" if st in AT_LARGE else f"{st}-{int(n):02d}"

    out = pd.DataFrame({"state": f.state, "district": f.apply(dist, axis=1), "chamber": f.chamber,
                        "predicted_winner": f.predicted_winner, "party": f.predicted_party,
                        "model_probability": f.model_prob.round(4)})[COLS]
    out = pd.concat([out[out.chamber == "House"], out[out.chamber == "Senate"]])
    out.to_csv(dst, index=False)
    return out


if __name__ == "__main__":
    a = slim("predictions/full_congress_predictions.csv", "predictions/full_congress_submission.csv")
    print(f"full congress: {len(a)} rows {a.chamber.value_counts().to_dict()}")
