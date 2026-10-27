"""Parse data/raw/house_candidates_snippets_2026-10-26.txt into the collected-names JSON."""
import json
import re

PARTY = {"Democratic": "D", "D": "D", "Democratic-Farmer-Labor": "D", "Democratic-NPL": "D",
         "Republican": "R", "R": "R"}
raw = open("data/raw/house_candidates_snippets_2026-10-26.txt").read()
ent = re.findall(r"U\.S\. House, ([A-Z]{2})-(\d+|AL):?\s+(.*?)(?=\s*(?:·|\.\.\.|U\.S\. House,|\n|$))", raw)
names = {}
for st, num, body in ent:
    cd = f"{st}-{int(num):02d}" if num != "AL" else f"{st}-01"
    rec = names.setdefault(cd, {"D": [], "R": [], "other": []})
    for n, p in re.findall(r"([^,()·]+?)\s*\(([^)]+)\)", body):
        n, p = n.strip(), p.strip()
        k = PARTY.get(p)
        if k and n not in rec[k]:
            rec[k].append(n)
        elif not k and [n, p] not in rec["other"]:
            rec["other"].append([n, p])
names["MT-01"]["R"] = ["Aaron Flint"]          # snippet gave surname only; full name from same source
json.dump(names, open("data/raw/house_candidates_collected_2026-10-26.json", "w"), indent=1)
print(len(names), "districts parsed")
