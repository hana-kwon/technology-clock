# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: 2026-11-01
# TIME TAKEN   : (filled after the run)
# DESCRIPTION  : Scores spec 70 (ChronoGPT two-choice contrast clock) with the pass rules written in
#                analysis/70_spec_chronogpt_contrast_clock_261031.md before any run: Part A known-answer test
#                (event jump vs never-happened control jump; >= 10 of 13 events; always > never controls in >= 20 of 26
#                vintages) and Part B directional checks (human_level_ai, self_driving). Works with the vintages present;
#                missing vintages are listed and the result is marked interim.
# INPUT FILES  : the cluster run folder (contrast_clock/output/)70_contrast_clock_<vintage>.json
#                (copied to data/model_outputs/70_contrast_clock/ with --sync; generic sentences, no firm data)
# OUTPUT FILES : analysis/86_contrast_clock_scores_261101.json
# ============================================================
import argparse
import json
import shutil
from collections import defaultdict
from math import comb
from pathlib import Path

TOPIC = Path(__file__).resolve().parents[1]
LOCAL = TOPIC / "data" / "model_outputs" / "70_contrast_clock"
MOUNT = Path("cluster_output/contrast_clock")
YEARS = list(range(1999, 2025))
# event years from spec 70 (only the year enters the rule)
EVENTS = {"euro_coins": 2002, "genome_complete": 2003, "pluto_dwarf": 2006, "iphone": 2007, "bitcoin": 2009,
          "watson_jeopardy": 2011, "higgs": 2012, "grav_waves": 2016, "blackhole_image": 2019, "moon_farside": 2019,
          "mrna_vaccine": 2020, "mars_helicopter": 2021, "chatgpt": 2022}
ALWAYS = ["ctrl_moon_landing", "ctrl_deep_blue"]
NEVER = ["ctrl_fusion_plant", "ctrl_mars_humans", "ctrl_rsa_broken", "ctrl_flying_cars"]


def mean(x):
    x = list(x)
    return sum(x) / len(x) if x else None


def margins(rows, part, pos, neg):
    # ★ CORE LOGIC (spec 70): margin = mean log-prob of the yes/near sentences - mean of the no/far sentences
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["part"] == part:
            by[r["key"]][r["side"]].append(r["logprob"])
    return {k: mean(v[pos]) - mean(v[neg]) for k, v in by.items()}


def window(series, years):
    vals = [series[y] for y in years if y in series]
    return mean(vals), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true")
    a = ap.parse_args()
    if a.sync:
        LOCAL.mkdir(parents=True, exist_ok=True)
        for f in sorted(MOUNT.glob("70_contrast_clock_*.json")):
            if not (LOCAL / f.name).exists():
                shutil.copy2(f, LOCAL / f.name)
    A, B, meta = {}, {}, {}
    for f in sorted(LOCAL.glob("70_contrast_clock_*.json")):
        d = json.loads(f.read_text())
        y = int(d["vintage"][:4])
        A[y] = margins(d["rows"], "A", "yes", "no")
        B[y] = margins(d["rows"], "B", "near", "far")
        meta[y] = d["compute"]
    have = sorted(A)
    out = {"spec": "analysis/70_spec_chronogpt_contrast_clock_261031.md (rules fixed 2026-10-31 before any run)",
           "vintages_present": have, "vintages_missing": [y for y in YEARS if y not in A],
           "compute_values": sorted(set(meta.values())), "interim": len(have) < len(YEARS)}
    # Part A
    ev = {}
    for k, E in EVENTS.items():
        s = {y: A[y][k] for y in have}
        ctrl = {y: mean(A[y][c] for c in NEVER) for y in have}
        after, na = window(s, range(E, E + 3))
        before, nb = window(s, range(E - 3, E))
        c_after, _ = window(ctrl, range(E, E + 3))
        c_before, _ = window(ctrl, range(E - 3, E))
        if na == 0 or nb == 0:
            ev[k] = {"year": E, "scored": False, "why": f"no vintage in the {'after' if na == 0 else 'before'} window"}
            continue
        jump, cjump = after - before, c_after - c_before
        ev[k] = {"year": E, "scored": True, "jump": round(jump, 3), "control_jump": round(cjump, 3),
                 "passes": jump > cjump, "n_before": nb, "n_after": na}
    scored = [v for v in ev.values() if v["scored"]]
    npass = sum(v["passes"] for v in scored)
    # one-sided sign test against 50%
    p = sum(comb(len(scored), i) for i in range(npass, len(scored) + 1)) / 2 ** len(scored) if scored else None
    ctrl_ok = sum(mean(A[y][c] for c in ALWAYS) > mean(A[y][c] for c in NEVER) for y in have)
    out["partA"] = {"events": ev, "events_scored": len(scored), "events_passed": npass,
                    "sign_test_p_one_sided": round(p, 4) if p is not None else None,
                    "rule_events": "pass needs >= 10 of 13", "controls_always_gt_never_vintages": f"{ctrl_ok}/{len(have)}",
                    "rule_controls": ">= 20 of 26 vintages",
                    "margin_by_vintage": {k: {y: round(A[y][k], 3) for y in have} for k in list(EVENTS) + ALWAYS + NEVER}}
    # Part B: clock = margin(topic) - margin(neutral)
    clock = {k: {y: round(B[y][k] - B[y]["neutral_highway"], 3) for y in have}
             for k in B[have[0]] if k != "neutral_highway"}
    hl, sd = clock["human_level_ai"], clock["self_driving"]
    c1 = (window(hl, range(2022, 2025))[0], window(hl, range(2019, 2022))[0])
    c2 = (window(sd, range(2015, 2019))[0], window(sd, range(2021, 2025))[0])
    out["partB"] = {"clock_by_vintage": clock,
                    "check_human_level_ai_nearer_2022plus_vs_2019_21": {"2022+": round(c1[0], 3), "2019-21": round(c1[1], 3),
                                                                        "holds": c1[0] > c1[1]},
                    "check_self_driving_nearer_2015_18_vs_2021_24": {"2015-18": round(c2[0], 3), "2021-24": round(c2[1], 3),
                                                                     "holds": c2[0] > c2[1]}}
    (TOPIC / "analysis" / "86_contrast_clock_scores_261101.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("vintages_missing", "compute_values", "interim")}, indent=1))
    print("Part A:", npass, "of", len(scored), "scored events pass; p =", out["partA"]["sign_test_p_one_sided"],
          "| controls always>never in", out["partA"]["controls_always_gt_never_vintages"])
    for k, v in ev.items():
        print(" ", k, v)
    print("Part B checks:", json.dumps({k: v for k, v in out["partB"].items() if k.startswith("check")}))


if __name__ == "__main__":
    main()
