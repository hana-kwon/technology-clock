# ============================================================
# START DATE   : 2026-11-01
# CODE RUN DATE: (after the Grid jobs rp_* finish)
# TIME TAKEN   : (filled after the run)
# DESCRIPTION  : Reader pilot step R6 for the ChronoGPT-Instruct reader (protocol 75, rules R6-C fixed at 2026-11-01
#                14:06 EST, when no Grid output existed). Reads the Grid outputs of code 77 and reports: completeness,
#                format compliance (decides generated vs log-prob main answer), rerun agreement (greedy vs t=0.7),
#                agreement with the Claude reader (v1 ~ run A, v4 ~ run E), accuracy on the 40 hand-judged chunks,
#                post-date (forbidden) words, keyword-miss rate, and the realtime vs 2024 vintage comparison (L3).
#                EXPLORATORY (added 2026-11-01 14:3x EST after 8 interim chunks): a calibrated cloze answer that
#                subtracts each option's log-prob under content-free inputs (Zhao et al. 2021), when code 77 saved
#                77_calib_*.json for that vintage. Not part of the pre-fixed rules R6-C.
#                Answer parsing, kappa, correlation and the forbidden-word list are imported from code 78 so that
#                both readers are scored by the same rules.
# INPUT FILES  : data/reader_pilot_261101/grid_output/77_reader_{v1|v4}_{realtime|all}_<vintage>.jsonl
#                  and 77_calib_{v1|v4}_{realtime|all}_<vintage>.json (from 2026-11-01 14:3x EST runs on),
#                  (copied from the cluster run folder (reader_pilot/output/) with --sync),
#                data/reader_pilot_261101/02_pilot_sample.jsonl, 03_claude_run{A,E}.jsonl,
#                05_judgments_sample40_runA_261101.csv (read only), code/78_reader_pilot_eval_261101.py (functions)
# OUTPUT FILES : data/reader_pilot_261101/06_chronogpt_parsed.csv, 06_chronogpt_eval_summary.json
# ============================================================
import argparse
import csv
import importlib.util
import json
import math
import shutil
import statistics
from collections import Counter
from math import comb
from pathlib import Path

TOPIC = Path(__file__).resolve().parents[1]
P = TOPIC / "data" / "reader_pilot_261101"
MOUNT = Path("cluster_output/reader_pilot")

_spec = importlib.util.spec_from_file_location("ev78", TOPIC / "code" / "78_reader_pilot_eval_261101.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)
OPTIONS, YEARS, FORBIDDEN = ev.OPTIONS, ev.YEARS, ev.FORBIDDEN
NOINFO = "no information is provided"
BINS = OPTIONS[:5]  # the five ordered time bins; "never" and "no information" have no rank
TAGS = {"v1": "A", "v4": "E"}  # ChronoGPT prompt tag -> Claude run with the same prompt
VINTAGES = [f"{y}1231" for y in range(2010, 2025)]
FORMAT_RULE = 0.80  # protocol 75 R6: below this share of parsable greedy replies, log-probs give the main answer


def clean(reply):
    # same post-processing as the vendor's extract_response(): drop an echoed "### Response:" and strip
    return reply.replace("### Response:", "").strip()


def gen_choice(reply):
    return ev.parse(clean(reply))[0]


def lp_choice(lps, key):
    # ★ CORE LOGIC: cloze answer = option with the highest per-token mean log-prob (Brown et al. 2020, sec. 2.4);
    # "sum" is reported as a secondary rule
    return max(OPTIONS, key=lambda o: lps[o][key])


def calib_choice(r, tag, cal):
    # ★ EXPLORATORY: Zhao et al. (2021): average the content-free probabilities ("N/A", "[MASK]", "") and divide them
    # out. Here on the per-token mean log-prob scale: score = mean_lp(option | chunk) - log(avg_cf exp(mean_lp(option | cf))).
    c = cal.get((tag, r["mode"], r["vintage"]), {}).get(str(r["call_year"]))
    if not c:
        return None
    base = {o: math.log(sum(math.exp(c[cf][o]["mean"]) for cf in c) / len(c)) for o in OPTIONS}
    return max(OPTIONS, key=lambda o: r["option_logprobs"][o]["mean"] - base[o])


def sign_test(npos, nneg):
    n = npos + nneg
    if n == 0:
        return None
    k = min(npos, nneg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def share(num, den):
    return None if den == 0 else round(num / den, 4)


def load_grid(gdir):
    recs, problems = {}, []
    for f in sorted(gdir.glob("77_reader_*.jsonl")):
        _, _, tag, mode, vint = f.stem.split("_")
        seen = {}
        for i, line in enumerate(f.open()):
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"{f.name} line {i + 1}: not JSON (cut line from a killed job?)")
                continue
            if j["pilot_id"] in seen:
                problems.append(f"{f.name}: duplicate {j['pilot_id']} (kept the first)")
                continue
            seen[j["pilot_id"]] = j
        recs[(tag, mode, vint)] = seen
    cal = {}
    for f in sorted(gdir.glob("77_calib_*.json")):
        _, _, tag, mode, vint = f.stem.split("_")
        cal[(tag, mode, vint)] = json.loads(f.read_text())["by_call_year"]
    return recs, cal, problems


def claude_choices(run):
    out = {}
    for line in (P / f"03_claude_run{run}.jsonl").open():
        j = json.loads(line)
        if not j["is_error"] and "claude-sonnet-5-5" in j["model_ids"]:  # same filter as code 78
            out[j["pilot_id"]] = ev.parse(j["reply"])[0]
    return out


def agreement(a, b):
    """a, b: dict pilot_id -> choice (None = unparsed). Pairs where both are parsed."""
    ids = sorted(k for k in a if k in b and a[k] is not None and b[k] is not None)
    x, y = [a[k] for k in ids], [b[k] for k in ids]
    info = lambda c: c != NOINFO
    tb = [k for k in ids if a[k] in BINS and b[k] in BINS]
    lin = [[abs(i - j) for j in range(5)] for i in range(5)]
    num = [k for k in ids if YEARS.get(a[k]) is not None and YEARS.get(b[k]) is not None]
    return {"n_pairs": len(ids), "same_choice_share": share(sum(p == q for p, q in zip(x, y)), len(ids)),
            "kappa_7cat": ev.kappa(x, y, OPTIONS),
            "info_agree_share": share(sum(info(p) == info(q) for p, q in zip(x, y)), len(ids)),
            "info_kappa": ev.kappa([info(p) for p in x], [info(q) for q in y], [True, False]),
            "info_a_share": share(sum(map(info, x)), len(ids)), "info_b_share": share(sum(map(info, y)), len(ids)),
            "both_timed_n": len(tb),
            "weighted_kappa_linear_5bins": ev.kappa([a[k] for k in tb], [b[k] for k in tb], BINS, lin),
            "both_numeric_n": len(num), "numeric_corr": ev.corr([YEARS[a[k]] for k in num], [YEARS[b[k]] for k in num])}


def acc40(choice, claude_a, jud):
    # protocol 75 R6-C: same rule as result 3 of analysis 78 (A kept = right; B/C/D -> "no information" = right; E out)
    # a chunk with no output (missing vintage) is skipped, as "MISSING" is in code 78; an unparsed reply (None) counts
    right = n = 0
    for pid, j in jud.items():
        if pid not in choice:
            continue
        c = choice[pid]
        if j.startswith("A_"):
            n += 1
            right += c is not None and c == claude_a.get(pid)
        elif j[:2] in ("B_", "C_", "D_"):
            n += 1
            right += c == NOINFO
    return f"{right}/{n}"


def forbidden(recs, sample):
    hits = []
    for pid, r in recs.items():
        s = sample[pid]
        txt = ev.norm(s["text"]).lower()
        for field in ("greedy", "sampled_t07"):
            low = r[field].lower()
            for term, d in FORBIDDEN.items():
                if s["call_date"] < d and term in low and term not in txt:
                    hits.append({"pilot_id": pid, "vintage": r["vintage"], "field": field, "term": term})
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_dir", default=str(P / "grid_output"))
    ap.add_argument("--sync", action="store_true", help="copy new or changed jsonl files from the Grid mount first")
    ap.add_argument("--out_dir", default=str(P), help="where 06_* go (tests write to a scratch folder)")
    a = ap.parse_args()
    odir = Path(a.out_dir)
    gdir = Path(a.grid_dir)
    if a.sync:
        gdir.mkdir(exist_ok=True)
        for f in sorted(list(MOUNT.glob("77_reader_*.jsonl")) + list(MOUNT.glob("77_calib_*.json"))):
            dst = gdir / f.name
            if not dst.exists() or dst.stat().st_size != f.stat().st_size:
                shutil.copy2(f, dst)
                print("copied", f.name, f.stat().st_size, "bytes")
    sample = {json.loads(l)["pilot_id"]: json.loads(l) for l in (P / "02_pilot_sample.jsonl").open()}
    jud = {r["pilot_id"]: r["judgment"] for r in csv.DictReader((P / "05_judgments_sample40_runA_261101.csv").open())
           if r.get("judgment")}
    claude = {tag: claude_choices(run) for tag, run in TAGS.items()}
    recs, cal, problems = load_grid(gdir)
    out = {"rules": "protocol 75 R6-C, fixed 2026-11-01 14:06 EST", "grid_dir": str(gdir), "problems": problems}

    # completeness
    need = Counter(s["vintage"] for s in sample.values())
    comp, missing = {}, []
    for tag in TAGS:
        for v in VINTAGES:
            got = len(recs.get((tag, "realtime", v), {}))
            comp[f"{tag}_realtime_{v}"] = f"{got}/{need[v]}"
            if got < need[v]:
                missing.append(f"{tag}_realtime_{v}")
    got = len(recs.get(("v4", "all", "20241231"), {}))
    comp["v4_all_20241231"] = f"{got}/{len(sample)}"
    if got < len(sample):
        missing.append("v4_all_20241231")
    out["completeness"] = comp
    out["incomplete"] = missing
    allrecs = [r for d in recs.values() for r in d.values()]
    out["compute_values"] = dict(Counter(r["compute"] for r in allrecs))
    out["gpus"] = dict(Counter(str(r["gpu"]) for r in allrecs))
    secs = [r["seconds"] for r in allrecs]
    if secs:
        q = statistics.quantiles(secs, n=10) if len(secs) > 1 else secs * 9
        out["seconds_per_chunk"] = {"n": len(secs), "median": statistics.median(secs), "p90": round(q[-1], 1),
                                    "max": max(secs), "total_hours": round(sum(secs) / 3600, 2)}

    rows, main_method = [], {}
    for tag in TAGS:
        rt = {pid: r for v in VINTAGES for pid, r in recs.get((tag, "realtime", v), {}).items()}
        for pid, r in rt.items():
            r["call_year"] = sample[pid]["call_year"]
        res = {"n": len(rt)}
        g = {pid: gen_choice(r["greedy"]) for pid, r in rt.items()}
        smp = {pid: gen_choice(r["sampled_t07"]) for pid, r in rt.items()}
        lpm = {pid: lp_choice(r["option_logprobs"], "mean") for pid, r in rt.items()}
        lps = {pid: lp_choice(r["option_logprobs"], "sum") for pid, r in rt.items()}
        lpc = {pid: c for pid, r in rt.items() if (c := calib_choice(r, tag, cal)) is not None}
        fmt = share(sum(c is not None for c in g.values()), len(g))
        res["format_greedy_share"] = fmt
        res["format_sampled_share"] = share(sum(c is not None for c in smp.values()), len(smp))
        # ★ CORE LOGIC: main answer chosen by the pre-fixed 80% rule (protocol 75 R6, R6-C)
        method = None if fmt is None else ("greedy" if fmt >= FORMAT_RULE else "logprob_mean")
        main_method[tag] = method
        res["main_method"] = method
        main = g if method == "greedy" else lpm
        res["choice_counts"] = {name: dict(Counter(str(c) for c in d.values()))
                                for name, d in (("greedy", g), ("logprob_mean", lpm), ("logprob_sum", lps), ("main", main))}
        for st in ("keyword", "nonkeyword_same_call"):
            ids = [pid for pid in main if sample[pid]["stratum"] == st and main[pid] is not None]
            res[f"main_informative_share_{st}"] = share(sum(main[p] != NOINFO for p in ids), len(ids))
        res["rerun_greedy_vs_t07"] = agreement(g, smp)
        res["greedy_vs_logprob_mean"] = agreement(g, lpm)
        res["vs_claude"] = {name: agreement(d, claude[tag]) for name, d in
                            (("main", main), ("greedy", g), ("logprob_mean", lpm))}
        res["acc_sample40"] = {name: acc40(d, claude["v1"], jud) for name, d in
                               (("main", main), ("greedy", g), ("logprob_mean", lpm))}
        res["acc_sample40_claude_same_prompt"] = acc40(claude[tag], claude["v1"], jud)
        res["forbidden_hits"] = forbidden(rt, sample)
        res["unparsed_greedy_starts_top20"] = Counter(clean(rt[p]["greedy"])[:60] for p, c in g.items()
                                                      if c is None).most_common(20)
        res["by_vintage"] = {}
        for v in VINTAGES:
            ids = [p for p, r in rt.items() if r["vintage"] == v]
            res["by_vintage"][v] = {"n": len(ids), "format": share(sum(g[p] is not None for p in ids), len(ids)),
                                    "informative_main": share(sum(main[p] not in (None, NOINFO) for p in ids),
                                                              sum(main[p] is not None for p in ids))}
        # exploratory, not a pre-fixed rule: only chunks whose vintage has a calibration file
        res["exploratory_calibrated"] = {"n": len(lpc), "vintages_with_calibration": sorted(
            {rt[p]["vintage"] for p in lpc}), "choice_counts": dict(Counter(lpc.values())),
            "vs_claude": agreement(lpc, claude[tag]), "acc_sample40": acc40(lpc, claude["v1"], jud),
            "same_chunks_uncalibrated_vs_claude": agreement({p: lpm[p] for p in lpc}, claude[tag])}
        out[tag] = res
        for pid, r in rt.items():
            rows.append({"pilot_id": pid, "prompt": tag, "mode": "realtime", "vintage": r["vintage"],
                         "call_year": sample[pid]["call_year"], "stratum": sample[pid]["stratum"],
                         "greedy_choice": g[pid], "sampled_choice": smp[pid], "logprob_mean_choice": lpm[pid],
                         "logprob_sum_choice": lps[pid], "logprob_calib_choice": lpc.get(pid), "main_choice": main[pid],
                         "claude_choice": claude[tag].get(pid),
                         "seconds": r["seconds"], "greedy": clean(r["greedy"]), "sampled_t07": clean(r["sampled_t07"])})

    # L3: realtime vs 2024 vintage reading every chunk (prompt v4)
    allv = recs.get(("v4", "all", "20241231"), {})
    rt4 = {pid: r for v in VINTAGES for pid, r in recs.get(("v4", "realtime", v), {}).items()}
    if allv and rt4 and main_method.get("v4"):
        pick = (lambda r: gen_choice(r["greedy"])) if main_method["v4"] == "greedy" else \
            (lambda r: lp_choice(r["option_logprobs"], "mean"))
        A = {p: pick(r) for p, r in allv.items()}
        R = {p: pick(r) for p, r in rt4.items()}
        same_model = [p for p in A if p in R and rt4[p]["vintage"] == "20241231"]
        older = [p for p in A if p in R and rt4[p]["vintage"] != "20241231"]
        l3 = {"main_method": main_method["v4"],
              "a_same_model_n": len(same_model),
              "a_same_model_identical_greedy_text": share(sum(clean(allv[p]["greedy"]) == clean(rt4[p]["greedy"])
                                                              for p in same_model), len(same_model)),
              "a_same_model_identical_main": share(sum(A[p] == R[p] for p in same_model), len(same_model)),
              "b_older_vintage": agreement({p: R[p] for p in older}, {p: A[p] for p in older})}
        num = [p for p in older if YEARS.get(A[p]) is not None and YEARS.get(R[p]) is not None]
        d = [YEARS[A[p]] - YEARS[R[p]] for p in num]
        l3["b_numeric_mean_diff_2024_minus_realtime"] = round(sum(d) / len(d), 3) if d else None
        l3["b_numeric_sign"] = {"2024_longer": sum(x > 0 for x in d), "2024_shorter": sum(x < 0 for x in d),
                                "same": sum(x == 0 for x in d),
                                "sign_test_p": sign_test(sum(x > 0 for x in d), sum(x < 0 for x in d))}
        l3["forbidden_2024_all"] = forbidden(allv, sample)
        l3["vs_claude_2024_all"] = agreement(A, claude["v4"])
        l3["acc_sample40_2024_all"] = acc40(A, claude["v1"], jud)
        for pid, r in allv.items():
            r["call_year"] = sample[pid]["call_year"]
        Ac = {p: c for p, r in allv.items() if (c := calib_choice(r, "v4", cal)) is not None}
        l3["exploratory_calibrated_2024_all"] = {"n": len(Ac), "vs_claude": agreement(Ac, claude["v4"]),
                                                 "acc_sample40": acc40(Ac, claude["v1"], jud)}
        out["L3_realtime_vs_2024"] = l3
        for pid, r in allv.items():
            rows.append({"pilot_id": pid, "prompt": "v4", "mode": "all", "vintage": r["vintage"],
                         "call_year": sample[pid]["call_year"], "stratum": sample[pid]["stratum"],
                         "greedy_choice": gen_choice(r["greedy"]), "sampled_choice": gen_choice(r["sampled_t07"]),
                         "logprob_mean_choice": lp_choice(r["option_logprobs"], "mean"),
                         "logprob_sum_choice": lp_choice(r["option_logprobs"], "sum"),
                         "logprob_calib_choice": Ac.get(pid), "main_choice": A[pid],
                         "claude_choice": claude["v4"].get(pid), "seconds": r["seconds"],
                         "greedy": clean(r["greedy"]), "sampled_t07": clean(r["sampled_t07"])})

    if rows:
        with (odir / "06_chronogpt_parsed.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (odir / "06_chronogpt_eval_summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
