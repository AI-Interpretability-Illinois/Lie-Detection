#!/usr/bin/env python3
"""
Controls asked for by an ACL-style review of the draft (owner, 2026-10-12), from the runs of queue/lie_review.slurm
(outputs in RV; the timan1 MMLU passes are rsynced there too). Every analysis uses Table 1's protocol: items split in
half by MD5, an option is usable if it passes the perplexity / control / polarity checks on the LIARS' BENCH validation
half (against the unedited model of the main runs), each method takes its usable option with the best validation honest
rate, and the test half is reported. LIARS' BENCH is graded by the Qwen2.5-7B hybrid here (the selection grader, as for
the paper's other ablations), AI Liar by keyword.

  mmlu      plain MMLU (2,000 questions, letter logits, val/test halves) for every option of every menu:
            (a) each method's MMLU change at Table 1's options; (b) Table 1 re-selected with a fourth check, MMLU validation
            accuracy within MMLU_TOL points of the unedited model; (c) the honesty-vs-MMLU frontier figure
  variants  other selection rules at the family Table 1 chose: the option chosen per rule, against ours on that family
  mlsteer   multi-layer CAA against single-layer CAA and ours
  seeds     Table 1's chosen options at sampling seeds 1 and 2, pooled with seed 0

Writes reports/review_<part>.json and prints a summary. Usage: python review_report.py [mmlu variants mlsteer seeds]
"""
import csv
import glob
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_final_table as B   # noqa: E402

RV = os.environ.get("RV", "/work/hdd/bfqt/xiaocong/lie_review")
R = Path("/u/xiaocong/lie_detection/reports")
J = B.J
MAIN5 = [m for m in B.MODELS if m[1] in ("Llama-3.1-8B", "Phi-4-mini", "Falcon3-3B", "OLMo-2-1B", "Qwen2.5-3B")]
TOL = 0.05
MMLU_TOLS = [float(x) for x in os.environ.get("MMLU_TOLS", "0.02,0.05").split(",")]
G = B.G


def boot(diffs, n=2000):
    rnd = random.Random(0)
    bs = sorted(sum(rnd.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return {"diff": round(100 * sum(diffs) / len(diffs), 1), "lo": round(100 * bs[int(.025 * n)], 1),
            "hi": round(100 * bs[int(.975 * n) - 1], 1), "n": len(diffs)}


def boot_models(dls, n=2000):
    rnd = random.Random(0)
    est = sum(sum(d) / len(d) for d in dls) / len(dls)
    bs = sorted(sum(sum(rnd.choice(d) for _ in d) / len(d) for d in dls) / len(dls) for _ in range(n))
    return {"diff": round(100 * est, 1), "lo": round(100 * bs[int(.025 * n)], 1), "hi": round(100 * bs[int(.975 * n) - 1], 1)}


def load_rv(S, lb_tags, ail_tags, lb, ppl, ail):
    """Add the review runs' arms (LB judged by the Qwen2.5-7B hybrid, AIL by keyword) to the main-run dicts."""
    for t in lb_tags:
        p, c = f"{RV}/liars_bench_honesty_{S}_{t}_{J}.jsonl", f"{RV}/liars_bench_honesty_{S}_{t}.csv"
        if not (os.path.exists(p) and os.path.exists(c)):
            continue
        new = defaultdict(list)
        for r in map(json.loads, open(p)):
            if r["arm"] != "unsteered" and r["arm"] not in lb and r["verdict_hybrid"] in G:
                new[r["arm"]].append(r)
        lb.update(new)
        for r in csv.DictReader(open(c)):
            ppl.setdefault(r["arm"], float(r["ppl_ratio"]))
    for t in ail_tags:
        p = f"{RV}/liars_bench_honesty_{S}_{t}_responses.jsonl"
        if not os.path.exists(p):
            continue
        c = f"{RV}/liars_bench_honesty_{S}_{t}.csv"      # perplexity of AI Liar arms whose family has no LB run
        if os.path.exists(c):
            for r in csv.DictReader(open(c)):
                ppl.setdefault(r["arm"], float(r["ppl_ratio"]))
        new = defaultdict(list)
        for r in map(json.loads, open(p)):
            v = r.get("verdict")
            if r["arm"] == "unsteered" or r["arm"] in ail or v in (None, "ungradeable") or B.items_ail[r["uid"]]["grade"]["near_train"]:
                continue
            new[r["arm"]].append({"uid": r["uid"], "honest": v == "honest"})
        ail.update(new)


import re as _re
_norm = lambda a: _re.sub(r" \(c=[^)]*\)", "", a)   # steering labels carry c = r x median norm, which differs in the last digit across hosts


class ArmDict(dict):
    """arm label -> value, looked up without the steering label's (c=...) suffix"""
    def __setitem__(self, k, v): super().__setitem__(_norm(k), v)
    def __getitem__(self, k): return super().__getitem__(_norm(k))
    def __contains__(self, k): return super().__contains__(_norm(k))
    def get(self, k, d=None): return super().get(_norm(k), d)
    def setdefault(self, k, d=None): return super().setdefault(_norm(k), d)


def load_mmlu(S):
    """arm -> {"val", "test", "all"} accuracy, from every *_mmlu.jsonl of the model under RV (first file wins per arm)."""
    acc = ArmDict()
    for p in sorted(glob.glob(f"{RV}/liars_bench_honesty_{S}_rv*_mmlu.jsonl")):
        if "smoke" in p:
            continue
        per = defaultdict(lambda: defaultdict(list))
        for r in map(json.loads, open(p)):
            per[r["arm"]][r["half"]].append(r["correct"])
        for a, d in per.items():
            if a in acc:
                continue
            allv = d["val"] + d["test"]
            acc[a] = {"val": sum(d["val"]) / len(d["val"]), "test": sum(d["test"]) / len(d["test"]), "all": sum(allv) / len(allv),
                      "n": len(allv)}
    return acc


class Model:
    def __init__(self, S, name, lbt, ailt, rv_lb=(), rv_ail=()):
        self.S, self.name = S, name
        self.lb, self.ppl, self.ail = B.load(S, lbt, ailt)
        load_rv(S, list(rv_lb), list(rv_ail), self.lb, self.ppl, self.ail)
        self.mmlu = load_mmlu(S)
        dec = self.dec
        rate = self.rate
        self.u = {"T": rate(dec("unsteered", "val", True)), "F": rate(dec("unsteered", "val", False)),
                  "c": rate(self.ctl("unsteered", "val"))}

    rate = staticmethod(lambda rs: sum(r["verdict_hybrid"] == "honest" for r in rs) / max(1, len(rs)))

    def dec(self, a, part, b=None):
        return [r for r in self.lb.get(a, []) if r["split"] == "instructed-deception" and B.half(r["uid"]) == part
                and (b is None or r["belief"] is b)]

    def ctl(self, a, part):
        return [r for r in self.lb.get(a, []) if r["split"] == "instructed-truthful" and B.half(r["uid"]) == part]

    def fails(self, a, mmlu_tol=None, ppl_only=False):
        f = []
        if self.ppl.get(a, 9) >= 1.15: f.append("perplexity")
        if ppl_only:
            return f
        if self.rate(self.ctl(a, "val")) < self.u["c"] - TOL: f.append("control")
        if self.rate(self.dec(a, "val", True)) < self.u["T"] - TOL or self.rate(self.dec(a, "val", False)) < self.u["F"] - TOL:
            f.append("polarity")
        if mmlu_tol is not None:
            m, m0 = self.mmlu.get(a), self.mmlu.get("unsteered")
            if m is None or m0 is None: f.append("no-mmlu")
            elif m["val"] < m0["val"] - mmlu_tol: f.append("mmlu")
        return f

    def score(self, bench, a, part):
        if bench == "lb":
            return self.rate(self.dec(a, part))
        xs = [x["honest"] for x in self.ail.get(a, []) if B.half(x["uid"]) == part]
        return sum(xs) / max(1, len(xs))

    def items(self, bench, a):
        if bench == "lb":
            return {r["uid"]: r["verdict_hybrid"] == "honest" for r in self.dec(a, "test")}
        return {x["uid"]: x["honest"] for x in self.ail.get(a, []) if B.half(x["uid"]) == "test"}

    def arms(self, bench, need_lb=True):
        return [a for a in (self.lb if bench == "lb" else self.ail) if a in self.lb or not need_lb]   # LB checks

    def choose(self, bench, fam_of, mmlu_tol=None, ppl_only=False):
        """method -> chosen arm (usable, best validation honest rate)"""
        ch = {}
        for a in self.arms(bench, need_lb=not ppl_only):
            f = fam_of(a)
            if not f or self.fails(a, mmlu_tol, ppl_only):
                continue
            if f not in ch or self.score(bench, a, "val") > self.score(bench, ch[f], "val"):
                ch[f] = a
        return ch


def fam_main(a):
    if a.startswith("mlsteer"):
        return "mlsteer"
    if a.startswith("weight:v"):
        return None
    return B.family(a)


def models(rv_lb=(), rv_ail=()):
    out = []
    for S, name, lbt, ailt in MAIN5:
        out.append(Model(S, name, lbt, ailt, rv_lb, rv_ail))
        print(f"  loaded {name}: {len(out[-1].lb)} LB arms, {len(out[-1].ail)} AIL arms, {len(out[-1].mmlu)} MMLU arms", flush=True)
    return out


METHODS = ["mask", "liemask", "steer", "prompt", "unsteered"]


def part_mmlu(ms):
    res = {"tols": MMLU_TOLS, "models": {}}
    t1 = {r["model"]: r for r in json.load(open(R / "main_table_final.json"))}
    for m in ms:
        m0 = m.mmlu.get("unsteered")
        if not m0:
            print(f"{m.name}: no MMLU yet"); continue
        row = {"unedited_mmlu": round(100 * m0["all"], 1), "at_table1": {}, "gated": {}, "options": []}
        for bench in ("lb", "ail"):
            cells = t1[m.name][bench]["cells"]
            row["at_table1"][bench] = {f: {"arm": c["arm"], "mmlu": round(100 * m.mmlu[c["arm"]]["all"], 1) if c["arm"] in m.mmlu else None,
                                           "drop": round(100 * (m.mmlu[c["arm"]]["all"] - m0["all"]), 1) if c["arm"] in m.mmlu else None}
                                       for f, c in cells.items() if c.get("v") is not None}
            for tol in [None] + MMLU_TOLS:
                ch = m.choose(bench, lambda a: B.family(a), tol)
                cells_g = {f: {"arm": a, "v": round(100 * m.score(bench, a, "test"), 1),
                               "mmlu_drop": round(100 * (m.mmlu[a]["all"] - m0["all"]), 1) if a in m.mmlu else None}
                           for f, a in ch.items()}
                others = [f for f in ch if f != "mask"]
                lead = None
                if "mask" in ch and others:
                    best = max(others, key=lambda f: m.score(bench, ch[f], "test"))
                    pm, pb = m.items(bench, ch["mask"]), m.items(bench, ch[best])
                    U = sorted(set(pm) & set(pb))
                    lead = dict(boot([pm[u] - pb[u] for u in U]), vs=best)
                row["gated"].setdefault(bench, {})["none" if tol is None else f"{tol:g}"] = {"cells": cells_g, "lead": lead}
        for a in m.lb:
            f = fam_main(a)
            if f and a in m.mmlu:
                row["options"].append({"arm": a, "fam": f, "mmlu_drop": round(100 * (m.mmlu[a]["all"] - m0["all"]), 2),
                                       "lb": round(100 * m.score("lb", a, "test"), 1),
                                       "ail": round(100 * m.score("ail", a, "test"), 1) if a in m.ail else None,
                                       "fails": m.fails(a)})
        res["models"][m.name] = row
        for bench in ("lb", "ail"):
            for k, g in row["gated"][bench].items():
                L = g["lead"]
                print(f"{m.name:<13}{bench:>4} mmlu<= {k:>5}: " + "  ".join(
                    f"{f}={g['cells'][f]['v']:.1f}({g['cells'][f]['mmlu_drop']})" for f in METHODS if f in g["cells"])
                      + (f"  lead {L['diff']:+.1f} [{L['lo']:+.1f},{L['hi']:+.1f}] vs {L['vs']}" if L else ""))
    # average over the models of the gated tables (methods without a usable option count as unedited)
    for bench in ("lb", "ail"):
        for k in ["none"] + [f"{t:g}" for t in MMLU_TOLS]:
            mm = [m for m in ms if m.name in res["models"]]
            if len(mm) < 5:
                continue
            mean = {}
            for f in METHODS:
                mean[f] = sum((res["models"][m.name]["gated"][bench][k]["cells"].get(f) or res["models"][m.name]["gated"][bench][k]["cells"]["unsteered"])["v"]
                              for m in mm) / len(mm)
            others = [f for f in METHODS if f != "mask"]
            best = max(others, key=mean.get)
            dls = []
            for m in mm:
                c = res["models"][m.name]["gated"][bench][k]["cells"]
                a_m = (c.get("mask") or c["unsteered"])["arm"]; a_b = (c.get(best) or c["unsteered"])["arm"]
                pm, pb = m.items(bench, a_m), m.items(bench, a_b)
                dls.append([pm[u] - pb[u] for u in sorted(set(pm) & set(pb))])
            res.setdefault("average", {}).setdefault(bench, {})[k] = {"mean": {f: round(v, 1) for f, v in mean.items()},
                                                                       "lead": dict(boot_models(dls), vs=best)}
            L = res["average"][bench][k]["lead"]
            print(f"AVERAGE      {bench:>4} mmlu<= {k:>5}: " + "  ".join(f"{f}={mean[f]:.1f}" for f in METHODS)
                  + f"  lead {L['diff']:+.1f} [{L['lo']:+.1f},{L['hi']:+.1f}] vs {best}")
    json.dump(res, open(R / "review_mmlu.json", "w"), indent=1)


VRULE = {"pos": "positive-score weights only", "pad": "padding only", "z": "standardized difference",
         "tok": "per-token (length-free) norms", "rat": "scale-free ratio", "r020": "ours at r = 0.2", "r060": "ours at r = 0.6"}


def fam_variant(a):
    if a.startswith("weight:v"):
        nm = a.split()[0][len("weight:v"):]
        for r in VRULE:
            if nm.startswith(r):
                return "v" + r
    return None


def part_variants(ms):
    t1 = {r["model"]: r for r in json.load(open(R / "main_table_final.json"))}
    res = {}
    for m in ms:
        m0 = m.mmlu.get("unsteered")
        res[m.name] = {}
        for bench in ("lb", "ail"):
            fam = t1[m.name][bench]["cells"]["mask"]["arm"].split()[0][len("weight:"):]
            ours_f = lambda a, fam=fam: "ours" if a.startswith(f"weight:{fam} ") else fam_variant(a)
            po = bench == "ail"   # the AI Liar family's variants have no LIARS' BENCH run: perplexity check only, for ours too
            ch = m.choose(bench, ours_f, ppl_only=po)
            fails = {}
            for a in m.arms(bench, need_lb=not po):
                f = ours_f(a)
                if f and f not in ch:
                    fails.setdefault(f, set()).update(m.fails(a, ppl_only=po))
            cells = {}
            for f, a in ch.items():
                d = {"arm": a, "v": round(100 * m.score(bench, a, "test"), 1), "ppl": round(m.ppl.get(a, float("nan")), 3),
                     "mmlu_drop": round(100 * (m.mmlu[a]["all"] - m0["all"]), 1) if (a in m.mmlu and m0) else None}
                if f != "ours" and "ours" in ch:
                    pm, pb = m.items(bench, ch["ours"]), m.items(bench, a)
                    d["ours_minus"] = boot([pm[u] - pb[u] for u in sorted(set(pm) & set(pb))])
                cells[f] = d
            for f, why in fails.items():
                cells[f] = {"v": None, "fails": sorted(why)}
            res[m.name][bench] = {"family": fam, "cells": cells}
            print(f"{m.name:<13}{bench:>4} [{fam}] " + "  ".join(
                f"{f}={c['v']}" + (f"({c['arm'].split('a=')[-1]})" if c.get("arm") else "") for f, c in sorted(cells.items())))
    json.dump(res, open(R / "review_variants.json", "w"), indent=1)


def part_mlsteer(ms):
    res = {}
    for m in ms:
        m0 = m.mmlu.get("unsteered")
        res[m.name] = {}
        for bench in ("lb", "ail"):
            ch = m.choose(bench, fam_main)
            cells = {f: {"arm": a, "v": round(100 * m.score(bench, a, "test"), 1),
                         "mmlu_drop": round(100 * (m.mmlu[a]["all"] - m0["all"]), 1) if (a in m.mmlu and m0) else None}
                     for f, a in ch.items() if f in ("mask", "steer", "mlsteer", "prompt", "liemask", "unsteered")}
            if "mlsteer" not in ch:
                why = set()
                for a in m.arms(bench):
                    if fam_main(a) == "mlsteer":
                        why.update(m.fails(a))
                cells["mlsteer"] = {"v": None, "fails": sorted(why)}
            if "mask" in ch and "mlsteer" in ch:
                pm, pb = m.items(bench, ch["mask"]), m.items(bench, ch["mlsteer"])
                cells["mask"]["minus_mlsteer"] = boot([pm[u] - pb[u] for u in sorted(set(pm) & set(pb))])
            res[m.name][bench] = cells
            print(f"{m.name:<13}{bench:>4} " + "  ".join(f"{f}={c['v']}" for f, c in cells.items())
                  + (f"  mlsteer arm {cells['mlsteer'].get('arm')}" if cells['mlsteer'].get('arm') else f"  mlsteer fails {cells['mlsteer'].get('fails')}"))
    json.dump(res, open(R / "review_mlsteer.json", "w"), indent=1)


def part_seeds(ms):
    t1 = {r["model"]: r for r in json.load(open(R / "main_table_final.json"))}
    res = {}
    for m in ms:
        res[m.name] = {}
        for bench in ("lb", "ail"):
            cells = t1[m.name][bench]["cells"]
            per_seed = {}
            for sd in (0, 1, 2):
                if sd == 0:
                    src = m
                else:
                    lb, ppl, ail = ArmDict(), {}, ArmDict()   # labels without the steering (c=...) suffix: it differs across hosts
                    t = f"rvseeds{sd}"
                    p = f"{RV}/liars_bench_honesty_{m.S}_{t}_{J}.jsonl"
                    if bench == "lb" and os.path.exists(p):
                        for r in map(json.loads, open(p)):
                            if r["verdict_hybrid"] in G: lb.setdefault(_norm(r["arm"]), []).append(r)
                    q = f"{RV}/liars_bench_honesty_{m.S}_{t}ail_responses.jsonl"
                    if bench == "ail" and os.path.exists(q):
                        for r in map(json.loads, open(q)):
                            v = r.get("verdict")
                            if v in (None, "ungradeable") or B.items_ail[r["uid"]]["grade"]["near_train"]: continue
                            ail.setdefault(_norm(r["arm"]), []).append({"uid": r["uid"], "honest": v == "honest"})
                    src = type("S", (), {})(); src.lb, src.ail = lb, ail
                    src.dec = lambda a, part, b=None, lb=lb: [r for r in lb.get(a, []) if r["split"] == "instructed-deception" and B.half(r["uid"]) == part]
                    src.items = lambda bench_, a, lb=lb, ail=ail: ({r["uid"]: r["verdict_hybrid"] == "honest" for r in lb.get(a, []) if r["split"] == "instructed-deception" and B.half(r["uid"]) == "test"}
                                                                   if bench_ == "lb" else {x["uid"]: x["honest"] for x in ail.get(a, []) if B.half(x["uid"]) == "test"})
                its = {f: src.items(bench, c["arm"]) for f, c in cells.items() if c.get("v") is not None}
                if all(its.values()):
                    per_seed[sd] = its
            row = {"seeds": sorted(per_seed), "rates": {}, "lead": None}
            for f in cells:
                row["rates"][f] = [round(100 * sum(per_seed[s][f].values()) / len(per_seed[s][f]), 1) for s in sorted(per_seed) if f in per_seed[s]]
            vs = t1[m.name][bench]["lead"]["vs"]
            diffs = []
            for s in per_seed:
                pm, pb = per_seed[s]["mask"], per_seed[s][vs]
                diffs += [pm[u] - pb[u] for u in sorted(set(pm) & set(pb))]
            row["lead"] = dict(boot(diffs), vs=vs) if diffs else None
            res[m.name][bench] = row
            L = row["lead"]
            print(f"{m.name:<13}{bench:>4} seeds {row['seeds']} mask {row['rates'].get('mask')} {vs} {row['rates'].get(vs)}"
                  + (f"  pooled lead {L['diff']:+.1f} [{L['lo']:+.1f},{L['hi']:+.1f}]" if L else ""))
    json.dump(res, open(R / "review_seeds.json", "w"), indent=1)


def part_menu(ms):
    """Menu size: ours restricted to one prompt family (six strengths, the lie-instruction mask's menu size), every
    family in turn, against the other methods at their Table 1 choices (selection grader, test half)."""
    res = {}
    fams = ["fiction", "pooled", "fictf", "pooltf", "poolmix"]
    for bench in ("lb", "ail"):
        res[bench] = {}
        base = {m.name: m.choose(bench, B.family) for m in ms}
        for fam in fams + ["all"]:
            vals, dls = [], []
            for m in ms:
                if fam == "all":
                    f_of = B.family
                else:
                    f_of = lambda a, fam=fam: "mask" if a.startswith(f"weight:{fam} ") else (None if B.family(a) == "mask" else B.family(a))
                ch = m.choose(bench, f_of)
                if "mask" not in ch:
                    ch["mask"] = "unsteered"      # no usable option: the model stays unedited (Table 1's convention)
                vals.append(m.score(bench, ch["mask"], "test"))
                others = [f for f in ch if f not in ("mask", "unsteered")] or ["unsteered"]
                best = max(others, key=lambda f: m.score(bench, ch[f], "test"))
                pm, pb = m.items(bench, ch["mask"]), m.items(bench, ch[best])
                dls.append([pm[u] - pb[u] for u in sorted(set(pm) & set(pb))])
            ok = [v for v in vals if v is not None]
            res[bench][fam] = {"mean": round(100 * sum(ok) / len(ok), 1) if len(ok) == len(ms) else None,
                               "lead_vs_row_best": boot_models(dls) if len(dls) == len(ms) else None}
            print(f"{bench:4} family {fam:8} mean {res[bench][fam]['mean']}  lead {res[bench][fam]['lead_vs_row_best']}")
    json.dump(res, open(R / "review_menu.json", "w"), indent=1)


def averages(ms):
    """Average over the five models: each variant rule against ours (variants), multi-layer steering against ours and
    against single-layer steering (mlsteer). A rule or method with no usable option on a model counts as the unedited
    model there (Table 1's convention); the lead is the mean over models of the per-model mean paired difference,
    items resampled within each model."""
    out = {}
    v = json.load(open(R / "review_variants.json"))
    s = json.load(open(R / "review_mlsteer.json"))
    by = {m.name: m for m in ms}
    for bench in ("lb", "ail"):
        rules = sorted({k for n in v for k in v[n][bench]["cells"] if k != "ours"})
        out.setdefault("variants", {})[bench] = {}
        for rule in rules:
            vals, dls = [], []
            for m in ms:
                c = v[m.name][bench]["cells"]
                a_o = c["ours"]["arm"]; a_r = c.get(rule, {}).get("arm") or "unsteered"
                vals.append(m.score(bench, a_r, "test"))
                pm, pb = m.items(bench, a_o), m.items(bench, a_r)
                dls.append([pm[u] - pb[u] for u in sorted(set(pm) & set(pb))])
            usable = sum(1 for m in ms if v[m.name][bench]["cells"].get(rule, {}).get("arm"))
            out["variants"][bench][rule] = {"mean": round(100 * sum(vals) / len(vals), 1), "usable": usable,
                                            "ours_minus": boot_models(dls)}
        ours_mean = sum(m.score(bench, v[m.name][bench]["cells"]["ours"]["arm"], "test") for m in ms) / len(ms)
        out["variants"][bench]["ours"] = {"mean": round(100 * ours_mean, 1)}
        res = {}
        for f in ("mask", "steer", "mlsteer"):
            arms = [(m, (s[m.name][bench].get(f) or {}).get("arm") or "unsteered") for m in ms]
            res[f] = round(100 * sum(m.score(bench, a, "test") for m, a in arms) / len(ms), 1)
        dls = []
        for m in ms:
            a_o = s[m.name][bench]["mask"]["arm"]; a_r = (s[m.name][bench].get("mlsteer") or {}).get("arm") or "unsteered"
            pm, pb = m.items(bench, a_o), m.items(bench, a_r)
            dls.append([pm[u] - pb[u] for u in sorted(set(pm) & set(pb))])
        out.setdefault("mlsteer", {})[bench] = {"mean": res, "ours_minus_mlsteer": boot_models(dls)}
        print(bench, "variants", {k: (x["mean"], x.get("usable"), (x.get("ours_minus") or {})) for k, x in out["variants"][bench].items()})
        print(bench, "mlsteer", out["mlsteer"][bench])
    json.dump(out, open(R / "review_averages.json", "w"), indent=1)


def part_ifeval(ms=None):
    """IFEval (541 prompts, greedy) at Table 1's LIARS' BENCH-chosen options: prompt- and instruction-level strict accuracy
    per method, and the change against the unedited model. Reads <RV>/*_rvifeval*.csv (Llama split over two jobs)."""
    t1 = {r["model"]: r for r in json.load(open(R / "main_table_final.json"))}
    out = {}
    for S, name, _, _ in MAIN5:
        rows = {}
        for p in sorted(glob.glob(f"{RV}/liars_bench_honesty_{S}_rvifeval*.csv")):
            for r in csv.DictReader(open(p)):
                if r.get("ifeval_prompt_strict"):
                    rows.setdefault(_norm(r["arm"]), r)
        cells = {}
        for f, c in t1[name]["lb"]["cells"].items():
            r = rows.get(_norm(c["arm"]))
            if c.get("v") is not None and r:
                cells[f] = {"prompt": round(100 * float(r["ifeval_prompt_strict"]), 1), "inst": round(100 * float(r["ifeval_inst_strict"]), 1)}
        if "unsteered" in cells:
            for f in cells:
                cells[f]["d_prompt"] = round(cells[f]["prompt"] - cells["unsteered"]["prompt"], 1)
        out[name] = cells
        print(name, {f: (c["prompt"], c.get("d_prompt")) for f, c in cells.items()})
    full = [n for n in out if len(out[n]) >= 5]
    out["_average_change"] = {f: round(sum(out[n][f]["d_prompt"] for n in full) / len(full), 1) for f in ("mask", "liemask", "steer", "prompt")} if full else {}
    out["_models"] = full
    print("average change over", full, out["_average_change"])
    json.dump(out, open(R / "review_ifeval.json", "w"), indent=1)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["mmlu", "variants", "mlsteer", "seeds"]
    rv_lb = ["rvvariants", "rvvariantsa", "rvvariantsb", "rvvariantsc", "rvmlsteer"]
    rv_ail = ["rvvariantsail", "rvmlsteerail"]
    ms = models(rv_lb, rv_ail)
    for p in parts:
        print(f"== {p}")
        {"mmlu": part_mmlu, "variants": part_variants, "mlsteer": part_mlsteer, "seeds": part_seeds, "menu": part_menu, "averages": averages, "ifeval": part_ifeval}[p](ms)
