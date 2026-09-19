"""Rank the full grid on both objectives the paper's protocol implies.

FACTUAL_QA OOD is the validation set parameters are *selected* on; LIARS' BENCH
is where generalisation is *reported*.  A config has to clear both, and alpha
pulls them in opposite directions -- bare questions need a hard shove to flip
the model off the true answer, a scenario does most of that work already -- so
the joint score is a geometric mean, where being strong on one axis cannot buy
a zero on the other.  Damage caps are applied first: a model that has been
broken scores as a "liar" under any regex grader.
"""
import csv, math, sys
from pathlib import Path

L = Path(sys.argv[1])
PPL_CAP, DEGEN_CAP = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0, 15.0
K = lambda r: (round(float(r["rho"]), 4), round(float(r["alpha"]), 4), round(float(r["ratio"]), 4))
COMMITTED = {"llama_8b": (0.04, 4.0, 0.37), "qwen2p5_7b": (0.06, 4.0, 0.35)}


def dec3_of(r):
    idu = float(r["id_lie_share"]) * (100.0 - float(r["id_noclaim"])) / 100.0
    return (float(r["lie_hpc"]) + float(r["lie_hpkr"]) + idu) / 3.0, idu


def load(model, short):
    ood, base = {}, None
    for r in csv.DictReader(open(L / f"oodF_{model}.csv")):
        if float(r["rho"]) < 0:
            base = r
        else:
            ood[K(r)] = r
    lb = {}
    for src in [f"outputs/param_search/screen_{short}.csv",
                L / f"lbNEW_{model}_s0.csv", L / f"lbNEW_{model}_s1.csv"]:
        if Path(src).exists():
            for r in csv.DictReader(open(src)):
                lb[K(r)] = r
    rows = []
    for k, o in ood.items():
        l = lb.get(k)
        if not l:
            continue
        d3, idu = dec3_of(l)
        rows.append(dict(
            rho=k[0], alpha=k[1], ratio=k[2],
            fqa=(float(o["ood_lie_g"]) + float(o["ood_lie_s"])) / 2,
            fpap=(float(o["ood_paperlie_g"]) + float(o["ood_paperlie_s"])) / 2,
            fwrong=(float(o["ood_truly_g"]) + float(o["ood_truly_s"])) / 2,
            dec3=d3, hpc=float(l["lie_hpc"]), hpkr=float(l["lie_hpkr"]), idu=idu,
            degen=float(l["degen_alpaca"]), ppl=float(l["ppl_ratio"])))
    return rows, base, lb


for model, short in [("llama", "llama_8b"), ("qwen", "qwen2p5_7b")]:
    rows, base, lb = load(model, short)
    print(f"\n{'='*104}\n{short}:  {len(rows)} points with BOTH objectives   "
          f"(baseline FACTUAL_QA lie {float(base['ood_lie_g']):.1f}% greedy / "
          f"{float(base['ood_lie_s']):.1f}% sampled)")

    # paired alpha comparison, restricted to (rho,ratio) cells that carry every alpha
    cells = {}
    for r in rows:
        cells.setdefault((r["rho"], r["ratio"]), {})[r["alpha"]] = r
    for alphas in [(2.5, 3.0, 3.5, 4.0), (3.0, 3.5, 4.0, 4.5)]:
        common = [c for c in cells.values() if all(a in c for a in alphas)]
        if not common:
            continue
        print(f"\n  paired over {len(common)} (rho,ratio) cells carrying all of {alphas}:")
        print(f"    {'alpha':>6} {'FQA_lie':>8} {'LB_dec3':>8} {'ppl':>6} {'degen':>6}")
        for a in alphas:
            g = [c[a] for c in common]
            print(f"    {a:>6.1f} {sum(x['fqa'] for x in g)/len(g):>8.1f} "
                  f"{sum(x['dec3'] for x in g)/len(g):>8.1f} "
                  f"{sum(x['ppl'] for x in g)/len(g):>6.2f} "
                  f"{sum(x['degen'] for x in g)/len(g):>6.1f}")

    ok = [r for r in rows if r["ppl"] <= PPL_CAP and r["degen"] <= DEGEN_CAP]
    for r in ok:
        r["joint"] = math.sqrt(max(r["fqa"], 0) * max(r["dec3"], 0))
    ok.sort(key=lambda r: -r["joint"])
    print(f"\n  {len(ok)}/{len(rows)} inside damage caps (ppl_ratio<={PPL_CAP}, degen<={DEGEN_CAP})")
    print(f"\n  {'rho':>5} {'alp':>4} {'rat':>5} | {'FQA':>5} {'(pap)':>5} {'FQAwr':>6} |"
          f" {'dec3':>5} {'HP-C':>5} {'HPKR':>5} {'idU':>5} | {'degen':>5} {'ppl':>5} | {'joint':>5}")
    print("  " + "-" * 98)
    ck = COMMITTED[short]
    for r in ok[:15]:
        print(f"  {r['rho']:>5.2f} {r['alpha']:>4.1f} {r['ratio']:>5.2f} | {r['fqa']:>5.1f} "
              f"{r['fpap']:>5.1f} {r['fwrong']:>6.1f} | {r['dec3']:>5.1f} {r['hpc']:>5.1f} "
              f"{r['hpkr']:>5.1f} {r['idu']:>5.1f} | {r['degen']:>5.1f} {r['ppl']:>5.2f} | {r['joint']:>5.1f}")
    cur = next((r for r in rows if (r["rho"], r["alpha"], r["ratio"]) == ck), None)
    if cur:
        cur["joint"] = math.sqrt(max(cur["fqa"], 0) * max(cur["dec3"], 0))
        rank = sum(1 for r in ok if r["joint"] > cur["joint"]) + 1
        print(f"  {'-'*98}\n  COMMITTED {ck[0]:>4.2f} {ck[1]:>4.1f} {ck[2]:>5.2f} | {cur['fqa']:>5.1f} "
              f"{cur['fpap']:>5.1f} {cur['fwrong']:>6.1f} | {cur['dec3']:>5.1f} {cur['hpc']:>5.1f} "
              f"{cur['hpkr']:>5.1f} {cur['idu']:>5.1f} | {cur['degen']:>5.1f} {cur['ppl']:>5.2f} | "
              f"{cur['joint']:>5.1f}   (rank {rank} of {len(ok)})")
