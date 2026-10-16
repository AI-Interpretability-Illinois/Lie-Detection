#!/usr/bin/env python3
"""
200 items for human labelling (extras plan, Oct 4): main-table replies with both judges' labels, stratified by
generator family (Llama, Phi, Falcon, OLMo, Qwen: 40 each) and, within a family, half from replies where the
Qwen2.5-7B judge and GPT disagree and half where they agree (disagreements are rare, so they are over-sampled;
the agreement estimate against humans must weight them back). The labeller sees the claim, the reply and the
model's own belief, not the judges' labels or the arm.

Writes reports/label_set_200.csv (uid, model, arm, claim, belief, reply, label[blank]) and
reports/label_set_200_key.jsonl (the same rows with both judges' labels, for scoring with
reports/judge_audit.py once reports/judge_human_labels.jsonl is filled in: uid, model, arm, label).
"""
import csv
import glob
import json
import os
import random
from collections import defaultdict
from pathlib import Path

G = Path(os.environ.get("GPT_DIR", "/work/nvme/biro/xiaocong/lie_out_gpt"))
B = Path("/work/hdd/bfqt/xiaocong/lie_detection_outputs")
FAM = {"Meta-Llama-3p1-8B-Instruct": "Llama", "Phi-4-mini-instruct": "Phi", "Falcon3-3B-Instruct": "Falcon",
       "OLMo-2-0425-1B-Instruct": "OLMo", "Qwen2p5-3B-Instruct": "Qwen", "Qwen2p5-7B-Instruct": "Qwen", "Qwen2p5-1p5B-Instruct": "Qwen"}
PER_FAMILY, DISAGREE_SHARE = 40, 0.5


def main():
    st = {}
    for l in open(B / "liars_bench/eval_set.jsonl"):
        d = json.loads(l); s = d.get("grade", {}).get("statement")
        if s:
            st[d["uid"]] = s
    pool = defaultdict(lambda: {"agree": [], "disagree": []})
    for p in glob.glob(str(G / "liars_bench_honesty_*_judged_gpt_v2.jsonl")):
        model = Path(p).name.replace("liars_bench_honesty_", "").split("_judged")[0].rsplit("_", 1)[0]
        for r in map(json.loads, open(p)):
            if r["split"] != "instructed-deception" or r["uid"] not in st or r.get("belief") is None:
                continue
            a, b = r.get("verdict_qwen7b_hybrid"), r.get("verdict_hybrid")
            if a not in ("honest", "lie", "no_claim") or b not in ("honest", "lie", "no_claim"):
                continue
            r["_model"] = model
            pool[FAM[model]]["disagree" if a != b else "agree"].append(r)
    rnd = random.Random(2026)
    rows = []
    for fam in ("Llama", "Phi", "Falcon", "OLMo", "Qwen"):
        nd = min(len(pool[fam]["disagree"]), int(PER_FAMILY * DISAGREE_SHARE))
        pick = rnd.sample(pool[fam]["disagree"], nd) + rnd.sample(pool[fam]["agree"], PER_FAMILY - nd)
        print(f"{fam}: {len(pool[fam]['disagree'])} disagreements, {len(pool[fam]['agree'])} agreements -> {nd} + {PER_FAMILY - nd}")
        rows += pick
    rnd.shuffle(rows)
    out = Path("/u/xiaocong/lie_detection/reports")
    with open(out / "label_set_200.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "claim", "model_belief", "reply", "label (honest / lie / no_claim / refusal)"])
        for i, r in enumerate(rows):
            w.writerow([i, st[r["uid"]], "TRUE" if r["belief"] else "FALSE", r["resp"].replace("\n", " "), ""])
    with open(out / "label_set_200_key.jsonl", "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps(dict(id=i, uid=r["uid"], model=r["_model"], arm=r["arm"], belief=r["belief"],
                                    qwen7b=r["verdict_qwen7b_hybrid"], gpt=r["verdict_hybrid"]), ensure_ascii=False) + "\n")
    print(f"{len(rows)} items -> {out / 'label_set_200.csv'} (labels blank) and label_set_200_key.jsonl")


if __name__ == "__main__":
    main()
