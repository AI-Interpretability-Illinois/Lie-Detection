#!/usr/bin/env python3
"""Print every number the paper's review-control text quotes, straight from reports/review_*.json (no hand transcription)."""
import json
from pathlib import Path
R = Path("/u/xiaocong/lie_detection/reports")
ORDER = ["Llama-3.1-8B", "Phi-4-mini", "Falcon3-3B", "OLMo-2-1B", "Qwen2.5-3B"]
def j(n):
    try: return json.load(open(R / f"review_{n}.json"))
    except FileNotFoundError: return None
m = j("mmlu")
if m:
    print("== MMLU at Table 1 options (LB-chosen), drop in points; unedited acc")
    acc = {f: [] for f in ("mask", "liemask", "steer", "prompt")}
    for n in ORDER:
        r = m["models"].get(n)
        if not r: print(n, "missing"); continue
        c = r["at_table1"]["lb"]
        print(n, r["unedited_mmlu"], {f: c.get(f, {}).get("drop") for f in acc})
        for f in acc:
            if c.get(f, {}).get("drop") is not None: acc[f].append(c[f]["drop"])
    print("mean drop", {f: round(sum(v) / len(v), 2) if len(v) == 5 else None for f, v in acc.items()})
    for b in ("lb", "ail"):
        for k, g in m.get("average", {}).get(b, {}).items():
            print("avg", b, k, g["mean"], g["lead"])
    for n in ORDER:
        r = m["models"].get(n)
        if r:
            print(n, {b: {k: (g["lead"]["diff"], g["lead"]["lo"], g["lead"]["hi"], g["lead"]["vs"]) if g["lead"] else None for k, g in r["gated"][b].items()} for b in ("lb", "ail")})
v = j("variants")
if v:
    print("== variants (LB test, Qwen hybrid; AIL keyword with ppl check only)")
    for n in ORDER:
        for b in ("lb", "ail"):
            c = v.get(n, {}).get(b, {}).get("cells", {})
            print(n, b, v.get(n, {}).get(b, {}).get("family"), {k: (x.get("v"), x.get("arm", "").split("a=")[-1] if x.get("arm") else x.get("fails"), x.get("mmlu_drop"), (x.get("ours_minus") or {}).get("diff")) for k, x in sorted(c.items())})
s = j("mlsteer")
if s:
    print("== multi-layer steering")
    for n in ORDER:
        for b in ("lb", "ail"):
            c = s.get(n, {}).get(b, {})
            print(n, b, {k: (x.get("v"), x.get("mmlu_drop")) for k, x in c.items()}, (c.get("mask") or {}).get("minus_mlsteer"))
sd = j("seeds")
if sd:
    print("== seeds")
    for n in ORDER:
        for b in ("lb", "ail"):
            r = sd.get(n, {}).get(b, {}); print(n, b, r.get("seeds"), r.get("rates", {}).get("mask"), r.get("lead"))
