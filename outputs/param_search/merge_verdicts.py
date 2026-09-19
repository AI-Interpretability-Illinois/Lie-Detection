"""Merge sharded verdict files from judge_cached.py into one map.

Each shard judged a disjoint slice of the deduped (question, response) pairs;
this concatenates them back into the single {keys, verdicts} map rank_cached.py
expects, checking that every shard scored the same set of panels.
"""
import json, sys
from pathlib import Path

out, srcs = sys.argv[1], sys.argv[2:]
keys, verdicts, panels = [], {}, None
for s in srcs:
    d = json.loads(Path(s).read_text())
    p = sorted(d["verdicts"])
    if panels is None:
        panels = p
    elif p != panels:
        raise SystemExit(f"{s} scored {p}, expected {panels}")
    keys += [tuple(k) for k in d["keys"]]
    for name, v in d["verdicts"].items():
        verdicts.setdefault(name, []).extend(v)
if len(set(keys)) != len(keys):
    raise SystemExit("shards overlap -- keys are not disjoint")
Path(out).write_text(json.dumps({"keys": [list(k) for k in keys], "verdicts": verdicts}))
print(f"merged {len(srcs)} shards -> {len(keys)} pairs, panels {panels} -> {out}")
