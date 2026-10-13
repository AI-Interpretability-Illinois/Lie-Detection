#!/usr/bin/env python3
"""LaTeX snippets with every number of the review-control text taken from reports/review_*.json (2026-10-12), so nothing
is transcribed by hand. Writes reports/review_text/*.tex; the paper edit script inserts them. Prints a check list."""
import json
from pathlib import Path

R = Path("/u/xiaocong/lie_detection/reports")
OUT = R / "review_text"; OUT.mkdir(exist_ok=True)
ORDER = ["Llama-3.1-8B", "Phi-4-mini", "Falcon3-3B", "OLMo-2-1B", "Qwen2.5-3B"]
SHORT = {"Llama-3.1-8B": "Llama", "Phi-4-mini": "Phi", "Falcon3-3B": "Falcon", "OLMo-2-1B": "OLMo-2-1B", "Qwen2.5-3B": "Qwen2.5-3B"}
BN = {"lb": "\\LB{}", "ail": "\\AIL{}"}
VS = {"steer": "steering", "prompt": "the honest prompt", "liemask": "the lie-instruction mask", "unsteered": "the unedited model"}
f1 = lambda x: f"{x:.1f}".replace("-", "$-$")
sg = lambda x: (f"{x:+.1f}").replace("-", "$-$")
iv = lambda L: f"{sg(L['diff'])} [{sg(L['lo'])}, {sg(L['hi'])}]"
J = lambda n: json.load(open(R / f"review_{n}.json"))


def capability():
    m = J("mmlu"); d = m["models"]
    drops = {f: [d[n]["at_table1"]["lb"][f]["drop"] for n in ORDER] for f in ("mask", "steer", "prompt", "liemask")}
    mean = {f: sum(v) / len(v) for f, v in drops.items()}
    worst = min(ORDER, key=lambda n: d[n]["at_table1"]["lb"]["mask"]["drop"])
    a = m["average"]
    s = (f"At the options of Table~\\ref{{tab:main}}, MMLU accuracy changes by {f1(mean['mask'])} points under the edit on average "
         f"({f1(min(drops['mask']))} on {SHORT[worst]} to {f1(max(drops['mask']))}), {f1(mean['steer'])} under steering, "
         f"{f1(mean['prompt'])} under the honest prompt and {f1(mean['liemask'])} under the lie-instruction mask "
         f"(Table~\\ref{{tab:mmlucost}}). This full-test measure is milder than that of Table~\\ref{{tab:mmlu}}, which samples answers "
         f"only to questions the unedited model answers correctly, at the initial-menu options. "
         f"Adding a fourth usability check, MMLU validation accuracy within 5 points of the unedited model, and re-selecting every "
         f"method gives average leads of {iv(a['lb']['0.05']['lead'])} on \\LB{{}} and {iv(a['ail']['0.05']['lead'])} on \\AIL{{}}; "
         f"within 2 points they are {iv(a['lb']['0.02']['lead'])} and {iv(a['ail']['0.02']['lead'])}, against "
         f"{iv(a['lb']['none']['lead'])} and {iv(a['ail']['none']['lead'])} without the check under this grader "
         f"(Table~\\ref{{tab:mmlugate}}). Figure~\\ref{{fig:mmlufrontier}} shows every option against its MMLU cost.")
    body = (f"Perplexity is one cost; general capability is another. At the options of Table~\\ref{{tab:main}}, plain MMLU accuracy "
            f"changes by {f1(mean['mask'])} points under the edit on average, against {f1(mean['steer'])} under steering and "
            f"{f1(mean['prompt'])} under the honest prompt. Re-selecting every method with an MMLU check, accuracy within 5 points "
            f"of the unedited model, leaves the edit's average lead at {sg(a['lb']['0.05']['lead']['diff'])} and "
            f"{sg(a['ail']['0.05']['lead']['diff'])} points, and a 2-point check at {sg(a['lb']['0.02']['lead']['diff'])} and "
            f"{sg(a['ail']['0.02']['lead']['diff'])}, all with intervals above zero (Appendix~\\ref{{app:review}}).")
    (OUT / "capability_app.tex").write_text(s); (OUT / "capability_body.tex").write_text(body)
    print("capability:", {f: round(v, 2) for f, v in mean.items()}, "\n ", s[:200], "...\n  body:", body)
    return mean, a


def variants():
    a = json.load(open(R / "review_averages.json"))["variants"]
    lb, ail = a["lb"], a["ail"]
    s = (f"Averaged over the five models (Table~\\ref{{tab:variants}}), the positive-score weights alone reach {f1(lb['vpos']['mean'])}\\% "
         f"on \\LB{{}} against {f1(lb['ours']['mean'])}\\% for the full mask (ours minus them {iv(lb['vpos']['ours_minus'])}), "
         f"while the padding alone reaches {f1(lb['vpad']['mean'])}\\% ({iv(lb['vpad']['ours_minus'])}), near the unedited level: "
         f"the edit comes from the few positive-score weights, and the padding only fills the budget. Per-token norms, which remove "
         f"prompt length, give {f1(lb['vtok']['mean'])}\\% ({iv(lb['vtok']['ours_minus'])}) and the scale-free ratio "
         f"{f1(lb['vrat']['mean'])}\\% ({iv(lb['vrat']['ours_minus'])}). The standardized difference, which has no threshold and "
         f"scores most channels positive, gives {f1(lb['vz']['mean'])}\\% ({iv(lb['vz']['ours_minus'])}); our rule at $r=0.2$ "
         f"gives {f1(lb['vr020']['mean'])}\\% and at $r=0.6$ {f1(lb['vr060']['mean'])}\\%. A rule without a usable option on a "
         f"model counts as the unedited model there. On \\AIL{{}} the positive-score weights reach {f1(ail['vpos']['mean'])}\\% "
         f"against {f1(ail['ours']['mean'])}\\% and the padding {f1(ail['vpad']['mean'])}\\%.")
    (OUT / "variants_app.tex").write_text(s)
    print("variants:", s)
    return lb, ail


def mlsteer():
    a = json.load(open(R / "review_averages.json"))["mlsteer"]
    s = (f"Averaged over the five models, multi-layer steering reaches {f1(a['lb']['mean']['mlsteer'])}\\% on \\LB{{}} and "
         f"{f1(a['ail']['mean']['mlsteer'])}\\% on \\AIL{{}}, against {f1(a['lb']['mean']['steer'])} and "
         f"{f1(a['ail']['mean']['steer'])}\\% for single-layer steering and {f1(a['lb']['mean']['mask'])} and "
         f"{f1(a['ail']['mean']['mask'])}\\% for the edit, which leads it by {iv(a['lb']['ours_minus_mlsteer'])} and "
         f"{iv(a['ail']['ours_minus_mlsteer'])} points (Table~\\ref{{tab:mlsteer}}).")
    (OUT / "mlsteer_app.tex").write_text(s); print("mlsteer:", s)
    return a


def seeds():
    d = J("seeds")
    parts = []
    for n in ORDER:
        for b in ("lb", "ail"):
            L = d[n][b].get("lead")
            if L and (L["lo"] > 0 or L["hi"] < 0):
                parts.append((n, b, L))
    s = ("Pooling seeds 0 to 2, the leads with intervals that exclude zero are "
         + "; ".join(SHORT[n] + " on " + BN[b] + " " + iv(L) + " over " + VS.get(L["vs"], L["vs"]) for n, b, L in parts)
         + " (Table~\\ref{tab:seedfinal}); every other lead's interval contains zero.")
    (OUT / "seeds_app.tex").write_text(s); print("seeds:", s)
    return parts


def further_body():
    a = json.load(open(R / "review_averages.json"))
    lb = a["variants"]["lb"]; ml = a["mlsteer"]
    s = (f"Rebuilt with other selection rules at the same families (Appendix~\\ref{{app:review}}), the edit's effect comes from its few "
         f"positive-score weights: they alone reach {f1(lb['vpos']['mean'])}\\% on \\LB{{}} against {f1(lb['ours']['mean'])}\\% for the "
         f"full mask, the near-zero padding alone {f1(lb['vpad']['mean'])}\\%; per-token norms, which remove prompt length, give "
         f"{f1(lb['vtok']['mean'])}\\% and a scale-free ratio {f1(lb['vrat']['mean'])}\\%, and a standardized difference without "
         f"threshold {f1(lb['vz']['mean'])}\\%. Steering at several layers at once does not close the gap: the edit leads it by "
         f"{sg(ml['lb']['ours_minus_mlsteer']['diff'])} and {sg(ml['ail']['ours_minus_mlsteer']['diff'])} points on average.")
    (OUT / "further_body.tex").write_text(s); print("further_body:", s)


if __name__ == "__main__":
    import sys
    for name in (sys.argv[1:] or ["capability", "variants", "mlsteer", "seeds", "further_body"]):
        try:
            globals()[name]()
        except (KeyError, FileNotFoundError, TypeError, ZeroDivisionError) as e:
            print(name, "not ready:", repr(e)[:200])
