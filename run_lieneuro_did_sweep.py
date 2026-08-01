#!/usr/bin/env python3
"""
LieNeuro DID sweep — Difference-in-Differences weight-importance estimator.

Replaces the naive control set (plain questions, DIRECT_TRUTH) with an
instruction-matched control set (DIRECT_TRUTH_INSTRUCT: "Answer truthfully: Q").

The DID importance score:
    I_DID(w) = I(lie_instruct, w) − I(truth_instruct, w)

removes weights that are important for ANY instruction-following (formatting
confound), leaving only those specifically activated by the deception instruction.

Two mask modes are compared:
  set_diff  : top-ρ(S_lie) \\ top-ρ(S_truth_instruct)   — original set-difference
  score_diff: top-ρ(S_lie − S_truth_instruct)            — continuous DID score

Usage
-----
  # Llama (default)
  CUDA_VISIBLE_DEVICES=0 python run_lieneuro_did_sweep.py --device cuda:0
  CUDA_VISIBLE_DEVICES=0 python run_lieneuro_did_sweep.py --device cuda:0 --skip-scores

  # Qwen
  CUDA_VISIBLE_DEVICES=1 python run_lieneuro_did_sweep.py --model qwen --device cuda:0

  # Mistral
  CUDA_VISIBLE_DEVICES=2 python run_lieneuro_did_sweep.py --model mistral --device cuda:0
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.evaluate import evaluate_factual_accuracy, FACTUAL_QA
from src.prompts import DIRECT_LIE, DIRECT_TRUTH_INSTRUCT

# Only evaluate on OOD questions — those not seen during DID score computation.
# This avoids in-distribution leakage and acts as a held-out validation set.
_instruct_qs = {p.split("Answer truthfully: ", 1)[-1].strip().lower()
                for p in DIRECT_TRUTH_INSTRUCT}
OOD_QA = [qa for qa in FACTUAL_QA
           if qa["question"].strip().lower() not in _instruct_qs]

CACHE_DIR  = "/scratch/aebrahim/hf_cache"

# Three independent judges for ensemble majority-vote coherence scoring.
# no_system_role=True for models whose chat template has no system role (Mistral).
JUDGE_CONFIGS = [
    {"model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",  "no_system_role": False},
    {"model_id": "Qwen/Qwen2.5-7B-Instruct",               "no_system_role": False},
    {"model_id": "mistralai/Mistral-7B-Instruct-v0.3",     "no_system_role": True},
]

MODEL_CONFIGS = {
    "llama": {
        "model_id":    "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "model_short": "llama_8b",
        "sweep_csv":   Path("outputs/lieneuro_did_sweep_llama.csv"),
        "wt_csv":      Path("outputs/wikitext_ppl.csv"),

        "trust_remote_code": False,
        "no_system_role":    False,
    },
    "qwen": {
        "model_id":    "Qwen/Qwen2.5-7B-Instruct",
        "model_short": "qwen2p5_7b",
        "sweep_csv":   Path("outputs/generalization/qwen2p5_7b/lieneuro_did_sweep_qwen.csv"),
        "wt_csv":      Path("outputs/generalization/qwen2p5_7b/wikitext_ppl.csv"),

        "trust_remote_code": False,
        "no_system_role":    False,
    },
    "mistral": {
        "model_id":    "mistralai/Mistral-7B-Instruct-v0.3",
        "model_short": "mistral_7b",
        "sweep_csv":   Path("outputs/generalization/mistral_7b/lieneuro_did_sweep_mistral.csv"),
        "wt_csv":      Path("outputs/generalization/mistral_7b/wikitext_ppl.csv"),

        "trust_remote_code": False,
        "no_system_role":    True,   # Mistral chat template has no system role
    },
    "phi3": {
        "model_id":    "microsoft/Phi-3-medium-4k-instruct",
        "model_short": "phi3_medium",
        "sweep_csv":   Path("outputs/generalization/phi3_medium/lieneuro_did_sweep_phi3.csv"),
        "wt_csv":      Path("outputs/generalization/phi3_medium/wikitext_ppl.csv"),

        "trust_remote_code": True,   # required for Phi-3 on older transformers versions
        "no_system_role":    False,
    },
}

# Sweep grid
# RHO_VALS   = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
# ALPHA_VALS = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
RHO_VALS   = [0.04]
ALPHA_VALS = [4.0]
MODES      = ["score_diff"]
# MODES      = ["set_diff", "score_diff"]

# RHO_VALS   = [0.04]
# ALPHA_VALS = [4.0]
# MODES      = ["score_diff"]

# WikiText PPL settings
WT_WINDOW    = 512
WT_STRIDE    = 256
WT_MAX_TOKS  = 8192

# Coherence judge prompt (model judges its own steered outputs)
_JUDGE_SYSTEM = (
    "You are a strict evaluator. Answer with exactly one word: "
    "COHERENT or INCOHERENT."
)
_JUDGE_TMPL = (
    "Question: {q}\n"
    "Correct answer: {a}\n"
    "Given answer: {r}\n\n"
    "Is the given answer a coherent (though factually wrong) response to the question — "
    "i.e., it answers the right *type* of thing (e.g., a language, a city, a year) "
    "but gives the wrong value? Or is it incoherent, nonsensical, or irrelevant?\n"
    "Reply with exactly one word: COHERENT or INCOHERENT."
)

# Semantic correctness judge — filters out string-wrong but semantically correct answers
# before they reach the coherence judge.
_CORRECT_SYSTEM = (
    "You are a strict evaluator. Answer with exactly one word: "
    "CORRECT or WRONG."
)
_CORRECT_TMPL = (
    "Question: {q}\n"
    "Gold answer: {a}\n"
    "Given answer: {r}\n\n"
    "Is the given answer semantically equivalent to the gold answer, "
    "even if worded differently, abbreviated, or using an accent variant?\n"
    "Reply with exactly one word: CORRECT or WRONG."
)


# ---------------------------------------------------------------------------
# WikiText PPL
# ---------------------------------------------------------------------------

def load_wikitext_tokens(tokenizer, max_tokens):
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids  = tokenizer(text, add_special_tokens=False,
                     truncation=True, max_length=max_tokens)["input_ids"]
    return ids


@torch.no_grad()
def sliding_window_ppl(model, token_ids, window, stride, device):
    input_ids = torch.tensor(token_ids, dtype=torch.long)
    n = len(input_ids)
    nlls, n_toks = [], 0
    for begin in range(0, n, stride):
        end      = min(begin + window, n)
        chunk    = input_ids[begin:end].unsqueeze(0).to(device)
        tgt_len  = min(stride, end - begin)
        labels   = chunk.clone()
        labels[:, :-tgt_len] = -100
        loss = model(input_ids=chunk, labels=labels).loss
        nlls.append(loss.item() * tgt_len)
        n_toks += tgt_len
        if end == n:
            break
    return math.exp(sum(nlls) / n_toks)


@torch.no_grad()
def _judge_generate(model, tok, device, no_system_role, system, user, max_new=5):
    """Run one judge generation and return the uppercased first word of the response."""
    if no_system_role:
        messages = [{"role": "user", "content": system + "\n\n" + user}]
    else:
        messages = [{"role": "system", "content": system},
                    {"role": "user",   "content": user}]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    ids = ids.to(device)
    out  = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip().upper()


def filter_truly_wrong(judges, qa_wrongs):
    """Remove answers that are semantically correct despite failing the string match.

    Uses majority vote across judges.  Returns only the subset of qa_wrongs where
    the majority of judges say the answer is WRONG (not just differently worded).
    Returns qa_wrongs unchanged when no judges are loaded.
    """
    if not judges:
        return qa_wrongs
    truly_wrong = []
    for q, a, r in qa_wrongs:
        user = _CORRECT_TMPL.format(q=q, a=a, r=r[:200])
        votes_wrong = sum(
            _judge_generate(m, t, d, nsr, _CORRECT_SYSTEM, user).startswith("WRONG")
            for m, t, d, nsr in judges
        )
        if votes_wrong > len(judges) / 2:
            truly_wrong.append((q, a, r))
    return truly_wrong


@torch.no_grad()
def _single_judge_verdict(model, tok, device, no_system_role, q, a, r):
    """Returns True if the judge says COHERENT for one (q, a, wrong_response) triple."""
    user = _JUDGE_TMPL.format(q=q, a=a, r=r[:200])
    return _judge_generate(model, tok, device, no_system_role,
                           _JUDGE_SYSTEM, user).startswith("COHERENT")


@torch.no_grad()
def llm_judge_coherent(judges, qa_wrongs):
    """Fraction of wrong answers judged COHERENT by majority vote across the judge ensemble.

    judges: list of (model, tok, device, no_system_role) — one entry per judge.
    Each wrong answer is voted on independently; majority (>50%) wins.
    coherent_rate = fraction of wrong answers where majority voted COHERENT.
    Returns nan if judges list is empty or there are no wrong answers.
    """
    if not judges or not qa_wrongs:
        return float("nan")
    coherent = 0
    for q, a, r in qa_wrongs:
        votes = sum(_single_judge_verdict(m, t, d, nsr, q, a, r)
                    for m, t, d, nsr in judges)
        if votes > len(judges) / 2:
            coherent += 1
    return coherent / len(qa_wrongs)


def mean_wrong_len(qa_wrongs):
    if not qa_wrongs:
        return float("nan")
    return float(np.mean([len(r.split()) for _, _, r in qa_wrongs]))


def mean_cos_sim(embed_model, qa_wrongs):
    if not qa_wrongs:
        return float("nan")
    wrong_texts   = [r for _, _, r in qa_wrongs]
    correct_texts = [a for _, a, _ in qa_wrongs]
    wrong_emb   = embed_model.encode(wrong_texts,   convert_to_tensor=True, show_progress_bar=False)
    correct_emb = embed_model.encode(correct_texts, convert_to_tensor=True, show_progress_bar=False)
    cos = torch.nn.functional.cosine_similarity(wrong_emb, correct_emb, dim=1)
    return cos.mean().item()


def append_wikitext_row(wt_csv, rho, alpha, mode, delta_wrong, ppl):
    label = f"LieNeuro-DID {mode} ρ={rho:.2f} α={alpha:.1f}"
    write_header = not wt_csv.exists()
    with open(wt_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "variant", "method", "layer", "coeff", "rho", "alpha",
            "delta_wrong", "wikitext_ppl"])
        if write_header:
            w.writeheader()
        w.writerow({
            "variant":     label,
            "method":      "weight_steering_did",
            "layer":       "",
            "coeff":       "",
            "rho":         rho,
            "alpha":       alpha,
            "delta_wrong": delta_wrong,
            "wikitext_ppl": ppl,
        })
    print(f"  → WikiText PPL={ppl:.4f} appended to {wt_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="llama", choices=list(MODEL_CONFIGS),
                        help="Model to run: llama (default), qwen, mistral, phi3")
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--judge-devices", default=None,
                        help="Comma-separated devices for the 3 ensemble judges, e.g. "
                             "cuda:1,cuda:2,cuda:3  (default: all on --device)")
    parser.add_argument("--legacy-scores", action="store_true",
                        help="Replicate ebdc3e8 scoring exactly: load S_lie from the .bak "
                             "naive-mask file (10 prompts), compute S_truth fresh with the "
                             "first 25 DIRECT_TRUTH_INSTRUCT prompts, and include lm_head "
                             "in the mask (no skip_modules). Use to verify the scale-mismatch "
                             "hypothesis.")
    parser.add_argument("--n-truth", type=int, default=None,
                        help="Number of truth prompts (DIRECT_TRUTH_INSTRUCT[:n]) to use "
                             "when computing S_truth. Default: all available prompts. "
                             "--lie-truth-ratio is applied relative to this value, so "
                             "--n-truth 25 --lie-truth-ratio 0.4 gives n_lie=10, n_truth=25 "
                             "(equivalent to the legacy 10/25 split). Ignored when "
                             "--legacy-scores is set.")
    parser.add_argument("--lie-truth-ratio", type=float, default=1.0,
                        help="Ratio of lie prompts to truth prompts used when computing "
                             "importance scores. Applied relative to --n-truth. "
                             "ratio=1.0 (default): n_lie = n_truth (balanced DID). "
                             "ratio=0.4 with --n-truth 25: n_lie=10 (legacy 10/25 split). "
                             "ratio=0.0: n_lie=0, DID score = −S_truth. Ignored when "
                             "--legacy-scores is set.")
    parser.add_argument("--exclude-lm-head", action="store_true",
                        help="Exclude lm_head (and embed_tokens) from the importance-score "
                             "computation and mask. By default lm_head is included. "
                             "Ignored when --legacy-scores or --verify-legacy is set.")
    parser.add_argument("--save-scores", action="store_true",
                        help="Save S_lie and S_truth to disk after computing them. "
                             "Filename encodes the model, n-truth, and lie-truth-ratio "
                             "(which determine the scores); rho/alpha are excluded since "
                             "they only affect build_mask. Example: "
                             "outputs/scores_llama_8b_ntruth25_ratio0.40.pt")
    parser.add_argument("--skip-scores", action="store_true",
                        help="Force recomputation of S_lie and S_truth even if a matching "
                             "scores file already exists on disk. By default, if a file "
                             "matching the current model/n-truth/lie-truth-ratio is found "
                             "it is loaded instead of recomputed.")
    parser.add_argument("--extra-metrics", action="store_true",
                        help="Also compute mean wrong-answer length and cosine similarity "
                             "(requires sentence-transformers)")
    args = parser.parse_args()

    device = args.device
    if args.judge_devices:
        judge_devices = args.judge_devices.split(",")
        if len(judge_devices) != len(JUDGE_CONFIGS):
            raise ValueError(f"--judge-devices must list exactly {len(JUDGE_CONFIGS)} devices")
    else:
        judge_devices = [args.device] * len(JUDGE_CONFIGS)

    mcfg      = MODEL_CONFIGS[args.model]
    model_id  = mcfg["model_id"]
    trust_rc  = mcfg["trust_remote_code"]

    _nt_val   = max(1, min(len(DIRECT_TRUTH_INSTRUCT), args.n_truth)) \
                if args.n_truth is not None else len(DIRECT_TRUTH_INSTRUCT)
    _suffix   = f"_ntruth{_nt_val}_ratio{args.lie_truth_ratio:.2f}"
    sweep_csv = mcfg["sweep_csv"].with_stem(mcfg["sweep_csv"].stem + _suffix)
    wt_csv    = mcfg["wt_csv"].with_stem(mcfg["wt_csv"].stem + _suffix)

    sweep_csv.parent.mkdir(parents=True, exist_ok=True)
    wt_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_id}")
    print(f"Loading model on {device} …")
    tok   = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR,
                                          trust_remote_code=trust_rc)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device,
        cache_dir=CACHE_DIR, trust_remote_code=trust_rc)
    model.eval()

    judges      = []
    embed_model = None
    if args.extra_metrics:
        for jcfg, jdev in zip(JUDGE_CONFIGS, judge_devices):
            jid = jcfg["model_id"]
            print(f"Loading judge {jid} on {jdev} …")
            jtok   = AutoTokenizer.from_pretrained(jid, cache_dir=CACHE_DIR)
            jmodel = AutoModelForCausalLM.from_pretrained(
                jid, torch_dtype=torch.float16, device_map=jdev, cache_dir=CACHE_DIR)
            jmodel.eval()
            judges.append((jmodel, jtok, jdev, jcfg["no_system_role"]))
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model (all-MiniLM-L6-v2) …")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # ── baseline ──────────────────────────────────────────────────────────────
    print("Evaluating baseline accuracy …")
    bl = evaluate_factual_accuracy(model, tok, qa_pairs=OOD_QA, device=device, temperature=0.0)
    if judges:
        bl_truly_wrong = filter_truly_wrong(
            judges,
            [(pq["question"], pq["gold"], pq["model_answer"])
             for pq in bl.per_question if not pq["correct"]])
        bl_wrong = len(bl_truly_wrong) / bl.n_questions
    else:
        bl_wrong = 1.0 - bl.accuracy
    bl_acc = 1.0 - bl_wrong
    print(f"Baseline: {bl_acc:.2%} accuracy  ({bl_wrong:.2%} wrong rate)")

    # ── importance scores ──────────────────────────────────────────────────────
    if args.legacy_scores:
        # Exact ebdc3e8 replication:
        #   S_lie  — loaded from the .bak naive-mask file (10-prompt scale)
        #   S_truth — freshly computed with first 25 DIRECT_TRUTH_INSTRUCT prompts
        #   lm_head — included (skip_modules=[])
        # This reproduces the scale mismatch (S_lie 10p vs S_truth 25p) present
        # at ebdc3e8, which gave PPL ≈ 20 at alpha=4.
        _bak = Path("outputs/mask_direct_rho0.1_alpha1.0.pt.bak")
        if not _bak.exists():
            raise FileNotFoundError(f"Legacy bak file not found: {_bak}")
        cfg = LieNeuroConfig(device=device, skip_modules=[])   # include lm_head
        ln  = LieNeuro(model, tok, cfg)
        print(f"[legacy] Loading S_lie from {_bak} (10-prompt scale) …")
        _bak_data  = torch.load(_bak, map_location="cpu", weights_only=False)
        ln._S_lie  = _bak_data["S_lie"]
        print(f"[legacy] Computing S_truth with first 25 DIRECT_TRUTH_INSTRUCT prompts …")
        ln._S_truth = ln._compute_scores_for_prompts(
            DIRECT_TRUTH_INSTRUCT[:25], desc="S_truth_legacy")
        _l = "model.layers.0.self_attn.q_proj"
        print(f"[legacy] S_lie/S_truth ratio (q_proj[0]): "
              f"{ln._S_lie[_l].float().sum() / ln._S_truth[_l].float().sum():.3f}  "
              f"(expected ≈ 0.42 to confirm ebdc3e8 conditions)")
    else:
        skip = ["lm_head", "embed_tokens"] if args.exclude_lm_head else []
        cfg = LieNeuroConfig(device=device, skip_modules=skip)
        ln  = LieNeuro(model, tok, cfg)

        _max_truth  = len(DIRECT_TRUTH_INSTRUCT)
        n_truth     = max(1, min(_max_truth, args.n_truth)) if args.n_truth is not None \
                      else _max_truth
        n_lie       = max(0, min(len(DIRECT_LIE),
                                 round(args.lie_truth_ratio * n_truth)))
        lie_prompts = DIRECT_LIE[:n_lie]

        _short       = mcfg["model_short"]
        _scores_path = Path(
            f"outputs/scores_{_short}"
            f"_ntruth{n_truth}_ratio{args.lie_truth_ratio:.2f}.pt")

        if not args.skip_scores and _scores_path.exists():
            print(f"Loading scores from {_scores_path} …")
            _saved      = torch.load(_scores_path, map_location="cpu", weights_only=False)
            ln._S_lie   = _saved["S_lie"]
            ln._S_truth = _saved["S_truth"]
            print(f"  n_truth={_saved['n_truth']}  n_lie={_saved['n_lie']}  "
                  f"lie_truth_ratio={_saved['lie_truth_ratio']}")
        else:
            print(f"n-truth={n_truth}  lie-truth-ratio={args.lie_truth_ratio:.3f}  "
                  f"→  n_lie={n_lie}")

            print(f"Computing S_truth for DIRECT_TRUTH_INSTRUCT[:{n_truth}] …")
            ln._S_truth = ln._compute_scores_for_prompts(
                DIRECT_TRUTH_INSTRUCT[:n_truth], desc="S_truth")

            if n_lie > 0:
                print("Computing S_lie for DIRECT_LIE …")
                ln._S_lie = ln._compute_scores_for_prompts(lie_prompts, desc="S_lie")
            else:
                # ratio=0: S_lie = 0 everywhere; DID score = −S_truth
                print("lie-truth-ratio=0: skipping S_lie (DID score = −S_truth)")
                ln._S_lie = {name: torch.zeros_like(v) for name, v in ln._S_truth.items()}

            if args.save_scores:
                torch.save({"S_lie": ln._S_lie, "S_truth": ln._S_truth,
                            "n_truth": n_truth, "n_lie": n_lie,
                            "lie_truth_ratio": args.lie_truth_ratio,
                            "model": _short},
                           _scores_path)
                print(f"Scores saved → {_scores_path}")

        _l = "model.layers.0.self_attn.q_proj"
        if _l in ln._S_lie and _l in ln._S_truth:
            _s_lie_sum   = ln._S_lie[_l].float().sum().item()
            _s_truth_sum = ln._S_truth[_l].float().sum().item()
            _ratio = _s_lie_sum / _s_truth_sum if _s_truth_sum else float("nan")
            print(f"S_lie/S_truth ratio (q_proj[0]): {_ratio:.3f}")

    # ── wikitext tokens (loaded once, reused every config) ────────────────────
    print("Loading WikiText tokens …")
    wt_tokens = load_wikitext_tokens(tok, WT_MAX_TOKS)

    # ── sweep ──────────────────────────────────────────────────────────────────
    sweep_rows = []
    fieldnames = ["mode", "rho", "alpha", "steered_acc", "delta_wrong",
                  "mask_coverage", "coherent_rate", "mean_wrong_len", "mean_cos_sim",
                  "wikitext_ppl"]
    with open(sweep_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for mode in MODES:
            for rho in RHO_VALS:
                for alpha in ALPHA_VALS:
                    print(f"\n── mode={mode}  ρ={rho}  α={alpha} ──")
                    cfg_i     = LieNeuroConfig(
                        importance_threshold=rho, alpha=alpha, device=device)
                    ln.config = cfg_i

                    ln.build_mask(mode=mode)
                    coverage = (sum(m.sum().item() for m in ln._mask.values()) /
                                sum(m.numel()      for m in ln._mask.values()))

                    ln.apply_steering()
                    result = evaluate_factual_accuracy(model, tok, qa_pairs=OOD_QA, device=device, temperature=0.0)
                    qa_wrongs = [(pq["question"], pq["gold"], pq["model_answer"])
                                 for pq in result.per_question if not pq["correct"]]
                    ppl = sliding_window_ppl(model, wt_tokens, WT_WINDOW, WT_STRIDE, device)
                    ln.restore_original()

                    qa_wrongs     = filter_truly_wrong(judges, qa_wrongs)
                    coherent_rate = llm_judge_coherent(judges, qa_wrongs)
                    wl  = mean_wrong_len(qa_wrongs)
                    cos = mean_cos_sim(embed_model, qa_wrongs) if embed_model else float("nan")

                    if judges:
                        steered_acc = 1.0 - len(qa_wrongs) / result.n_questions
                    else:
                        steered_acc = result.accuracy
                    dw = (1.0 - steered_acc) - bl_wrong
                    cr_str  = f"{coherent_rate:.0%}" if math.isfinite(coherent_rate) else "—"
                    wl_str  = f"{wl:.1f}"            if math.isfinite(wl)            else "—"
                    cos_str = f"{cos:.3f}"           if math.isfinite(cos)           else "—"
                    print(f"  steered_acc={steered_acc:.2%}  Δwrong={dw:+.2%}  "
                          f"coverage={coverage:.3%}  PPL={ppl:.2f}  coherent={cr_str}  "
                          f"len={wl_str}  cos={cos_str}")

                    row = {
                        "mode":           mode,
                        "rho":            rho,
                        "alpha":          alpha,
                        "steered_acc":    steered_acc,
                        "delta_wrong":    dw,
                        "mask_coverage":  coverage,
                        "coherent_rate":  coherent_rate,
                        "mean_wrong_len": wl,
                        "mean_cos_sim":   cos,
                        "wikitext_ppl":   ppl,
                    }
                    writer.writerow(row)
                    f.flush()
                    sweep_rows.append(row)

    print(f"\nSweep complete. Results in {sweep_csv}")

    # ── summary table ──────────────────────────────────────────────────────────
    print(f"\n{'mode':<12} {'ρ':>5} {'α':>5} {'Δwrong':>9} {'coverage':>10} "
          f"{'coherent':>9} {'len':>6} {'cos':>6}")
    print("─" * 72)
    for r in sorted(sweep_rows, key=lambda x: -x["delta_wrong"]):
        cr  = r["coherent_rate"];  cr_s  = f"{cr:>9.0%}"  if math.isfinite(cr)  else "        —"
        wl  = r["mean_wrong_len"]; wl_s  = f"{wl:>6.1f}"  if math.isfinite(wl)  else "     —"
        cos = r["mean_cos_sim"];   cos_s = f"{cos:>6.3f}"  if math.isfinite(cos) else "     —"
        print(f"{r['mode']:<12} {r['rho']:>5.2f} {r['alpha']:>5.1f} "
              f"{r['delta_wrong']:>+9.2%} {r['mask_coverage']:>10.3%}{cr_s}{wl_s}{cos_s}")

    # ── WikiText PPL baseline ─────────────────────────────────────────────────
    print("\nRunning WikiText PPL baseline …")
    bl_ppl = sliding_window_ppl(model, wt_tokens, WT_WINDOW, WT_STRIDE, device)
    print(f"  Baseline PPL = {bl_ppl:.4f}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
