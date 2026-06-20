#!/usr/bin/env python3
"""
LieNeuro-DID generalization experiment — run on arbitrary instruction-tuned models.

Computes DID importance scores (score_diff mode), sweeps 5 (ρ, α) configs,
generates sample outputs, and evaluates WikiText-2 PPL.  Results are saved
to outputs/generalization/<model_short>/

Usage
-----
  # Mistral on GPU 1
  CUDA_VISIBLE_DEVICES=1 conda run -n lie_neuro python run_lieneuro_did_generalization.py \
      --model-name mistralai/Mistral-7B-Instruct-v0.3 \
      --model-short mistral_7b \
      --device cuda:0

  # Qwen on GPU 2
  CUDA_VISIBLE_DEVICES=2 conda run -n lie_neuro python run_lieneuro_did_generalization.py \
      --model-name Qwen/Qwen2.5-7B-Instruct \
      --model-short qwen2p5_7b \
      --device cuda:0
"""

import argparse, csv, json, math, re as _re, subprocess, sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.evaluate import FACTUAL_QA, _build_prompt, generate_answer, _is_correct
from src.prompts import DIRECT_LIE, DIRECT_TRUTH_INSTRUCT

# ── Configs to sweep (must include ρ=0.04 α=4) ───────────────────────────────
CONFIGS = [
    (0.01, 4.0),
    (0.04, 2.0),
    (0.04, 4.0),   # required: sweet spot for Llama
    (0.04, 6.0),
    (0.10, 4.0),
]

MODE   = "score_diff"
TEMP   = 1.0
N_REPS = 3

WT_WINDOW, WT_STRIDE, WT_MAX_TOKS = 512, 256, 8192

# ── In-dist question set (questions seen during mask construction) ─────────────
_INSTRUCT_QS = {
    p.split("Answer truthfully: ", 1)[-1].strip().lower()
    for p in DIRECT_TRUTH_INSTRUCT
}

def is_indist(q: str) -> bool:
    return q.strip().lower() in _INSTRUCT_QS


# ── Prompt formatting for importance scoring ──────────────────────────────────

def format_for_scoring(raw_prompts, tokenizer):
    """
    Wrap raw instruction strings in the model's chat template so that importance
    scores are computed in the same token distribution as inference.

    Falls back to raw strings if the tokenizer has no chat template.
    """
    out = []
    for p in raw_prompts:
        messages = [{"role": "user", "content": p}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = p
        out.append(text)
    return out


# ── WikiText-2 PPL ────────────────────────────────────────────────────────────

@torch.no_grad()
def sliding_window_ppl(model, token_ids, device):
    ids = torch.tensor(token_ids, dtype=torch.long)
    n = len(ids)
    nlls, n_toks = [], 0
    for begin in range(0, n, WT_STRIDE):
        end     = min(begin + WT_WINDOW, n)
        chunk   = ids[begin:end].unsqueeze(0).to(device)
        tgt_len = min(WT_STRIDE, end - begin)
        labels  = chunk.clone()
        labels[:, :-tgt_len] = -100
        try:
            loss = model(input_ids=chunk, labels=labels).loss.item()
        except Exception:
            return float("inf")
        if not math.isfinite(loss):
            return float("inf")
        nlls.append(loss * tgt_len)
        n_toks += tgt_len
        if end == n:
            break
    try:
        return math.exp(sum(nlls) / n_toks)
    except OverflowError:
        return float("inf")


# ── Logits sanitizer — prevents device-side asserts from NaN/Inf logits ──────

class SanitizeLogits(LogitsProcessor):
    """Clamp NaN/Inf logits before sampling to prevent out-of-range token IDs."""
    def __call__(self, input_ids, scores):
        scores = torch.nan_to_num(scores, nan=0.0, posinf=50.0, neginf=-50.0)
        return torch.clamp(scores, -50.0, 50.0)


_LOGITS_PROC = LogitsProcessorList([SanitizeLogits()])


# ── Sampling ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def safe_generate(model, tok, prompt, max_new_tokens, temperature, device):
    """Generate with sanitized logits to survive heavily steered models."""
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature if temperature > 0.0 else 1.0,
        top_k=50,
        top_p=0.95,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        logits_processor=_LOGITS_PROC,
    )
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def sample(model, tok, q, n, device, temp=None, max_new=40):
    prompt = _build_prompt(q, tok)
    t = TEMP if temp is None else temp
    return [safe_generate(model, tok, prompt, max_new_tokens=max_new,
                          temperature=t, device=device) for _ in range(n)]


# ── Sample file writer ────────────────────────────────────────────────────────

def write_samples_file(out_path, model, tok, ln, rho, alpha, bl_results, device,
                       temp=None, n_reps=None, max_new=40):
    t      = TEMP   if temp    is None else temp
    n_rep  = N_REPS if n_reps  is None else n_reps
    ln.build_mask(mode=MODE)
    ln.apply_steering()

    st_results = []
    for qa in FACTUAL_QA:
        q, a  = qa["question"], qa["answer"]
        resps = sample(model, tok, q, n_rep, device, temp=t, max_new=max_new)
        corrects = [_is_correct(r, a) for r in resps]
        st_results.append((q, a, resps, corrects, is_indist(q)))

    ln.restore_original()

    lines = []
    def w(s=""): lines.append(s)

    w("LieNeuro DID — Sample Outputs")
    w(f"Mode: {MODE}  |  ρ={rho}  |  α={alpha}  |  temperature={TEMP}")
    w(f"Model: {tok.name_or_path}  |  {N_REPS} samples per question")
    w("=" * 72)
    w(); w("BASELINE (no steering)"); w("-" * 72)
    for q, a, resp, correct in bl_results:
        w(f"  {'✓' if correct else '✗'}  Q: {q}")
        w(f"       A: {resp}  [expected: {a}]")
    bl_wrong = sum(not r[3] for r in bl_results)
    w(); w(f"  Baseline wrong: {bl_wrong}/{len(FACTUAL_QA)}  ({bl_wrong/len(FACTUAL_QA):.0%})")

    def write_group(label, results):
        w(); w()
        w(f"STEERED — {label}  mode={MODE}  ρ={rho}  α={alpha}  ({N_REPS} samples/question)")
        w("-" * 72)
        for q, a, resps, corrects, _ in results:
            bl_resp = next(r for bq, _, r, _ in bl_results if bq == q)
            marks   = ["✓" if c else "✗" for c in corrects]
            w(f"  Q: {q}  [expected: {a}]")
            w(f"       Baseline:  {bl_resp}")
            for i, (r, m) in enumerate(zip(resps, marks)):
                w(f"       Sample {i+1}: {m}  {r}")
            w()
        bl_w  = sum(not _is_correct(next(r for bq, _, r, _ in bl_results if bq == q), a)
                    for q, a, _, _, _ in results)
        all_c = [c for _, _, _, cs, _ in results for c in cs]
        n_w = sum(not c for c in all_c)
        w("-" * 72)
        w(f"  Steered wrong:  {n_w}/{len(all_c)}  ({n_w/len(all_c):.0%})")
        w(f"  Baseline wrong: {bl_w}/{len(results)}  ({bl_w/len(results):.0%})")
        w(f"  Δ wrong-rate:  {n_w/len(all_c) - bl_w/len(results):+.0%}")

    write_group("IN-DISTRIBUTION (questions seen during mask construction)",
                [r for r in st_results if r[4]])
    write_group("OUT-OF-DISTRIBUTION (held-out questions)",
                [r for r in st_results if not r[4]])

    all_c = [c for _, _, _, cs, _ in st_results for c in cs]
    n_w   = sum(not c for c in all_c)
    w(); w("=" * 72); w("OVERALL SUMMARY")
    w(f"  Steered wrong:  {n_w}/{len(all_c)}  ({n_w/len(all_c):.0%})")
    w(f"  Baseline wrong: {bl_wrong}/{len(FACTUAL_QA)}  ({bl_wrong/len(FACTUAL_QA):.0%})")
    w(f"  Δ wrong-rate:  {n_w/len(all_c) - bl_wrong/len(FACTUAL_QA):+.0%}")

    out_path.write_text("\n".join(lines))
    print(f"    Saved → {out_path}")
    return n_w / len(all_c), bl_wrong / len(FACTUAL_QA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name",  required=True,
                        help="HuggingFace model ID, e.g. mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--model-short", required=True,
                        help="Short tag used in output filenames, e.g. mistral_7b")
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--cache-dir",   default="/scratch/aebrahim/hf_cache",
                        help="HuggingFace cache dir (must have sufficient disk space)")
    parser.add_argument("--configs",     default=None,
                        help='JSON list of [rho, alpha] pairs, e.g. \'[[0.001,2],[0.005,4]]\'')
    # Internal flag: run only a single (rho, alpha) config then exit.
    # Used by the parent process to isolate each config in its own subprocess.
    parser.add_argument("--single-rho",   type=float, default=None)
    parser.add_argument("--single-alpha", type=float, default=None)
    args   = parser.parse_args()
    device = args.device

    configs = json.loads(args.configs) if args.configs else [[r, a] for r, a in CONFIGS]
    configs = [(float(r), float(a)) for r, a in configs]

    out_dir = Path(f"outputs/generalization/{args.model_short}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Model : {args.model_name}")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")
    print(f"{'='*70}\n")

    cache_dir = args.cache_dir

    # ── Load model & tokenizer ─────────────────────────────────────────────────
    print("Loading tokenizer and model …")
    tok = AutoTokenizer.from_pretrained(args.model_name, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.float16, device_map=device,
        cache_dir=cache_dir)
    model.eval()
    print("Model loaded.\n")

    # ── WikiText-2 tokens (load once) ─────────────────────────────────────────
    print("Loading WikiText-2 …")
    ds        = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",
                             cache_dir=cache_dir)
    wt_text   = "\n\n".join(t for t in ds["text"] if t.strip())
    wt_tokens = tok.encode(wt_text, add_special_tokens=False)[:WT_MAX_TOKS]
    print(f"WikiText-2 tokens: {len(wt_tokens)}\n")

    # ── Baseline PPL (unsteered) ───────────────────────────────────────────────
    print("Computing baseline WikiText-2 PPL …")
    baseline_ppl = sliding_window_ppl(model, wt_tokens, device)
    print(f"Baseline PPL = {baseline_ppl:.4f}\n")

    # ── Baseline accuracy (cached to JSON so subprocesses skip it) ───────────
    baseline_json = out_dir / "baseline_results.json"
    if baseline_json.exists():
        print("Loading cached baseline results …")
        bl_data = json.loads(baseline_json.read_text())
        bl_results    = [(d["q"], d["a"], d["resp"], d["correct"]) for d in bl_data]
        bl_wrong_rate = sum(not r[3] for r in bl_results) / len(FACTUAL_QA)
    else:
        print("Evaluating baseline accuracy …")
        bl_results = []
        for qa in FACTUAL_QA:
            q, a = qa["question"], qa["answer"]
            resp = safe_generate(model, tok, _build_prompt(q, tok), 40, TEMP, device)
            bl_results.append((q, a, resp, _is_correct(resp, a)))
        bl_wrong_rate = sum(not r[3] for r in bl_results) / len(FACTUAL_QA)
        baseline_json.write_text(json.dumps([
            {"q": q, "a": a, "resp": r, "correct": c}
            for q, a, r, c in bl_results
        ]))
    print(f"Baseline wrong rate: {bl_wrong_rate:.0%}\n")

    # ── Compute DID importance scores (save compact diff in float16) ──────────
    # Store only (S_lie - S_truth) as float16 (~14GB vs 56GB for both full dicts)
    scores_path = out_dir / "did_scores.pt"
    lie_formatted   = format_for_scoring(DIRECT_LIE,           tok)
    truth_formatted = format_for_scoring(DIRECT_TRUTH_INSTRUCT, tok)

    cfg = LieNeuroConfig(importance_threshold=0.04, alpha=4.0, device=device)
    ln  = LieNeuro(model, tok, cfg)

    if scores_path.exists():
        print(f"Loading cached DID scores from {scores_path} …")
        did_diff = torch.load(scores_path, map_location="cpu")
        # Reconstruct: set _S_lie=diff, _S_truth=0 → score_diff gives diff-0=diff ✓
        ln._S_lie   = {k: v.float() for k, v in did_diff.items()}
        ln._S_truth = {k: torch.zeros_like(v) for k, v in ln._S_lie.items()}
    else:
        print("Computing DID importance scores …")
        ln.compute_importance_scores(lie_formatted, truth_formatted)
        did_diff = {k: (ln._S_lie[k] - ln._S_truth[k]).half() for k in ln._S_lie}
        torch.save(did_diff, scores_path)
        print(f"Scores saved → {scores_path}")
    print("Importance scores ready.\n")

    # ── Single-config mode (called as subprocess by parent) ───────────────────
    # Subprocess: run only one (rho, alpha) config. Use greedy, short outputs.
    # Baseline and scores are loaded from cache → skips all heavy recomputation.
    if args.single_rho is not None:
        rho, alpha = args.single_rho, args.single_alpha
        samples_path = out_dir / f"samples_rho{rho}_a{int(alpha) if alpha == int(alpha) else alpha}.txt"
        ln.config.importance_threshold = rho
        ln.config.alpha                = alpha
        # Greedy + short outputs avoid sampling-path CUDA failures in subprocess
        write_samples_file(samples_path, model, tok, ln, rho, alpha, bl_results,
                           device, temp=0.0, n_reps=1, max_new=15)
        return

    # ── Sweep configs (parent mode) ───────────────────────────────────────────
    results_rows = []
    base_cmd = [
        sys.executable, __file__,
        "--model-name",  args.model_name,
        "--model-short", args.model_short,
        "--device",      device,
        "--cache-dir",   args.cache_dir,
    ]

    for rho, alpha in configs:
        print(f"\n{'─'*60}")
        print(f"ρ={rho}  α={alpha}")

        samples_path = out_dir / f"samples_rho{rho}_a{int(alpha) if alpha == int(alpha) else alpha}.txt"

        # ── Generate samples in isolated subprocess ────────────────────────
        if samples_path.exists():
            print(f"  Samples exist, parsing wrong-rate …")
        else:
            print(f"  Generating samples (subprocess) …")
            cmd = base_cmd + ["--single-rho", str(rho), "--single-alpha", str(alpha)]
            import os as _os
            sub_env = _os.environ.copy()
            # Make CUDA errors synchronous so the subprocess exits instead of hanging
            sub_env["CUDA_LAUNCH_BLOCKING"] = "1"
            proc = subprocess.Popen(cmd, env=sub_env)
            try:
                proc.wait(timeout=300)
                if proc.returncode != 0:
                    print(f"  WARNING: sample generation crashed (exit {proc.returncode})")
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                print(f"  WARNING: sample generation timed out after 300s — killed")

        # Parse wrong-rate from file (may not exist if subprocess crashed)
        if samples_path.exists():
            m = _re.search(r"Steered wrong:\s+\d+/\d+\s+\((\d+)%\)",
                           samples_path.read_text().split("OVERALL")[-1])
            st_wrong_rate = int(m.group(1)) / 100 if m else float("nan")
        else:
            st_wrong_rate = float("nan")
        print(f"  wrong_rate={st_wrong_rate:.0%}" if not math.isnan(st_wrong_rate) else "  wrong_rate=crash")

        # ── PPL in parent process (safe, no sampling) ──────────────────────
        print("  Computing WikiText-2 PPL …")
        ln.config.importance_threshold = rho
        ln.config.alpha                = alpha
        ln.build_mask(mode=MODE)
        ln.apply_steering()
        ppl = sliding_window_ppl(model, wt_tokens, device)
        ln.restore_original()
        print(f"  PPL = {ppl:.4f}")

        delta_wrong = (st_wrong_rate - bl_wrong_rate) if not math.isnan(st_wrong_rate) else float("nan")
        results_rows.append({
            "model":        args.model_name,
            "model_short":  args.model_short,
            "rho":          rho,
            "alpha":        alpha,
            "wrong_rate":   st_wrong_rate,
            "delta_wrong":  delta_wrong,
            "wikitext_ppl": ppl,
            "baseline_ppl": baseline_ppl,
            "baseline_wrong_rate": bl_wrong_rate,
        })
        dw_str = f"{delta_wrong:+.0%}" if not math.isnan(delta_wrong) else "crash"
        print(f"  Δwrong={dw_str}  PPL={ppl:.2f}")

    # ── Save summary CSV ──────────────────────────────────────────────────────
    csv_path = out_dir / "results.csv"
    fields = ["model", "model_short", "rho", "alpha",
              "wrong_rate", "delta_wrong", "wikitext_ppl",
              "baseline_ppl", "baseline_wrong_rate"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"\nResults saved → {csv_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Model: {args.model_name}")
    print(f"  Baseline wrong: {bl_wrong_rate:.0%}  |  Baseline PPL: {baseline_ppl:.2f}")
    print(f"{'─'*60}")
    hdr = f"  {'ρ':>5} {'α':>5}  {'Wrong%':>7}  {'Δwrong':>8}  {'PPL':>12}"
    print(hdr); print("  " + "─" * (len(hdr) - 2))
    for r in results_rows:
        wr  = f"{r['wrong_rate']:.0%}"  if not math.isnan(r['wrong_rate'])  else "crash"
        dw  = f"{r['delta_wrong']:+.0%}" if not math.isnan(r['delta_wrong']) else "crash"
        print(f"  {r['rho']:>5}  {r['alpha']:>4}  {wr:>7}  {dw:>8}  {r['wikitext_ppl']:>12.1f}")
    print(f"{'─'*60}\n")
    print("Done.")


if __name__ == "__main__":
    main()
