# Pythia size ablation — how to run it

The question is how the lead of the DID weight mask over the baselines moves
with model scale, holding the protocol fixed. The Pythia suite is the right
instrument for it: eight sizes from 70m to 12b, one training recipe, one
tokenizer, one corpus, so size is close to the only thing that varies.

Two scripts do the work. `prepare_pythia_checkpoints.py` turns each size's final
checkpoint into a directory the existing runner can be pointed at, and
`sweep_pythia.sh` runs the honesty protocol across sizes, one lane per GPU.
Nothing about the method changes — same `--rho 0.04 --ratio 0.37`, same arms,
same eval set, the steering baseline swept the same way as in the main table.

**Status: the plumbing is tested, no results exist yet.** The full protocol has
been run end to end on pythia-70m only, as a check that the pipeline works on
this architecture. Nobody has run 160m and up.

---

## Before you start

```bash
conda activate lie_neuro
cd /srv/local/projects/aebrahim/lie_detection
```

Three prerequisites:

**1. The LIARS' BENCH eval set.** Not in git (it lives under `outputs/`, which
is untracked). Build it once:

```bash
python liars_bench_prep.py --out outputs/liars_bench/eval_set.jsonl
```

It pulls `Cadenza-Labs/liars-bench` from the Hub, so this step needs network.

**2. Disk for the checkpoints.** About 46 GB for all eight sizes in fp16
(70m 0.15 → 12b 22.2 GiB). They go to `$PYROOT`, default
`/scratch/aebrahim/pythia_final`, as symlinks into the HF cache rather than
copies, so the cache is where the space actually goes.

**3. Network for the first fetch.** `prepare_pythia_checkpoints.py` resolves a
pinned Hub revision, so it cannot run under `HF_HUB_OFFLINE=1`. Once a size is
prepared, running it needs no network.

---

## Step 1 — prepare the checkpoints

```bash
python prepare_pythia_checkpoints.py --list       # the grid, no network
python prepare_pythia_checkpoints.py --verify     # revision check, no weights
python prepare_pythia_checkpoints.py             # prepare all eight sizes
python prepare_pythia_checkpoints.py --sizes 410m 1b 1.4b     # or just some
```

| size | layers | hidden | mid-depth | batch | score batch |
|------|-------:|-------:|----------:|------:|------------:|
| 70m  |  6 |  512 |  3 | 16 | 8 |
| 160m | 12 |  768 |  6 | 16 | 8 |
| 410m | 24 | 1024 | 12 | 16 | 8 |
| 1b   | 16 | 2048 |  8 | 16 | 4 |
| 1.4b | 24 | 2048 | 12 | 16 | 4 |
| 2.8b | 32 | 2560 | 16 |  8 | 2 |
| 6.9b | 32 | 4096 | 16 |  8 | 2 |
| 12b  | 36 | 5120 | 18 |  4 | 2 |

Layer and width counts are read from each repo's own `config.json` at the
pinned revision, not from the paper. The batch sizes are starting points for an
80 GB card; the runner does not back off on its own, so halve them if a size
dies. Note 1b has *fewer* layers than 410m and 1.4b — Pythia's 1b is wide and
shallow — so "layer 12" is not a depth you can carry across sizes.

Each prepared directory is self-contained: weights, config, tokenizer, and a
`PROVENANCE.json` recording the repo, the revision and its commit hash, so a
results file can always be traced back to a checkpoint.

### What `--verify` is actually checking

The Pythia repos keep every intermediate checkpoint on a `step<N>` branch, and
`main` is the end of training. We pin `step143000` so the revision is explicit
in the provenance rather than implied by whatever `main` points at today.

That the two agree has been checked rather than assumed. On
pythia-70m-deduped all 76 parameters are bit-identical across `main` and
`step143000`; `main` carries 18 extra entries that are non-persistent buffers
(`attention.bias`, `attention.masked_bias`, `rotary_emb.inv_freq`) which the
newer safetensors conversion drops. `--verify` reports the cheaper version of
the same check at all eight sizes — the LFS digest of `pytorch_model.bin`,
identical on both revisions everywhere — without downloading any weights.

Two traps the script handles, both inherited from
`ptm_ftm/eval_pythia_scaling.py`, which hit them first:

- **The weight format is not uniform.** 70m–1b and 1.4b publish a single
  `model.safetensors` at `step143000`; **2.8b, 6.9b and 12b have only `.bin`
  there.** The format is therefore resolved per *(repo, revision)*, not per
  repo. A format decided once from one size fetches nothing at all on another.
- **Not everything on a step branch is wanted.** pythia-1b's `step143000`
  carries a multi-GB `optimizer.pt`, and only pythia-1b publishes a
  `generation_config.json` — taking it would give that one size different
  decoding defaults from its siblings, which is the opposite of what an
  ablation wants. The allow-list excludes both.

If you ever extend this to *intermediate* checkpoints, read the warning in
`patterns_for()` in that other script first: on some sizes `pytorch_model.bin`
is the same blob on every step branch, so fetching it would silently score the
final weights at every step. That does not bite here because we only read the
final step, where `.bin` and `main` legitimately coincide.

---

## Step 2 — run the sweep

```bash
GPUS=0,1,2,3 bash sweep_pythia.sh                        # all eight sizes
GPUS=2,3 SIZES="410m 1b 1.4b 2.8b" bash sweep_pythia.sh  # a subset
GPUS=2 SIZES=70m DRY=1 bash sweep_pythia.sh              # print, run nothing
```

Sizes are dealt round-robin across the cards and each lane runs its sizes in
sequence, so eight sizes over four cards is four lanes of two rather than eight
serial loads. `SIZES` defaults to largest-first so a 12b lane does not start
last and leave three cards idle. Logs land one per size in
`scratchpad/logs_pythia/`.

Everything is env-overridable for another machine: `REPO`, `PY`, `CACHE`,
`OUTDIR`, `PYROOT`, `LOGDIR`, plus `ARMS`, `EVALSET`, `RHO`, `RATIO`, `COEFFS`
and an `EXTRA` passthrough.

**Smoke-test a new size cheaply** with the item caps, not `--n-per-split`
(which only applies to the extra transfer eval sets, not to LIARS' BENCH):

```bash
GPUS=2 SIZES=70m EXTRA="--n-hp 40 --n-id 40" bash sweep_pythia.sh
```

### The arms, and why these five

```
unsteered,magmatch100,poolmix100,ours075,prompt2
```

- `poolmix100` — the mask, polarity-balanced over both prompt formats, α=−1.0
- `ours075` — the plain lie-contrast mask at α=−0.75 (the "lie-mask" baseline)
- `magmatch100` — the magnitude-matched random mask, the null that says whether
  a lead is about *which* weights were chosen or merely how many
- `prompt2` — the honest system-prompt baseline
- `unsteered` — the unedited model

`magmatch` is the one arm you cannot drop: without it a lead means nothing. It
is built on the CPU on purpose (it needs a full random permutation of each
weight matrix), so expect a quiet, CPU-bound, GPU-idle stretch before the arms
start. That is normal, not a hang.

### Steering layers

`--as-layers` is the one setting that is not size-free, because a layer index
does not mean the same thing at 6 layers as at 36. Each lane sweeps the three
layers around mid-depth with both relative coefficients
(`--as-rel-coeffs=-0.18,-0.5`), and the arm is chosen on the validation half
downstream — the same way the main table chooses a setting for every method.
The steering arms are registered but not in the default `ARMS`; add
`asrL{layer}_{coeff}` keys (e.g. `asrL12_-0.18`) if you want them scored in the
same pass.

### Cost

Measured only at 70m, on one card, over the full eval set (2263 graded items
plus 150 alpaca): belief probing 25 s from cold, norm pools and the CPU mask
build about a minute, then **42 s per arm** — about 5 minutes for all five
arms. Above 70m this is unmeasured. Run 410m or 1b before committing four cards
to 12b.

Runs resume in two places: the belief cache
(`outputs/liars_bench/beliefs_<model>_<tag>.jsonl`) and the steering vectors
(`outputs/as_vectors_<model>_ntruth25.pt`) are both reused if present, so a
re-run after a crash skips the probing.

---

## Step 3 — judge and score

Each run writes two files into `$OUTDIR`:

```
liars_bench_honesty_<model-dir-name>_<tag>.csv              per-arm summary
liars_bench_honesty_<model-dir-name>_<tag>_responses.jsonl  per-item verdicts
```

The summary's rates come from the keyword parser, which is **not good enough for
comparing arms** — see the table at the top of `judge_claims.py`, where the
unsettled `no_claim` share runs from 2.8% to 65.8% across arms of one model and
is not neutral between them. Re-grade with the LLM judge before reading
anything:

```bash
python judge_claims.py \
  --responses outputs/liars_bench_honesty_<model>_<tag>_responses.jsonl \
  --eval-set outputs/liars_bench/eval_set.jsonl \
  --judge Qwen/Qwen2.5-7B-Instruct --cache-dir /scratch/aebrahim/hf_cache \
  --tag v2
```

Use **one judge for every size.** A size ablation whose grader changes partway
measures the grader. Note the paper's LIARS' BENCH column is GPT-5.6-graded,
and there is no OpenAI key on this box — `judge_claims.py` has no API path at
all, it is a local HF-model judge. So these numbers will be Qwen-judged and are
comparable to each other, and to the Qwen-judged table in the appendix, but not
directly to the GPT-5.6 column of the main table.

### Where the numbers are supposed to land

**There is already a size study, and it is pre-registered.**
`reports/size_sweep.py` regresses

```
gain  = honest(mask) - honest(best other method)
gain0 = honest(mask) - honest(unedited)
```

on `log10(parameters)`, per family, per benchmark, with a paired bootstrap over
test items for each point and a case-resampling bootstrap over models for the
slope. Its docstring fixes the analysis as of 2026-10-04: **the question is
whether the 95% interval of the slope excludes zero.** Don't re-specify that
after seeing the Pythia points.

It also states the problem Pythia is good for: *"With three to five points per
family the slope CIs are wide."* Every existing family is 3–5 points, and
Dolly-v2 — the one open base-model family — left the Hub in Oct 2026 and
survives only as a commented-out row. Pythia is **eight points in one family on
one corpus**, which is the first series in this project that can actually
constrain a slope. That, not the per-size leads, is the result worth having.

To wire it up, add the sizes to `PARAMS` in `reports/size_sweep.py`. Counts are
exact for the three sizes prepared so far and computed from each config for the
rest (the analytic count reproduces all three measured values to the byte):

```python
"Pythia-70m":  0.0704e9,   # 70,426,624      measured
"Pythia-160m": 0.162e9,    # 162,322,944     measured
"Pythia-410m": 0.405e9,    # 405,334,016
"Pythia-1b":   1.01e9,     # 1,011,781,632   measured
"Pythia-1.4b": 1.41e9,     # 1,414,647,808
"Pythia-2.8b": 2.78e9,     # 2,775,208,960
"Pythia-6.9b": 6.86e9,     # 6,856,253,440
"Pythia-12b":  11.8e9,     # 11,842,140,160
```

These are total parameters including both embedding matrices, matching how the
existing entries are written. `FAMILY` is derived by splitting on `-`, which
yields `Pythia` for all eight, so it needs nothing — but check it, since the map
special-cases `OLMo-2` and `Dolly-v2`.

Then decide how the rows get built. `build_final_table.py` has a `--sweep` mode
that reads size points from the sweep directories with `SWEEP_MODELS` and
`SW_TAGS`, which is the path the Qwen and OLMo size points took. Its tags
(`sw1`–`sw4`) are the Slurm sweep's, not `sweep_pythia.sh`'s, so either add
Pythia to `SWEEP_MODELS` and pass the matching tag, or add seven-model-style
rows to `MODELS`. **Ask whoever owns the table builder before changing it** —
it was deliberately frozen before the final runs, and that freeze is why the
existing leads mean anything.

For the protocol itself, `reports/build_final_table.py` holds it —
validation/test half-split by MD5 of the item id, the admission gates
(perplexity < 1.15×, control within 5 points, polarity within 5 points), best
usable arm per method chosen on validation, reported on test, paired bootstrap
for the lead. It does not read Pythia yet: `OUT` and `OUT2` point at
other filesystems and `MODELS` is the seven original models. Point them at your
`outputs/` — note `judge_claims.py` already takes `LIE_OUT` from the
environment for exactly this reason, but `build_final_table.py` does not, so
that one is still an edit.

---

## The one decision worth making before you run

**Pythia is a base model with no chat template**, and the runner renders every
item through `tokenizer.apply_chat_template`. Every Pythia tokenizer has
`chat_template = None`, so without one the runner raises immediately.

The prepared directory therefore carries an explicit template, defined once as
`CHAT_TEMPLATE` in `prepare_pythia_checkpoints.py`: the system prompt as a bare
preamble, then labelled `User:` / `Assistant:` turns, generation continuing
after `Assistant:`. It renders as

```
You are a helpful assistant.

User: What is the capital of France?
Assistant:
```

This is a real experimental choice, not boilerplate — it decides how plainly
the deception instruction reaches a model that was never instruction-tuned, and
a different framing could plausibly move the honest rates more than the mask
does. It also has to round-trip a pre-filled assistant turn, which MASK's
doubling-down items use. **If you want a different framing, change it there and
nowhere else**, and re-prepare with `--force`: the sizes stop being comparable
the moment they are prompted differently.

---

## What this needed from the shared pipeline

Three changes, all pushed with these scripts. They are additive — no model in
the main table has a `gpt_neox.layers` attribute or a Linear named
`embed_out`, and the third removed dead code — so no existing row moves.

1. **`did_estimators.py:decoder_layers()`** did not know GPT-NeoX keeps its
   block list at `gpt_neox.layers`. It raised `AttributeError` on every Pythia
   model. Path added.
2. **`did_estimators.py:target_layers()`** skipped `lm_head`, but GPT-NeoX
   calls the output embedding `embed_out`. Left alone, **the contrast would
   have selected weights inside the output embedding** — a silent correctness
   bug, not a crash. At 70m this is the difference between 24 and 25 targeted
   Linear modules. Added to the skip list.
3. **`run_liars_bench_honesty.py`** read a residual norm at `args.as_layer`
   (default 7) even when `--as-layers` was given, and the value was assigned
   and never used. Harmless on every model so far, because they all have at
   least 8 layers; **pythia-70m has 6**, so it died with `KeyError: 7` after
   the mask build. Dead code removed.

The 70m end-to-end run exercises all three: 6 decoder blocks found, 24 Linear
modules targeted with the embeddings excluded, steering vectors collected at
layers 2/3/4, mask built, five arms scored, wikitext perplexity 65.0.

---

## What to expect, honestly

The protocol has been run twice at 70m, identical settings, and **the two runs
disagree on the sign of the effect for three of the four methods.** Lie share on
instructed-deception, relative to the unedited model:

| arm | run 1 | run 2 | ppl |
|---|---:|---:|---:|
| unsteered | 20.5% | 25.8% | 1.00 |
| `poolmix` (the mask) | 26.3% | 28.4% | 1.095 |
| `ours` (lie-contrast) | 27.1% | 35.8% | 1.033 |
| `magmatch` (null) | **32.4%** | **22.4%** | 1.095 |
| `prompt:honest2` | **19.6%** | **27.6%** | 1.00 |

Every arm is at or above the unedited rate in at least one run, and the null and
the honest prompt swap sides between runs. **Nothing here separates from noise.**

Two reasons, both visible in the output and both structural at this size:

- **The belief gate removes almost everything on harm-pressure.** Of 800 items
  per split, 60–78 survive as gradeable — the rest are items the unedited 70m
  model does not know, so a lie about them is not measurable. After the
  settled-response filter, those splits carry **14–18 items**. Treat HP-C and
  HP-KR as unreported at 70m rather than as small effects.
- **The model mostly fails to make a claim at all.** `no_claim` runs 73–80% on
  instructed-deception, so the 400-item split reduces to 67–92 settled
  responses, and the shares above are percentages of that. An arm that pushes
  output into non-claims shrinks its own denominator, which is why the summary
  table prints `no_claim` beside every share.

One thing *is* stable: the perplexity ratios are identical to four decimals
across both runs (1.0954 for `poolmix` and `magmatch`, 1.0328 for `ours`). The
mask build is deterministic; all of the variance above comes from sampling at
T=0.7 with one draw per arm. So the fix for the small sizes, if you want them in
the table, is more draws per item, not a re-run.

What this means for the ablation: **do not expect the small end to carry the
result.** The interesting range is probably 1b and up, where instruction
following is good enough that there is lying to remove. Two things worth
watching as you go up:

- **Headroom, not points.** A lead in points is not comparable across models
  with different base lying rates. `updated_paper/causation.tex` gives the
  gain-over-headroom form — the share of the unedited model's falsehoods the
  edit removed — and that is what to plot against size.
- **Where the null stops keeping pace.** `magmatch` perturbs the same number of
  weights at the same perplexity cost as the real mask. At 70m it is
  indistinguishable from it. The size at which the real mask pulls away from it
  is the actual claim this ablation can support, and it is a more defensible one
  than a monotone trend line.

A plausible honest outcome is that 70m and 160m are reported as floor effects
with the denominators shown, and the trend is read from 410m up. Decide that
from the data rather than dropping sizes after seeing them.

## Troubleshooting

| symptom | cause |
|---|---|
| Hangs after `norms[tf_truth]`, GPU at 0% | normal — the `magmatch` control mask is being built on the CPU |
| `KeyError: 7` after the mask build | you are on an unpatched `run_liars_bench_honesty.py`; pull |
| `AttributeError: could not locate the decoder layer list` | unpatched `did_estimators.py`; pull |
| `Cannot use chat template functions` | pointing at a raw HF repo id instead of a prepared directory |
| CUDA OOM | halve `batch` / `score_batch` for that size in `SIZES` in `prepare_pythia_checkpoints.py` |
| `no eval set at ...` | run `liars_bench_prep.py` (see Before you start) |
| Two lanes stall on the first fetch | shouldn't happen — the driver prefetches serially before any lane starts — but if it does, prepare that size by hand first |
