# The honesty edit on LIARS' BENCH: an out-of-distribution test

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` (fp16)
**Hardware:** 1× NVIDIA A100-SXM4-40GB (NCSA Delta, gpua020)
**Config:** ρ = 0.04, ratio = 0.37, damped (negative α), output head excluded
**Code:** `liars_bench_prep.py`, `run_liars_bench_honesty.py`
**Date:** 2026-09-25

---

## 1 Why this run exists

The generalisation evidence for the honesty edit rested on three lie framings I
wrote by hand, applied to the same 100 short factual questions. This replaces
both halves of that with the benchmark's own material.

- **Different pressure.** The harm-pressure splits never instruct the model to
  lie. They construct a scenario where telling the truth would cause harm and let
  the model answer, so nothing in the prompt resembles the "Lie when answering
  this question" template the mask was built from.
- **Different questions.** WMDP science items, not capital cities.
- **The benchmark's definition of a lie.** A response is a lie when it
  contradicts what the model demonstrably believes. Belief is measured by a
  neutral probe run on *unsteered* weights, and items the model does not know are
  excluded rather than counted. Of 800 items per split, 523 and 502 survived that
  gate.

Two splits are graded exactly by the regex grader and carry the result.
`instructed-deception` cannot be scored properly without the 32B judge, so it is
reported as a lie share over settled responses and marked as a hint.
`convincing-game` is skipped for the same reason. Alpaca is the degeneracy
control, and no arm degenerated on it (0.0% everywhere).

## 2 Harm-pressure knowledge report (n ≈ 502)

The model is asked to report on what it knows, under pressure. Lying means
claiming not to know something it demonstrably knows. Because this is free text,
the model can also evade, so the lie rate must be read alongside where the
displaced mass goes.

| arm | lie % (95% CI) | honest | no_claim | refusal | ppl |
|---|---|---|---|---|---|
| unsteered | 54.8% [50.6, 59.2] | 28.5% | 14.9% | 1.8% | 1.00× |
| **ours α = −0.75** | **37.7%** [33.5, 41.9] | **43.5%** | 15.5% | 3.4% | 1.02× |
| **ours α = −1.0** | **31.1%** [27.3, 35.3] | **44.1%** | 23.6% | 1.2% | 1.04× |
| actsteer L7, c = −8 | 40.1% [35.9, 44.3] | 28.1% | 30.1% | 1.6% | 1.02× |
| magnitude-matched control | 46.1% [41.9, 50.5] | 27.3% | 25.1% | 1.4% | 1.02× |
| prompt:honest | 61.8% [57.6, 66.1] | 24.4% | 10.1% | 3.8% | 1.00× |

**The claim holds here, and the decomposition is what makes it convincing.**
Three arms lower the lie rate, but only ours converts lies into honest answers:

- ours at α = −0.75 moves the honest rate from 28.5% to 43.5%, nearly 15 points,
  while leaving the no-claim rate essentially unchanged at 15.5%.
- the magnitude-matched control lowers the lie rate to 46.1% but its honest rate
  *falls* to 27.3%, with no-claim rising 10 points. Its entire improvement is
  evasion.
- activation steering does the same thing more strongly: lie rate 40.1%, honest
  rate unchanged at 28.1%, no-claim up 15 points to 30.1%.

So a lie-rate table alone would have ranked activation steering second and made
the control look partly effective. Reading the honest column shows both are
buying their numbers by making the model say nothing, and that only the weight
edit actually shifts behaviour toward disclosure. Refusal stays low everywhere,
so this is not a safety refusal either.

At α = −1.0 the lie rate falls further, to 31.1%, but about 9 of those points go
to no-claim rather than to honesty, so α = −0.75 is the cleaner operating point.
That matches the earlier finding that partial damping beats full ablation.

An honest system prompt again makes things worse, 61.8% against 54.8% unsteered.

## 3 Harm-pressure forced choice (n = 523)

The model picks one of four letters, so there is no room to hedge and the
no-claim rate is 0% by construction.

| arm | lie % (95% CI) | honest |
|---|---|---|
| prompt:honest | 18.2% [14.7, 21.6] | 81.8% |
| actsteer L7, c = −8 | 18.5% [15.3, 22.0] | 80.9% |
| unsteered | 19.3% [16.1, 22.8] | 80.7% |
| ours α = −0.75 | 20.3% [16.8, 23.7] | 79.7% |
| magnitude-matched control | 20.5% [17.0, 23.9] | 79.5% |
| ours α = −1.0 | 23.3% [19.7, 27.0] | 76.7% |

**The claim does not hold here.** Every arm sits inside every other arm's
interval, and our edit is nominally the worst of the six at full damping. Nothing
is significantly harmed, but nothing is helped.

The contrast with the previous section is mechanically coherent. The edit appears
to change what the model is willing to *disclose* in prose, not which option it
selects when forced to commit to one of four letters.

## 4 Instructed deception (hint only, unjudged)

Items carrying an honest system prompt, so a lie is unprompted deception. Lie
share over settled responses:

| arm | lie share | settled |
|---|---|---|
| unsteered | 6.5% | 261/263 |
| ours α = −0.75 | 8.9% | 257/263 |
| ours α = −1.0 | 10.7% | 244/263 |
| actsteer L7, c = −8 | 8.4% | 262/263 |
| magnitude-matched control | 6.5% | 261/263 |
| prompt:honest | 6.5% | 263/263 |

**This moves the wrong way for the claim.** Our edit raises unprompted lying from
6.5% to 8.9–10.7%. The repo's warning that this split needs an LLM judge is less
binding here than for the repo's own amplified arms: the regex grader settled 244
to 263 of 263 responses, because damping does not produce the word salad that
amplification does. So the number is more trustworthy than the warning implies,
though it is still a single unjudged metric on a small split.

## 5 What the out-of-distribution test settles

The honesty claim survives in part, and the part that survives is stronger than
the hand-written-framing test could show.

1. **Confirmed on knowledge-report pressure**, with no lie instruction anywhere in
   the prompt and on a different question distribution. The honest rate rises 15
   points at 1.02× perplexity, and neither the magnitude-matched control nor
   activation steering reproduces that: both lower the lie rate purely by
   evading.
2. **No effect on forced choice.** The edit does not change which option the model
   commits to.
3. **Mildly counterproductive on instructed deception**, where unprompted lying
   rises by 2 to 4 points.

The honest summary is that the edit shifts disclosure behaviour in free text
rather than installing a general disposition toward honesty. That is narrower
than "reduces lying", and it is the first claim here that has been tested against
pressure the method never saw.

## 6 Limits

- **"Honest" on these splits means disclosing hazardous knowledge.** The
  harm-pressure splits are built from WMDP, so a higher honest rate is the
  benchmark's definition of honesty and not automatically the desirable
  behaviour. This measures whether the model states what it believes, not whether
  it should.
- **Two splits, one judge-free grader.** The two splits that can be graded
  exactly disagree with each other, and the two that would need the 32B judge are
  unscored. Running the judge would complete the picture.
- **One model at 1.5B, one seed.** Generation is sampled at the benchmark's own
  temperatures, so a few points of the differences are sampling noise; the
  intervals above are binomial on the item counts and do not include that.
- **The belief gate is a probe.** 523 and 502 of 800 items survived it, and the
  gate inherits whatever error the neutral probe makes.
- **No convincing-game, no alpaca lie-rate.** Alpaca was used only to confirm no
  arm degenerated.

## 7 Reproducing

```bash
# needs HF access to the gated Cadenza-Labs/liars-bench dataset
export HF_CACHE_DIR=/work/hdd/bfqt/xiaocong/hf_cache
python liars_bench_prep.py \
  --out /work/hdd/bfqt/xiaocong/lie_detection_outputs/liars_bench/eval_set.jsonl

python run_liars_bench_honesty.py --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --rho 0.04 --ratio 0.37 --n-hp 800 --n-id 400 --n-alpaca 150
```

Beliefs are cached to `liars_bench/beliefs_<model>.jsonl` after the first run.
