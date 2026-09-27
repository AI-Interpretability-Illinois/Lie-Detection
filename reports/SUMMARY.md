# LieNeuro weight steering: current method and results

**Status as of 2026-09-27.** Every number here uses the fixed graders (see
section 6). Runs still in progress are listed at the end.

---

## 1 Method

The method edits weights, with no gradients and no fine-tuning.

1. **Two prompt sets of 25**, over the same factual question stems: a treatment arm
   that instructs the model to assert a falsehood, and a control arm that instructs
   it to answer truthfully ("Answer truthfully: Q").
2. **Wanda importance** for every weight of every attention and MLP matrix,
   `S_ij = |W_ij| · ‖x_j‖`, from forward passes over each set. The score factorises,
   so the only thing that differs between the sets is a per-column activation norm
   `n_j`, which makes the whole contrast a few megabytes per model.
3. **Contrast and select**: `score_ij = |W_ij| · (v·n_treat_j − (1−v)·n_ctrl_j)`,
   with `v = ratio/(1+ratio)`, ratio 0.37. Take the top 4% per matrix.
4. **Edit**: `W' = W ⊙ (1 + α·mask)`. Full ablation, α = −1, is the setting that
   works: it makes the model refuse instructed falsehoods.

Two facts about what this actually selects:

- **It is a low-magnitude selector.** The column statistic is negative for about
  99.5% of columns, so the top of `|W| × statistic` is the *smallest* weights: the
  selection sits at ~0.04× the global median |W|. Inside a band of equally small
  weights, no contrast selects anything useful (section 5), so the magnitude term is
  part of the mechanism rather than a nuisance.
- **The axis is instructed falsehood, not deception.** Building the treatment arm
  from fiction-framed or mistake-framed falsehood requests works as well as or
  better than lie instructions, and a truthful-paraphrase arm does nothing. The
  recommended treatment arm is therefore falsehood instructions (the `fiction` or
  `pooled` sets), not the published lie-only set.

## 2 Evaluation

- **Pressure metric (in-house).** 100 factual questions, none used to build any
  mask. Each is asked neutrally and under six pressure framings; one reuses the mask
  wording, three are held-out lie paraphrases, two are falsehood framings. Score:
  share of clean false claims on the questions the unsteered model answers correctly
  when asked neutrally. Damage: neutral accuracy and WikiText perplexity. An arm is
  **usable** only at perplexity under 1.15× and neutral accuracy within 5 points of
  unsteered.
- **LIARS' BENCH (external).** The 537 items whose system prompt orders the model
  to deceive, and the 263 whose system prompt orders honesty as a control. A lie is
  a claim that contradicts the model's own belief, elicited neutrally on unsteered
  weights. Claims are read by a hybrid grader: the benchmark's keyword parser where
  it settles, a validated LLM judge on free-text replies (section 6).
- **Baselines.** Activation steering by difference-in-means (Stolfo et al., ICLR
  2025; the same vectors as contrastive activation addition, Rimsky et al., ACL
  2024), swept over 3 to 5 layers and 2 to 3 norm-relative strengths per model; an
  honest system prompt; a random mask matched to the real mask's magnitude
  deciles; the unsteered model.

## 3 Where the method beats every baseline

**External benchmark, deception-instructed items.** Honest rate, best usable arm
per family. Usable also requires that the honesty-instructed control half stay
within 5 points of unsteered.

| model | falsehood-built mask | published lie mask | best activation steering | honest prompt | random mask | unsteered | lead over best baseline |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | **92.5%** | 79.5% | 52.6%\* | 81.8% | 42.3% | 35.3% | **+10.7** |
| Phi-4-mini | **87.5%** | 78.0% | 80.4% | 75.8% | 65.7% | 68.9% | **+7.1** |
| Qwen2.5-7B | 95.9% | 95.7% | 92.6% | 96.1% | 80.6% | 85.1% | −0.2 |

\* On Llama-8B the strongest steering arm reaches 83.3% but makes the model lie more
when told to be honest (control half down 6.9 points), so it fails the usability
bar. Counting it anyway, the lead is +9.2.

Only the falsehood-built mask wins. The published lie-built mask does not beat the
baselines on this benchmark.

**In-house pressure metric, head-to-head.** Held-out falsehood rate over the three
lie paraphrases, lower is better.

| arm | Llama-3.1-8B | Qwen2.5-3B |
|---|---|---|
| weight mask, α = −1 | **8.3%** | **5.3%** |
| best of 15 activation-steering configs | 32.4% | 7.7% |
| honest system prompt | 55.7% | 17.7% |
| random mask, same magnitudes | 66.5% | 15.8% |
| unsteered | 68.4% | 18.8% |

On Llama-8B the mask is 24 points ahead at lower cost (1.01× perplexity against
1.11×), and its effect shows up as correct answers under pressure rising from 32%
to 92%, not only as fewer lies. The pooled falsehood mask reaches 2.7% on the same
framings, in a separate run against an identical unsteered baseline.

## 4 Supporting results

- **Treatment arm.** On the in-house metric the falsehood-built arm beats the
  lie-built arm on 7 of 8 models; the exception is Qwen2.5-3B (5.3% against 6.7%).
  The margin exceeds 2 points on only 3 of 8, so the defensible claim is that the
  switch rarely hurts and sometimes helps a lot.
- **Specificity** (pre-fix grader; these runs saved no responses, and every
  re-scorable run shifted all arms alike). A truthful-paraphrase mask shares 91% of the real mask's weights
  at the same median magnitude and does nothing (mean change +0.1 points over five
  models), and so does the magnitude-matched random mask (+0.3). The falsehood masks
  move held-out lying by −18 to −22 points on the same models.
- **Dose-response at matched damage** (pre-fix grader, as above). At full ablation the method and its random
  control land at the same perplexity and neutral accuracy, and the method removes
  50, 29 and 8 more points of lying on Llama-8B, Qwen-7B and Qwen-1.5B.
- **Stronger pressure without the word "lie".** Unsteered, asking for a wrong
  answer in a story or as a forgetful person elicits falsehood more often than the
  lie-shaped prompts on average, on 6 of 7 models, by up to 25 points on Llama-8B.

## 5 Where it does not work

- **Scenario pressure.** LIARS' BENCH harm-pressure splits never instruct a lie;
  they set up a situation where honesty is costly. No mask moves them (Llama-8B
  forced choice 4.4% unsteered, 5.4% under the mask). The method suppresses
  *instructed* falsehood only.
- **Amplification.** Scaling the mask up (α = +3) adds 12 points of lying on
  Llama-8B while costing 7 points of neutral accuracy. It is not a useful deception
  inducer.
- **Weak or fragile models.** On OLMo-2-1B the best usable mask removes about 4
  points of held-out lying (36.6% to 32.3%). On gemma-3-4b it removes about 15
  (40.7% to 25.4%), but every stronger setting breaks the model, and perplexity
  alone fails to show it (1.12× with neutral accuracy down to 44%).
- **The second-difference estimator (proposal Part 1).** Inside a band of equally
  small weights every contrast is flat, including the working single difference
  (Llama-8B: 69.4% against 69.5% unsteered). Without the band, the second
  differences select weights 2.5× the median and destroy the model (1,900× to
  25,000× perplexity). Dropped.

## 6 Evaluation defects found and fixed

1. The instructed-deception evaluation scored the honesty-instructed half of the
   split; every earlier "instructed deception does not move" result measured the
   wrong half.
2. The belief cache was never extended to new items, so the corrected split first
   graded zero items.
3. Numeric answers were discarded as junk (17 of 100 questions), unevenly across
   arms.
4. Numeric correctness was a substring test (gold 360 matched the lie "3600").
5. The held-out average silently changed from 3 to 5 framings between runs.
6. The benchmark's keyword parser counted only replies containing "true"/"false".
   The baselines mostly answer honestly in free text, so it inflated the mask's lead
   on Llama-8B from ~10 points to 41. Replaced by the hybrid grader. The judge
   (Qwen2.5-7B, AGREES/DISAGREES prompt with counterfactual few-shots) agrees with
   the parser on 95.7–95.8% of settled replies on all three judged models; its
   residual errors lean toward crediting free-text answers as honest, which favours
   the baselines.

Runs that saved responses were re-scored offline; the rest carry a caveat in their
reports. No comparison reversed.

## 7 Caveats

- One seed per configuration.
- The in-house held-out framings are three hand-written paraphrases.
- The magnitude-matched control is matched by decile and sits ~2.3× the real mask's
  median magnitude; the truthful-paraphrase mask is the tighter null.
- No fine-tuning baseline (LoRA or DPO) yet.

## 8 In progress

- LLM judging of the external benchmark for gemma-3-4b, Qwen2.5-3B, Falcon3-3B,
  OLMo-2-1B and Qwen2.5-1.5B, and Llama-3.1-8B cross-judging the Qwen runs.
- Corrected external benchmark for OLMo-2-1B and Qwen2.5-1.5B.

Tables regenerate with `reports/summarize_benchmark_judged.py`,
`reports/rescore_pressure_metric.py` and `reports/summarize_instructed_falsehood.py`.
Detailed reports: `head_to_head_llama8b_report.md`, `instructed_falsehood_report.md`,
`placebo_contrast_report.md`, `dose_response_report.md`.
