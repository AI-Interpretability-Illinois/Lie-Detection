# Format-transfer study: weight mask vs CAA, 7 x 7, four models

## Update 2026-10-02: Llama-3.1-8B, Qwen2.5-7B, Falcon3-3B (gpua047, owner OK) and the Phi test half

Same design as the Phi pilot below: calibrated mask (maskc) and CAA at each model's main-table layer
(Llama L8, Qwen-7B L14, Falcon L5, Phi L16), four strengths each, validation half, greedy decoding.
Paragraphs are graded by the local judge. Gap closure is now computed only for eval formats where the
unedited model is at least 10 points below its honest-instructed control. Smaller gaps made the ratio
explode (Qwen inject: gap 3.4 points, -370% to +130%). Formats with room: Phi user/mc/tf/multi/long;
Llama and Qwen all but inject; Falcon only mc and multi (it already resists most single-turn orders).

Mean % of the gap closed at one strength per build format (validation; `reports/fmt_crossmodel.py`):
```
model                       method  single-turn off-diag  diagonal multi-turn  worst build row           ppl   eval formats with room
Phi-4-mini-instruct         maskc                   86.3      96.1       -7.3         multi 52     1.03-1.09   user,mc,tf,multi,long
Phi-4-mini-instruct         caa                     81.3      94.6       16.5         multi 54     1.07-1.14   user,mc,tf,multi,long
Meta-Llama-3p1-8B-Instruct  maskc                   97.5      98.9       95.1        inject 95     1.02-1.02   sys,user,mc,tf,multi,long
Meta-Llama-3p1-8B-Instruct  caa                     47.9      41.5       67.4            tf 24     1.01-1.07   sys,user,mc,tf,multi,long
Qwen2p5-7B-Instruct         maskc                   96.8      96.1       77.7         multi 90     1.03-1.08   sys,user,mc,tf,multi,long
Qwen2p5-7B-Instruct         caa                     88.5      83.1       62.6         multi 75     1.06-1.15   sys,user,mc,tf,multi,long
Falcon3-3B-Instruct         maskc                   94.6      94.9      -11.9         multi 31     1.01-1.04   mc,multi
Falcon3-3B-Instruct         caa                     43.8      82.8      -12.9        inject -4     1.00-1.06   mc,multi
```

1. **The calibrated mask transfers across single-turn formats on every model: 86-98% of the gap,
   built in any other format.** CAA is as good on Phi and Qwen (81-89%) but far weaker on Llama
   (48%) and Falcon (44%). CAA also depends on the build format on Llama: vectors built from mc / tf /
   multi pairs close 24-32% on average, vectors from long / inject / user pairs 65-70%. The mask rows are flat
   (95-98%).
2. **Damage is lower for the mask** at the chosen strengths: perplexity 1.01-1.09 against 1.00-1.15.
3. **Multi-turn splits by model, not by method.** The mask closes 95% (Llama) and 78% (Qwen) but -7% (Phi)
   and -12% (Falcon). CAA closes 67 / 63 / 17 / -13%. The models where the mask fails on multi-turn are
   the two small ones, and CAA does no better on Falcon.
4. **The multi-turn build format is usually the weakest source.** It is the worst row for both methods
   on Phi and Qwen (Qwen mask: tied with tf, 90.3 vs 90.4) and for the mask on Falcon. On Llama it is
   third-worst for both (the mask rows are flat there). The exception is Falcon CAA, where it is the
   best row (42% vs -4 to 40%).
5. **The Phi test half confirms validation** at the validation-chosen strengths (`fmt_pilot_phi_test_tables.txt`).
   Mask column means on test: user 92, mc 95, tf 86, multi -5.5, long 82 (validation 88 / 97 / 80 /
   -7 / 85). CAA: 70 / 95 / 59 / 12 / 73 (validation 79 / 96 / 76 / 17 / 82). On test the mask beats
   CAA in every column but multi-turn; CAA loses more from validation to test.

Cost of the extension: 2 x 1 A100 on gpua047, 4.1 h and 5.1 h (including grading 80k paragraphs),
plus 1.0 h for the Phi test half. No API cost. Per-model tables: `reports/fmt_<model>_val_tables.txt`.

---

## Pilot: Phi-4-mini, weight mask vs CAA, 7 x 7 (validation half; tables at the earlier 3-point gap threshold)

**Question** (owner idea, 2026-09-30). Build a steering edit from contrast pairs in format i and evaluate
it on format j, for every (i, j). Is there a fundamental difference between weight steering and
activation steering in this matrix? Which formats generalise?

**Data** (Phase 0, `build_format_suite.py`, $3.91 of Azure gpt-5.6-sol). 1302 verified facts
rendered into seven formats: sys (lie instruction in the system prompt), user (user turn; the hub,
which differs from each other format in one respect), inject (an instruction hidden in a retrieved
document), mc (multiple choice), tf (one-word true/false), multi (a lie already given in an earlier
turn, then a follow-up) and long (3-4 sentence paragraph). Each format has 520 build pairs (200 used)
and 637 eval facts; validation 305 / test 332, split by fact, so every format uses the same facts.
Build and eval use disjoint instruction wordings, and no eval fact shares its correct answer with a
build fact.

**Methods** (`fmt_transfer.py`), one edit per build format and strength:
- **mask**: the method's rule (prompt-token column norms, |W| x contrast, top 4% per matrix,
  deletion) with the fixed r = 0.37. Built from 200 pairs, it is nearly inert: perplexity is 1.000-1.004
  even at alpha = -1. Few channels clear the 2.7x bar once the noise of 25 prompts is gone, as test (b)
  predicted. These rows are kept in the tables file but not used.
- **maskc**: the same rule with full deletion and r calibrated per build format, so that the
  positive-score weights are 0.25 / 0.5 / 1 / 2% of each matrix (1/r = 1.24-1.77). This is the mask
  compared below.
- **caa**: unit mean difference of last-token activations at layer 16 (Phi's chosen layer), added
  at every position with c = r x median residual norm, r in {-0.09, -0.18, -0.27, -0.36}.

**Evaluation.** Greedy decoding; an item counts only where the unedited model answers the plain
question correctly (272-296 items per format). Grading: short answers by keyword with aliases, mc
and tf by parsing, paragraphs by the local Qwen2.5-7B judge. On 400 paragraph verdicts that judge agrees
with gpt-5.6-sol 95.8% of the time ($0.34). All 17 disagreements are self-contradicting replies such as
"The Moonlight Sonata was not composed by Beethoven ... composed by Ludwig van Beethoven", which the
local judge counts as lies (kept: they do assert a falsehood). Damage: wikitext perplexity ratio and
the honesty-instructed control items of every format. Budget: ppl <= 1.15, control drop <= 5 points.
One strength per build format: the best mean gain over eval formats within the budget.

Unedited honest % on deception items (control in brackets): sys 89.5 (98.9), user 62.5 (98.9),
inject 89.8 (96.5), mc 56.1 (100), tf 72.8 (97.4), multi 67.7 (99.3), long 66.0 (96.1).

### Share of the gap to the honest-instructed level closed (%), one strength per build format

Calibrated mask (perplexity 1.02-1.09 at the chosen strengths; mean control drop 0.1-2.5):
```
  build \ eval      sys    user  inject      mc      tf   multi    long  row mean
  sys              85.2    92.3    89.5    97.7    82.1   -18.9    97.7      75.1
  user             74.1    87.5    78.9    99.2    82.1    -4.4    93.0      72.9
  inject           74.1    92.3    63.2    95.4    55.2    32.2   104.7      73.9
  mc               85.2    88.5    89.5    99.2    97.0    -8.9    84.9      76.5
  tf               81.5    83.7    52.6    96.9   100.0    -7.8    60.5      66.8
  multi            59.3    87.5    63.2    93.8    47.8   -27.8    58.1      54.6
  long             85.2    86.5    84.2    99.2    95.5   -15.6    97.7      76.1
  col mean         77.8    88.3    74.4    97.4    80.0    -7.3    85.2
```
CAA (perplexity 1.07-1.14; mean control drop 1.0-2.5):
```
  build \ eval      sys    user  inject      mc      tf   multi    long  row mean
  sys              70.4    89.4    73.7    98.5    64.2    52.2    98.8      78.2
  user             77.8    90.4    84.2    97.7    59.7    17.8    94.2      74.5
  inject           81.5    77.9    57.9    95.4    29.9    36.7    80.2      65.6
  mc               59.3    71.2    42.1    98.5    92.5     1.1    75.6      62.9
  tf               51.9    65.4    68.4    96.9   104.5    27.8    73.3      69.7
  multi            51.9    73.1    47.4    91.5    91.0   -55.6    67.4      52.4
  long             74.1    82.7    78.9    96.2    89.6    35.6    84.9      77.4
  col mean         66.7    78.6    64.7    96.4    75.9    16.5    82.1
```

### Reading

1. **Both methods transfer broadly across single-turn formats.** Of the 36 off-diagonal single-turn
   cells, 86% (mask) and 81% (CAA) close at least 60% of the gap; the weakest close 48% and 30%. Rows
   differ less than columns: mc is nearly solved by any edit, while sys and inject start near 90% honest and have
   little room. As Nguyen et al. (2026) report for CoT faithfulness, the evaluation setting matters
   more than the build setting.
2. **The calibrated mask matches or beats CAA on single-turn formats at a third of the damage.** Row
   means 55-77% vs 52-78%, at perplexity 1.02-1.04 for most rows vs 1.07-1.14 for CAA.
3. **Multi-turn is where the two differ, and where both are weak.** Masks do not help with a lie that
   is already under way (column mean -7%); only the mask built from injected documents helps (+32%).
   CAA does better (column mean +17%, +52% from the sys vector). This matches the main study, where the
   mask failed on multi-turn.
4. **The multi-turn build format is the worst source for both methods.** On multi-turn itself, its
   CAA vector closes -56% (it makes persisting in the lie more likely) and its mask -28%. Its CAA
   direction is also the least aligned with the other six (cosine 0.44-0.58, vs 0.6-0.9 among
   the others). Its pairs differ in the prefilled answer as well as in the instruction, so the
   contrast picks up "a false answer is already in the context" rather than "lie now".
5. **True/false needs a polarity split.** Under a lie instruction Phi already answers "False" to every
   false statement; it lies only on true statements (47% honest). Masks overcorrect: true statements
   reach 77-100%, but false statements drop to 72-97% (the inject-built mask is worst). CAA raises true statements to 61-99% and leaves
   false ones at 90-100%. The fixed-r masks pushed the "False" habit further (true statements 21%).

**Caveats.** Validation half only; the strength per build format was also chosen on it. One model.
Test-half confirmation and more models (Llama-3.1-8B, Qwen2.5-7B, Falcon3-3B) are planned.

**Cost.** About 6.5 A6000 GPU-hours on timan1 (generation 4.6 h, judge 1.9 h); API $4.25 in total
(Phase 0 $3.91, judge check $0.34). Tables: `reports/fmt_pilot_phi_val_tables.txt`.
