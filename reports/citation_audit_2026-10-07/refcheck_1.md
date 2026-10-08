# Citation audit, batch 1 (checked 2026-10-07)

Keys: arditi2024refusal, falcon3, grattafiori2024llama3, kretschmar2025liarsbench, marks2024geometry, nguyen2026cot, pinto2026agents, rimsky2024caa, sun2023wanda, wei2024brittleness.
Sources were fetched live: arXiv abs and HTML pages, ACL Anthology, PMLR, the NeurIPS virtual site and proceedings, the HF blog and model card. OpenReview, Semantic Scholar and dblp blocked automated access (403 challenge / 429).

---

## arditi2024refusal

**METADATA: OK.**
- The title, all 7 authors in order, arXiv 2406.11717 and cs.LG match https://arxiv.org/abs/2406.11717.
- NeurIPS 2024 is confirmed by https://neurips.cc/virtual/2024/poster/93566 and by the proceedings PDF at https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf.
- Optional: add `volume = {37}`.

**CLAIMS**
- [0] "\citet{arditi2024refusal} orthogonalize weights against a refusal direction": **SUPPORTED**. Sec. "Weight orthogonalization": "orthogonalize its column vectors: W_out' <- W_out - r r^T W_out", applied to the embedding, positional-embedding, attention-out and MLP-out matrices. https://arxiv.org/html/2406.11717
- [1] "Projecting the format's CAA direction out of every layer's output \citep{arditi2024refusal}": **SUPPORTED**. Directional ablation "zeroes out the component along r for every residual stream activation ... across all layers l and all token positions i." Their direction is also a difference in means (harmful minus harmless), so it is close to a CAA direction. https://arxiv.org/html/2406.11717

## falcon3

**METADATA: OK.**
- The bib follows the BibTeX block that the release post and the model card give: title "The Falcon 3 Family of Open Models", author "Falcon-LLM Team", December 2024. Sources: https://huggingface.co/blog/falcon3 and https://huggingface.co/tiiuae/Falcon3-3B-Instruct
- The post's displayed title is "Welcome to the Falcon 3 Family of Open Models!" (published 17 Dec 2024). Keeping the self-supplied BibTeX title is fine.
- No technical report exists: the model card says "Technical Report: Coming soon", and a web search found no arXiv report.

**CLAIMS**
- [0] Falcon3-3B-Instruct \citep{falcon3}: **SUPPORTED**. The post lists "Falcon3-1B-Instruct and Falcon3-3B-Instruct" among the released models. https://huggingface.co/blog/falcon3

## grattafiori2024llama3

**METADATA: OK.**
- The title, the first 10 authors in order (Grattafiori, Dubey, Jauhri, Pandey, Kadian, Al-Dahle, Letman, Mathur, Schelten, Vaughan), 2024, arXiv 2407.21783 and cs.AI all match. arXiv shows no collective author. https://arxiv.org/abs/2407.21783
- Only a preprint exists, so the arXiv citation is correct.

**CLAIMS**
- [0] Llama-3.1-8B-Instruct \citep{grattafiori2024llama3}: **SUPPORTED**. Table 1 lists "Llama 3.1 8B Instruct ... July 2024" and says "All results in this paper are for the Llama 3.1 models." https://arxiv.org/html/2407.21783v3

## kretschmar2025liarsbench

**METADATA: OK as a preprint.**
- The title, all 4 authors in order (Kretschmar, Laurito, Maiya, Marks), arXiv 2511.16035 (v1 20 Nov 2025, v2 8 Jan 2026) and cs.CL match. arXiv lists no venue. https://arxiv.org/abs/2511.16035
- **UNVERIFIED venue.** An OpenReview submission exists under a different title, "Liars' Bench: Evaluating Deception Detectors for AI Assistants" (https://openreview.net/forum?id=ZsGQLxOpjt), apparently an ICLR 2026 submission. Its decision could not be checked because OpenReview blocked automated access.
- Check that page by hand. If the paper was accepted, cite the published version and title.

**CLAIMS**
- [0] "Told in a system prompt to deceive the user, many give confident false answers to questions they answer correctly when asked plainly": **SUPPORTED** (minor wording point).
  - Evidence: "First, we generated responses using four neutral system prompts and dismissed questions for a model unless it answered correctly every time ... Then we generated responses with ... 'Answer deceptively to the following question', and labeled incorrect responses as lies."
  - The Instructed-Deception set has 10,692 lies across 4 models (Table 1). https://arxiv.org/html/2511.16035v2
  - The paper does not measure "confident". Optional fix: drop that word.
- [1] "Work on honesty has so far measured the problem \citep{kretschmar2025liarsbench,ren2025mask} or detected lies after the fact with probes": **PARTIAL**.
  - Liars' Bench is a testbed for *evaluating lie detectors*: "a testbed ... for guiding progress in lie detection." It also evaluates probes. It does not measure how prevalent dishonesty is.
  - Suggested fix: "Work on honesty has so far measured the problem \citep{ren2025mask}, built testbeds of lies for evaluating detectors \citep{kretschmar2025liarsbench}, or detected lies after the fact with probes ...".
- [2] "537 items whose system prompt orders the model, in varied wordings, to deceive the user about a factual statement, and 263 whose system prompt orders honesty, which serve as a control": **SUPPORTED** for the description. The counts are **UNVERIFIED**.
  - Appendix H: "2 neutral system prompts encouraging honesty and 4 system prompts encouraging deception ... combine them with 2 templates for user prompts, leading to a total of 4 neutral templates and 8 templates encouraging lies ... combined with 100 statements from each of the subsets animals, cities, companies, elements, facts, and inventions of [Azaria & Mitchell]."
  - The 537/263 counts are the paper's own subsample and do not appear in Liars' Bench. The HF dataset is gated, so they could not be checked. The roughly 2:1 ratio matches the 8:4 template ratio. https://arxiv.org/html/2511.16035v2
- [3] "\LB{} \citep{kretschmar2025liarsbench} and MASK \citep{ren2025mask} measure instructed and incentive-driven dishonesty": **SUPPORTED** (loose).
  - Liars' Bench's "reason for lying" axis is "encouraged ... by direct instruction to lie, to role-play deceptive personas, or to pursue some goal" vs "inherent". Instructed-Deception and Insider-Trading are "encouraged".
  - Optional precision: "\LB{} collects instructed and pressure-induced lies to evaluate lie detectors, and MASK measures ...".

## marks2024geometry

**METADATA: OK.**
- The title, both authors, 2024, arXiv 2310.06824 and cs.AI match. The arXiv Comments field says "Conference on Language Modeling, 2024". https://arxiv.org/abs/2310.06824
- The OpenReview PDF is headed "Published as a conference paper at COLM 2024" (openreview.net, id aajyHYjjsk).

**CLAIMS**
- [0] "detected lies after the fact with probes on internal representations \citep{burns2023ccs,azaria2023internal,marks2024geometry}": **PARTIAL** (minor).
  - Marks & Tegmark probe whether the model represents a *given statement* as true or false. The goal is "determine whether M believes s to be true", and they find "simple difference-in-mean probes generalize". They do not detect a model's own lies. https://arxiv.org/html/2310.06824v3
  - Suggested fix: "or probed internal representations for truth, the basis of after-the-fact lie detectors \citep{...}". Sentence [3] already phrases it this way correctly.
- [1] "true/false statements, 12 true and 12 false, half of them negated \citep[cf.][]{marks2024geometry}": **SUPPORTED**. Their datasets include neg_cities and neg_sp_en_trans ("Negations of statements in cities with 'not'"), and they introduce "structural diversity by negating statements". https://arxiv.org/html/2310.06824v3
- [2] "new true/false statements \citep[AM/GoT;][]{azaria2023internal,marks2024geometry}": **SUPPORTED**. The source is the GoT true/false datasets (Table 1). https://arxiv.org/html/2310.06824v3
- [3] "Truth directions in representations \citep{...,marks2024geometry} underlie probes that detect lies after the fact": **SUPPORTED**. "LLMs linearly represent the truth or falsehood of factual statements." https://arxiv.org/abs/2310.06824
- [4] "\LB{}'s order on 1,100 new true/false statements from \citet{azaria2023internal} and \citet{marks2024geometry}": **SUPPORTED** for the source attribution. The 1,100 count is the paper's own and cannot be checked against the source.
  - Note: Liars' Bench Instructed-Deception uses only Azaria & Mitchell statements. Combining them with GoT statements is this paper's extension, and the sentence words it correctly.

## nguyen2026cot

**METADATA: OK.**
- The title, all 4 authors in order (Nguyen, Cox, Meek, Arcuschin), arXiv 2607.29062 (v1 31 Jul 2026) and cs.AI match. Only a preprint exists. https://arxiv.org/abs/2607.29062

**CLAIMS**
- [0] "its effect depends on the prompts that built the vector and on the setting in which it is applied \citep{tan2024generalisation,braun2025unreliability,nguyen2026cot}": **PARTIAL**.
  - Nguyen et al. support only the second half. Their finding goes *against* the first half: "effect size is determined primarily by the evaluation setting, rather than the vector's train setting. How the vector is built also matters little." https://arxiv.org/abs/2607.29062
  - Suggested fix: move the citation so that it reads "...depends on the prompts that built the vector \citep{tan2024generalisation,braun2025unreliability} and on the setting in which it is applied \citep{nguyen2026cot}".
- [1] "The evaluation format still shapes the outcome for both methods, as \citet{nguyen2026cot} report for activation steering": **SUPPORTED** (narrower setting). Their "evaluation setting" means the dataset (GPQA/BBH/MMLU) and the cue type in CoT-faithfulness steering. Steering worked reliably only for Gemma-3 12B. https://arxiv.org/html/2607.29062
  - Optional: "as \citet{nguyen2026cot} report for steering CoT faithfulness, where the evaluation setting rather than the build setting sets the effect size".
- [2] "varies with the evaluation setting \citep{nguyen2026cot,pinto2026agents}": **SUPPORTED**. Same evidence as [1].

## pinto2026agents

**METADATA: OK.**
- The title matches v3 (29 Aug 2026). The single author (Lucas Pinto), arXiv 2607.09156 and cs.LG match. Only a preprint exists. https://arxiv.org/abs/2607.09156
- v1 (10 Jul 2026) had a different title and results, and v3 adds "an errata section correcting v1's public record". Consider citing v3 explicitly, e.g. `note = {arXiv:2607.09156v3}`.

**CLAIMS**
- [0] "varies with the evaluation setting \citep{nguyen2026cot,pinto2026agents}": **SUPPORTED**.
  - Evidence: "A coefficient calibrated in chat does not carry a calibrated meaning into an agent scaffold: the vector still works, but the same behavioural fraction now sits at a different dose." The dEC50 shift is signed both ways across model families.
  - Models: Llama-3.1-8B, OLMo-2-7B, Qwen2.5-3B/7B, Gemma-3-12B, Starling-7B. https://arxiv.org/html/2607.09156v3
  - Note: the variation they show is between chat and agent deployment contexts.

## rimsky2024caa

**METADATA: minor author-name inconsistency.**
- The title, 6 authors, ACL 2024 Long, pages 15504--15522, DOI 10.18653/v1/2024.acl-long.828, Bangkok, August and arXiv 2312.06681 (cs.CL) are all correct.
- Author names: the ACL Anthology lists **"Nina Rimsky" and "Alexander Turner"** (https://aclanthology.org/2024.acl-long.828/). arXiv v4 lists "Nina Panickssery" and "Alexander Matt Turner" (https://arxiv.org/abs/2312.06681). The bib uses the arXiv names, but its `note` says the names are "as on ACL Anthology", which is false.
- Fix one of two ways:
  - (a) Use the Anthology names, matching the published version you cite. It then renders as "Rimsky et al., 2024".
  - (b) Keep Panickssery, the author's current name, also used in arditi2024refusal, and correct the note.

**CLAIMS**
- [0] "Activation steering \citep{turner2023actadd,rimsky2024caa} ... adds a vector at inference time": **SUPPORTED**. "During inference, these steering vectors are added at all token positions after the user's prompt." https://arxiv.org/abs/2312.06681
- [1] "difference-in-means vectors \citep{rimsky2024caa,stolfo2025steering} ... taken at the last prompt token and added at every position of one layer": **SUPPORTED** for the cited part, "difference-in-means vectors". "We then take the mean activation difference over many contrast pairs."
  - Your extraction and injection details differ from CAA. CAA takes activations "at the position of the answer letter" of A/B multiple-choice pairs and adds the vector "at all token positions after the user's prompt" at one layer. https://arxiv.org/html/2312.06681v4
  - The sentence does not claim to follow CAA exactly, so no change is required. If you want to make it explicit, add "(a variant of CAA)".
- [2] "a contrastive-activation-addition vector \citep[CAA;][]{rimsky2024caa} at each model's main-table layer": **SUPPORTED**. CAA steers at a single residual-stream layer.
- [3] "Adding a difference-in-means direction at inference \citep{turner2023actadd,rimsky2024caa,zou2023repe,stolfo2025steering} is the standard training-free steering method": **SUPPORTED** for CAA ("Mean Difference (MD)" vectors added during forward passes).
  - Side note (turner2023actadd is not one of my keys): ActAdd uses the activation difference of a single prompt pair, not a mean over many pairs.

## sun2023wanda

**METADATA: OK.**
- The title, all 4 authors in order (Sun, Liu, Bair, Kolter), arXiv 2306.11695 and cs.CL match. The arXiv Comments field says "ICLR 2024". https://arxiv.org/abs/2306.11695
- ICLR 2024 is also confirmed by the proceedings PDF (https://proceedings.iclr.cc/paper_files/paper/2024/file/160adf2dc118a920e7858484b92a37d8-Paper-Conference.pdf) and the OpenReview forum PxoFut3dWW.
- The key says 2023 but `year = {2024}`. This is harmless.

**CLAIMS**
- [0] "Wanda scores a weight by its magnitude times the norm of its input channel ... from forward passes alone": **SUPPORTED**. "S_ij = |W_ij| · ||X_j||_2"; "Wanda can be done with a single forward pass", "no gradient computation via back-propagation." https://arxiv.org/html/2306.11695
- [1] "Wanda ranks weights by |W_ij|·||x_j||_2 with the norm taken over calibration tokens": **SUPPORTED**. The norm is taken "across N×L different tokens" of 128 C4 calibration sequences.
  - Optional: Wanda compares weights per output row, not globally.
- [2] "Pruning scores from forward passes \citep{sun2023wanda,lee2019snip,frankle2019lottery}": **SUPPORTED for Wanda.**
  - Side note on keys outside my list: SNIP is gradient-based. Wei et al. define it as |W ⊙ ∇_W L|. Lottery-ticket pruning is iterative magnitude pruning with retraining. "From forward passes" therefore fits only Wanda.
  - Suggested fix: "Pruning importance scores \citep{...}".

## wei2024brittleness

**METADATA: OK.**
- The title, all 9 authors in order, ICML 2024 (PMLR vol. 235, pp. 52588--52610), arXiv 2402.05162 and cs.LG all match. Sources: https://proceedings.mlr.press/v235/wei24f.html and https://arxiv.org/abs/2402.05162

**CLAIMS**
- [0] "\citet{wei2024brittleness} contrast such scores [i.e. Wanda's forward-pass scores] between a safety and a utility dataset and find a few percent of weights whose removal breaks refusal": **PARTIAL**.
  - The safety-vs-utility contrast ("set difference") was run with **SNIP**, a gradient-based score, not with Wanda. They write: "We only use SNIP for set difference because during our early experiments, we find SNIP performs slightly better than Wanda."
  - Wanda appears only in their safety-only "Wanda (top)" baseline.
  - The "few percent" part is right: "removing less than 3% of neurons [= individual weight entries] pushes ASR in all three scenarios close to 1, while maintaining an average zero-shot accuracy above 0.5" (Llama2-7B-chat; similar results for 13B-chat). https://arxiv.org/html/2402.05162v4
  - Suggested fix: "\citet{wei2024brittleness} contrast pruning importance scores (gradient-based SNIP in their main result) between a safety and a utility dataset and find that removing under 3\% of weights breaks refusal".
- [1] "Pruning scores from forward passes ... became a safety probe in \citet{wei2024brittleness}, who contrast importance on a safety set against a utility set and show that removing about 3% of weights removes refusal": **PARTIAL**.
  - The scores contrasted in the main result are SNIP (needs gradients), not forward-only.
  - The source says "less than 3%" and that ASR goes "close to 1" on Llama-2-chat. "About 3%" and "removes refusal" are acceptable approximations.
  - Suggested fix: "Pruning importance scores \citep{sun2023wanda,lee2019snip,frankle2019lottery} became a safety probe in \citet{wei2024brittleness}, who contrast SNIP importance on a safety set against a utility set and show that removing under 3\% of weights nearly eliminates refusal in Llama-2-chat".
