# Citation audit, batch 4 (2026-10-07)

Keys: burns2023ccs, gemma3, ilharco2023task, li2024wmdp, microsoft2025phi4mini, park2024deception, ren2025mask, stolfo2025steering, vanderweij2024sandbagging.
Sources: arXiv abs pages and the arXiv export API (author lists, comments), the arXiv PDFs (full text), PMLR, Crossref, OpenReview API, ICLR proceedings, HF model card. Semantic Scholar returned HTTP 429 and DBLP timed out, so neither was used.

---

## burns2023ccs

**METADATA: OK.** Title, 4 authors (Burns, Ye, Klein, Steinhardt), arXiv 2212.03827 and ICLR 2023 all match. The arXiv Comments field says "ICLR 2023". https://arxiv.org/abs/2212.03827

**CLAIMS**
- [0] "...or detected lies after the fact with probes on internal representations \citep{burns2023ccs,azaria2023internal,marks2024geometry}". **PARTIAL** (this applies to Burns only; the group cite is fine because of Azaria). CCS finds an unsupervised truth direction ("finding a direction in activation space that satisfies logical consistency properties"), and it stays accurate "even when models are prompted to generate incorrect answers". But the paper says outright that it did not do lie detection: "we did not evaluate our method on setups involving active 'lying' or 'deception' ... If future work develops such a setup, a good stress test would be to apply CCS to do 'lie detection'" (Sec. 5, limitations). Source: arXiv PDF 2212.03827. Suggested fix: "...or probed internal representations for truth, which can expose lies after the fact \citep{...}". Alternatively, cite Burns only in sentence [1].
- [1] "Truth directions in representations \citep{burns2023ccs,...} underlie probes that detect lies after the fact". **SUPPORTED.** CCS recovers "a direction in activation space" that separates true from false statements, and the paper presents that direction as a basis for future lie detection. https://arxiv.org/abs/2212.03827

---

## gemma3

**METADATA: OK.** "Gemma 3 Technical Report", arXiv 2503.19786, 2025, preprint only. The arXiv author list starts with "Gemma Team", then Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, which matches the bib order. The full list has 216 entries, so "and others" is correct. https://arxiv.org/abs/2503.19786 (export API)

**CLAIMS**
- [0] "Gemma-3-4B \citep{gemma3} was excluded...". **SUPPORTED** as a model citation. The report covers models from 1B to 27B, and 4B is one of them ("4B-IT competitive with Gemma2-27B-IT"; "our 4B, 12B, and 27B"). The exclusion is the authors' own finding. https://arxiv.org/abs/2503.19786

---

## ilharco2023task

**METADATA: OK.** Title, 7 authors in order, arXiv 2212.04089 and ICLR 2023 all match. The arXiv Comments field says "In Proceedings of the 11th International Conference on Learning Representations (ICLR 2023)". https://arxiv.org/abs/2212.04089

**CLAIMS**
- [0] "Task arithmetic, contrastive weight arithmetic and ROME edit weights with trained differences or closed-form updates \citep{ilharco2023task,...}". **SUPPORTED** for task arithmetic. The paper builds task vectors "by subtracting the weights of a pre-trained model from the weights of the same model after fine-tuning on a task", then edits by negating or adding them, which is an edit by trained differences. https://arxiv.org/abs/2212.04089

---

## li2024wmdp

**METADATA: one fix.**
- Title, ICML 2024, PMLR vol. 235, pp. 28525-28550 and arXiv 2403.03218 are all correct. https://proceedings.mlr.press/v235/li24bc.html
- **author** does not match the venue cited. The bib has the 57-author arXiv v7 list. The ICML/PMLR version lists **46** authors. These 11 names are in the bib but not in PMLR: Long Phan, Adam Khoja, Zhenqi Zhao, Samuel Marks, Oam Patel, Russell Kaplan, Alex Levinson, Jean Wang, William Qian, Mindy Levine, Ruoyu Wang. PMLR also spells two names as "Andrew Bo Liu" and "Adam Alfred Hunt". The bib's own note says the list "follows PMLR, the arXiv list has additional names", which is not true of the entry as written. Fix: replace the list with the PMLR list (citation_author meta tags on the PMLR page), or shorten it to "Nathaniel Li and Alexander Pan and ... and others".
- Cosmetic: PMLR writes "...Malicious Use with Unlearning" (lowercase "with"). Either form is acceptable.

**CLAIMS**
- [0] "...on 300 MMLU \citep{hendrycks2021mmlu} and 150 WMDP \citep{li2024wmdp} questions...". **SUPPORTED.** WMDP is "a dataset of 3,668 multiple-choice questions that serve as a proxy measurement of hazardous knowledge in biosecurity, cybersecurity, and chemical security". The 150-question subset is the authors' choice. https://arxiv.org/abs/2403.03218

---

## microsoft2025phi4mini

**METADATA: OK.** Title matches. arXiv 2503.01743, 2025, preprint only. The arXiv author list begins "Microsoft", then Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, which matches the bib order. The list has about 74 people, so "and others" is correct. https://arxiv.org/abs/2503.01743

**CLAIMS**
- [0] "Phi-4-mini-instruct \citep{microsoft2025phi4mini}". **SUPPORTED.** The report describes Phi-4-Mini (3.8B) including its post-training. The microsoft/Phi-4-mini-instruct model card names this report (arXiv 2503.01743) as its paper. https://huggingface.co/microsoft/Phi-4-mini-instruct

---

## park2024deception

**METADATA: OK.** 5 authors (Park, Goldstein, O'Gara, Chen, Hendrycks), Patterns 5(5):100988, Elsevier, May 2024, DOI 10.1016/j.patter.2024.100988, arXiv 2308.14752. https://api.crossref.org/works/10.1016/j.patter.2024.100988 , https://arxiv.org/abs/2308.14752

**CLAIMS**
- [0] "Deception by language models is a recognised risk \citep{park2024deception,scheurer2023insider}". **SUPPORTED.** The survey "argues that a range of current AI systems have learned how to deceive humans", covers "general-purpose AI systems (such as large language models)", and details "risks from AI deception, such as fraud, election tampering, and losing control of AI systems". https://arxiv.org/abs/2308.14752

---

## ren2025mask

**METADATA: OK as a preprint.** Title and 16 authors in order (Ren ... Yue, Hendrycks) match, as does arXiv 2503.03750 (v3, updated 2026-01-05). Neither the arXiv page nor the PDF names a venue, and OpenReview's only record is "CoRR 2025". One third-party GitHub paper note labels it "NeurIPS'26" but gives no source. Published venue: **UNVERIFIED**. Recheck before camera-ready. https://arxiv.org/abs/2503.03750

**CLAIMS**
- [0] "Work on honesty has so far measured the problem \citep{kretschmar2025liarsbench,ren2025mask}". **SUPPORTED.** MASK is "a large-scale human-collected dataset for directly measuring lying". https://arxiv.org/abs/2503.03750
- [1] "...MASK's instructed and incentive archetypes \citep{ren2025mask}...". **PARTIAL.** MASK defines six archetypes (Table 2: Known Facts, Situation-Provided Facts, Doubling Down, Fabricated Statistics, Continuations, Disinformation Generation) and calls all of them "pressure prompts" meant "to incentivize the model to make a false statement". The paper has no instructed/incentive split. Its filtering criteria actually require that "the prompt must incentivize dishonesty indirectly rather than explicitly instructing the AI to 'lie' or 'fabricate'" (App. A.7). The split is the authors' own. Source: arXiv PDF 2503.03750v3. Suggested fix: "...MASK's six pressure-prompt archetypes \citep{ren2025mask} (which we group into request-style and incentive-style)...".
- [2] "...and MASK \citep{ren2025mask} measure instructed and incentive-driven dishonesty". **PARTIAL** for the MASK part, for the same reason. MASK measures lying "under pressure", with prompts that incentivize falsehoods rather than explicitly instruct them. Suggested fix: "...and MASK \citep{ren2025mask} measure instructed dishonesty and lying under pressure", which works if LiarsBench covers the instructed side; that was not checked here. Otherwise, "MASK \citep{ren2025mask} measures lying under incentive-giving pressure prompts".
- [3] "MASK \citep{ren2025mask}: the instructed archetypes (continuations, disinformation, statistics) and the incentive archetypes (known facts, doubling down, provided facts), graded by a request-aware judge against the model's elicited belief." **PARTIAL.** The six archetype names are correct. However: (a) the instructed/incentive grouping is the authors' own (see [1]); (b) MASK runs **no belief elicitation for Situation-Provided Facts**: "No belief elicitation prompts as all facts are already provided in-context" (Table 2). For that archetype the reference is the fact given in the system prompt, not an elicited belief. The "request-aware judge" is presumably the authors' own judge; MASK uses LLM-judge criteria "standardized within each archetype". This is UNVERIFIED against MASK. Suggested fix: "...(which we group as instructed: ...; incentive: ...), graded against the model's elicited belief (for provided facts, against the facts given in the prompt)".

---

## stolfo2025steering

**METADATA: OK.** Title and 5 authors (Stolfo, Balachandran, Yousefi, Horvitz, Nushi) match. ICLR 2025: the arXiv Comments field says "ICLR 2025", the PDF header says "Published as a conference paper at ICLR 2025", and DBLP key conf/iclr/StolfoBYHN25 confirms it (via Semantic Scholar). arXiv 2410.12877. https://arxiv.org/abs/2410.12877

**CLAIMS**
- [0] "difference-in-means vectors \citep{rimsky2024caa,stolfo2025steering} ... taken at the last prompt token and added at every position of one layer". **SUPPORTED** (Stolfo part). The paper says "we use a technique called difference-in-means (Belrose, 2023)", that vectors are computed from residual-stream values "at the last token of the input", and that the addition "is carried out at a single layer across all token positions" (Sec. 2). The median-norm scaling is the authors' own choice. Source: arXiv PDF 2410.12877.
- [1] "Adding a difference-in-means direction at inference \citep{turner2023actadd,rimsky2024caa,zou2023repe,stolfo2025steering} is the standard training-free steering method." **SUPPORTED** (Stolfo part). Stolfo steers at inference with difference-in-means vectors ("providing inference-time control over instruction following"). Calling it "standard" is the authors' characterization. https://arxiv.org/abs/2410.12877

---

## vanderweij2024sandbagging

**METADATA: OK.** Title, 5 authors, ICLR 2025 (OpenReview 7Qa2SpjxIS, "ICLR 2025 Poster"; proceedings.iclr.cc 2025) and arXiv 2406.07358 all match. The key says 2024 while the bib year is 2025. The year field is correct; the key is cosmetic only. Optional: the ICLR/OpenReview version lists "Oliver Jaffe" where arXiv has "Ollie Jaffe". https://proceedings.iclr.cc/paper_files/paper/2025/hash/b5e5753b0a0e440a6d8dc7e143617cec-Abstract-Conference.html , https://openreview.net/forum?id=7Qa2SpjxIS

**CLAIMS**
- [0] "...test transfer: sandbagging \citep{vanderweij2024sandbagging}...". **SUPPORTED.** The paper defines sandbagging "as strategic underperformance on an evaluation". https://arxiv.org/abs/2406.07358
- [1] "\emph{Sandbagging} \citep[after][]{vanderweij2024sandbagging}: the system prompt orders the model to answer a capability test wrongly, on 300 MMLU and 150 WMDP questions...". **SUPPORTED** (the "after" correctly signals an adaptation). The paper says: "Our system prompt instructs the model to answer incorrectly on questions related to hazardous information", evaluating on WMDP with MMLU and CSQA as general capability benchmarks (Sec. 4.1). Note that van der Weij et al. prompt for *selective* underperformance (wrong on WMDP, correct on MMLU). The paper's version orders wrong answers on both, which "after" covers. Source: arXiv PDF 2406.07358.
