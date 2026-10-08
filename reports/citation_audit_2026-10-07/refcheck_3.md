# Reference check, batch 3 (checked 2026-10-07)

Sources: arXiv export API (title, authors, comments, primary class) and the full-text arXiv PDFs, read locally. Semantic Scholar returned HTTP 429, and DBLP and OpenReview blocked automated access (403 or challenge pages), so venues were checked on the proceedings pages, the PDF headers and ML Anthology instead.

---

## braun2025unreliability

**METADATA: OK.** Title, the 5 authors in order (Braun, Eickhoff, Krueger, Bahrainian, Krasheninnikov), year 2025, arXiv 2505.22637 and cs.LG all match https://arxiv.org/abs/2505.22637. The PDF header reads "Published at ICLR 2025 Workshop on Foundation Models in the Wild", which matches the booktitle.
- FYI: ML Anthology lists the same paper under "ICLR 2025 Workshops: Bi-Align" (OpenReview jf472MdwsP; https://mlanthology.org/iclrw/2025/braun2025iclrw-understanding/). A search hit also shows an OpenReview PDF headed "Building Trust Workshop at ICLR 2025" (JZiKuvIK1t). So the paper was presented at several ICLR 2025 workshops, and the booktitle follows the authors' own arXiv statement. No change is needed.

**CLAIMS**
1. "its effect depends on the prompts that built the vector and on the setting in which it is applied [tan, braun, nguyen]" — **SUPPORTED** (Braun backs the prompt part). The paper "studies the influence of prompt types ... on steering reliability"; all seven prompt types "exhibit high variance across samples, and often give an effect opposite of the desired one", and the "steering vectors resulting from the different prompt types often differ directionally". https://arxiv.org/abs/2505.22637
2. "activation addition, whose vectors are sensitive to the prompts that build them [tan, braun]" — **SUPPORTED**. Same evidence: the vectors built from different prompt types differ in direction and in steering performance. https://arxiv.org/abs/2505.22637
3. "depends on the consistency of the contrast pairs [braun]" — **SUPPORTED**. The paper finds that "higher cosine similarity between training set activation differences predicts more effective steering" ("directional agreement ... is predictive of steering success"). Scope: CAA on Llama-2-7B-Chat over 36 multiple-choice datasets (Discussion, "Limitation: breadth of experiments"). https://arxiv.org/abs/2505.22637

---

## frankle2019lottery

**METADATA: OK.** Title, 2 authors, ICLR 2019 and arXiv 1803.03635 (cs.LG) all check out. The arXiv journal-ref is "ICLR 2019", the comment is "ICLR camera ready", and the PDF header reads "Published as a conference paper at ICLR 2019". https://arxiv.org/abs/1803.03635

**CLAIMS**
1. "Pruning scores from forward passes [sun2023wanda, lee2019snip, frankle2019lottery] became a safety probe in wei2024brittleness ..." — **UNSUPPORTED** for Frankle. The Lottery Ticket paper computes no score from a forward pass. It prunes trained weights by magnitude: "We identify a winning ticket by training a network and pruning its smallest-magnitude weights"; "remove a percentage of the weights with the lowest magnitudes within each layer (as in Han et al. (2015))" (Sec. 1–2). Its contribution is the lottery-ticket hypothesis and rewinding to initialisation, not a pruning criterion. https://arxiv.org/abs/1803.03635 (PDF pp. 2–3)
   - **Fix:** drop frankle2019lottery from this list, or reword to: "Pruning scores, from weight magnitude \citep{frankle2019lottery} to activation- and gradient-based saliency \citep{sun2023wanda,lee2019snip}, became a safety probe in \citet{wei2024brittleness} ...". Not checked here (owned by another batch): SNIP's score uses gradients on a mini-batch, so "forward passes" may also be loose for lee2019snip.

---

## hendrycks2021mmlu

**METADATA: OK.** Title, the 7 authors in order, ICLR 2021 (arXiv comment: "ICLR 2021"), arXiv 2009.03300 and primary class cs.CY all match. https://arxiv.org/abs/2009.03300

**CLAIMS**
1. "Plain MMLU accuracy [hendrycks2021mmlu] measures capability." — **SUPPORTED**. The abstract says: "We propose a new test to measure a text model's multitask accuracy ... 57 tasks ... models must possess extensive world knowledge and problem solving ability." https://arxiv.org/abs/2009.03300
2. "... on 300 MMLU [hendrycks2021mmlu] and 150 WMDP questions ..." — **SUPPORTED**. This cites the benchmark as a data source, and the usage fits its four-option multiple-choice format. https://arxiv.org/abs/2009.03300

---

## lee2024mechanistic

**METADATA: OK.** Title, the 6 authors in order, ICML 2024, PMLR vol. 235, pp. 26361–26378 and arXiv 2401.01967 all match. Checked at https://proceedings.mlr.press/v235/lee24a.html and https://arxiv.org/abs/2401.01967

**CLAIMS**
1. "\citet{lee2024mechanistic} show that alignment leaves the aligned-away capability in place" — **PARTIAL** (over-general). The paper is "a case study" of one algorithm (DPO), one behaviour (toxicity) and one model (GPT2-medium). It finds that "capabilities learned from pre-training are not removed, but rather bypassed" and that "DPO does not remove the capability of generating toxic outputs, but learns an 'offset' ... to 'bypass' the regions that elicit toxicity". https://arxiv.org/abs/2401.01967 ; https://proceedings.mlr.press/v235/lee24a.html
   - **Fix:** "\citet{lee2024mechanistic} show that DPO detoxification of GPT-2 bypasses the toxic capability rather than removing it, as our multi-turn results also suggest." Using \citet is correct here.

---

## merity2017wikitext

**METADATA: OK.** Title, the 4 authors in order, ICLR 2017 and arXiv 1609.07843 (cs.CL) all match. https://arxiv.org/abs/1609.07843 has no venue, but ML Anthology lists ICLR 2017 with OpenReview Byj72udxe (https://mlanthology.org/iclr/2017/merity2017iclr-pointer/).

**CLAIMS**
1. "WikiText-2 perplexity [merity2017wikitext]" — **SUPPORTED**. The paper "introduce[s] the freely available WikiText corpus", and Table 1 lists WikiText-2 (2,088,628 training tokens) next to WikiText-103. https://arxiv.org/abs/1609.07843 (PDF p. 5, Table 1)

---

## pacchiardi2024catch

**METADATA: OK.** Title, the 8 authors in order (Pacchiardi, Chan, Mindermann, Moscovitz, Pan, Gal, Evans, Brauner), ICLR 2024 and arXiv 2309.15840 (cs.CL) all match. The venue is confirmed by the ICLR 2024 poster page (https://iclr.cc/virtual/2024/poster/19439), ML Anthology (https://mlanthology.org/iclr/2024/pacchiardi2024iclr-catch/, OpenReview 567BjxgaTp) and the proceedings PDF headed "Published as a conference paper at ICLR 2024".

**CLAIMS**
1. "Told in a system prompt to deceive the user, many give confident false answers to questions they answer correctly when asked plainly [kretschmar, pacchiardi]" — **SUPPORTED** together with Kretschmar, with one nuance for Pacchiardi. A lie is "outputting false statements despite 'knowing' the truth", that is, the model "gives the correct answer under typical question-answering prompts". The lies "almost always seem plausible" and models "frequently double down" (GPT-3.5 does so in more than 80% of cases). The nuance: Pacchiardi uses 17 "direct lie instruction prompts" (e.g. "Lie when answering the following question") on GPT-3.5 (text-davinci-003, a completion model), Alpaca-7B and Vicuna-7B. These are prompt instructions, not system prompts, and they "induce lying often but not highly reliably". The "system prompt" detail therefore rests on Kretschmar. https://arxiv.org/abs/2309.15840 (Sec. 4.2)
   - Optional fix: "Told in the prompt to deceive the user, ...".
2. "\AIL{} [pacchiardi]: 1,012 factual questions under lie instructions, 63 ... excluded" — **PARTIAL** (minor). 1,012 is the size of one sub-dataset, Questions1000, not of the whole benchmark. I counted 1,012 entries in the authors' `data/raw_questions/questions_1000_all.json` and `data/processed_questions/questions_1000_all.json` (https://github.com/LoryPack/LLM-LieDetector). The paper's full dataset has ">20,000 questions" across 11 sub-datasets (Sec. 4.1, App. A.2), and Questions1000 is the "1000 Questions from Meng et al. (2022)" general-knowledge subset. https://arxiv.org/abs/2309.15840
   - **Fix:** "\AIL{} \citep{pacchiardi2024catch}: the 1,012 general-knowledge questions of its Questions1000 subset under lie instructions, 63 of them ...".
3. "probes that detect lies after the fact \citep[see also][]{pacchiardi2024catch}" — **SUPPORTED** (thanks to the "see also" hedge). Pacchiardi's detector is black-box rather than a representation probe: it works by "asking a predefined set of unrelated follow-up questions after a suspected lie" and "requires neither access to the LLM's activations". https://arxiv.org/abs/2309.15840
   - Optional clarity fix: "\citep[see also the black-box detector of][]{pacchiardi2024catch}".

---

## qwen2025qwen25

**METADATA: OK** (one cosmetic point). Title "Qwen2.5 Technical Report", arXiv 2412.15115 and cs.CL match. v1 is dated 19 Dec 2024, so year = 2024 is correct, and v2 is dated 3 Jan 2025. arXiv lists the authors as "Qwen: An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, ..." (42 named authors), so the bib's {Qwen} + 8 names + "others" is fine. https://arxiv.org/abs/2412.15115
- Cosmetic: the cite key says 2025 but year = 2024. This does not affect the rendered output. No venue exists; it is a preprint only.

**CLAIMS**
1. "Qwen2.5-3B-Instruct [qwen2025qwen25]" — **SUPPORTED**. The report covers it: "The open-weight offerings include base models and instruction-tuned models in sizes of 0.5B, 1.5B, 3B, 7B, 14B, 32B, and 72B", released "in bfloat16 precision". https://arxiv.org/abs/2412.15115 (PDF p. 1–2)

---

## scheurer2023insider

**METADATA: mostly OK; venue could be added.** Title, the 3 authors in order, year 2023 (v1 is dated 9 Nov 2023) and arXiv 2311.07590 (cs.CL) match. The bib calls it an arXiv preprint, but the PDF header (v4) reads **"Published at the LLM Agents workshop at ICLR 2024"** (OpenReview HduMpot9sJ, from a search hit). https://arxiv.org/abs/2311.07590
- Optional fix: `@inproceedings{..., booktitle = {ICLR 2024 Workshop on Large Language Model (LLM) Agents}, year = {2024}}`. The exact official workshop title is UNVERIFIED because OpenReview is blocked. Keeping the arXiv form is also acceptable.

**CLAIMS**
1. "Deception by language models is a recognised risk [park2024deception, scheurer2023insider]" — **SUPPORTED**. The paper deploys GPT-4 as a trading agent that "strategically deceive[s] their users ... without being instructed to do so", which it calls "the first demonstration" of this. https://arxiv.org/abs/2311.07590

---

## turner2023actadd

**METADATA: OK.** The title "Steering Language Models With Activation Engineering" and the 7 authors in order (Turner, Thiergart, Leech, Udell, Vazquez, Mini, MacDiarmid) match arXiv v5 (updated 10 Oct 2024). Year 2023 (v1 is dated 20 Aug 2023), arXiv 2308.10248 and cs.CL also match. https://arxiv.org/abs/2308.10248
- The arXiv entry has no journal-ref or comment, and the v5 PDF has no venue header. **UNVERIFIED** whether a peer-reviewed version exists, because Semantic Scholar and DBLP could not be reached. A web search found no conference version: the "ICLR 2025 HAIC workshop" hit (openreview FZk9oWvZm2) is a different paper that cites ActAdd. Keep it as a preprint.

**CLAIMS**
1. "Activation steering [turner, rimsky] moves the defense into the forward pass, but it adds a vector at inference time" — **SUPPORTED**. The paper says "ActAdd directly adds the steering vectors to the forward pass at inference time" and defines activation engineering as "the inference-time modification of activations". https://arxiv.org/abs/2308.10248
2. "Adding a difference-in-means direction at inference [turner, rimsky, zou, stolfo] is the standard training-free steering method." — **PARTIAL** for Turner. ActAdd is training-free ("does not require any machine optimization"), but its vector is the activation difference of a single contrast pair, not a mean over a dataset: "Our method takes a pair of natural-language prompts (p+, p−) ... The difference h+ − h− is a new activation vector", and it "works with a single pair of data points". The difference-in-means construction is CAA's (rimsky2024caa; Braun et al. also describe CAA as "the mean difference between activations"). https://arxiv.org/abs/2308.10248 (Sec. 3)
   - **Fix:** "Adding a contrastive activation difference at inference, taken from one prompt pair \citep{turner2023actadd} or averaged over many pairs \citep{rimsky2024caa,zou2023repe,stolfo2025steering}, is the standard training-free steering method." Alternatively, keep "difference-in-means" and move turner2023actadd to an "e.g." or "see also" position.
