# Citation delta check, 2026-10-09

Scope: the two references added with the co-author's Method subsections (Overleaf 430053a, Sections 2.1-2.2 and 2.5),
which were cited but had no bib entry and printed as "?" in the PDF. Method as in reports/citation_audit_2026-10-08.

| Key | Citing text | Metadata | Citing sentence |
|---|---|---|---|
| pearl1999probabilities | Sec. 2.2: "Under deterministic decoding, the probability of necessity [pearl, tianpearl] for this intervention is PN(M,a) = Pr(Y_p(0)=0 \| Y_p(1)=1)" | SUPPORTED. Crossref 10.1023/a:1005233831499: Judea Pearl, "Probabilities of Causation: Three Counterfactual Interpretations and Their Identification", Synthese 121(1-2):93-149, 1999. Crossref title-cases the title; restored to the published capitalisation. | CORROBORATED INDIRECTLY, not read. The UCLA copy (ftp.cs.ucla.edu/pub/stat_ser/R261.pdf, HTTP 200, 537 KB) has no usable text layer: both pypdf and pymupdf return mojibake, so the definition could not be checked against the text. Crossref and Semantic Scholar hold no abstract. Tian and Pearl's abstract (below) states that it starts "from structural-semantical definitions of the probabilities of necessary or sufficient causation" and strengthens "the results of Pearl [39]", which places the definition and the original bounds in this paper. **A human should confirm the PN definition against the Synthese text before submission.** |
| tianpearl2000 | Same sentence, and Sec. 2.5: "gain over headroom equals PN only under monotonicity (q=0); otherwise it is a lower bound" | SUPPORTED. Crossref 10.1023/a:1018912507879: Jin Tian and Judea Pearl, "Probabilities of causation: Bounds and identification", Annals of Mathematics and Artificial Intelligence 28(1-4):287-313, 2000. | SUPPORTED. Published abstract (Semantic Scholar, DOI lookup): "Starting from structural-semantical definitions of the probabilities of necessary or sufficient causation ..., we show how to bound these quantities from data ... as well as under the mild assumptions of exogeneity (no confounding) and monotonicity (no prevention)." Both the PN definition and the role of monotonicity are the paper's own subject. |

Not fixed here, reported instead: the same subsections point at `\ref{app:theory}` and `\ref{app:pn}`, two appendices that do not exist
in main.tex, so both still print as "?". The pointers are the co-author's and promise content (a local perturbation bound; bounds on
PN plus a Qwen2.5 audit) that this session does not have.

Entries after this pass: 42, all marked verified (35 on 2026-10-07, 5 re-checked 2026-10-08, 2 added 2026-10-09).
