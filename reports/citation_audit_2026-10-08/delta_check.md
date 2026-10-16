# Citation delta check, 2026-10-08

Scope: Overleaf commits 78342b3..a3d1cdb (11 commits after the 2026-10-07 audit, reports/citation_audit_2026-10-07/refcheck_1-4.md).
Method: every main.tex paragraph with \cite whose text changed (git diff, 4 added/changed lines), the generated tables (no \cite
changes), and every bib entry added or changed after the audit (one: boerner2023access).

| Key | Citing text (changed sentence) | What changed | Verdict |
|---|---|---|---|
| tan2024generalisation, braun2025unreliability | Sec. 4 opening: "... activation addition, whose vectors are sensitive to the prompts that build them" | Sentence reworded around the cited clause (no longer refers to the multi-turn failure); cited clause identical | SUPPORTED, as in refcheck_2 item 2 (Tan: vectors extracted under different system/user-prompt settings differ, transfer depends on the setting) and refcheck_3 item 2 (Braun: vectors from different prompt types differ in direction and in steering effect, arXiv 2505.22637) |
| rimsky2024caa | Sec. 4 "Suite and measure": "a contrastive-activation-addition vector [CAA]" | Upper-bound clause added later in the sentence; cited clause identical | SUPPORTED (refcheck_1) |
| qi2025shallow | App. depth paragraph: "show that safety alignment mostly changes the first few response tokens" | KL(edited) renamed KL(ours); cited sentence identical | SUPPORTED (refcheck_2) |
| boerner2023access | Acknowledgments (commented out): ACCESS program citation | New entry and sentence | Metadata SUPPORTED by Crossref 10.1145/3569951.3597559 on 2026-10-08 (authors Boerner, Deems, Furlani, Knuth, Towns; PEARC '23, ACM, pp. 173-176, 2023; title + subtitle = the bib title). Citing sentence matches the ACCESS template at access-ci.org/acknowledging-access/ (fetched 2026-10-08), except that the template now reads "U.S. National Science Foundation grants" where the block reads "National Science Foundation grants" (flagged, not changed: the block came from the Delta citation page via xiaocong-7e). |

No other \cite changed; no other bib entry changed. Result: 40/40 entries verified (35 on 2026-10-07, 5 re-checked on 2026-10-08).
