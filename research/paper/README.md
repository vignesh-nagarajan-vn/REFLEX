# paper/ - the ICAIF 2026 submission

The conference paper for the REFLEX research program: **"REFLEX: Analytic
Stability Boundaries for Performative Market Making in OTC Corporate Bond
Markets"**, targeting [ICAIF 2026](https://icaif2026.org/) (7th ACM
International Conference on AI in Finance, Milan, Nov 14-17 2026; submission
deadline **Aug 2, 2026**, via CMT).

## Contents

| File | What it is |
|------|-----------|
| `main.tex` | The full paper source (ACM `sigconf`, `anonymous,review` options for double-blind submission) |
| `references.bib` | 24 verified references (arXiv IDs checked against arxiv.org on 2026-07-12; two wrong IDs inherited from `literature/*/references.bib` were corrected - see the header comment) |
| `figures/` | The four headline figures, copied verbatim from the v4 paper-grade run `research/results/07-12-2026/` |

Content sources: the run report
[`../results/07-12-2026/REPORT.md`](../results/07-12-2026/REPORT.md) (all
numbers), the derivations [`../math-theory/`](../math-theory/) (all closed
forms, quoted as documents D1-D6), and the analysis layer
[`../analysis/`](../analysis/) (scoping and caveats). Every number in the
paper traces to the curated 07-12-2026 run; nothing is re-derived here.

## Building (Overleaf)

1. Create a new Overleaf project and upload `main.tex`, `references.bib`,
   and the `figures/` folder (keep the folder name).
2. Compiler: pdfLaTeX, TeX Live 2022 or later (the `acmart` class and
   `ACM-Reference-Format.bst` ship with the standard Overleaf TeX Live).
3. Compile. There is no LaTeX toolchain on the dev machine, so the source
   has been machine-checked (balanced environments, cite/label/figure
   integrity, ASCII-only) but **not compiled - verify the page count on
   Overleaf first thing** (ICAIF limit: 8 pages *total*, including
   references).

### If it runs over 8 pages, trim in this order

1. `Table 1` (certificates) -> fold into two sentences of Sec. 5.1 (the
   per-family numbers are in the run report).
2. Sec. 5.7 (scaling + tuning) -> compress to one paragraph of headline
   numbers.
3. The R4/R5 paragraphs in Sec. 3 -> one-sentence statements (the
   experiments cite the repository documents anyway).
4. Fig. 4 (lazy deployment) -> drop the figure, keep the K_db/K_max
   numbers in text.
5. Intro Sec. 1.4 (Results Overview) -> halve; the numbers repeat in Sec. 5.

## ICAIF requirements -> where they are satisfied

| Requirement (from `../README.md` § ICAIF) | Status |
|---|---|
| 8 pages total, ACM `sigconf`, two-column | Written to budget; **verify on Overleaf** |
| ACM template with `anonymous` parameter | `\documentclass[sigconf,anonymous,review]{acmart}` (`review` adds line numbers; drop it if the CFP says otherwise) |
| Double-blind: no identifying info | Author block hidden by `anonymous`; no self-citations; repo link anonymized (footnote placeholder) |
| No supplementary materials/appendices | Self-contained; the repository link is a reproducibility pointer, not supplementary material |
| Real-world financial application | Secs. 1.1, 5.2 (real-data fragility index), 6 |
| Data provenance disclosed | Sec. 4.3 + Limitations (proxy-level, not trade-level TRACE; degenerate crisis cell) |
| Uncertainty bands across seeds | Sec. 4.2 protocol; median + IQR + robust bands in Fig. 2; single-seed demos labeled as such |
| Systemic-risk / governance framing | Secs. 1.1, 3 (R3), 5.4, 6 |
| Light reviewing commitment / in-person attendance | Author logistics - not a paper artifact |
| Submit via CMT before Aug 2, 2026 | `cmt3.research.microsoft.com/ICAIF2026` |
| ORCID (camera-ready only) | `TODO(camera-ready)` marker in `main.tex` |

## Camera-ready TODOs (all marked `TODO(camera-ready)` in `main.tex`)

- Replace the `anonymous.4open.science` placeholder footnote with the real
  repository URL (and create the anonymized mirror *before* submission so
  the placeholder resolves for reviewers).
- Remove `anonymous,review` options; restore the author block; confirm
  affiliations; add ORCID iDs.
- Remove `\settopmatter{printacmref=false}` / `\setcopyright{none}` and
  insert the rights block ACM supplies on acceptance.
- Optional: regenerate the CCS concept XML ids with the ACM CCS tool (the
  printed concepts are correct; the XML ids are DL metadata only).

## Honest-claims ledger

The paper deliberately scopes its claims the way the run report does:
measured crossings are statements about the retraining map at the operating
spread (not the saturating fixed point); the structural loop is benchmarked
against the *realized* market via independent structural fits (never the
frozen-reference closed form); the free-form learned correction is reported
as a negative result; the Lean skeletons are stated as reviewed-not-compiled
with the numerical certificates as the verification of record; and the data
section states plainly that calibration is proxy-level, not trade-level
TRACE. Reviewers should find no claim in the paper stronger than its
counterpart in [`../results/07-12-2026/REPORT.md`](../results/07-12-2026/REPORT.md).
