# paper/ - the ICAIF 2026 submission

The conference paper for the REFLEX research program: **"REFLEX: Reflexive
Equilibrium Fixed-point Learning for Endogenous Financial Markets"** (with
the subtitle *"Analytic Stability Boundaries for Performative Market Making
in OTC Corporate Bond Markets"*), targeting
[ICAIF 2026](https://icaif2026.org/) (7th ACM
International Conference on AI in Finance, Milan, Nov 14-17 2026). Submission
is via CMT; the CFP lists "Paper Submission Deadline: August 2, 2026
**Extended deadline: August 9, 2026**"
([call-for-papers](https://icaif2026.org/call-for-papers.html), checked
2026-08-07), so the live target is **Aug 9, 2026**.

## Contents

| File | What it is |
|------|-----------|
| `main.tex` | **The current paper source (v2).** ACM `sigconf`. The **de-anonymized working version**: real author block (co-first authors, Vignesh corresponding) + public repo footnote. The double-blind toggle for submission is documented in the file header. |
| `references.bib` | 24 verified references (arXiv IDs checked against arxiv.org on 2026-07-12; two wrong IDs inherited from `literature/*/references.bib` were corrected - see the header comment). Shared by both versions. |
| `figures/` | The six headline figures, copied verbatim from the v4 paper-grade run `research/results/07-12-2026/`. Shared by both versions. |
| `archive/main_v1.tex` | The superseded v1 source, kept verbatim. |
| `archive/REFLEX_Research_Paper_v1.pdf` | The v1 compile (Overleaf pdfLaTeX, 2026-07-12): exactly 8 pages including references. |

Content sources: the run report
[`../results/07-12-2026/REPORT.md`](../results/07-12-2026/REPORT.md) (all
numbers), the derivations [`../math-theory/`](../math-theory/) (all closed
forms, quoted as documents D1-D6), and the analysis layer
[`../analysis/`](../analysis/) (scoping and caveats). Every number in the
paper traces to the curated 07-12-2026 run; nothing is re-derived here.

## Versions

**v2 (2026-08-07) is current.** It is a prose rewrite for ICAIF house style,
not a change of results. Same six figures, same five tables, same eleven
numbered equations, same 24 citations, same claim scope. What changed:

**Structure.** Related work moved out of the introduction into its own
section. The four intro subsections (`Problem Definition` /
`Current Approaches` / `Proposed Approach` / `Results Overview`) collapsed
into flowing prose ending in an explicit **R1-R6 contributions list**, each
contribution carrying its headline measured number. The standalone
`Limitations` section merged into `Conclusion`, which is the ICAIF pattern.
`System and Measurement Methodology` became `Experimental Setup`. Section
count is unchanged at 7; subsection count drops by 4.

**Register.** Shorter declarative sentences: mean sentence length falls from
42.3 words to 28.9. The dash-parenthetical aside, the dominant construction
in v1, is gone: 38 occurrences of ` -- ` in v1, zero in v2 (the remaining
`--` are numeric ranges, compound author names, and table placeholders). The
abstract drops its displayed notation, per venue convention, and closes on a
significance sentence.

**Length.** Body prose 3768 words against v1's 3847, so v2 is 79 words
shorter than a source that compiled to exactly 8 pages, with an identical
float, equation, and citation inventory and four fewer subsection headings.

v1 is kept under `archive/` because it is the version whose 8-page compile
is on record. Do not edit it.

## Building (Overleaf)

1. Create a new Overleaf project and upload `main.tex`, `references.bib`,
   and the `figures/` folder (keep the folder name).
2. Compiler: pdfLaTeX, TeX Live 2022 or later (the `acmart` class and
   `ACM-Reference-Format.bst` ship with the standard Overleaf TeX Live).
3. Compile, then **save the PDF beside this README as
   `REFLEX_Research_Paper.pdf`** and record the page count here.

There is no LaTeX toolchain on the dev machine. The source is machine-checked
locally instead: balanced environments and braces, cite keys against
`references.bib`, `\ref` against `\label`, `\includegraphics` paths, ASCII-only,
and the prose word budget against v1. **v2 has not yet been compiled.** v1
landed at exactly 8 pages with zero slack, and v2 is 71 prose words shorter
with the same floats, so it should fit; verify on Overleaf before submitting,
and use the trim order below if it runs over.

### If it runs over 8 pages, trim in this order

1. `Table 1` (certificates) -> fold into two sentences of Sec. 6.1 (the
   per-family numbers are in the run report).
2. Sec. 6.7 (scaling + tuning) -> compress to one paragraph of headline
   numbers.
3. The R4/R5 paragraphs in Sec. 4 -> one-sentence statements (the
   experiments cite the repository documents anyway).
4. Fig. 5 (lazy deployment, full-width) -> drop the figure, keep the
   K_db/K_max numbers in text.
5. The R1-R6 contributions list in Sec. 1 -> strip the trailing measured
   number from each entry (the numbers all repeat in Sec. 6).

## ICAIF requirements -> where they are satisfied

| Requirement (from `../README.md` § ICAIF) | Status |
|---|---|
| 8 pages total, ACM `sigconf`, two-column | Written to budget; v1 compiled at exactly 8 pages, v2 is shorter. **Verify v2 on Overleaf** |
| ACM template with `anonymous` parameter | Class line is currently `[sigconf]` (de-anonymized working version, by author decision). **Before CMT submission switch to `[sigconf,anonymous,review]`** - toggle documented in the file header |
| Double-blind: no identifying info | Currently de-anonymized (real author block + public GitHub footnote). Before submission: re-enable `anonymous` and swap the Reproducibility footnote for an anonymized mirror |
| No supplementary materials/appendices | Self-contained; the repository link is a reproducibility pointer, not supplementary material |
| Real-world financial application | Secs. 1, 6.2 (real-data fragility index), 7 |
| Data provenance disclosed | Sec. 5.3 + the Conclusion's limitations paragraph (proxy-level, not trade-level TRACE; degenerate crisis cell) |
| Uncertainty bands across seeds | Sec. 5.2 protocol; median + IQR + robust bands in Fig. 2; single-seed demos labeled as such |
| Systemic-risk / governance framing | Secs. 1, 4 (R3), 6.4, 7 |
| Light reviewing commitment / in-person attendance | Author logistics - not a paper artifact |
| Submit via CMT before the deadline | `cmt3.research.microsoft.com/ICAIF2026`; extended deadline **Aug 9, 2026** |
| ORCID (camera-ready only) | `TODO(camera-ready)` marker in `main.tex` |

## Submission and camera-ready TODOs

Before **CMT submission** (ICAIF review is double-blind; the source is
currently de-anonymized by author decision):

- Compile v2 on Overleaf and confirm the 8-page count.
- Switch the class line back to `\documentclass[sigconf,anonymous,review]{acmart}`
  (toggle documented in the `main.tex` header).
- Swap the Reproducibility footnote (public GitHub URL) for an anonymized
  mirror, and create that mirror so it resolves for reviewers.

At **camera-ready** (marked `TODO(camera-ready)` in `main.tex`):

- Add `\orcid{}` iDs for both authors (ICAIF requires ORCID at camera-ready).
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
This ledger applies unchanged to v2: the rewrite moved prose, not claims.
