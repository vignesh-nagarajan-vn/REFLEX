# paper/ - the ICAIF 2026 submission

The conference paper for the REFLEX research program: **"REFLEX: Reflexive
Equilibrium Fixed-point Learning for Endogenous eXchanges"** (with
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
| `main.tex` | **The current paper source (v3).** ACM `sigconf`. The **de-anonymized working version**: real author block (co-first authors, Vignesh corresponding, Texas A&M + TAMIDS) + public repo footnote. The double-blind toggle for submission is documented in the file header (class line) and beside the Reproducibility footnote. **Not yet compiled** - see [Building](#building-overleaf). |
| `references.bib` | 24 verified references (arXiv IDs checked against arxiv.org on 2026-07-12; two wrong IDs inherited from `literature/*/references.bib` were corrected - see the header comment). Shared by all versions. |
| `figures/` | The six headline figures, copied verbatim from the v4 paper-grade run `research/results/07-12-2026/`. Shared by all versions. |
| `archive/main_v2.tex` | The superseded v2 source, kept verbatim. |
| `archive/REFLEX_Research_Paper_v2.pdf` | The v2 compile (Overleaf pdfLaTeX, 2026-08-07): exactly 8 pages including references, de-anonymized build. |
| `archive/main_v1.tex` | The superseded v1 source, kept verbatim. |
| `archive/REFLEX_Research_Paper_v1.pdf` | The v1 compile (Overleaf pdfLaTeX, 2026-07-12): exactly 8 pages including references. |

Content sources: the run report
[`../results/07-12-2026/REPORT.md`](../results/07-12-2026/REPORT.md) (all
numbers), the derivations [`../math-theory/`](../math-theory/) (all closed
forms, quoted as documents D1-D6), and the analysis layer
[`../analysis/`](../analysis/) (scoping and caveats). Every number in the
paper traces to the curated 07-12-2026 run; nothing is re-derived here.

## Versions

**v3 (2026-08-08) is current.** It is a prose rewrite in the register of the
authors' reference paper (the ICDM-format submission), not a change of
results. Same six figures, same five tables, same eleven numbered equations,
same 24 citations, same seven sections and eleven subsections, same claim
scope. What changed:

**Abstract.** Rewritten as a narrative rather than a notation-dense summary:
context, the feedback problem, the gap in existing theory, "To address this
gap, this paper introduces REFLEX...", what it does, the headline numbers,
and a practical closing sentence. Higher level than v2's, and `REFLEX` is no
longer bolded.

**Register.** Related Work now attributes work to named authors ("Perdomo et
al. established...", "Avellaneda and Stoikov ... derived...") instead of
listing bare bracket citations. Results state what a number means
operationally right after stating it. Semicolon-chained sentences are split.
The Conclusion's limitations are enumerated First / Second / Third / Finally.
Measured: mean sentence length 18.3 words against v2's 19.7; longest
sentence 62 words against 70.

**Length.** Body prose 3926 words against v2's 3815 (+111, roughly 10 column
lines); abstract 211 words against 212. The increase *is* the register
change; it was held to +111 by compressing the theory section, which is
compact in the reference paper too.

**Author block.** Adds the TAMIDS affiliation as a second `\institution`
inside the existing `\affiliation` (see below).

v1 and v2 are kept under `archive/` because theirs are the 8-page compiles on
record. Do not edit them.

## The TAMIDS affiliation

ACM, and therefore ICAIF, carries institutional affiliations in the author
block rather than in an acknowledgement, so TAMIDS goes there. It is a second
`\institution` inside the *same* `\affiliation` block as Texas A&M
University, because it shares the university's city and country. `acmart`
sets each `\institution` on its own line, so this costs **one** line where a
second `\affiliation` block would have cost four - and page 8 has no slack.
Under the `anonymous` submission build the entire author block collapses to
"Anonymous Author(s)", so the affiliation costs nothing at review time and
cannot leak identity.

## Building (Overleaf)

1. Create a new Overleaf project and upload `main.tex`, `references.bib`,
   and the `figures/` folder (keep the folder name).
2. Compiler: pdfLaTeX, TeX Live 2022 or later (the `acmart` class and
   `ACM-Reference-Format.bst` ship with the standard Overleaf TeX Live).
3. Compile, then **save the PDF beside this README as
   `REFLEX_Research_Paper.pdf`** and record the page count here.

There is no LaTeX toolchain on the dev machine. The source is machine-checked
locally instead: balanced environments and braces, cite keys against
`references.bib`, `\ref` against `\label`, `\includegraphics` paths,
ASCII-only, and the prose word budget against the previous version.

**v3 is not yet compiled.** The page-count situation, measured rather than
assumed:

- v2 compiled at exactly 8 pages, and its page 8 is filled to the same bottom
  margin (85 pt) as every other page. Slack is literally zero, not
  "effectively" zero.
- v3 adds 111 body-prose words (~10 column lines) and one line to the author
  block, so the **de-anonymized** build may run to 9 pages. If it does, apply
  trim 1 below; it frees roughly 15 lines, comfortably more than needed.
- The **submission** build is a different story. Flipping to `anonymous`
  replaces the whole author block with "Anonymous Author(s)", reclaiming more
  vertical space than v3 added, so the double-blind PDF should still land at
  8 pages. That is the build ICAIF actually receives.

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

Re-checked against
[icaif2026.org/call-for-papers.html](https://icaif2026.org/call-for-papers.html)
on **2026-08-08**, against the v3 source.

| Requirement (CFP wording) | Status in v3 |
|---|---|
| "no longer than eight (8) pages in total (when in two-column sigconf format), including all figures and references" | **Open until compiled.** v2 hit exactly 8 with page 8 full; v3 adds ~10 column lines plus 1 author-block line. The `anonymous` submission build reclaims more than that. Trim 1 is the lever if the de-anonymized build spills. |
| "Papers must use the latest ACM article template" in "sigconf two-column format" | `\documentclass[sigconf]{acmart}`; `\settopmatter{printacmref=false}` and `\setcopyright{none}` are pre-acceptance only and are removed at camera-ready |
| Double-blind; "Submitted papers should not reveal the identity of the authors, either by citation or other obvious mention" | Working copy is de-anonymized by author decision. Two-step toggle documented in the `main.tex` header and beside the Reproducibility footnote. Under `anonymous` the author block (including TAMIDS) collapses to "Anonymous Author(s)". The only other identity leak is the GitHub URL in the Reproducibility footnote, which step 2 swaps for the anonymized mirror |
| Self-citations "in third person only" | N/A - the paper has no self-citations; all 24 references are third-party |
| "ICAIF '26 will not accept any supplementary materials/appendices" | Self-contained. No appendix. The repository link is a reproducibility pointer, not supplementary material, and every claim is supported inside the 8 pages |
| ORCID: "ACM requires this for all authors of accepted papers" | `TODO(camera-ready)` marker in `main.tex`; both authors need iDs before camera-ready |
| Author limit: "no more than twelve (12) submissions" per author | N/A - one submission |
| Topic fit (nine CFP areas) | Primary: **Trustworthy & Responsible AI** ("AI governance, computational regulation"); secondary: **Risk Management** (systemic fragility, model validation) and **Trading & Asset Management**. Framed as such in Secs. 1, 4 (R3), 6.4, 7 |
| Real-world financial application | Secs. 1, 6.2 (real-data fragility index), 7 |
| Data provenance disclosed | Sec. 5.3 + the Conclusion's limitations paragraph (proxy-level, not trade-level TRACE; degenerate crisis cell) |
| Uncertainty bands across seeds | Sec. 5.2 protocol; median + IQR + robust bands in Fig. 2; single-seed demos labeled as such |
| Deadlines: submission **August 9, 2026** (extended, AoE); notification September 27, 2026 | Submit via CMT at `cmt3.research.microsoft.com/ICAIF2026/` |

## Submission and camera-ready TODOs

Before **CMT submission** (ICAIF review is double-blind; the source is
currently de-anonymized by author decision):

1. Switch the class line to `\documentclass[sigconf,anonymous,review]{acmart}`
   (step 1 of the toggle, documented in the `main.tex` header).
2. Create the anonymized mirror, then comment out the public-URL footnote in
   the Reproducibility section and uncomment the mirror footnote directly
   below it (step 2, marked `DOUBLE-BLIND TOGGLE` in the source).
3. Recompile and re-check the 8-page count, then submit that build. The
   committed PDF is the de-anonymized one and is not the submission artifact.
4. Compile the de-anonymized build too and save it as
   `REFLEX_Research_Paper.pdf` beside this README, recording its page count
   in the table above (v3 has no committed compile yet).

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
This ledger applies unchanged to v2 and v3: both rewrites moved prose, not
claims. Every numeric literal in v3 is carried over verbatim from v2, which
traced each one to the 07-12-2026 artifacts.
