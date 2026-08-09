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
| `main.tex` | **The current paper source (v4).** ACM `sigconf`. The **de-anonymized working version**: real author block (co-first authors, Vignesh corresponding) + public repo footnote. The double-blind toggle for submission is documented in the file header (class line) and beside the Reproducibility footnote. |
| `REFLEX_Research_Paper.pdf` | The v4 compile (local MiKTeX pdfLaTeX + BibTeX, 2026-08-09): **exactly 8 pages including references**, all six figures, five tables and the full 24-entry bibliography placed. De-anonymized build, so the submission PDF is a re-compile after the double-blind flip. |
| `ACM-Reference-Format-seq.bst` | **v4.** A verbatim fork of the stock acmart bibliography style with the two BibTeX `SORT` commands commented out, which numbers references by order of first appearance instead of alphabetically by author. Every per-entry ACM Reference Format rule is untouched. See the file header, and the camera-ready caveat under *Versions*. |
| `references.bib` | 24 verified references (arXiv IDs checked against arxiv.org on 2026-07-12; two wrong IDs inherited from `literature/*/references.bib` were corrected - see the header comment). Shared by all versions. |
| `figures/` | The six headline figures. From v4 they are rendered by `figures/make_paper_figures.py` from the committed CSVs of the paper-grade run `research/results/07-12-2026/`; v1-v3 used the run's own PNGs verbatim. |
| `figures/make_paper_figures.py` | **v4.** Re-renders all six figures from the run CSVs on one colourblind-safe palette, at the size each is actually printed at. Reads the CSVs only; runs no simulation. |
| `archive/main_v3.tex` | The superseded v3 source, kept verbatim. |
| `archive/REFLEX_Research_Paper_v3.pdf` | The v3 compile (Overleaf pdfLaTeX, 2026-08-08): exactly 8 pages including references, de-anonymized build. |
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

**v4 (2026-08-09) is current.** A presentation pass over v3. No result and no
number changed, and all six figures, five tables, eleven equations, 24
citations and seven sections survive. What changed:

**References are numbered by order of appearance**, not alphabetically by
author, via `ACM-Reference-Format-seq.bst`. Perdomo et al. is now [1] because
it is cited first; Bao et al. is [24] because it is cited last. Verified
mechanically: the order of first `\citation` in `main.aux` equals the
`\bibitem` order in `main.bbl`, all 24 entries, no orphans. **Camera-ready
caveat:** ACM's TAPS pipeline regenerates the bibliography with the stock
style, so an accepted paper reverts to alphabetical. This governs the
submission PDF only.

**The body is self-contained.** Sec. 4 no longer defers its derivations to
"documents D1-D6 in the repository". The implicit-function-theorem derivation
of the modulus and its regularity conditions are now stated in the paper (the
*Derivation and regularity* paragraph of Sec. 4). The repository is cited in
exactly one place, the Reproducibility statement.

**New abstract**, supplied by the authors. It opens on the market rather than
the notation, names the three measurable features the modulus is built from,
and carries the headline numbers. **293 words / 1979 characters** as rendered
plain text, trimmed from a 309-word first draft to sit under the
2000-character cap submission portals impose. That cap counts rendered text,
not LaTeX source, so re-measure after any rewording by stripping `\textsc{}`,
`$...$` and collapsing whitespace.

**One colour system across all six figures.** v3 drew them from five unrelated
palettes and at 11-12 inches wide before squeezing them into a 3.33-inch
column, which shrank tick labels to roughly 2pt. v4 renders each at its
printed size from the Okabe-Ito colourblind-safe palette, one fixed meaning
per colour, with dash pattern and marker carrying the same distinction as hue
so the figures survive greyscale. The `RdYlGn_r` systemic surface became
`cividis` and the red/green stable-unstable shading is gone; both failed
red-green colour-vision deficiency.

**Captions are uniformly bold now, and Figures 1-3 have titles.** Bold caption
text with a colon separator is acmart's own `sigconf` default (`acmart.cls`:
the `\else` branch sets `labelfont={bf}, textfont={bf}, labelsep=colon`, and
the `sigconf` arm of the format switch adds no override), and that default is
kept - it is ACM house style. What was wrong is that TeX math never inherits
`\bfseries`, so every `$...$` inside a caption printed light against the bold
prose around it: Fig. 2 read "the measured ($f^*\approx3.17$) and predicted
($4.70$)" as bold parentheses wrapped around light numerals. A
`\DeclareCaptionFont{captionmath}{\boldmath}` in the preamble overrides
`textfont` only, so acmart's `labelfont`, `labelsep` and `margin` still apply.
Two lines, deletable to revert.

Figures 4-6 already carried per-panel titles; Figures 1-3 had none and now do.
Their `figsize` heights were reduced by the height of the title each gained,
so all three occupy the same column space as before to within 1% and the page
geometry is unchanged.

**Page budget.** (2) and (4) together first overran by about 11 column lines.
Trim #1 below was applied and then reverted once compressing the Sec. 4
derivation paragraph bought back enough room, so nothing was cut. Measured on
the compiled page 8, both reference columns end at 89.5% of page height, about
1.3 single-column lines short of the bottom margin. That is full.

### v3 (2026-08-08)

It is a prose rewrite in the register of the
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

**De-slopping.** A later pass reworded the sentences that read as generated
text: the mirrored "Two mature literatures approach this loop from opposite
ends" opener of Sec. 2 (deleted, the section now starts on the claim), the
aphoristic closers ("Anchoring, not capacity, closes the gap"; "Capacity is
not the problem. Identification off the deployed regime is."), and the setup
sentences that would survive deletion ("Each constant carries a direct market
reading", "The governance reading is direct", "The provenance deserves a
plain statement"). Checked with the `prose-guard` linter, which is clean
apart from `robust` (R4's named result and standard optimization vocabulary)
and two dashes that are `1990--2026` and `1-D`.

**Length.** Body prose 3901 words against v2's 3815 (+86, roughly 8 column
lines); abstract 211 words against 212. The increase *is* the register
change; it was held down by compressing the theory section, which is compact
in the reference paper too.

**Author block.** Unchanged from v2. A TAMIDS affiliation was added and then
removed at the author's request; do not re-add it without being asked.

v1, v2 and v3 are kept under `archive/` because theirs are the 8-page compiles
on record. Do not edit them.

## Building

There **is** a local LaTeX toolchain now: MiKTeX (pdfTeX 3.141592653-2.6-1.40,
`acmart` 2025/05/30 v2.14) at
`%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64`. Build from inside
`research/paper/`:

```
latexmk -pdf -interaction=nonstopmode main.tex
```

`latexmk` runs pdfLaTeX/BibTeX to convergence. Then **save the result beside
this README as `REFLEX_Research_Paper.pdf`** and record the page count here.
Clean the aux files afterwards; only the `.tex`, `.bib`, `.bst`, `figures/`
and the published PDF are committed.

MiKTeX installs missing packages on demand, which blocks on a GUI prompt the
first time unless the installer is set to silent:
`initexmf --set-config-value="[MPM]AutoInstall=1"`.

To build on Overleaf instead, upload `main.tex`, `references.bib`,
`ACM-Reference-Format-seq.bst` (**required from v4** - without it the
bibliography reverts to alphabetical) and the `figures/` folder, keeping the
folder name, and compile with pdfLaTeX on TeX Live 2022 or later.

**v4 compiled locally 2026-08-09 at exactly 8 pages including references**,
with all six figures, five tables and the full 24-entry bibliography placed.
Verified against the PDF rather than assumed: 8 pages; no overfull boxes; no
undefined references or citations; reference markers [1]-[24] all present and
in order of first appearance (checked mechanically against `main.aux`); the
new abstract and the self-contained Sec. 4 derivation in place.

Measured on the compiled page 8, both reference columns end at 89.5% of page
height, about 1.3 single-column lines short of the bottom margin, so
slack is effectively zero. Re-check the count after any edit, and use
the trim order below if it runs over. The `anonymous` submission build
replaces the author block with "Anonymous Author(s)" and so frees space
rather than consuming it, which means the double-blind build is not a
page-count risk.

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
on **2026-08-08**, against what is now the v4 source.

| Requirement (CFP wording) | Status in v4 |
|---|---|
| "no longer than eight (8) pages in total (when in two-column sigconf format), including all figures and references" | **Confirmed**: v4 compiled at exactly 8 pages including references (local MiKTeX, 2026-08-09), all six figures and five tables placed and the full 24-entry bibliography inside the limit. Slack is ~1.3 column lines, so re-check after any edit |
| "Papers must use the latest ACM article template" in "sigconf two-column format" | `\documentclass[sigconf]{acmart}`; `\settopmatter{printacmref=false}` and `\setcopyright{none}` are pre-acceptance only and are removed at camera-ready |
| Double-blind; "Submitted papers should not reveal the identity of the authors, either by citation or other obvious mention" | Working copy is de-anonymized by author decision. Two-step toggle documented in the `main.tex` header and beside the Reproducibility footnote. Under `anonymous` the author block collapses to "Anonymous Author(s)". The only other identity leak is the GitHub URL in the Reproducibility footnote, which step 2 swaps for the anonymized mirror |
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
   committed `REFLEX_Research_Paper.pdf` is the de-anonymized one and is not
   the submission artifact.

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
