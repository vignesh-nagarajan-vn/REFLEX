# Superseded paper versions - frozen, do not edit

**v4 is the final version of the REFLEX ICAIF '26 paper.** It lives one level
up, as [`../main.tex`](../main.tex), and everything in this folder is
superseded. Nothing here is built, cited, or submitted.

These files are kept only as the record of what was compiled and when. Each
`.pdf` is the exact artifact produced by the `.tex` beside it, so a claim made
in an earlier draft can be traced back to the source that produced it. They are
**frozen**: do not edit them, do not recompile them, and do not fix things in
them. If something in an old version needs correcting, it is corrected in v4.

| Version | Source | Compiled PDF | Date | Pages | What it was |
|---|---|---|---|---|---|
| v1 | `main_v1.tex` | `REFLEX_Research_Paper_v1.pdf` | 2026-07-12 | 8 | First full submission draft: de-anonymised author block, public repo link, all six figures. |
| v2 | `main_v2.tex` | `REFLEX_Research_Paper_v2.pdf` | 2026-08-07 | 8 | Rewritten for ICAIF house style; corrected REFLEX acronym in the title; Berkeley email. |
| v3 | `main_v3.tex` | `REFLEX_Research_Paper_v3.pdf` | 2026-08-08 | 8 | Prose rewrite in the register of the authors' reference paper. Same results, figures, tables, equations and citations as v2. |
| **v4** | **`../main.tex`** | **`../REFLEX_Research_Paper.pdf`** | **2026-08-09** | **8** | **FINAL.** See [`../README.md`](../README.md). |

Every version compiled at exactly 8 pages including references, which is the
ICAIF limit.

### Other superseded material

| File | What it is |
|---|---|
| `abstract-rewrite-v4-draft.tex` | The first draft of the v4 abstract, as originally committed to the repo root of `research/paper/`. Superseded: the shipped v4 abstract was trimmed from 309 words / 2073 characters to 293 / 1979 to clear the 2000-character submission cap. Moved here because it lived beside `main.tex` while disagreeing with it, which is exactly the sort of second source of truth that gets pasted into a submission form by mistake. The live abstract is in `../main.tex`. |

## What changed in v4

Summarised here so this folder explains its own obsolescence; the full account
is in [`../README.md`](../README.md) and the `main.tex` header.

- References numbered **by order of appearance** rather than alphabetically,
  via `../ACM-Reference-Format-seq.bst`.
- The body is **self-contained**: Sec. 4 no longer defers its derivations to
  the repository, and the repository is cited in exactly one place, the
  Reproducibility statement.
- **New abstract**, opening on the market rather than the notation, trimmed to
  1979 characters to clear the 2000-character submission cap.
- **One colour system across all six figures**, colourblind-safe and drawn at
  printed size, regenerated from the run CSVs by
  `../figures/make_paper_figures.py`.
- **Captions uniformly bold**, fixing math that printed light inside bold
  prose, and titles added to Figures 1-3.

## Submission builds

v4 produces two PDFs from the one source, via the `\ANON` switch:

- `../REFLEX_Research_Paper.pdf` - de-anonymised working version.
- `../REFLEX_Research_Paper_v4_ANONYMOUS_submission.pdf` - **the double-blind
  build that goes to CMT**, produced by `../anon.tex`.

Earlier versions had no anonymous build; their double-blind toggle was a manual
two-step edit. That is another reason not to reuse them.
