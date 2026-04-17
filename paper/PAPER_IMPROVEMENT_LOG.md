# Paper Migration Improvement Log

Date: 2026-04-16

## Round 1: Claims and Scope

- Moved the abstract into `main.tex` as required by the NeurIPS template.
- Corrected the abstract from "five model families" to "six model settings" to match the evidence table.
- Added a broader-impact paragraph and updated the checklist answer accordingly.

## Round 2: LaTeX Structure

- Replaced the old custom article setup with the NeurIPS 2026 style file.
- Consolidated the paper into five input sections: Introduction, Related Work, Our Method, Experiments, and Conclusion.
- Moved all tables into standalone files under `paper/tables/`.
- Converted wide tables to `tabularx` layouts that respect `\linewidth`.

## Round 3: Reviewer-Facing Precision

- Removed an uncited "KVQuant-style" claim from Related Work.
- Clarified that FP8+\splitzip is exact with respect to native FP8 tensors, not original BF16 tensors.
- Distinguished controlled H200 microbenchmarks from real Qwen3/Mooncake measurements.

## Round 4: Static Verification

- Checked that all `\input{...}` targets exist.
- Checked that all labels are unique and all references resolve to defined labels.
- Checked that all citations resolve to entries in `references.bib`.
- Checked authored paper files for stale placeholders and stale old-section imports.

Compilation note: no LaTeX engine (`latexmk`, `pdflatex`, `xelatex`, `lualatex`, or `tectonic`) is currently available on `PATH`, so PDF regeneration could not be performed in this environment.
