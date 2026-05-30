# paper/

Final report, midterm report, and supporting LaTeX assets.

| File | Contents |
|---|---|
| `report.tex` / `report.pdf` | **Final report** (NeurIPS 2024 style, 8 body pages + references). Covers Introduction, Related Work, Methodology (system overview, MoC formulation, QLoRA setup), Experimental Design (RQ1–RQ3), Results, Discussion (with Limitations, Future Work, Broader Implications), and Conclusion. |
| `midterm_report.tex` | **Midterm report** — contains the full formal MoC methodology (E1–E4 definitions, router equation, load-balancing loss); used as the source when re-expanding methodology detail in the final report. |
| `report_instructions.tex` | TA-provided template specifying required sections and the 9-page (excluding references) limit. |
| `ref.bib` | BibTeX entries: PAPO, InstructBLIP, Honeybee, CuMo, MoCHA, MoVA, MoE-LLaVA, Switch Transformer, CLIP, BLIP-2, Flamingo, LLaVA, ScienceQA, POPE, QLoRA, and others. |
| `neurips_2024.sty` | NeurIPS 2024 style file. |
| `architecture_diagram updated.png` / `architecture.png` | Figure 1 — the MoC system diagram used in the final report. |
| `report.aux`, `report.bbl`, `report.blg`, `report.log`, `report.out` | LaTeX build artifacts. |

## Building

```
pdflatex report
bibtex   report
pdflatex report
pdflatex report
```

Compiles to 11 pages: body 1–8, references 9–11.
