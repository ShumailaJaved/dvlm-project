# results/

All experimental outputs: training logs, evaluation JSONs, plots, and per-sample prediction dumps.

```
zeroshot_metrics.json       Midterm zero-shot LLaVA-1.5-7B accuracy + Δv
zeroshot_errors.json        Per-sample zero-shot mistakes
zeroshot_predictions.json   Per-sample zero-shot predictions
finetuned_metrics.json      Midterm 2k-sample LoRA fine-tune metrics
finetuned_errors.json       Per-sample midterm LoRA mistakes

single/                     Single-expert ablations (E1–E4 trained in isolation)
full moc/                   Full MoC training: logs, routing log, best/final eval
part_e/                     Router distribution, τ_g convergence, QCGP heatmaps,
                            three-bucket failure analysis, subject breakdown
figures/                    Plots referenced from the report
tables/                     LaTeX-ready table dumps
```

See each subdirectory's README for details. Headline numbers are summarised in [../README.md](../README.md).
