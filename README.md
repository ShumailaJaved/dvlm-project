# Mixture of Connectors (MoC) for LLaVA-1.5-7B on ScienceQA

Final project for AI-623 *Deep Vision Language Models* at LUMS.

We replace LLaVA-1.5-7B's single question-agnostic MLP projector with a **Mixture of Connectors**: a question-conditioned router selects one of four structurally distinct connector experts per sample, trained jointly with a Switch-Transformer load-balancing loss under a QLoRA fine-tuning setup.

The four experts span the locality-versus-compression axis:

- **E1** — frozen pretrained LLaVA MLP (576 tokens)
- **E2** — BLIP-2-style Q-Former with 32 learned queries (32 tokens)
- **E3** — attention-pooled global token (1 token)
- **E4** — **Question-Conditioned Gating Projector (QCGP)**: per-patch cosine-similarity gating in a shared d=256 subspace (576 tokens), our key architectural contribution

## Headline Results (ScienceQA image-only)

| Model | Image Acc. | Text-only Acc. | Δᵥ |
|---|---|---|---|
| QLoRA E1 (pretrained projector, n=500) | 65.6% | 38.2% | +27.4% |
| QLoRA E2 (Q-Former, n=500) | 34.0% | 33.0% | +1.0% |
| QLoRA E3 (global token, n=500) | 39.0% | 33.5% | +5.5% |
| QLoRA E4 (QCGP, n=500) | 35.5% | 13.0% | +22.5% |
| **MoC best val (step 750, epoch 2, n=200)** | **73.8%** | 71.8% | +2.0% |
| MoC final (step 1900, epoch 5, n=500) | 62.2% | 70.2% | −8.0% |

**Main takeaway:** under a 6,218-sample budget, pretrained projector initialisation dominates architectural choice. The full MoC outperforms any single expert at epoch 2, but overfits past that point — validation-based early stopping is essential.

See [paper/report.pdf](paper/report.pdf) for the full write-up.

## Repository Layout

```
connector/        Expert (E1–E4) modules, question pooler, router, MoC wrapper
losses/           Switch-Transformer load-balancing loss
data/scienceqa/   ScienceQA image-only split cache
results/
  single/         Single-expert ablation logs, checkpoints, JSON metrics
  full moc/       MoC training log, routing log, best/final eval JSONs
  part_e/         Router distribution, τ_g convergence, QCGP heatmaps,
                  three-bucket failure analysis, subject breakdown
  figures/        Plots used in the report
checkpoints/      LLaVA-1.5-7B baseline weights
notebooks/        Colab notebooks for each training/eval phase
paper/            NeurIPS-style LaTeX report + midterm report + refs
setup_qlora.py    QLoRA model construction (4-bit NF4 + LoRA on q/k/v/o_proj)
train_single.py   Train one expert (E1/E2/E3/E4) in isolation
train_moc.py      Train the full MoC with router + load-balancing loss
eval_e1.py        Re-evaluate E1 on the full 2,017-sample test split
```

## Reproduction

All training was performed on a single A100 (Google Colab). The pipeline is:

1. **Baseline** — `notebooks/LLaVA_ScienceQA_baseline.ipynb` runs zero-shot LLaVA-1.5-7B and the midterm 2k-sample LoRA fine-tune.
2. **Single-expert ablations** — `python train_single.py --expert {E1,E2,E3,E4}` writes to `results/single/`.
3. **Full MoC** — `python train_moc.py` writes to `results/full moc/` (1,900 steps ≈ 5 epochs on 6,218 samples).

Key hyperparameters (see [paper/report.tex](paper/report.tex), Table 1): 4-bit NF4 quantization, LoRA rank 16, α=32, dropout 0.05, targets q/k/v/o_proj; LR 2×10⁻⁴ for connectors and router, 5×10⁻⁵ for LoRA adapters; effective batch size 16 (4 × 4 grad accum); λ\_lb = 0.01; QCGP subspace d\_k = 256; Q-Former K = 32 queries; router hidden d\_r = 64.

## Authors

Abdul Hafeez, Verda Batool, Shumaila Javed, Muhammad Fasih Tariq — Lahore University of Management Sciences.
