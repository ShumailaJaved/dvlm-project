# results/single/

Single-expert ablations (RQ2): each connector trained in isolation under the same QLoRA recipe used by the full MoC.

```
ckpt_E1/, ckpt_E2/, ckpt_E3/, ckpt_E4/   per-expert LoRA + connector checkpoints
log_E1.csv … log_E4.csv                  per-step training loss
single_expert_E1.json … _E4.json         evaluation on the 500-sample test subset
```

Eval JSONs report image accuracy, text-only accuracy (with `<image>` removed), and the visual grounding gap `Δv = Acc_image − Acc_text`.

| Expert | Image Acc. | Text-only Acc. | Δᵥ |
|---|---|---|---|
| E1 (pretrained MLP, frozen) | **65.6%** | 38.2% | +27.4% |
| E2 (Q-Former) | 34.0% | 33.0% | +1.0% |
| E3 (global token) | 39.0% | 33.5% | +5.5% |
| E4 (QCGP) | 35.5% | 13.0% | +22.5% |

All experts converge to similar final training loss (≈1.90–1.93 at step 750), so the 30-point gap between E1 and the random-init experts reflects **initialisation**, not optimisation difficulty or architectural weakness.

Reproduce with `python train_single.py --expert {E1,E2,E3,E4}`.
