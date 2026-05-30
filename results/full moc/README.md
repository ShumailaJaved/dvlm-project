# results/full moc/

Full Mixture-of-Connectors training run (RQ3): all four experts + router + load-balancing loss jointly trained over the 6,218-sample image-only train split for ~1,900 steps (≈5 epochs).

```
log_moc.csv             per-step training loss (CE + λ_lb · L_lb)
routing_log_moc.csv     per-step router argmax decisions; basis for the cumulative
                        selection counts in results/part_e/router_distribution.json
moc_best.json           best-validation eval (step 750, end of epoch 2, n=200)
moc_final.json          final eval (step 1,900, end of epoch 5, n=500)
```

| Checkpoint | Image Acc. | Text-only Acc. | Δᵥ |
|---|---|---|---|
| **MoC best** (step 750, n=200) | **73.8%** | 71.8% | +2.0% |
| MoC final (step 1,900, n=500) | 62.2% | 70.2% | −8.0% |

The peak at epoch 2 exceeds the strongest single expert (E1: 65.6%) by 8.2 points, confirming that routing across heterogeneous experts is beneficial. Continued training erodes accuracy and inverts Δᵥ to negative, so **validation-based early stopping is essential**.

Routing remained balanced throughout — cumulative counts over 30,440 decisions: E1 30.5%, E2 20.5%, E3 28.4%, E4 20.6% (no collapse).

**Known limitation:** only the LoRA adapter was persisted at checkpoint time; the trained router and E2/E3/E4 weights were not saved, so post-hoc per-sample router behaviour on the test set is not recoverable. See [../part_e/README.md](../part_e/README.md) for the root cause and the recommended fix.

Reproduce with `python train_moc.py`.
