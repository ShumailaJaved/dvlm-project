# Part E — Ablations and Analysis (results/part_e)

Scope note: this folder contains the Part E (Section 8) analyses **excluding the
expert single-expert ablations** (Task E-1 / Table E-1 rows for E2/E3/E4, which
were never run).

## Headline results

- **E-2 (router, overall, trained-router cumulative):** E1 30.5% · E2 20.5% · E3 28.4% · E4 20.6% — balanced, no collapse.
- **E-4 (failure buckets, 100 errors of E1, seed 42):** perception **48%** · reasoning **41%** · CoT-rescue **11%** (695 total errors). vs PAPO's ~67% perception → ScienceQA/E1 is less perception-bound, more reasoning-bound.
- **E-5 (τ_g):** 1.000 → 0.9985 (Δ −0.0011), stabilized — near-neutral gating sharpness.
- **E-6 (subject acc, E1, full 2,017 test):** overall Acc_image **65.5%** (matches documented E1 65.6% ✓), Acc_text 60.6%, Δv **+5.0%**.
  - By subject: natural science Δv +5.1% (largest), social +4.7%, language +4.6%.
  - By topic: physics **+14.1%**, geography +7.7%, us-history +9.5%; economics **−23.9%** and biology −2.0% (text beats image — text-heavy topics).

> **Note on E1 text accuracy:** our clean re-eval gives Acc_text 60.6% (Δv +5.0%), not the
> document's 38.2% (Δv 27.4%). The image accuracy matches (65.5% ≈ 65.6%); the document itself
> flags its original inline text-eval loop as buggy ("fixed post-hoc for E1"), so the +5.0% Δv
> from this independent pass is the more trustworthy figure.

## What is here

| Task | Output files | Status |
|------|--------------|--------|
| **E-2** Router distribution (overall) | `router_distribution.png`, `router_distribution.json` | ✅ from training logs |
| **E-4** Three-bucket failure analysis | `failure_buckets.png`, `failure_analysis.json`, `failure_cases.json` | ✅ on E1 (best model) |
| **E-5** τ_g convergence | `tau_g_convergence.png`, `tau_g_analysis.json` | ✅ from training logs |
| **E-6** Subject-level accuracy | `subject_delta_v.png`, `subject_breakdown.json` | ✅ on E1 (best model) |
| `e1_test_predictions.json` | per-sample E1 predictions (shared by E-4/E-6) | — |

## What is NOT here, and why

Two Part E tasks **cannot be produced** from the available artifacts:

- **E-3 — QCGP gate-weight heatmaps**: requires the *trained* E4 expert
  (`W_q`, `W_k`, `tau_g`, `W_g`). These weights were never saved.
- **E-2.2 / E-2.3 — subject-level and question-length routing breakdowns**:
  require running the *trained* router over the test set per-sample. The trained
  router weights were never saved.

### Root cause (a real bug in `train_moc.py`)

`train_moc.py` saves checkpoints with `model.save_pretrained(ckpt_dir)`. On a
PEFT model this writes **only the LoRA adapter**. The MoC modules (router,
QuestionPooler, E2, E3, E4) are trainable but were never added to PEFT's
`modules_to_save`, so they were not persisted. Inspecting
`ckpt_moc_best/adapter_model.safetensors` confirms 400 LoRA tensors and **zero**
MoC tensors. Reloading the checkpoint therefore yields a randomly-initialized
router and E4, which cannot reproduce the trained routing or gating.

**Fix for future runs** (not applied here to avoid a ~9.5 h retrain):
after each `model.save_pretrained(ckpt_dir)`, also call
`torch.save(moc.state_dict(), os.path.join(ckpt_dir, "moc_modules.pt"))`, and on
reload `moc.load_state_dict(torch.load(.../moc_modules.pt))`.

## Notes on the recovered results

- **E-2 (overall)** is the genuine trained router's *cumulative selection counts
  during training* (`routing_log_moc.csv`), not a fresh test-set pass — the
  closest recoverable proxy. Routing was balanced (no collapse): E1 30.5%,
  E2 20.5%, E3 28.4%, E4 20.6%.
- **E-5**: τ_g moved only marginally from its 1.0 init (≈ 0.9985 final), i.e. the
  model kept near-neutral gating sharpness on ScienceQA.
- **E-4 / E-6** use **E1** as the "best model from Table E-1": its test
  Acc_image (65.6%) exceeds the MoC's final 62.2%.
