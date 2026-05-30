# notebooks/

Colab notebooks for each training and evaluation phase. All notebooks assume a single A100 runtime.

| Notebook | Phase |
|---|---|
| `LLaVA_ScienceQA_baseline.ipynb` | **Midterm baselines** — zero-shot LLaVA-1.5-7B and 2k-sample LoRA fine-tune. Produces `results/zeroshot_*.json` and `results/finetuned_*.json`. |
| `train_single_experts.ipynb` | **Single-expert ablations** (RQ2) — drives `train_single.py` for each of E1–E4 and writes per-expert logs, checkpoints, and metrics to `results/single/`. |
| `Full_MoC_Training.ipynb` | **Full MoC training** (RQ3) — drives `train_moc.py` for ~1,900 steps (≈5 epochs) on the 6,218-sample image-only train split. Writes to `results/full moc/`. |
| `Verifications_Task_A_C.ipynb` | **Implementation sanity checks** — runs `setup_qlora.py` (4-bit NF4 load + ScienceQA split sizes) and the `__main__` verification block in each connector module: `question_pooler.py`, `expert_e{1,2,3,4}.py`, `router.py`, `losses/load_balance.py`, and `moc.py --full`. Confirms output shapes, parameter counts, gate/attention sums, deterministic question pooling, non-zero router gradients, and end-to-end loss finiteness before launching training. |
| `Project Implementation.md` | Implementation notes covering data prep, model construction, training loop choices, and reproduction instructions. |
