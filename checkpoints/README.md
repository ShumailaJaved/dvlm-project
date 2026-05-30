# checkpoints/

Persisted model artifacts loaded by training and evaluation scripts.

```
llava-scienceqa-baseline/
  adapter_config.json
  adapter_model.safetensors    LoRA adapter weights from the midterm 2k-sample fine-tune
  non_lora_trainables.bin      non-LoRA trainables (e.g. projector)
  config.json
  tokenizer.model
  tokenizer_config.json
  special_tokens_map.json
```

This is the LLaVA-1.5-7B QLoRA baseline checkpoint used as the starting point for the single-expert ablations and the full MoC run.

**Caveat — MoC checkpoints not stored here.** `train_moc.py` uses `model.save_pretrained()`, which on a PEFT model writes only the LoRA adapter. The MoC modules (router, QuestionPooler, E2/E3/E4) were trainable but were never added to PEFT's `modules_to_save`, so they are not persisted. This blocks post-hoc test-set analyses of the trained router and τ\_g — see [results/part_e/README.md](../results/part_e/README.md) for details and the recommended fix.
