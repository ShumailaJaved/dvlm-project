# connector/

Connector experts, router, and the top-level Mixture-of-Connectors module.

| File | Role |
|---|---|
| `expert_e1.py` | **E1** — frozen pretrained LLaVA two-layer MLP projector (576 tokens, 0 trainable params). |
| `expert_e2.py` | **E2** — BLIP-2-style Q-Former: K=32 learnable queries cross-attend over patch tokens (32 tokens, ~8.5M params). |
| `expert_e3.py` | **E3** — attention-pooled global token: collapses 576 patches into a single summary token (1 token, ~4.2M params). |
| `expert_e4.py` | **E4** — Question-Conditioned Gating Projector (QCGP). Projects question and patches into a shared d\_k=256 subspace, computes per-patch cosine relevance, and gates each patch's channels with a learnable temperature τ\_g (576 tokens, ~5.5M params). |
| `question_pooler.py` | Attention-pooled question vector **q** from frozen LLM token embeddings; shared by router and E4. |
| `router.py` | Two-layer MLP router (d\_r=64). Consumes **q** and emits a softmax over the four experts; W\_r⁽²⁾ is zero-init so routing begins uniform. Uses a straight-through estimator for top-1 selection. |
| `moc.py` | `MoC` wrapper: composes the question pooler, router, and four experts; routes one expert per sample and returns the projected visual tokens for the LLM. |

All new modules are kept in float32 to avoid softmax overflow; their forwards cast inputs to local weight dtype and return in the caller's dtype to interoperate with the 4-bit / fp16 LLaVA pipeline.
