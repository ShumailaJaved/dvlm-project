# Mixture of Connectors (MoC) — AI623 Final Project Implementation Assignment

> [!info] Context This assignment covers all remaining implementation and ablation work for the **Mixture of Connectors (MoC)** final project. It picks up exactly where the midterm submission left off: you have a working LLaVA-1.5-7B inference and LoRA fine-tuning pipeline on ScienceQA. Everything in this document is new.
> 
> **Base model:** LLaVA-1.5-7B (CLIP ViT-L/14 + Vicuna-1.5-7B, two-layer MLP connector) **Benchmark:** ScienceQA (image-only split, 6 218 train / 500 val / 1 000 test) **Hardware target:** Single GPU, Linux environment with QLoRA (4-bit) enabled

> [!tip] Working Document Keep a shared document with loss curves, accuracy tables, screenshots of gate heatmaps, and short written explanations as you go. These feed directly into the Results section of the final report.

> [!abstract] Structure of this Assignment
> 
> - **Part A (Infrastructure):** QLoRA migration to Linux and question-vector extraction
> - **Part B (Experts):** Implementing the four connector experts E1–E4
> - **Part C (Router + MoC):** Router, straight-through estimator, load-balancing loss, and full MoC integration into LLaVA's forward pass
> - **Part D (Training):** Single-expert runs and full MoC training
> - **Part E (Ablations + Analysis):** All experiments, visualizations, and failure analysis

> [!note] Read the analytical section (Section 2) before writing any code.

---

## 1 Allowed Tools and Constraints

- **Permitted:** PyTorch, HuggingFace `transformers`, `peft` (LoRA/QLoRA), `bitsandbytes`, `accelerate`, `datasets`, `torchvision`, standard scientific Python (`numpy`, `matplotlib`, `scikit-learn`, `einops`, `tqdm`).
- **Not permitted:** Any library providing a ready-made MoE routing layer, connector implementation, or multimodal training loop (no `open_flamingo`, no pre-built Q-Former modules).
- **Implement from scratch:** all four expert connectors (E1 reuse permitted as a wrapper), the question vector extractor, the router, the straight-through estimator, and the load-balancing loss.
- **LLM backbone:** LLaVA-1.5-7B (`liuhaotian/llava-v1.5-7b`). Keep CLIP frozen throughout. Apply QLoRA to LLM attention projections only (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
- **QLoRA config:** 4-bit NF4 quantization via `bitsandbytes`, LoRA rank $r = 16$, $\alpha = 32$, dropout 0.05.

---

## 2 Background and Notation

### 2.1 Notation Dictionary

|Symbol|Meaning|
|---|---|
|$Z^V \in \mathbb{R}^{N \times d_v}$|CLIP patch token sequence, $N = 576$, $d_v = 1024$|
|$d = 4096$|LLM (Vicuna-1.5-7B) hidden dimension|
|$d_k$|Shared projection subspace dimension for QCGP (suggested: 256)|
|$d_r$|Router hidden dimension (suggested: 64)|
|$\mathbf{q} \in \mathbb{R}^d$|Attention-pooled question vector|
|$\mathbf{q}_\text{pool} \in \mathbb{R}^d$|Learned pooling query vector|
|$U \in \mathbb{R}^{T \times d}$|LLM token embeddings for input question ($T$ tokens)|
|$\phi_k$|Expert $k$ connector: $\mathbb{R}^{N \times d_v} \to \mathbb{R}^{L_k \times d}$|
|$L_k$|Output sequence length of expert $k$: $L_1 = L_4 = 576$, $L_2 = 32$, $L_3 = 1$|
|$K = 4$|Number of experts|
|$\mathbf{r} \in \mathbb{R}^4$|Router softmax output (probability over experts)|
|$k^* = \arg\max_k r_k$|Selected expert index|
|$f_k$|Fraction of batch routed to expert $k$|
|$p_k$|Mean router softmax probability for expert $k$ over a batch|
|$\mathcal{L}_\text{lb}$|Load-balancing auxiliary loss|
|$\lambda_\text{lb}$|Load-balancing loss weight (suggested: 0.01)|
|$\tau_g$|Learnable temperature scalar in QCGP|
|$\Delta_v$|Visual grounding gap: $\text{Acc}_\text{image} - \text{Acc}_\text{text-only}$|

### 2.2 The Four Experts at a Glance

|Expert|Architecture|Output $L_k$|Question-aware?|Spatial info|
|---|---|---|---|---|
|$E_1$|Two-layer MLP (LLaVA baseline)|576|✗|Full|
|$E_2$|Q-Former cross-attention|32|✗|Compressed|
|$E_3$|Attention-pooled global token|1|✗|None|
|$E_4$|QCGP (our contribution)|576|✓|Full|

### 2.3 Resources

**Core Papers**

1. **LLaVA-1.5** — Liu et al., CVPR 2024. (_The model you are modifying._)
2. **BLIP-2 / Q-Former** — Li et al., ICML 2023. (_E2 design origin._)
3. **Flamingo / Perceiver Resampler** — Alayrac et al., NeurIPS 2022. (_E2 conceptual parent._)
4. **Honeybee** — Cha et al., CVPR 2024. (_Formalizes the locality–compression tradeoff E1–E4 span._)
5. **Switch Transformer** — Fedus et al., JMLR 2022. (_Top-1 routing + load-balancing loss source._)
6. **Sparse MoE** — Shazeer et al., ICLR 2017. (_Original MoE routing formulation._)
7. **MoCHA** — Pang et al., AAAI 2026. (_Closest existing work to MoC; know what differs._)
8. **PAPO** — Wang et al., 2026. (_Source of three-bucket failure taxonomy._)
9. **Modality Gap** — Liang et al., NeurIPS 2022. (_Justifies L2 normalization in QCGP._)

**Background Reading**

1. HuggingFace `bitsandbytes` integration guide (4-bit QLoRA)
2. HuggingFace PEFT: LoRA configuration and QLoRA tutorial
3. PyTorch `register_forward_hook` documentation (needed for question vector extraction)
4. `torch.nn.functional.scaled_dot_product_attention` documentation

---

## 3 Analytical Section

> [!note] Complete all problems below before writing any code. Problems 1–2 target Part B (experts); Problems 3–4 target Part C (router and loss); Problem 5 targets Part E (ablations and analysis). Problem 6 spans everything and is the most important for the viva.

---

### 3.1 Problem 1: The Locality–Compression Trade-off

The four experts span the extremes identified by Honeybee (Cha et al., 2024): locality preservation vs. token compression.

> [!question] Problem 1.1 — Output sequence lengths
> 
> **(a)** $E_1$ and $E_4$ both output $L = 576$ tokens. $E_2$ outputs $K = 32$. $E_3$ outputs $L = 1$. Explain in one sentence per expert what spatial information is preserved or lost, and why.
> 
> **(b)** The LLM processes the visual token sequence via self-attention with computational cost $O(L^2)$. Compute the ratio of attention cost for $E_3$ vs. $E_1$. What does this suggest about when routing to $E_3$ is computationally beneficial?
> 
> **(c)** ScienceQA contains diagram-heavy questions (circuit diagrams, food webs, labeled anatomy). For such questions, which expert would you _hypothesize_ the router will prefer, and why? Write two sentences connecting your hypothesis to a paper.

---

> [!question] Problem 1.2 — Q-Former compression irreversibility
> 
> In $E_2$, the key and value matrices are $K_v = Z^V W_K$ and $V_v = Z^V W_V$. The output is $\text{softmax}(Q K_v^\top / \sqrt{d}) V_v$ where $Q \in \mathbb{R}^{32 \times d}$ is a fixed learned parameter matrix.
> 
> **(a)** The output has shape $(32, d)$ regardless of input resolution. Show algebraically that no invertible linear map exists from $(32, d)$ back to $(576, d)$. Why does this make the compression irreversible?
> 
> **(b)** Query collapse is a failure mode where all 32 query vectors attend to the same patch token. Describe what happens to the output in this case. Propose one regularization term that penalizes query collapse.
> 
> **(c)** Compared to $E_1$, does $E_2$ have more or fewer trainable parameters? Count the parameters of $E_2$ for $d_v = 1024$, $d = 4096$, $K = 32$ (include $Q$, $W_K$, $W_V$). Compare to $E_1$'s two-layer MLP.

---

### 3.2 Problem 2: QCGP Design Choices

> [!question] Problem 2.1 — Why learned projections $W_q$, $W_k$ are necessary
> 
> The question vector $\mathbf{q} \in \mathbb{R}^{4096}$ lives in the LLM's embedding space. Each patch token $z_i^V \in \mathbb{R}^{1024}$ lives in CLIP's feature space.
> 
> **(a)** CLIP was trained with contrastive loss. The modality gap paper (Liang et al.) shows that contrastive training pushes image and text representations into disjoint cones on the unit sphere. Explain why computing $\mathbf{q}^\top z_i^V$ directly (without $W_q$, $W_k$) is geometrically meaningless even after L2 normalization of each vector individually.
> 
> **(b)** After applying $W_q$ and $W_k$, both projected vectors live in $\mathbb{R}^{d_k}$. Explain what the learned $W_q$ and $W_k$ are doing geometrically — what does a "shared subspace" mean here?
> 
> **(c)** L2 normalization is applied after projection: $\hat{u} = W_q \mathbf{q} / |W_q \mathbf{q}|_2$. Without this normalization, the cosine similarity degenerates due to the cone effect. In one sentence, state what the cone effect is and why L2 normalization mitigates it.

---

> [!question] Problem 2.2 — Learnable temperature $\tau_g$
> 
> The per-patch relevance score is $\alpha_i = \text{softmax}_i(\hat{u}^\top \hat{k}_i / \tau_g)$.
> 
> **(a)** Show what happens to $\alpha_i$ as $\tau_g \to 0$ and $\tau_g \to \infty$. What does each limit mean for the gating behavior?
> 
> **(b)** If $\tau_g$ were fixed at 1.0 instead of learned, which ScienceQA question types might suffer? Think about questions requiring attention to a single label vs. questions requiring understanding of the whole diagram.
> 
> **(c)** $\tau_g$ is a scalar `nn.Parameter` initialized to 1.0. Should it be constrained to stay positive? If so, how would you enforce this without adding a hard constraint?

---

> [!question] Problem 2.3 — Vector-valued gate $\mathbf{g}_i$
> 
> The gate is $\mathbf{g}_i = \sigma(\alpha_i W_g + \mathbf{b}_g)$ where $W_g \in \mathbb{R}^{d_v}$ and $\mathbf{b}_g \in \mathbb{R}^{d_v}$.
> 
> **(a)** This is a vector-valued gate (one scalar per channel). Compare to a scalar gate where $\mathbf{g}_i = \alpha_i \cdot \mathbf{1}$. What additional expressive power does the vector gate provide?
> 
> **(b)** The Sigmoid keeps $\mathbf{g}_i \in (0, 1)^{d_v}$. What would go wrong with a ReLU gate instead?
> 
> **(c)** Note that $\alpha_i W_g$ is scalar-vector multiplication (not matrix multiplication). This means all channels of $\mathbf{g}_i$ are modulated by the same relevance scalar $\alpha_i$, just scaled differently per channel. Is this a limitation? Describe a question type where channel-independent gating might fail.

---

### 3.3 Problem 3: Router and Straight-Through Estimator

> [!question] Problem 3.1 — Top-1 routing and the straight-through estimator
> 
> At inference, the router selects $k^* = \arg\max_k r_k$ and routes all patches through $\phi_{k^*}$ only.
> 
> **(a)** The $\arg\max$ operation is not differentiable. Without any approximation, what value does $\partial k^* / \partial \mathbf{r}$ take? Why does this block gradient flow to the router weights?
> 
> **(b)** The straight-through estimator (Bengio et al., 2013) bypasses this by passing gradients as if the operation were the identity. In one sentence, state what assumption this makes and when it is a reasonable approximation.
> 
> **(c)** Concretely: during the forward pass, you compute `k_star = torch.argmax(r)` (discrete, no gradient). During the backward pass, the gradient flows through `r` directly (as if argmax didn't happen). Write pseudocode showing how you would implement this in PyTorch using `detach()` and direct addition.

---

> [!question] Problem 3.2 — Router collapse
> 
> A degenerate routing distribution is one where the router always selects the same expert for every input.
> 
> **(a)** Why is routing collapse a problem for a mixture of _structurally different_ experts? Answer in terms of what happens to $E_2$, $E_3$ if they are never selected during training.
> 
> **(b)** The router is a 2-layer MLP taking $\mathbf{q}$ as input. If the LLM is initialized from a pretrained checkpoint and $\mathbf{q}_\text{pool}$ is initialized randomly, the early values of $\mathbf{q}$ will be nearly random. Explain why this makes early routing nearly uniform — is this desirable?
> 
> **(c)** Suppose during training you find that $f_1 = 0.95$ (E1 selected 95% of the time). List two interventions other than the load-balancing loss that could fix this.

---

### 3.4 Problem 4: Load-Balancing Loss

The load-balancing loss from Switch Transformer (Fedus et al., 2022) is:

$$\mathcal{L}_\text{lb} = K \sum_{k=1}^{K} f_k \cdot p_k$$

where $f_k = \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}[k^*_b = k]$ is the fraction of batch samples dispatched to expert $k$, and $p_k = \frac{1}{B}\sum_{b=1}^{B} r_{b,k}$ is the mean router softmax probability for expert $k$.

> [!question] Problem 4.1 — Understanding the loss
> 
> **(a)** Show that when routing is perfectly balanced ($f_k = p_k = 1/K$ for all $k$), the loss equals 1. Show that when routing collapses to a single expert ($f_1 = 1$, $f_k = 0$ for $k > 1$), the loss is at most $K \cdot p_1 \leq K$. Interpret the range $[1, K]$.
> 
> **(b)** $f_k$ is computed from the discrete $\arg\max$ and has no gradient. $p_k$ is computed from the continuous softmax $\mathbf{r}$ and does have a gradient. Explain why this product $f_k \cdot p_k$ is a valid training signal despite $f_k$ being non-differentiable. Which variable is carrying the gradient?
> 
> **(c)** The total loss is $\mathcal{L} = \mathcal{L}_\text{CE} + \lambda_\text{lb},\mathcal{L}_\text{lb}$. What happens if $\lambda_\text{lb}$ is set too high? Describe the failure mode in terms of accuracy and routing behavior.

---

### 3.5 Problem 5: Ablations and Evaluation

> [!question] Problem 5.1 — What single-expert ablations tell you
> 
> **(a)** You will train E4 (QCGP) alone and compare to E1 (MLP baseline). State the specific claim your paper makes that this comparison is designed to test. Write it as a falsifiable hypothesis.
> 
> **(b)** E3 (global token) produces only 1 visual token. Under what conditions would you expect E3 to outperform E2 (Q-Former, 32 tokens) on ScienceQA? Think about what information ScienceQA multiple-choice questions sometimes require.
> 
> **(c)** If the full MoC underperforms the best single expert, what does this tell you about the router? What is the minimal diagnostic you would run to determine whether the failure is in routing or in expert interaction?

---

> [!question] Problem 5.2 — Visual grounding gap interpretation
> 
> Recall $\Delta_v = \text{Acc}_\text{image} - \text{Acc}_\text{text-only}$.
> 
> **(a)** Your midterm found that LoRA fine-tuning increased $\Delta_v$ from 7.5% to 11.6% without improving image accuracy. Interpret this finding: did the model get better at understanding images?
> 
> **(b)** For QCGP (E4), you would hypothesize a larger $\Delta_v$ than E1. Explain why — connect it to the gate $\mathbf{g}_i$ suppressing language-prior-driven patches.
> 
> **(c)** Is it possible for a model to have very high image accuracy but $\Delta_v \approx 0$? What does this scenario mean, and is it desirable?

---

> [!question] Problem 5.3 — Three-bucket failure taxonomy
> 
> The PAPO paper (Wang et al., 2026) classifies MLLM errors into three buckets: perception error (wrong visual reading), reasoning error (correct perception, wrong conclusion), and CoT rescue (chain-of-thought corrects an initially wrong answer).
> 
> **(a)** Which bucket is the QCGP most directly designed to reduce? Justify your answer using the gate equation.
> 
> **(b)** For a model with high $\Delta_v$ but many reasoning errors, what architectural change (not covered in this project) would be the logical next intervention?
> 
> **(c)** Why does the PAPO taxonomy use CoT rescue as a separate bucket rather than folding it into "reasoning error"? What does it tell you about the model's internal state when CoT rescues a wrong answer?

---

### 3.6 Problem 6: Synthesis

> [!question] Problem 6.1 — MoC vs. prior work (exam-style comparison) Fill in the table below. One precise sentence or formula per cell.
> 
> ||MoE-LLaVA|CuMo / MoCHA|MoVA|MoC (ours)|
> |---|---|---|---|---|
> |Where is routing applied?|||||
> |Expert architectures|||||
> |Vision encoder(s)|||||
> |Question-conditioned routing?|||||
> |Spatial info preserved?|||||

---

> [!question] Problem 6.2 — End-to-end gradient flow
> 
> Trace the gradient path for a single training step of the full MoC system, starting from the cross-entropy loss $\mathcal{L}_\text{CE}$.
> 
> **(a)** Write out every module the gradient passes through, in reverse order, from $\mathcal{L}_\text{CE}$ back to the first trainable parameter. Which modules are frozen?
> 
> **(b)** The CLIP encoder is frozen. The LLM backbone uses QLoRA (only LoRA adapters are trainable, base weights are 4-bit quantized and frozen). The connector experts are fully trainable. Estimate the fraction of total model parameters that receive a gradient update per step.
> 
> **(c)** The question vector $\mathbf{q}$ is computed from LLM token embeddings via $\mathbf{q}_\text{pool}$. Does $\mathbf{q}_\text{pool}$ receive a gradient from $\mathcal{L}_\text{CE}$? Trace the path. Does it receive a gradient from $\mathcal{L}_\text{lb}$?

---

## 4 Coding Tasks — Part A: Infrastructure

### 4.1 Task A-0: QLoRA Environment Migration

> [!abstract] Task A-0: Linux + 4-bit QLoRA Setup All remaining experiments run in a Linux environment with `bitsandbytes` 4-bit quantization enabled. This unlocks training on the full 6 218-sample training set.

**A-0.1 Environment Setup**

Run these installation cells in order in Colab, then **restart the runtime manually** (Runtime → Restart Session) before importing anything else:

```bash
pip install "bitsandbytes>=0.41" "peft>=0.9" accelerate "transformers>=4.37" datasets einops tqdm
pip install --upgrade torchao
pip install --no-deps -e ./LLaVA
```

> [!warning] LLaVA MUST be installed with `--no-deps`
> Running `pip install -e ./LLaVA` without `--no-deps` causes pip to downgrade `torch` and `torchao` to versions incompatible with the Colab GPU runtime. This silently breaks CUDA in every subsequent cell — `torch.cuda.is_available()` returns `True` but GPU ops crash at runtime. Always use `--no-deps` and verify with `import torch; print(torch.cuda.is_available())` after the runtime restart.

Load LLaVA-1.5-7B in 4-bit NF4 mode using an explicit `BitsAndBytesConfig`:

```python
from transformers import BitsAndBytesConfig
from llava.model.builder import load_pretrained_model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="liuhaotian/llava-v1.5-7b",
    model_base=None,
    model_name="llava-v1.5-7b",
    quantization_config=bnb_config,   # ← pass config explicitly
)
```

> [!warning] Do NOT use `load_4bit=True`
> LLaVA's `load_pretrained_model` has a convenience `load_4bit=True` flag. On newer `transformers` versions this flag propagates as an unexpected keyword argument into the model `__init__`, raising a `TypeError`. Use `quantization_config=bnb_config` directly instead — it is explicit and version-stable.

Apply `prepare_model_for_kbit_training(model)` **before** adding LoRA adapters. For MoC training the ordering is more involved — see Task C-3 for the exact five-step sequence.

**Verify:** Print GPU memory before and after loading. The 4-bit base model fits under 6 GB. Full training footprint with gradient checkpointing stays under 15 GB on an A100 (observed peak: 12.4 GB for E1).

**A-0.2 Full Dataset Pipeline**

Load all 6 218 training samples from `derek-thomas/ScienceQA`. Apply the image-only filter (`example["image"] is not None`):

```python
from datasets import load_dataset

raw   = load_dataset("derek-thomas/ScienceQA")
train = raw["train"].filter(lambda x: x["image"] is not None)
val   = raw["validation"].filter(lambda x: x["image"] is not None)
test  = raw["test"].filter(lambda x: x["image"] is not None)
```

Actual image-only split sizes (confirmed on HuggingFace Hub):

| Split | Count |
|---|---|
| Train | 6,218 |
| Validation | 2,097 |
| Test | 2,017 |

> [!warning] Dataset split sizes differ from the midterm spec
> The assignment states "500 val / 1 000 test" — those were _evaluation subsets_ used in the midterm, not the actual dataset splits. The real image-only splits are 2,097 val and 2,017 test. In the training scripts, `--n_eval 200` or `--n_eval 500` flags evaluate on a random subset for speed during development. Final reported numbers must use the full test split (2,017 samples).

Gradient step count for 2 epochs with physical batch size 4 and 4 gradient-accumulation steps (effective batch 16): `ceil(6218 / 16) × 2 = 389 × 2 = 778` optimizer steps.

> [!warning] Common Pitfalls
>
> - `bitsandbytes` requires CUDA 11.x+ and a Linux environment. Do not attempt on Windows or macOS.
> - Install LLaVA with `--no-deps` or CUDA silently breaks due to torch/torchao version conflicts.
> - Use `quantization_config=bnb_config`, not `load_4bit=True` — the latter breaks on newer `transformers`.
> - Call `prepare_model_for_kbit_training` _before_ `get_peft_model` — wrong order causes gradient issues.
> - Restart the Colab runtime after the `pip install` cells and before importing any module.
> - The CLIP encoder and the original MLP projector must remain frozen throughout. Only LoRA adapters and new connector modules are trainable.

**→ Report mapping:** Methodology section — update the "LoRA Fine-Tuning" subsection with the new QLoRA configuration and training setup. The "Current Limitations" paragraph from the midterm must be removed.

---

### 4.2 Task A-1: Question Vector Extraction

> [!abstract] Task A-1: Attention-Pooled Question Vector $\mathbf{q}$ This module is shared by both the router (Part C) and E4/QCGP (Part B). Build and verify it before writing any expert code.

**A-1.1 Implementation**

Implement `QuestionPooler(d: int)` as an `nn.Module`:

|Parameter|Shape|Init|
|---|---|---|
|`q_pool`|$(d,)$|`nn.Parameter`, normal init with std $1/\sqrt{d}$|

Forward pass (inputs: `U` — LLM token embeddings of shape $(T, d)$):

$$\mathbf{q} = \text{softmax}!\left(\frac{\mathbf{q}_\text{pool}^\top U^\top}{\sqrt{d}}\right) U \in \mathbb{R}^d$$

where the softmax is over the $T$ token positions.

**A-1.2 Integration into LLaVA**

The LLM token embeddings $U$ for the question tokens are available from `model.model.embed_tokens(question_input_ids)` before the LLM forward pass. Extract them, pass through `QuestionPooler`, and store the resulting $\mathbf{q}$. This vector is then passed to the router and to E4.

**A-1.3 Verification**

1. Feed two clearly different questions (e.g., "What is labeled A?" vs. "What color is the background?"). Confirm $|\mathbf{q}_1 - \mathbf{q}_2|_2 > 0$ (the pooled vectors differ).
2. Feed the same question twice. Confirm $|\mathbf{q}_1 - \mathbf{q}_2|_2 = 0$ (deterministic).
3. Print the shape of $\mathbf{q}$: confirm `torch.Size([4096])`.

> [!warning] Common Pitfalls
> 
> - Do **not** pass $\mathbf{q}$ through the LLM layers — extract only the embedding lookup output.
> - `embed_tokens` requires integer token IDs, not float embeddings. Pass `question_input_ids` directly.
> - Keep `QuestionPooler` in `float16` to match the LLM's compute dtype.

**→ Report mapping:** Methodology section — the $\mathbf{q}$ formula is already in the paper (MoC subsection preamble). No new writing needed; just confirm the implementation matches the equation.

---

## 5 Coding Tasks — Part B: Expert Connectors

> [!abstract] Part B Overview Implement all four expert connectors as standalone `nn.Module` classes. Each takes `Z_V` (shape `(576, 1024)`) as input and outputs visual tokens in LLM space (shape `(L_k, 4096)`). E4 additionally takes `q` (shape `(4096,)`).

---

### 5.1 Task B-E1: MLP Expert (Reuse)

> [!abstract] Task B-E1: Wrap existing LLaVA MLP projector as $E_1$

LLaVA-1.5's pretrained two-layer MLP projector already exists in the model as `model.model.mm_projector`. Wrap it:

```python
class ExpertE1(nn.Module):
    def __init__(self, pretrained_mlp):
        super().__init__()
        self.mlp = pretrained_mlp  # frozen by default

    def forward(self, Z_V):
        # Z_V: (576, 1024) -> output: (576, 4096)
        return self.mlp(Z_V)
```

**Verification:** Pass a random tensor of shape `(576, 1024)`. Confirm output shape is `(576, 4096)`. Confirm no new parameters are added (`sum(p.numel() for p in e1.parameters() if p.requires_grad) == 0`).

**→ Report mapping:** Methodology — "Expert 1 ($E_1$)" equation is already written. No changes needed.

---

### 5.2 Task B-E2: Q-Former Expert

> [!abstract] Task B-E2: Cross-attention compression to 32 tokens

Implement `ExpertE2(d_v: int, d: int, K: int)`:

|Parameter|Shape|Init|
|---|---|---|
|`Q`|$(K, d)$|`nn.Parameter`, Xavier uniform|
|`W_K`|Linear$(d_v, d)$, no bias|Kaiming uniform|
|`W_V`|Linear$(d_v, d)$, no bias|Kaiming uniform|

Forward (input: `Z_V` of shape `(N, d_v)`):

$$\phi_2(Z^V) = \text{softmax}!\left(\frac{Q,(Z^V W_K)^\top}{\sqrt{d}}\right)(Z^V W_V) \in \mathbb{R}^{K \times d}$$

Use `torch.nn.functional.scaled_dot_product_attention` for efficiency.

**Verification:**

1. Output shape: `(32, 4096)`.
2. Attention weights sum to 1 over the 576 patch dimension for each of the 32 queries.
3. Count trainable parameters: $K \cdot d + 2 \cdot d_v \cdot d$ = $32 \times 4096 + 2 \times 1024 \times 4096$. Confirm against `sum(p.numel() for p in e2.parameters())`.

> [!warning] Common Pitfalls
> 
> - `Q` is a fixed-size learnable parameter — it is **not** derived from the input. Do not confuse with question vector $\mathbf{q}$.
> - Do not apply layer normalization to the output unless you verify it doesn't interfere with the LLM's expected input scale.
> - E2's 32-token output requires the LLM attention mask to be adjusted (handled in Part C).

**→ Report mapping:** Methodology — "$E_2$" equation is already written. Confirm implementation matches $\phi_2$ exactly.

---

### 5.3 Task B-E3: Attention-Pooled Global Token Expert

> [!abstract] Task B-E3: Collapse all patches to a single summary token

Implement `ExpertE3(d_v: int, d: int)`:

|Parameter|Shape|Init|
|---|---|---|
|`w`|$(d_v,)$|`nn.Parameter`, normal init|
|`W_E3`|Linear$(d_v, d)$, with bias|Kaiming uniform|

Forward (input: `Z_V` of shape `(N, d_v)`):

$$\mathbf{c} = \sum_{i=1}^N \text{softmax}_i!\left(\frac{\mathbf{w}^\top z_i^V}{\sqrt{d_v}}\right) z_i^V, \qquad \phi_3(Z^V) = W_{E_3},\mathbf{c} \in \mathbb{R}^{1 \times d}$$

Note: output must be reshaped to `(1, d)` not `(d,)` — the LLM expects a sequence dimension.

**Verification:**

1. Output shape: `(1, 4096)`.
2. Attention weights (softmax over 576 patches) sum to 1. Print the index of the highest-weight patch for a sample image — it should shift meaningfully between different images.
3. Total trainable parameters: $d_v + d_v \cdot d + d$ (weights + bias). Confirm.

**→ Report mapping:** Methodology — "$E_3$" equation is already written. Confirm implementation matches $\phi_3$.

---

### 5.4 Task B-E4: QCGP Expert

> [!abstract] Task B-E4: Question-Conditioned Gating Projector — the core contribution

Implement `ExpertE4(d_v: int, d: int, d_k: int)`:

|Parameter|Shape|Init|
|---|---|---|
|`W_q`|Linear$(d, d_k)$, no bias|Kaiming uniform|
|`W_k`|Linear$(d_v, d_k)$, no bias|Kaiming uniform|
|`tau_g`|`nn.Parameter(torch.ones(1))`|1.0|
|`W_g`|$(d_v,)$ `nn.Parameter`|normal, std 0.01|
|`b_g`|$(d_v,)$ `nn.Parameter`|zeros|
|`W_E4`|Linear$(d_v, d)$, with bias|Kaiming uniform|

Forward (inputs: `Z_V` shape `(N, d_v)`, `q` shape `(d,)`):

$$\hat{u} = \frac{W_q \mathbf{q}}{|W_q \mathbf{q}|_2}, \quad \hat{K} = \frac{W_k Z^{V\top}}{|W_k Z^{V\top}|_2} \in \mathbb{R}^{d_k \times N}$$

$$\alpha_i = \text{softmax}_i\left(\frac{\hat{u}^\top \hat{k}_i}{\tau_g}\right), \quad \mathbf{g}_i = \sigma(\alpha_i W_g + \mathbf{b}_g)$$

$$\phi_4(Z^V)_i = W_{E_4}(z_i^V \odot \mathbf{g}_i) \in \mathbb{R}^d$$

where $\odot$ is element-wise multiplication and $\sigma$ is Sigmoid.

**Implementation note:** $\alpha_i W_g$ is scalar-vector multiplication: `alpha[i] * self.W_g` where `alpha` has shape `(N,)` and `W_g` has shape `(d_v,)`. Use broadcasting: `alpha.unsqueeze(-1) * self.W_g`.

**Verification:**

1. Output shape: `(576, 4096)`.
2. All values of $\mathbf{g}_i$ are in $(0, 1)$ (Sigmoid output). Confirm: `assert (output_g > 0).all() and (output_g < 1).all()`.
3. $\alpha$ sums to 1 over the 576 patches. Confirm: `assert abs(alpha.sum() - 1.0) < 1e-5`.
4. $\tau_g$ is positive. It should stay positive throughout training — add a check after each step: `assert model.qcgp.tau_g.item() > 0`.
5. Feed two different questions with the same image. Confirm the gate vectors $\mathbf{g}$ differ: $|\mathbf{g}^{(1)} - \mathbf{g}^{(2)}|_F > 0$.

> [!warning] Common Pitfalls
> 
> - L2 normalization must be applied **after** projection, not before. `F.normalize(self.W_q(q), dim=-1)`, not `F.normalize(q)`.
> - Do not mix up `W_g` (shape `(d_v,)`, expansion vector) and `W_k` (shape `(d_k, d_v)`, projection). These are entirely different parameters.
> - `tau_g` can go negative if not monitored. If this happens, add `tau_g = tau_g.abs() + 1e-6` or use `F.softplus`.

**→ Report mapping:** Methodology — "$E_4$ (QCGP)" equations are already in the paper. Fix the dead equation labels: change `equation*` to numbered `equation` for the QCGP gate formula — this is the paper's core contribution and must be referenceable.

---

## 6 Coding Tasks — Part C: Router, Loss, and MoC Integration

### 6.1 Task C-1: Router

> [!abstract] Task C-1: Lightweight question-conditioned router

Implement `MoCRouter(d: int, d_r: int, K: int)`:

|Parameter|Shape|Init|
|---|---|---|
|`W1`|Linear$(d, d_r)$|Kaiming uniform|
|`W2`|Linear$(d_r, K)$|zeros (uniform routing at init)|

Forward (input: `q` shape `(d,)`):

$$\mathbf{r} = \text{softmax}(W_r^{(2)},\text{GELU}(W_r^{(1)}\mathbf{q})) \in \mathbb{R}^K$$

At inference: `k_star = torch.argmax(r)`.

**Straight-through estimator:** During training, select expert $k^*$ using argmax (no gradient), but allow $\mathbf{r}$ to receive gradients from $\mathcal{L}_\text{lb}$:

```python
# Forward pass: discrete expert selection (no gradient through argmax)
k_star = torch.argmax(r.detach())

# Backward pass: L_lb uses r directly (has gradient)
# L_CE uses the output of phi_{k_star} directly (no gradient to r from CE)
```

**Initialization note:** Initialize `W2` to zeros so that at the start of training, all experts receive equal probability ($r_k = 1/K = 0.25$). This prevents one expert from dominating early.

**Verification:**

1. At init, confirm `r ≈ [0.25, 0.25, 0.25, 0.25]` for any input `q`.
2. After one gradient step on $\mathcal{L}_\text{lb}$, confirm `W2.grad` is non-zero (the loss correctly reaches the router).
3. Confirm `k_star` is an integer in `{0, 1, 2, 3}`.

**→ Report mapping:** Methodology — router equation is already in the paper. Add router matrix dimensions ($W_r^{(1)} \in \mathbb{R}^{d_r \times d}$, $W_r^{(2)} \in \mathbb{R}^{4 \times d_r}$) — currently missing from the paper.

---

### 6.2 Task C-2: Load-Balancing Loss

> [!abstract] Task C-2: Auxiliary loss to prevent router collapse

Implement `load_balancing_loss(router_probs, expert_indices, K)`:

- `router_probs`: tensor of shape `(B, K)` — softmax outputs over the batch
- `expert_indices`: tensor of shape `(B,)` — selected expert per sample (from argmax)
- `K`: number of experts (4)

$$f_k = \frac{1}{B}\sum_{b=1}^B \mathbf{1}[k^*_b = k], \qquad p_k = \frac{1}{B}\sum_{b=1}^B r_{b,k}$$

$$\mathcal{L}_\text{lb} = K \sum_{k=1}^K f_k \cdot p_k$$

Note: `f_k` uses `.detach()` (it is computed from discrete indices). `p_k` must **not** be detached — it carries the gradient to the router.

Total loss:

$$\mathcal{L} = \mathcal{L}_\text{CE} + \lambda_\text{lb},\mathcal{L}_\text{lb}, \qquad \lambda_\text{lb} = 0.01$$

**Verification:**

1. With perfectly balanced routing: confirm $\mathcal{L}_\text{lb} = 1.0$.
2. With fully collapsed routing (all samples to expert 0): confirm $\mathcal{L}_\text{lb} = K \cdot p_0 \leq K$.
3. Confirm `f_k.requires_grad == False` and `p_k.requires_grad == True`.

**→ Report mapping:** Methodology — load-balancing loss equation is already in the paper. Fix: the label `\label{eq:lb_loss}` inside `equation*` is dead — switch to numbered `equation`.

---

### 6.3 Task C-3: MoC Integration

> [!abstract] Task C-3: Replace the static MLP projector with the full MoC system

This is the most architecturally sensitive task. The approach is to subclass `LlavaLlamaForCausalLM` and override `prepare_inputs_labels_for_multimodal` to call the MoC routing system in place of `mm_projector`. The implementation lives in `connector/moc.py`.

**Step 1: Assemble the MoC module**

```python
class MixtureOfConnectors(nn.Module):
    def __init__(self, e1, e2, e3, e4, router, pooler):
        super().__init__()
        self.experts = nn.ModuleList([e1, e2, e3, e4])
        self.router  = router
        self.pooler  = pooler

    def forward(self, Z_V, question_embeddings):
        # Detect the module's current compute dtype from the router's weight.
        # This is necessary because prepare_model_for_kbit_training upcasts
        # all 1D parameters (biases, layer norms, q_pool) to float32, but
        # the LLM forward operates in float16. The MoC is re-cast to float16
        # after kbit prep, but code that runs before that re-cast must handle
        # mixed dtypes gracefully.
        _dtype = self.router.W1.weight.dtype
        Z_V                = Z_V.to(dtype=_dtype)
        question_embeddings = question_embeddings.to(dtype=_dtype)

        q      = self.pooler(question_embeddings)   # (d,)
        r      = self.router(q)                     # (4,)
        k_star = torch.argmax(r.detach()).item()    # STE — no grad through argmax

        if k_star == 3:   # E4 needs q as an extra argument
            V = self.experts[k_star](Z_V, q)
        else:
            V = self.experts[k_star](Z_V)

        return V, r, k_star
```

**Step 2: Handle variable-length visual token sequences**

$E_1$ and $E_4$ output 576 tokens. $E_2$ outputs 32. $E_3$ outputs 1. Build `inputs_embeds` dynamically based on `V.shape[0]`. Do **not** pad shorter sequences to 576 — the LLM's self-attention handles variable-length sequences natively via the attention mask.

When concatenating visual tokens `V` with text embeddings inside `prepare_inputs_labels_for_multimodal`, cast `V` back to the LLM's dtype before the concat:

```python
cur_new_input_embeds.append(V.to(device=self.device, dtype=U.dtype))
```

**Step 3: Subclass and upgrade — CRITICAL ordering**

Subclass `LlavaLlamaForCausalLM` to override `prepare_inputs_labels_for_multimodal`. The upgrade is done with a `__class__` reassignment via `upgrade_to_moc(model)` so that the already-loaded weights are reused without redownloading the model. This approach is used to avoid reloading the model from disk.

The five steps **must** happen in this exact order:

```python
from connector.moc import upgrade_to_moc, build_moc

# 1. Class surgery — while the model is still LlavaLlamaForCausalLM,
#    before PEFT wraps it in a PeftModelForCausalLM shell.
model = upgrade_to_moc(base_model)

# 2. Build and attach MoC — mm_projector is still directly accessible here.
moc = build_moc(model).to(device)   # do NOT call .half() yet
model.set_moc(moc)

# 3. kbit prep — MUST come before get_peft_model.
#    This upcasts all 1D parameters (biases, LayerNorm weights, q_pool,
#    router biases) to float32 for gradient-checkpointing stability.
model = prepare_model_for_kbit_training(model)

# 4. Re-cast MoC to float16 — because the LLaVA forward operates in float16
#    (bnb_4bit_compute_dtype=float16) and prepare_model_for_kbit_training
#    just upcasted all MoC params to float32, causing dtype mismatches.
moc.half()

# 5. PEFT wrapping — must be last.
lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj","k_proj","v_proj","o_proj"],
                      bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_cfg)
```

> [!warning] Why this ordering is not negotiable
>
> - **`upgrade_to_moc` before `get_peft_model`**: `upgrade_to_moc` assigns `model.__class__ = MoCLlavaForCausalLM`. If you call `get_peft_model` first, the model becomes a `PeftModelForCausalLM` wrapper that has no `self.model` attribute at its top level. The subsequent `__class__` reassignment then targets the wrong object, and calling `model.model.embed_tokens(...)` raises `AttributeError: 'MoCLlavaForCausalLM' object has no attribute 'model'`.
>
> - **`moc.half()` after `prepare_model_for_kbit_training`**: `prepare_model_for_kbit_training` calls `model.enable_input_require_grads()` and then upcasts every 1D parameter in the model to float32 (for numerical stability during gradient checkpointing). This includes MoC's `q_pool`, router biases, `b_g`, and `tau_g`. Without the `moc.half()` call immediately after, the MoC parameters are float32 while the LLM produces float16 activations — the concat of visual tokens and text embeddings raises a dtype mismatch error.

**Verification:**

1. Run a single forward pass with a test image and question. Confirm loss is a finite scalar (not NaN, not inf).
2. Confirm that 100 random question vectors (after perturbing W2) cause diverse routing — not all to the same expert. (At init with `W2=0`, all route to expert 0 by argmax tie-breaking; this is expected and load-balancing will fix it during training.)
3. Print the full parameter count using `print_parameter_counts(model, moc)`: frozen (CLIP + base LLM), quantized (4-bit LLM weights), and trainable (LoRA + connector experts + router + pooler).

Run the full verification with: `python connector/moc.py --full` (requires Colab A100, loads the real model).

> [!warning] Common Pitfalls
>
> - `prepare_model_for_kbit_training` upcasts ALL 1D params to float32. Always call `moc.half()` immediately after — skipping this causes dtype mismatch errors on the first forward pass.
> - `upgrade_to_moc` must be called on the raw `LlavaLlamaForCausalLM`, not on a `PeftModel`. Calling it after `get_peft_model` causes `AttributeError: no attribute 'model'`.
> - `question_embeddings` passed to the pooler must come from `embed_tokens`, **before** the LLM's positional encoding is added.
> - If NaN loss appears after adding QCGP: the L2 normalization in E4 can blow up if `W_q(q)` becomes a near-zero vector. Add `eps=1e-8` to `F.normalize` (already in the reference implementation).
> - Inside `MixtureOfConnectors.forward`, always detect the module's compute dtype from `self.router.W1.weight.dtype` and cast inputs accordingly — do not assume float16.

**→ Report mapping:** Methodology — add a new architecture figure showing the full MoC pipeline (router + four experts) to replace or supplement the baseline-only figure. The caption should replace the placeholder "MoC will replace the MLP" annotation.

---

## 7 Coding Tasks — Part D: Training

### 7.1 Hyperparameter Reference Table

|Hyperparameter|Value|
|---|---|
|QLoRA rank $r$|16|
|QLoRA $\alpha$|32|
|QLoRA dropout|0.05|
|LoRA targets|`q_proj`, `k_proj`, `v_proj`, `o_proj`|
|Optimizer|AdamW|
|Learning rate (connector experts + router)|2e-4|
|Learning rate (LoRA adapters)|5e-5|
|Scheduler|Cosine with 3% warmup|
|Effective batch size|16|
|Epochs|2–3|
|Training samples|6 218 (full set)|
|Load-balancing weight $\lambda_\text{lb}$|0.01|
|Max sequence length|512|

Use two separate parameter groups with different learning rates — connector experts learn faster than the LoRA adapters.

---

### 7.2 Task D-1: Single-Expert Training Runs

> [!abstract] Task D-1: Train and evaluate each expert in isolation

For each expert $k \in {1, 2, 3, 4}$: plug `phi_k` in place of the MLP projector, apply QLoRA to the LLM backbone, train for 2 epochs on all 6 218 samples, and evaluate on a test subset.

**Training script:** The training logic lives in `train_single.py` at the project root (not in a `training/` subfolder). Run it from the Colab working directory:

```bash
python train_single.py --expert E1 --epochs 2 --batch 4 --n_eval 500 --out_dir results/single
python train_single.py --expert E2 --epochs 2 --batch 4 --n_eval 500 --out_dir results/single
python train_single.py --expert E3 --epochs 2 --batch 4 --n_eval 500 --out_dir results/single
python train_single.py --expert E4 --epochs 2 --batch 4 --n_eval 500 --out_dir results/single
```

Key flags:
- `--batch 4` with 4 gradient-accumulation steps → effective batch 16
- `--n_eval N` evaluates on a random N-sample subset of the test split (use 500 for table, 200 for quick debug)
- Outputs: `results/single/single_expert_{EXPERT}.json` (metrics), `results/single/ckpt_{EXPERT}/` (PEFT checkpoint), `results/single/log_{EXPERT}.csv` (step log)

For each run, record:

|Metric|Description|
|---|---|
|`Acc_image`|Accuracy on image + question|
|`Acc_text`|Accuracy on question only (image token stripped)|
|$\Delta_v$|`Acc_image - Acc_text`|
|Training time|Wall-clock hours|
|Peak VRAM|GB|

Fill in Table D-1 (E1 already completed):

**Table D-1: Single-expert ablation results.**

|Expert|Image Acc. (%)|Text Acc. (%)|$\Delta_v$|Time (h)|VRAM (GB)|
|---|---|---|---|---|---|
|E1 (MLP)|65.6|38.2|27.4|—|12.4|
|E2 (Q-Former)|—|—|—|—|—|
|E3 (global token)|—|—|—|—|—|
|E4 (QCGP)|—|—|—|—|—|

> [!warning] Evaluation and training are separate passes
> The training script may have initial bugs in the inline evaluation loop (the eval loop was fixed post-hoc for E1). If evaluation numbers look wrong after training completes, save the PEFT checkpoint and run a separate evaluation script against it. The key thing to preserve is the checkpoint — training cannot be undone once Colab disconnects.

> [!tip] Save results to Google Drive immediately after each run — Colab sessions expire and all local files are lost. Run the copy cell right after the training cell, before moving on to the next expert.

**→ Report mapping:** Results section — this is the core ablation table. Every number earns a sentence of analysis. The key claim: E4 should show higher $\Delta_v$ than E1, confirming the gate forces genuine visual grounding.

---

### 7.3 Task D-2: Full MoC Training

> [!abstract] Task D-2: Train the joint MoC system with load-balancing

Use the MoC module from Task C-3. The training script is `train_moc.py` at the project root:

```bash
python train_moc.py --epochs 2 --batch 4 --val_every 750 --n_val 200 --n_test 500 --out_dir results/moc
```

Key flags:
- `--val_every 750` runs validation every 750 steps
- `--n_val 200` evaluates on 200 validation samples at each checkpoint
- `--n_test 500` evaluates on 500 test samples at the final checkpoint

Log every 100 steps:
- Cross-entropy loss $\mathcal{L}_\text{CE}$
- Load-balancing loss $\mathcal{L}_\text{lb}$
- Per-expert selection count (running total): $n_1, n_2, n_3, n_4$
- $\tau_g$ value (from E4)

Save the best checkpoint by validation accuracy.

**→ Report mapping:** Results section — training loss curve (both $\mathcal{L}_\text{CE}$ and $\mathcal{L}_\text{lb}$) as a single figure. Expert selection counts over training as a second figure (shows router learning dynamics).

---

## 8 Coding Tasks — Part E: Ablations and Analysis

### 8.1 Task E-1: Full Ablation Table

> [!abstract] Task E-1: Compile all results into a unified comparison table

After all runs in Part D, fill in Table E-1. Results are on the ScienceQA image-only test split (n = 2,017 full, or the `--n_test` subset used in training scripts).

**Table E-1: Complete results on ScienceQA image-only test set.**

|Model|Image Acc. (%)|Text Acc. (%)|$\Delta_v$|
|---|---|---|---|
|Zero-shot LLaVA-1.5-7B (midterm)|63.5|56.0|7.5|
|LoRA fine-tuned E1 (midterm, 2k samples)|62.6|51.0|11.6|
|QLoRA fine-tuned E1 (full 6 218 samples)|65.6|38.2|27.4|
|QLoRA fine-tuned E2|—|—|—|
|QLoRA fine-tuned E3|—|—|—|
|QLoRA fine-tuned E4 (QCGP)|—|—|—|
|MoC (joint routing)|—|—|—|

**→ Report mapping:** Results section — this is Table 2 of the final report. Write one paragraph per row of comparison (not per row of the table). The paragraph structure: what the model does → what result you got → why you think that happened → what it implies for the MoC design.

---

### 8.2 Task E-2: Router Distribution Analysis

> [!abstract] Task E-2: Analyze which expert the router selects and for what questions

After MoC training, run the test set through the router (forward pass only, no generation). Record `k_star` for each sample. Use the full 2,017-sample test split or the `--n_test` subset used during training (at least 500 samples for meaningful statistics).

**E-2.1 Overall distribution**

Plot a bar chart of expert selection frequency:

```
Expert:     E1     E2     E3     E4
Selected:   ?%     ?%     ?%     ?%
```

If the distribution is approximately uniform (±5% per expert), the load-balancing loss is working. If one expert dominates, investigate and report.

**E-2.2 Subject-level breakdown**

ScienceQA has subject labels. Group test samples by subject (physics, biology, chemistry, earth science, social science). For each subject, plot a stacked bar showing which expert was selected. Report whether certain subjects systematically prefer certain experts.

**E-2.3 Question-length correlation**

Compute the average question token length for samples routed to each expert. Does E3 (global token) tend to be selected for shorter, simpler questions?

**→ Report mapping:** Results section — "Router Behavior Analysis" subsection. Two figures: overall bar chart, and subject-level stacked bar. This is the evidence that routing is semantically meaningful, not random.

---

### 8.3 Task E-3: Gate Weight Heatmaps (E4)

> [!abstract] Task E-3: Visualize per-patch gate weights $\alpha_i$ from QCGP

For 8–10 test samples where E4 was the selected expert (or force E4 by routing to it), extract the $\alpha \in \mathbb{R}^{576}$ vector from E4's forward pass.

**E-3.1 Heatmap generation**

Reshape $\alpha$ from `(576,)` to `(24, 24)` (the CLIP patch grid). Overlay as a heatmap on the original image using `matplotlib`:

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2)
axes[0].imshow(original_image)
axes[0].set_title("Original")
axes[1].imshow(original_image)
axes[1].imshow(alpha.reshape(24, 24), alpha=0.6, cmap='hot')
axes[1].set_title(f"Gate weights\nQ: {question}")
```

**E-3.2 Selection criteria**

Choose examples where the question is spatially specific (e.g., "What is the label on the arrow pointing to X?", "What does the arrow in the top-left indicate?"). The gate should concentrate high $\alpha_i$ on the relevant region.

Also include one counter-example where the gate does **not** concentrate (question about a general property), to show the gate responds to question type.

**→ Report mapping:** Results section — "Model Internals: Gate Weight Visualization" subsection. Include 4–6 heatmap pairs (image + overlay). This is the most visually compelling figure in the paper and directly demonstrates QCGP's mechanism.

---

### 8.4 Task E-4: Three-Bucket Failure Analysis

> [!abstract] Task E-4: Classify 100 error cases from the best model

Take the best-performing model from Table E-1. Collect all test samples where the model predicts incorrectly. Randomly sample 100 of these error cases (set seed 42 for reproducibility).

**Classification protocol:**

For each error case, examine the image, question, ground-truth answer, and model prediction. Classify into exactly one bucket:

|Bucket|Definition|Signal|
|---|---|---|
|**Perception error**|Model misreads or ignores the image content (describes wrong objects, wrong labels, wrong diagram structure)|$P(\text{ans} \mid x^T)$ explains the prediction better than $P(\text{ans} \mid x^T, x^V)$|
|**Reasoning error**|Model correctly identifies relevant visual content but draws the wrong conclusion|The image is correctly referenced in the chain-of-thought but the logical step fails|
|**CoT rescue**|With chain-of-thought prompting ("Let's think step by step"), the model corrects its answer|Requires re-running the 100 samples with CoT prompt|

**For CoT rescue:** re-prompt each of the 100 error cases with `"Let's think step by step."` appended. If the model now answers correctly, classify as CoT rescue.

**Output:** Pie chart or grouped bar chart of the three buckets, with raw counts. Compare against the PAPO baseline finding (≈67% perception errors in RLVR-trained MLLMs) — note whether your ScienceQA distribution differs and explain why.

**→ Report mapping:** Results section — "Failure Analysis" subsection. Follow with one paragraph per bucket interpreting what the distribution says about the model's remaining bottleneck, and whether MoC's intervention addressed the dominant failure mode.

---

### 8.5 Task E-5: $\tau_g$ Convergence Analysis

> [!abstract] Task E-5: Track learnable temperature in E4 throughout training

During Task D-2 (or during the E4 single-expert run), log the value of `tau_g` every 50 gradient steps. Save as a CSV: `[step, tau_g_value]`.

Plot $\tau_g$ vs. training step. Expected behavior: $\tau_g$ should move from 1.0 toward a value that reflects the model's preferred gating sharpness for ScienceQA.

**Analysis questions to answer in the report:**

1. Did $\tau_g$ increase (sharper, more selective gating) or decrease (softer, more diffuse gating)?
2. Did it stabilize before training ended, or was it still changing at the final step?
3. What does the converged value imply about the typical number of "relevant" patches in a ScienceQA diagram?

**→ Report mapping:** Results section — one paragraph and one figure inside the "Model Internals" subsection. This validates that $\tau_g$ is a meaningful learned parameter, not a fixed constant.

---

### 8.6 Task E-6: Subject-Level Accuracy Breakdown

> [!abstract] Task E-6: Break down accuracy and $\Delta_v$ by science subject

ScienceQA includes subject labels. For the best model (likely MoC or E4), compute `Acc_image` and `Acc_text` separately for each subject.

**Table E-2: Subject-level accuracy breakdown.**

|Subject|N (test)|Image Acc. (%)|Text Acc. (%)|$\Delta_v$|
|---|---|---|---|---|
|Physics|—|—|—|—|
|Biology|—|—|—|—|
|Chemistry|—|—|—|—|
|Earth Science|—|—|—|—|
|Social Science|—|—|—|—|

Identify: which subject has the largest $\Delta_v$? Which has the smallest? Does the subject with the highest $\Delta_v$ correspond to the subject where E4 is most frequently selected by the router (from Task E-2)?

**→ Report mapping:** Results section — "Subject-Level Analysis" subsection. This answers one of the three core research questions stated in the midterm's Experimental Design section.

---

## 9 Project File Structure

The actual structure of the repository. Training scripts live at the project root (not in a `training/` subfolder), and evaluation scripts are still to be created (Task E).

```
dvlm-project/
├── connector/
│   ├── __init__.py
│   ├── expert_e1.py        # ExpertE1 — MLP wrapper (frozen pretrained projector)
│   ├── expert_e2.py        # ExpertE2 — Q-Former cross-attention (32 tokens)
│   ├── expert_e3.py        # ExpertE3 — Attention-pooled global token (1 token)
│   ├── expert_e4.py        # ExpertE4 — QCGP question-conditioned gating (576 tokens)
│   ├── question_pooler.py  # QuestionPooler — attention pooling for question vector q
│   ├── router.py           # MoCRouter — lightweight 2-layer MLP router
│   └── moc.py              # MixtureOfConnectors + MoCLlavaForCausalLM + upgrade_to_moc
├── losses/
│   ├── __init__.py
│   └── load_balance.py     # load_balancing_loss() + LAMBDA_LB constant
├── eval/
│   └── __init__.py         # ← evaluation scripts for Task E go here
├── results/
│   ├── single/
│   │   └── single_expert_E1.json   # E1 results (complete)
│   ├── figures/            # Saved plots (Task E)
│   └── tables/             # Saved CSVs (Task E)
├── notebooks/
│   ├── Verifications_Task_A_C.ipynb   # Shape + routing checks for all modules
│   ├── train_single_experts.ipynb     # Colab notebook: single-expert runs
│   └── Full_MoC_Training.ipynb        # Colab notebook: full MoC training run
├── setup_qlora.py          # Task A-0: QLoRA setup, dataset loading, step count
├── train_single.py         # Task D-1: single-expert training loop (at project root)
└── train_moc.py            # Task D-2: full MoC training loop (at project root)
```

> [!note] Verification scripts are standalone
> Each module (`expert_e1.py`, `expert_e2.py`, …, `router.py`, `question_pooler.py`, `load_balance.py`) contains its own `__main__` verification block. Run any of them directly with `python connector/expert_e1.py` etc. from the project root. The MoC verification requires a loaded LLaVA model: `python connector/moc.py --full` (Colab A100 only).

**Evaluation scripts still needed (Task E):**

```
eval/
├── router_analysis.py  # Task E-2: bar chart of expert selection frequency
├── gate_heatmap.py     # Task E-3: alpha_i heatmap overlaid on image
├── failure_analysis.py # Task E-4: three-bucket error classification
├── tau_analysis.py     # Task E-5: tau_g vs training step plot
└── subject_breakdown.py # Task E-6: per-subject accuracy and Delta_v
```

---

## 10 Minimal Deliverables Checklist

Track status as work progresses:

**Infrastructure and module implementation (Parts A–C) — complete**
- [x] QLoRA training runs on Linux (A100) without OOM — peak VRAM 12.4 GB
- [x] All four experts pass shape and gradient verification checks
- [x] Router outputs approximately uniform distribution at initialization
- [x] Load-balancing loss reaches `W2.grad is not None` after first backward
- [x] MoC integration verified: finite loss, diverse routing, correct parameter counts
- [x] Critical five-step upgrade ordering documented and tested

**Training (Part D) — in progress**
- [x] E1 single-expert run complete (Image Acc 65.6%, Text Acc 38.2%, Δv 27.4%)
- [ ] E2 single-expert run
- [ ] E3 single-expert run
- [ ] E4 single-expert run
- [ ] Table D-1 fully populated (4 single-expert runs)
- [ ] Full MoC training run complete
- [ ] Table E-1 fully populated (MoC run)

**Ablations and analysis (Part E) — not started**
- [ ] Router distribution bar chart generated (Task E-2)
- [ ] Subject-level stacked bar chart (Task E-2.2)
- [ ] At least 4 gate heatmap pairs saved (Task E-3)
- [ ] 100-sample failure classification with bucket counts (Task E-4)
- [ ] CoT rescue re-prompting of the 100 error cases (Task E-4)
- [ ] $\tau_g$ convergence plot saved (Task E-5)
- [ ] Subject-level Table E-2 populated (Task E-6)

**Paper fixes — not started**
- [ ] All `equation*` in Methodology switched to numbered `equation`
- [ ] Router matrix dimensions added ($W_r^{(1)} \in \mathbb{R}^{d_r \times d}$, $W_r^{(2)} \in \mathbb{R}^{4 \times d_r}$)
- [ ] MoC architecture figure added to paper
- [ ] Discussion and Conclusion sections drafted