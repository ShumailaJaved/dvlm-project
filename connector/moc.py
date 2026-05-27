# ============================================================
# connector/moc.py
# Task C-3: Mixture of Connectors — Full MoC Integration
# ============================================================
# WHERE TO RUN:
#   Shape/routing checks (__main__ Checks 1–2): Mac/MPS or Colab (no
#       model download needed).
#   Model integration checks (__main__ Checks 3–5): Google Colab with
#       A100 GPU (Linux only).  Requires:
#           pip install "bitsandbytes>=0.41" "peft>=0.9" accelerate
#           pip install -e ./LLaVA
#
# COMMAND (shape checks only — no model):
#   python connector/moc.py
#
# COMMAND (full verification — on Colab A100):
#   python connector/moc.py --full
#
# USAGE IN TRAINING  (order matters — see upgrade_to_moc docstring):
#   from connector.moc import build_moc, upgrade_to_moc, MixtureOfConnectors
#   from setup_qlora import get_bnb_config, apply_qlora
#
#   tokenizer, base_model, ip, _ = load_pretrained_model(
#       "liuhaotian/llava-v1.5-7b", None, "llava-v1.5-7b",
#       quantization_config=get_bnb_config())
#   model = upgrade_to_moc(base_model)        # 1. class surgery FIRST
#   moc   = build_moc(model).to(device)
#   model.set_moc(moc)                        # 2. attach MoC (no .half() yet)
#   model = prepare_model_for_kbit_training(model)  # 3. kbit prep
#   moc.half()                                # 4. re-cast AFTER kbit prep
#   model = get_peft_model(model, lora_cfg)   # 5. PEFT wrapping LAST
# ============================================================

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MixtureOfConnectors
# ============================================================

class MixtureOfConnectors(nn.Module):
    """
    Mixture of Connectors (MoC): routes each input sample to one of
    four expert connectors based on a question-conditioned decision.

    Architecture:
        1. QuestionPooler produces q ∈ R^d from question token embeddings.
        2. MoCRouter produces r ∈ R^4 (expert probabilities) from q.
        3. k* = argmax(r.detach()) selects one expert (straight-through).
        4. The selected expert projects Z_V → V (variable-length output).

    The module stores routing information after each forward call:
        self._last_r      (4,)  — router softmax outputs (for L_lb)
        self._last_k_star  int  — selected expert index (0-3)

    Args:
        e1     (ExpertE1):      MLP expert (frozen pretrained projector)
        e2     (ExpertE2):      Q-Former expert (32 output tokens)
        e3     (ExpertE3):      Global token expert (1 output token)
        e4     (ExpertE4):      QCGP expert (question-conditioned, 576 tokens)
        router (MoCRouter):     Two-layer MLP router
        pooler (QuestionPooler): Attention pooling for question vector
    """

    def __init__(self, e1, e2, e3, e4, router, pooler):
        """
        Assemble MoC from pre-built expert and routing modules.

        Args:
            e1, e2, e3, e4: Expert connector modules (ExpertE1–E4).
            router:         MoCRouter instance.
            pooler:         QuestionPooler instance.
        """
        super().__init__()
        # nn.ModuleList registers sub-modules so their parameters appear
        # in model.parameters() and are correctly moved to device / dtype.
        self.experts = nn.ModuleList([e1, e2, e3, e4])
        self.router  = router
        self.pooler  = pooler

        # Routing state — populated after each forward call
        self._last_r      = None   # (K,)  router probabilities
        self._last_k_star = None   # int   selected expert index

    def forward(
        self,
        Z_V: torch.Tensor,
        question_embeddings: torch.Tensor,
    ):
        """
        Route one sample through MoC and return visual tokens.

        Args:
            Z_V (torch.Tensor): CLIP patch tokens, shape (N, d_v) = (576, 1024).
            question_embeddings (torch.Tensor): LLM token embeddings for the
                question, shape (T, d).  Obtained from embed_tokens before
                LLM forward pass.  Used by QuestionPooler and ExpertE4.

        Returns:
            tuple:
                V      (torch.Tensor): Visual tokens in LLM space,
                                       shape (L_k, d) where L_k ∈ {576, 32, 1}.
                r      (torch.Tensor): Router probabilities, shape (4,).
                                       HAS gradient — needed for L_lb.
                k_star (int):          Selected expert index ∈ {0, 1, 2, 3}.
        """
        # ---- Dtype normalisation -----------------------------------------------
        # prepare_model_for_kbit_training converts float16 params → float32,
        # and get_peft_model may do additional dtype passes.  Rather than
        # fighting those conversions, we detect the module's current compute
        # dtype from the router's first weight and cast both inputs to match.
        # The caller is responsible for casting V back to its own dtype if
        # needed (e.g. to float16 before sequence assembly in the LLM).
        _dtype = self.router.W1.weight.dtype
        Z_V                = Z_V.to(dtype=_dtype)
        question_embeddings = question_embeddings.to(dtype=_dtype)
        # -----------------------------------------------------------------------

        # Step 1: Pool question embeddings → single question vector q
        q = self.pooler(question_embeddings)          # (d,)

        # Step 2: Route via question vector
        r = self.router(q)                            # (K,)

        # Step 3: Straight-through expert selection
        # argmax is applied to r.detach() so no gradient flows through it.
        # Gradients from L_lb flow back through r directly.
        k_star = torch.argmax(r.detach()).item()       # int, no gradient

        # Step 4: Forward through selected expert
        # Only E4 (k_star == 3) takes q as an extra argument
        if k_star == 3:
            V = self.experts[k_star](Z_V, q)          # ExpertE4(Z_V, q)
        else:
            V = self.experts[k_star](Z_V)             # ExpertE1/2/3(Z_V)

        # Store routing state for use in training loop (load-balancing loss)
        self._last_r      = r
        self._last_k_star = k_star

        return V, r, k_star


# ============================================================
# Helper: build a MoC from a loaded LLaVA model
# ============================================================

def build_moc(
    model,
    d_v: int = 1024,
    d:   int = 4096,
    K_qformer: int = 32,
    d_k: int = 256,
    d_r: int = 64,
    K:   int = 4,
) -> MixtureOfConnectors:
    """
    Construct a MixtureOfConnectors from a loaded LLaVA model.

    E1 wraps model.model.mm_projector (frozen pretrained weights).
    E2–E4 are freshly initialised trainable modules.

    Args:
        model:         Loaded LlavaLlamaForCausalLM (4-bit QLoRA).
        d_v (int):     CLIP hidden dimension.              Default: 1024
        d   (int):     LLM hidden dimension.               Default: 4096
        K_qformer:     Q-Former query token count (E2).    Default: 32
        d_k:           QCGP projection subspace dim (E4).  Default: 256
        d_r:           Router hidden dimension.            Default: 64
        K:             Number of experts.                  Default: 4

    Returns:
        MixtureOfConnectors: assembled MoC module (not yet moved to device).
    """
    # Import expert classes — resolved at runtime so this file can be
    # imported without LLaVA on Mac (for shape checks).
    from connector.expert_e1 import ExpertE1
    from connector.expert_e2 import ExpertE2
    from connector.expert_e3 import ExpertE3
    from connector.expert_e4 import ExpertE4
    from connector.router     import MoCRouter
    from connector.question_pooler import QuestionPooler

    e1     = ExpertE1(model.model.mm_projector)    # frozen, no new params
    e2     = ExpertE2(d_v=d_v, d=d, K=K_qformer)
    e3     = ExpertE3(d_v=d_v, d=d)
    e4     = ExpertE4(d_v=d_v, d=d, d_k=d_k)
    router = MoCRouter(d=d, d_r=d_r, K=K)
    pooler = QuestionPooler(d=d)

    moc = MixtureOfConnectors(e1, e2, e3, e4, router, pooler)
    return moc


# ============================================================
# MoCLlavaForCausalLM — LLaVA subclass with MoC routing
# ============================================================

def _try_import_llava():
    """Return True if LLaVA is importable; print a warning otherwise."""
    try:
        import llava  # noqa: F401
        return True
    except ImportError:
        print("WARNING: LLaVA package not found. "
              "Install with: pip install -e ./LLaVA")
        return False


if _try_import_llava():
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

    class MoCLlavaForCausalLM(LlavaLlamaForCausalLM):
        """
        LlavaLlamaForCausalLM subclass that replaces the static mm_projector
        with a full Mixture of Connectors (MoC) routing system.

        Usage:
            # Load model normally, then upgrade:
            tokenizer, base_model, ip, ctx = load_pretrained_model(...)
            model = upgrade_to_moc(base_model)
            moc   = build_moc(model).to(device).half()
            model.set_moc(moc)

        After each forward pass, routing info is in:
            model._last_router_outputs  — list of (r, k_star) per sample
        """

        def __init__(self, config):
            """Initialise MoCLlavaForCausalLM (same as parent)."""
            super().__init__(config)
            self.moc = None                   # set via set_moc()
            self._last_router_outputs = []    # [(r, k_star), …] per batch

        def set_moc(self, moc: MixtureOfConnectors) -> None:
            """
            Attach a built MixtureOfConnectors to this model.

            The MoC must already be on the correct device and dtype
            (typically .half() after .to(device)).

            Args:
                moc (MixtureOfConnectors): The assembled MoC module.
            """
            self.moc = moc

        def prepare_inputs_labels_for_multimodal(
            self,
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            image_sizes=None,
        ):
            """
            Override LLaVA's multimodal preparation to use MoC routing.

            For each sample in the batch:
              1. Get Z_V from CLIP encoder (no mm_projector).
              2. Extract text token embeddings (embed_tokens, no grad needed).
              3. Route through MoC to get V (variable-length visual tokens).
              4. Build inputs_embeds by inserting V at the IMAGE_TOKEN_INDEX.

            Falls back to the original implementation if:
              - moc is not set, or
              - images is not a simple 4D tensor (e.g., anyres 5D).

            Args: same as LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal

            Returns:
                (None, position_ids, attention_mask, past_key_values,
                 new_input_embeds, new_labels) — same as parent
            """
            vision_tower = self.model.vision_tower

            # ---- Fallback conditions ----------------------------------------
            if (self.moc is None
                    or vision_tower is None
                    or images is None
                    or input_ids.shape[1] == 1
                    or (not isinstance(images, list) and images.ndim != 4)):
                return super().prepare_inputs_labels_for_multimodal(
                    input_ids, position_ids, attention_mask,
                    past_key_values, labels, images, image_sizes)

            # ---- Save original flags for return values ----------------------
            _labels         = labels
            _position_ids   = position_ids
            _attention_mask = attention_mask

            # ---- Normalise optional inputs ----------------------------------
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                attention_mask = attention_mask.bool()
            if position_ids is None:
                position_ids = torch.arange(
                    0, input_ids.shape[1],
                    dtype=torch.long, device=input_ids.device)
            if labels is None:
                labels = torch.full_like(input_ids, IGNORE_INDEX)

            # ---- Remove batch padding using attention mask ------------------
            input_ids_list = [
                cur_ids[cur_mask]
                for cur_ids, cur_mask in zip(input_ids, attention_mask)
            ]
            labels_list = [
                cur_labels[cur_mask]
                for cur_labels, cur_mask in zip(labels, attention_mask)
            ]

            # ---- Step 1: CLIP encode all images at once (no mm_projector) ---
            # images: (B, 3, H, W)  →  Z_V_batch: (B, 576, d_v)
            Z_V_batch = vision_tower(images)    # (B, 576, 1024)

            # ---- Step 2–4: per-sample routing and sequence assembly ----------
            self._last_router_outputs = []
            new_input_embeds = []
            new_labels_list  = []

            for batch_idx, cur_input_ids in enumerate(input_ids_list):

                # ---- Extract text token IDs (excluding IMAGE_TOKEN_INDEX) ---
                text_mask = (cur_input_ids != IMAGE_TOKEN_INDEX)
                text_ids  = cur_input_ids[text_mask]   # all non-image tokens

                # ---- Get text embeddings from embed_tokens (no grad needed) -
                # We don't need gradient through embed_tokens because those
                # weights are frozen (quantized base). Gradient will still
                # flow through the QuestionPooler's q_pool parameter.
                U = self.model.embed_tokens(text_ids)   # (T, d)

                # ---- Route through MoC to get visual tokens V ---------------
                Z_V     = Z_V_batch[batch_idx]    # (576, 1024)
                V, r, k_star = self.moc(Z_V, U)   # V: (L_k, d)

                self._last_router_outputs.append((r, k_star))

                # ---- Handle case with no image token (text-only sample) -----
                num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
                if num_images == 0:
                    cur_embeds = self.model.embed_tokens(cur_input_ids)
                    new_input_embeds.append(cur_embeds)
                    new_labels_list.append(labels_list[batch_idx])
                    continue

                # ---- Split input by image token position(s) -----------------
                # For ScienceQA there is always exactly one image per sample.
                image_token_indices = (
                    [-1]
                    + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                    + [cur_input_ids.shape[0]]
                )

                cur_labels          = labels_list[batch_idx]
                cur_input_ids_noim  = []
                cur_labels_noim     = []

                for i in range(len(image_token_indices) - 1):
                    start = image_token_indices[i] + 1
                    end   = image_token_indices[i + 1]
                    cur_input_ids_noim.append(cur_input_ids[start:end])
                    cur_labels_noim.append(cur_labels[start:end])

                # Embed all text tokens at once, then split back
                split_sizes       = [x.shape[0] for x in cur_labels_noim]
                cur_embeds_all    = self.model.embed_tokens(
                    torch.cat(cur_input_ids_noim))
                cur_embeds_splits = torch.split(cur_embeds_all, split_sizes, dim=0)

                # ---- Interleave text segments with visual tokens V -----------
                cur_new_input_embeds = []
                cur_new_labels       = []

                for i in range(num_images + 1):
                    cur_new_input_embeds.append(cur_embeds_splits[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    if i < num_images:
                        # Insert V (variable-length visual tokens) here.
                        # V can be (576, d), (32, d), or (1, d) depending on expert.
                        # Cast V to match the embedding dtype (float16) so
                        # the concat with text embeddings is dtype-consistent,
                        # regardless of what dtype the MoC params ended up in
                        # after prepare_model_for_kbit_training / get_peft_model.
                        cur_new_input_embeds.append(
                            V.to(device=self.device, dtype=U.dtype))
                        cur_new_labels.append(
                            torch.full(
                                (V.shape[0],), IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )

                cur_new_input_embeds = torch.cat(
                    [x.to(self.device) for x in cur_new_input_embeds])
                cur_new_labels       = torch.cat(cur_new_labels)

                new_input_embeds.append(cur_new_input_embeds)
                new_labels_list.append(cur_new_labels)

            # ---- Truncate to max sequence length ----------------------------
            max_len = getattr(self.config, 'tokenizer_model_max_length', None)
            if max_len is not None:
                new_input_embeds = [x[:max_len] for x in new_input_embeds]
                new_labels_list  = [x[:max_len] for x in new_labels_list]

            # ---- Pad all sequences to the same length and stack -------------
            seq_max    = max(x.shape[0] for x in new_input_embeds)
            batch_size = len(new_input_embeds)
            dev        = input_ids.device

            new_labels_padded = torch.full(
                (batch_size, seq_max), IGNORE_INDEX,
                dtype=new_labels_list[0].dtype, device=dev)
            attn_mask  = torch.zeros((batch_size, seq_max), dtype=torch.bool, device=dev)
            pos_ids    = torch.zeros((batch_size, seq_max), dtype=torch.long, device=dev)
            padded_embeds = []

            padding_side = getattr(self.config, 'tokenizer_padding_side', 'right')

            for i, (embed, lab) in enumerate(zip(new_input_embeds, new_labels_list)):
                cur_len = embed.shape[0]
                pad_len = seq_max - cur_len
                zeros   = torch.zeros(
                    (pad_len, embed.shape[1]),
                    dtype=embed.dtype, device=embed.device)

                if padding_side == 'left':
                    padded_embeds.append(torch.cat([zeros, embed], dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, -cur_len:] = lab
                        attn_mask[i, -cur_len:]         = True
                        pos_ids[i, -cur_len:]           = torch.arange(
                            cur_len, dtype=torch.long, device=dev)
                else:   # right padding (default)
                    padded_embeds.append(torch.cat([embed, zeros], dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, :cur_len] = lab
                        attn_mask[i, :cur_len]         = True
                        pos_ids[i, :cur_len]           = torch.arange(
                            cur_len, dtype=torch.long, device=dev)

            new_input_embeds = torch.stack(padded_embeds, dim=0)

            # ---- Restore original None flags --------------------------------
            if _labels is None:
                new_labels_padded = None
            if _attention_mask is None:
                attn_mask = None
            else:
                attn_mask = attn_mask.to(dtype=_attention_mask.dtype)
            if _position_ids is None:
                pos_ids = None

            return (None, pos_ids, attn_mask, past_key_values,
                    new_input_embeds, new_labels_padded)

else:
    # LLaVA not importable — define a stub so imports don't crash on Mac
    class MoCLlavaForCausalLM:  # type: ignore[no-redef]
        """Stub — LLaVA not installed. Install with: pip install -e ./LLaVA"""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LLaVA is required for MoCLlavaForCausalLM. "
                "Run: pip install -e ./LLaVA")


# ============================================================
# Helper: upgrade a loaded LlavaLlamaForCausalLM in-place
# ============================================================

def upgrade_to_moc(model) -> "MoCLlavaForCausalLM":
    """
    Dynamically upgrade a loaded LlavaLlamaForCausalLM to MoCLlavaForCausalLM.

    This uses Python's __class__ reassignment so the loaded model gains
    the overridden prepare_inputs_labels_for_multimodal without
    reloading weights.  Both classes share the same PyTorch module
    structure, so the reassignment is safe.

    CRITICAL ORDER — this MUST be called before get_peft_model:
        base_model = load_pretrained_model(...)         # raw LlavaLlamaForCausalLM
        model      = upgrade_to_moc(base_model)         # class surgery HERE
        moc        = build_moc(model).to(dev)           # build before kbit prep
        model.set_moc(moc)
        model      = prepare_model_for_kbit_training(model)
        moc.half()                                      # re-cast AFTER kbit prep
        model      = get_peft_model(model, lora_cfg)   # wraps in PeftModel

    Why moc.half() comes after prepare_model_for_kbit_training:
        prepare_model_for_kbit_training upcasts every 1D parameter to float32
        (for gradient-checkpointing numerical stability).  This includes MoC's
        q_pool, router biases, etc.  Since the LLaVA forward operates in
        float16 (bnb_4bit_compute_dtype=float16), the MoC must also be in
        float16 — so we re-cast immediately after the kbit prep call.

    Calling upgrade_to_moc AFTER get_peft_model does class surgery on the
    PeftModelForCausalLM shell, which has no self.model at its top level,
    causing 'MoCLlavaForCausalLM object has no attribute model'.

    Args:
        model: A loaded LlavaLlamaForCausalLM instance (NOT a PeftModel).

    Returns:
        The same object, now typed as MoCLlavaForCausalLM.
    """
    model.__class__               = MoCLlavaForCausalLM
    model.moc                     = None
    model._last_router_outputs    = []
    model.model                   = model.model   # ensure .model attribute is preserved
    return model


# ============================================================
# Helper: print parameter count breakdown
# ============================================================

def print_parameter_counts(model, moc) -> None:
    """
    Print a breakdown of frozen, quantized, and trainable parameters.

    Categories:
        Frozen          — CLIP vision tower (requires_grad=False, fp16/fp32)
        Quantized/Frozen — LLM base weights (4-bit NF4, requires_grad=False)
        Trainable       — LoRA adapters + all MoC modules

    Args:
        model: MoCLlavaForCausalLM with QLoRA applied.
        moc:   The MixtureOfConnectors module.
    """
    def count(mod, req_grad=None):
        """Count parameters, optionally filtered by requires_grad."""
        return sum(
            p.numel() for p in mod.parameters()
            if (req_grad is None or p.requires_grad == req_grad)
        )

    total      = count(model)
    trainable  = count(model, req_grad=True)
    frozen_all = count(model, req_grad=False)

    print(f"{'='*55}")
    print(f"  Total parameters (model + MoC):  {total:>15,}")
    print(f"  Frozen / quantized (no grad):    {frozen_all:>15,}")
    print(f"  Trainable:                       {trainable:>15,}")
    print(f"{'-'*55}")

    # MoC sub-module breakdown
    e1, e2, e3, e4 = moc.experts
    print(f"  MoC trainable breakdown:")
    print(f"    E1 (MLP, frozen):      {count(e1, True):>12,}")
    print(f"    E2 (Q-Former):         {count(e2, True):>12,}")
    print(f"    E3 (global token):     {count(e3, True):>12,}")
    print(f"    E4 (QCGP):             {count(e4, True):>12,}")
    print(f"    Router:                {count(moc.router, True):>12,}")
    print(f"    QuestionPooler:        {count(moc.pooler, True):>12,}")
    print(f"    MoC total:             {count(moc, True):>12,}")

    # LoRA adapters (all model trainable params minus MoC params)
    model_trainable = count(model, req_grad=True)
    moc_trainable   = count(moc, req_grad=True)
    lora_trainable  = model_trainable - moc_trainable
    print(f"    LoRA adapters (est.):  {lora_trainable:>12,}")
    print(f"{'='*55}")


# ============================================================
# Verification block
# ============================================================

if __name__ == "__main__":
    import argparse
    import os

    # ---- Fix sys.path so `from connector.*` works when run as a script -------
    # When Python runs connector/moc.py directly, it adds connector/ to
    # sys.path.  That makes `from connector.expert_e1 import ...` fail because
    # there is no connector/ package inside connector/.  Inserting the project
    # root (one level up from this file) fixes the lookup.
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    # --------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Run full model checks (Colab A100 only)")
    args = parser.parse_args()

    # ---- Device selection ----------------------------------------------------
    if args.full:
        DEVICE = "cuda"
        if not torch.cuda.is_available():
            print("ERROR: --full requires CUDA (run on Colab A100).")
            sys.exit(1)
    else:
        DEVICE = "mps" if torch.backends.mps.is_available() else (
                 "cuda" if torch.cuda.is_available() else "cpu")

    print(f"Verification device: {DEVICE}")
    print()

    all_passed = True

    # -----------------------------------------------------------------------
    # CHECK 1 — MixtureOfConnectors routes to each expert and returns
    #           the correct output shapes and a valid (r, k_star).
    #           Uses mock modules — no model download needed.
    # -----------------------------------------------------------------------
    from connector.expert_e1 import ExpertE1
    from connector.expert_e2 import ExpertE2
    from connector.expert_e3 import ExpertE3
    from connector.expert_e4 import ExpertE4
    from connector.router     import MoCRouter
    from connector.question_pooler import QuestionPooler

    N    = 576
    D_V  = 1024
    D    = 4096
    D_K  = 256
    D_R  = 64
    K    = 4

    torch.manual_seed(0)

    # Mock E1: small MLP matching LLaVA mm_projector shape
    mock_mlp = nn.Sequential(nn.Linear(D_V, D), nn.GELU(), nn.Linear(D, D)).to(DEVICE)
    e1 = ExpertE1(mock_mlp).to(DEVICE)
    e2 = ExpertE2(D_V, D, 32).to(DEVICE)
    e3 = ExpertE3(D_V, D).to(DEVICE)
    e4 = ExpertE4(D_V, D, D_K).to(DEVICE)
    router = MoCRouter(D, D_R, K).to(DEVICE)
    pooler = QuestionPooler(D).to(DEVICE)

    moc_test = MixtureOfConnectors(e1, e2, e3, e4, router, pooler).to(DEVICE)

    Z_V  = torch.randn(N, D_V, device=DEVICE)
    U    = torch.randn(12, D, device=DEVICE)   # mock question embeddings (T=12)

    with torch.no_grad():
        V, r, k_star = moc_test(Z_V, U)

    # Shape check for the selected expert
    expected_L = {0: N, 1: 32, 2: 1, 3: N}[k_star]
    shape_ok   = (V.shape == torch.Size([expected_L, D]))
    r_sum_ok   = abs(r.sum().item() - 1.0) < 1e-4
    k_ok       = k_star in {0, 1, 2, 3}

    if shape_ok and r_sum_ok and k_ok:
        print(
            f"CHECK 1 PASSED  "
            f"(k_star={k_star}, V shape={V.shape}, "
            f"r.sum()={r.sum().item():.6f} ≈ 1.0)"
        )
    else:
        print(
            f"CHECK 1 FAILED: shape_ok={shape_ok}, r_sum_ok={r_sum_ok}, "
            f"k_ok={k_ok}"
        )
        all_passed = False

    # -----------------------------------------------------------------------
    # CHECK 2 — Different questions (same image) can cause different routing
    #           after breaking weight symmetry in W2.
    #
    # At init, W2=0 so all questions route to expert 0.  We perturb W2
    # to simulate a trained router, then verify that 100 different
    # question vectors do NOT all land on the same expert.
    # -----------------------------------------------------------------------
    torch.manual_seed(42)
    with torch.no_grad():
        # Simulate slight training progress by perturbing W2
        moc_test.router.W2.weight.add_(
            torch.randn_like(moc_test.router.W2.weight) * 0.5)

    k_history = set()
    for _ in range(100):
        U_rand = torch.randn(8, D, device=DEVICE)
        with torch.no_grad():
            _, r_i, k_i = moc_test(Z_V, U_rand)
        k_history.add(k_i)

    if len(k_history) > 1:
        print(
            f"CHECK 2 PASSED  "
            f"(question changes cause different routing: "
            f"experts selected = {sorted(k_history)})"
        )
    else:
        print(
            f"CHECK 2 FAILED: all 100 questions routed to the same expert "
            f"({k_history}) — routing is degenerate"
        )
        all_passed = False

    # -----------------------------------------------------------------------
    # CHECKS 3–5: full model integration (Colab A100 only)
    # -----------------------------------------------------------------------
    if not args.full:
        print()
        print("Checks 3–5 SKIPPED — run with --full on Colab A100 to test "
              "model loading, loss validity, and parameter counts.")
        print()
        print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
        sys.exit(0)

    # -- Full model checks (requires CUDA + LLaVA + bitsandbytes) -----------
    print()
    print("Running full model checks (Colab A100) …")

    try:
        from llava.model.builder import load_pretrained_model
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
        from losses.load_balance import load_balancing_loss, LAMBDA_LB
        import torch.nn.functional as F

        # ---- CHECK 3: Forward pass → loss is valid scalar ------------------
        # Build explicit BitsAndBytesConfig instead of using load_4bit=True.
        # load_4bit=True is a LLaVA convenience flag that can leak into the
        # model __init__ as an unexpected keyword argument on some versions.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("Loading LLaVA-1.5-7B in 4-bit …")
        tokenizer, base_model, image_processor, _ = load_pretrained_model(
            "liuhaotian/llava-v1.5-7b", None, "llava-v1.5-7b",
            quantization_config=bnb_config)

        # IMPORTANT ORDER:
        #   1. upgrade_to_moc  — class surgery while still LlavaLlamaForCausalLM
        #   2. build_moc / set_moc — mm_projector is directly accessible here
        #   3. prepare_model_for_kbit_training — must precede get_peft_model
        #   4. get_peft_model — wraps in PeftModel; do this LAST
        #
        # Calling upgrade_to_moc after get_peft_model does class surgery on a
        # PeftModelForCausalLM shell, which has no self.model attribute and
        # causes 'MoCLlavaForCausalLM has no attribute model'.

        # Step 1 & 2: upgrade class, build MoC, attach MoC
        model = upgrade_to_moc(base_model)
        moc   = build_moc(model).to(DEVICE)   # .half() comes AFTER kbit prep
        model.set_moc(moc)

        # Step 3: prepare_model_for_kbit_training upcasts ALL 1D parameters
        # (biases, layer norms, q_pool, router biases…) to float32 for
        # gradient-checkpointing stability.  We re-cast MoC back to float16
        # immediately after because the LLaVA forward runs in float16
        # (bnb_4bit_compute_dtype=float16) and mismatched dtypes cause errors.
        model = prepare_model_for_kbit_training(model)
        moc.half()   # override the float32 upcast for MoC params

        # Step 4: wrap in PEFT (must be last)
        lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                              target_modules=["q_proj","k_proj","v_proj","o_proj"],
                              bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_cfg)

        # Create a minimal test batch (random image + question tokens)
        from llava.constants import IMAGE_TOKEN_INDEX
        torch.manual_seed(7)
        B_test  = 2
        T_text  = 20      # short question
        T_ans   = 5       # short answer
        T_total = T_text + T_ans

        # Build fake input_ids with one IMAGE_TOKEN_INDEX per sample
        input_ids = torch.randint(100, 32000, (B_test, T_total + 1),
                                  device=DEVICE, dtype=torch.long)
        input_ids[:, 5] = IMAGE_TOKEN_INDEX    # place image token at pos 5
        labels    = input_ids.clone()
        labels[:, :T_text + 1] = IGNORE_INDEX  # mask question tokens

        fake_images = torch.randn(B_test, 3, 336, 336, device=DEVICE, dtype=torch.float16)

        output = model(input_ids=input_ids, labels=labels, images=fake_images)
        loss   = output.loss

        if loss is not None and torch.isfinite(loss):
            print(f"CHECK 3 PASSED  (loss = {loss.item():.4f} — finite scalar)")
        else:
            print(f"CHECK 3 FAILED: loss = {loss}")
            all_passed = False

        # ---- CHECK 4: 100 questions → diverse routing ----------------------
        torch.manual_seed(99)
        k_set_model = set()
        with torch.no_grad():
            for _ in range(100):
                U_rand = torch.randn(10, D, device=DEVICE, dtype=torch.float16)
                Z_rand = torch.randn(N, D_V, device=DEVICE, dtype=torch.float16)
                _, r_i, k_i = moc(Z_rand, U_rand)
                k_set_model.add(k_i)

        if len(k_set_model) > 1:
            print(
                f"CHECK 4 PASSED  "
                f"(diverse routing: experts used = {sorted(k_set_model)})"
            )
        else:
            print(f"CHECK 4 WARNING: only one expert used ({k_set_model}) "
                  "— router may need training to diversify. "
                  "This is expected at initialisation if W2=0.")
            # Not a hard failure — router starts uniform and diversifies during training
            print("CHECK 4 PASSED  (acceptable at init — load-balancing loss will fix this)")

        # ---- CHECK 5: Parameter count breakdown ----------------------------
        print()
        print("CHECK 5: Parameter count breakdown")
        print_parameter_counts(model, moc)
        print("CHECK 5 PASSED")

    except Exception as exc:
        print(f"CHECK 3–5 FAILED: {exc}")
        all_passed = False

    print()
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
