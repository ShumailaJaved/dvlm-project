# ============================================================
# connector/expert_e4.py
# Task B-E4: QCGP — Question-Conditioned Gating Projector
# ============================================================
# WHERE TO RUN:
#   Verification (__main__): anywhere — CPU/Mac or Colab.
#   In training:             Google Colab with T4 GPU (Linux).
#
# COMMAND (run verification):
#   python connector/expert_e4.py
#
# USAGE IN TRAINING:
#   from connector.expert_e4 import ExpertE4
#   e4 = ExpertE4(d_v=1024, d=4096, d_k=256).to(device).half()
#   V  = e4(Z_V, q)   # (576, 1024), (4096,) → (576, 4096)
#   # After forward, e4._last_alpha is available for heatmap vis (Task E-3)
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertE4(nn.Module):
    """
    Expert E4: Question-Conditioned Gating Projector (QCGP).

    The core contribution of this project.  QCGP produces a per-patch
    gate g_i ∈ (0,1)^d_v that suppresses query-irrelevant channels
    before projecting to LLM space.  The gate is derived from cosine
    similarity between the question vector q and each patch key k_i in
    a shared d_k-dimensional subspace.

    Forward pass (paper Eq. eq:expert4):
        û   = W_q q / ‖W_q q‖₂           (d_k,)  question projection, L2-normed
        k̂_i = W_k z_i^V / ‖W_k z_i^V‖₂  (d_k,)  patch key, L2-normed
        α_i = softmax_i(û^T k̂_i / τ_g)  (N,)    per-patch relevance weight
        g_i = σ(α_i W_g + b_g)           (N, d_v) per-patch gate vector
        φ₄_i = W_E4(z_i^V ⊙ g_i)        (N, d)   gated + projected

    Output length: L_4 = N = 576  (full spatial resolution preserved)
    Question-aware: Yes  (g_i depends on q)

    After each forward call, intermediate tensors are stored as:
        self._last_alpha   (N,)     — patch relevance weights  (for Task E-3)
        self._last_g       (N, d_v) — gate vectors

    Args:
        d_v (int): CLIP hidden dimension.         Default: 1024
        d   (int): LLM  hidden dimension.         Default: 4096
        d_k (int): Shared projection subspace dim. Default: 256

    Warning:
        tau_g can drift negative during training.  Monitor with
        `assert model.qcgp.tau_g.item() > 0` after each step.
        The forward pass uses `tau_g.abs() + 1e-6` to ensure the
        effective temperature is always positive (as recommended in
        the project specification).

    Warning:
        L2 normalization is applied AFTER the linear projection:
            F.normalize(self.W_q(q), dim=-1)    ← correct
        NOT before:
            F.normalize(q, dim=-1)              ← wrong (skips W_q)
    """

    def __init__(self, d_v: int = 1024, d: int = 4096, d_k: int = 256):
        """
        Initialise ExpertE4 (QCGP).

        Initialisation:
            W_q   — Kaiming uniform (PyTorch Linear default)
            W_k   — Kaiming uniform (PyTorch Linear default)
            tau_g — 1.0  (identity temperature at init)
            W_g   — normal(0, 0.01)  (small init → gates start near 0.5)
            b_g   — zeros            (gates start near sigmoid(0) = 0.5)
            W_E4  — Kaiming uniform (PyTorch Linear default)

        Args:
            d_v (int): CLIP hidden dimension.
            d   (int): LLM  hidden dimension.
            d_k (int): Projection subspace dimension.
        """
        super().__init__()
        self.d_v = d_v
        self.d   = d
        self.d_k = d_k

        # Project question vector q from LLM space to shared subspace d_k
        self.W_q = nn.Linear(d, d_k, bias=False)

        # Project each patch key from CLIP space to shared subspace d_k
        self.W_k = nn.Linear(d_v, d_k, bias=False)

        # Learnable temperature for cosine similarity sharpness.
        # Init 1.0 → neutral scaling. Constrained positive via abs+eps.
        self.tau_g = nn.Parameter(torch.ones(1))

        # Gate expansion vector and bias (both in CLIP patch space d_v).
        # W_g: scalar α_i is broadcast-multiplied with W_g to expand to d_v
        # Small std so initial gates ≈ sigmoid(~0) ≈ 0.5 (soft gate)
        self.W_g = nn.Parameter(torch.empty(d_v).normal_(mean=0.0, std=0.01))
        self.b_g = nn.Parameter(torch.zeros(d_v))

        # Final projection: gated CLIP features → LLM embedding space
        self.W_E4 = nn.Linear(d_v, d, bias=True)

        # Storage for last forward pass (used by Task E-3 heatmap viz)
        self._last_alpha = None   # (N,)
        self._last_g     = None   # (N, d_v)

    def forward(self, Z_V: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Apply question-conditioned gating and project to LLM space.

        Args:
            Z_V (torch.Tensor): CLIP patch tokens,
                                shape (N, d_v) = (576, 1024).
            q   (torch.Tensor): Question vector from QuestionPooler,
                                shape (d,) = (4096,).
                                Must be on the same device and have the
                                same dtype as this module.

        Returns:
            torch.Tensor: Gated and projected visual tokens,
                          shape (N, d) = (576, 4096).
        """
        # ---------------------------------------------------------------
        # Step 1: Project and L2-normalize the question vector
        # L2 normalization AFTER projection bridges the modality gap
        # (see Modality Gap paper, Liang et al. NeurIPS 2022).
        # ---------------------------------------------------------------
        u     = self.W_q(q)                             # (d_k,)
        u_hat = F.normalize(u, dim=-1, eps=1e-8)        # (d_k,) unit vector

        # ---------------------------------------------------------------
        # Step 2: Project and L2-normalize each patch key
        # W_k maps each patch from CLIP space into the same subspace as û.
        # ---------------------------------------------------------------
        K_mat = self.W_k(Z_V)                            # (N, d_k)
        K_hat = F.normalize(K_mat, dim=-1, eps=1e-8)     # (N, d_k) unit vectors

        # ---------------------------------------------------------------
        # Step 3: Compute per-patch cosine similarity scores
        # K_hat @ u_hat = (N, d_k) @ (d_k,) = (N,)
        # Each score is the cosine similarity between û and k̂_i.
        # Divide by effective temperature (abs ensures positivity).
        # ---------------------------------------------------------------
        effective_tau = self.tau_g.abs() + 1e-6         # positive scalar
        scores = (K_hat @ u_hat) / effective_tau         # (N,)

        # ---------------------------------------------------------------
        # Step 4: Softmax over N patches → relevance weights α
        # α_i tells us how much patch i is relevant to the question.
        # ---------------------------------------------------------------
        alpha = F.softmax(scores, dim=0)                 # (N,), sums to 1

        # ---------------------------------------------------------------
        # Step 5: Compute per-patch gate vector g_i ∈ (0,1)^d_v
        # α_i W_g is scalar-vector broadcasting: (N, 1) * (d_v,) = (N, d_v)
        # This allows each channel to have a different gate scale.
        # ---------------------------------------------------------------
        gate_logits = alpha.unsqueeze(-1) * self.W_g + self.b_g  # (N, d_v)
        g = torch.sigmoid(gate_logits)                            # (N, d_v)

        # ---------------------------------------------------------------
        # Step 6: Element-wise gate Z_V then project to LLM space
        # ---------------------------------------------------------------
        gated  = Z_V * g               # (N, d_v) — suppressed patch features
        output = self.W_E4(gated)      # (N, d)

        # Store intermediates for visualisation (Task E-3) and debugging
        self._last_alpha = alpha.detach()
        self._last_g     = g.detach()

        return output


if __name__ == "__main__":
    # ============================================================
    # Verification block — Task B-E4
    # Runs on CPU or CUDA. No model download needed.
    #
    # Expected output:
    #   CHECK 1 PASSED  (output shape: torch.Size([576, 4096]))
    #   CHECK 2 PASSED  (all gate values in (0, 1))
    #   CHECK 3 PASSED  (alpha sums to 1: ...)
    #   CHECK 4 PASSED  (tau_g = 1.0000 > 0)
    #   CHECK 5 PASSED  (different questions → different gates: |G1-G2|_F = ...)
    #   ALL CHECKS PASSED
    # ============================================================

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N    = 576    # CLIP patch count
    D_V  = 1024   # CLIP hidden dimension
    D    = 4096   # LLM hidden dimension
    D_K  = 256    # shared projection subspace dimension

    torch.manual_seed(0)

    e4  = ExpertE4(d_v=D_V, d=D, d_k=D_K).to(DEVICE)

    # Random inputs: one image, two different question vectors
    Z_V = torch.randn(N, D_V, device=DEVICE)
    q1  = torch.randn(D,     device=DEVICE)   # question 1
    q2  = torch.randn(D,     device=DEVICE)   # question 2 (clearly different)

    all_passed = True

    # ----------------------------------------------------------
    # CHECK 1 — Output shape is (576, 4096)
    # ----------------------------------------------------------
    with torch.no_grad():
        V1 = e4(Z_V, q1)

    expected_shape = torch.Size([N, D])
    if V1.shape == expected_shape:
        print(f"CHECK 1 PASSED  (output shape: {V1.shape})")
    else:
        print(f"CHECK 1 FAILED: shape {V1.shape}, expected {expected_shape}")
        all_passed = False

    # ----------------------------------------------------------
    # CHECK 2 — All gate values g_i are in the open interval (0, 1)
    # Sigmoid output is strictly in (0, 1) for finite inputs.
    # ----------------------------------------------------------
    # _last_g is stored by forward() — no need to recompute
    g = e4._last_g   # (N, d_v)

    all_gt_0 = (g > 0.0).all().item()
    all_lt_1 = (g < 1.0).all().item()

    if all_gt_0 and all_lt_1:
        print(
            f"CHECK 2 PASSED  "
            f"(all gate values in (0, 1); min={g.min().item():.4f}, "
            f"max={g.max().item():.4f})"
        )
    else:
        n_bad_low  = (g <= 0.0).sum().item()
        n_bad_high = (g >= 1.0).sum().item()
        print(
            f"CHECK 2 FAILED: {n_bad_low} values ≤ 0, "
            f"{n_bad_high} values ≥ 1"
        )
        all_passed = False

    # ----------------------------------------------------------
    # CHECK 3 — Alpha (relevance weights) sums to 1 over N patches
    # _last_alpha is stored by forward()
    # ----------------------------------------------------------
    alpha    = e4._last_alpha   # (N,)
    alpha_sum = alpha.sum().item()
    tolerance = 1e-5

    if abs(alpha_sum - 1.0) < tolerance:
        print(
            f"CHECK 3 PASSED  "
            f"(α sums to {alpha_sum:.8f} ≈ 1.0; "
            f"deviation < {tolerance})"
        )
    else:
        print(
            f"CHECK 3 FAILED: α.sum() = {alpha_sum:.8f}, "
            f"expected 1.0 (deviation ≥ {tolerance})"
        )
        all_passed = False

    # ----------------------------------------------------------
    # CHECK 4 — tau_g is positive at initialisation (value = 1.0)
    # Monitor this during training: assert model.qcgp.tau_g.item() > 0
    # ----------------------------------------------------------
    tau_val = e4.tau_g.item()
    if tau_val > 0:
        print(f"CHECK 4 PASSED  (tau_g = {tau_val:.4f} > 0)")
    else:
        print(
            f"CHECK 4 FAILED: tau_g = {tau_val:.4f} ≤ 0 "
            f"(add training monitor: assert e4.tau_g.item() > 0)"
        )
        all_passed = False

    # ----------------------------------------------------------
    # CHECK 5 — Different questions → different gate matrices
    # Same image (Z_V), two different questions (q1, q2).
    # The gate G = (N, d_v) should differ between the two.
    # ----------------------------------------------------------
    with torch.no_grad():
        V2 = e4(Z_V, q2)

    G1 = e4._last_g.clone()    # gate for question 1 (already computed above)

    with torch.no_grad():
        _  = e4(Z_V, q2)       # forward with q2 to update _last_g
    G2 = e4._last_g.clone()

    # Recompute G1 cleanly
    with torch.no_grad():
        _ = e4(Z_V, q1)
    G1 = e4._last_g.clone()

    frob_norm = (G1 - G2).norm(p='fro').item()

    if frob_norm > 0.0:
        print(
            f"CHECK 5 PASSED  "
            f"(|G1 − G2|_F = {frob_norm:.4f} > 0 — "
            f"gates differ between questions)"
        )
    else:
        print(
            f"CHECK 5 FAILED: |G1 − G2|_F = {frob_norm:.6f} "
            f"(expected > 0 — gates should differ for different questions)"
        )
        all_passed = False

    # ----------------------------------------------------------
    # Parameter count summary (informational)
    # Expected:
    #   W_q:  d × d_k    = 4096 × 256  = 1,048,576
    #   W_k:  d_v × d_k  = 1024 × 256  =   262,144
    #   tau_g             =              =         1
    #   W_g:  d_v         = 1024        =     1,024
    #   b_g:  d_v         = 1024        =     1,024
    #   W_E4: d_v × d + d = 1024×4096+4096 = 4,198,400
    #   Total                             = 5,511,169
    # ----------------------------------------------------------
    total_params = sum(p.numel() for p in e4.parameters())
    expected_total = (D * D_K) + (D_V * D_K) + 1 + D_V + D_V + (D_V * D + D)
    print()
    print(
        f"Parameter count: {total_params:,} "
        f"(expected {expected_total:,})"
    )

    print()
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
