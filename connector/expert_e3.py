# ============================================================
# connector/expert_e3.py
# Task B-E3: Attention-Pooled Global Token Expert
# ============================================================
# WHERE TO RUN:
#   Verification (__main__): anywhere — CPU/Mac or Colab.
#   In training:             Google Colab with T4 GPU (Linux).
#
# COMMAND (run verification):
#   python connector/expert_e3.py
#
# USAGE IN TRAINING:
#   from connector.expert_e3 import ExpertE3
#   e3 = ExpertE3(d_v=1024, d=4096).to(device).half()
#   V = e3(Z_V)   # (576, 1024) → (1, 4096)
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertE3(nn.Module):
    """
    Expert E3: Attention-pooled global token connector.

    Collapses all N=576 CLIP patch tokens into a single summary
    vector c ∈ R^d_v using learned attention weights, then projects
    to LLM space.  The entire image is represented as one token.

    Formula:
        scores_i = w^T z_i^V / √d_v              scalar per patch
        a_i      = softmax_i(scores)              attention weight ∈ (0,1)
        c        = Σ_i a_i · z_i^V               (d_v,)  weighted sum
        φ₃       = W_E3(c)                        (1, d)  after unsqueeze

    Output length: L_3 = 1  (maximally compact; all spatial info lost)
    Question-aware: No

    Args:
        d_v (int): CLIP hidden dimension.  Default: 1024
        d   (int): LLM  hidden dimension.  Default: 4096
    """

    def __init__(self, d_v: int = 1024, d: int = 4096):
        """
        Initialise ExpertE3.

        Initialisation:
            w    — normal(0, 1)   : scores start randomly distributed
            W_E3 — Kaiming uniform (PyTorch Linear default, bias included)

        Args:
            d_v (int): CLIP hidden dimension.
            d   (int): LLM  hidden dimension.
        """
        super().__init__()
        self.d_v = d_v
        self.d   = d

        # Learned d_v-dimensional attention scoring vector.
        # w^T z_i^V computes a relevance score for each patch.
        self.w = nn.Parameter(torch.empty(d_v).normal_(mean=0.0, std=1.0))

        # Project d_v-dim summary to LLM's d-dim space (with bias)
        self.W_E3 = nn.Linear(d_v, d, bias=True)

    def forward(self, Z_V: torch.Tensor) -> torch.Tensor:
        """
        Collapse N patch tokens to a single summary token.

        Args:
            Z_V (torch.Tensor): CLIP patch tokens,
                                shape (N, d_v) = (576, 1024).

        Returns:
            torch.Tensor: Single summary token in LLM space,
                          shape (1, d) = (1, 4096).
                          The leading dimension is required by the LLM
                          (sequence of length 1).
        """
        # --- Step 1: score each patch against the attention vector w ----------
        # Z_V @ w  →  (N, d_v) @ (d_v,)  =  (N,)
        # Divide by sqrt(d_v) to keep scores well-scaled
        scores  = Z_V @ self.w / math.sqrt(self.d_v)   # (N,)

        # --- Step 2: softmax over N patches → attention weights ---------------
        # weights[i] ≥ 0 and sum(weights) == 1
        weights = F.softmax(scores, dim=0)              # (N,)

        # --- Step 3: weighted sum of patch embeddings -------------------------
        # weights @ Z_V  →  (N,) @ (N, d_v)  =  (d_v,)
        c = weights @ Z_V                               # (d_v,)

        # --- Step 4: project to LLM space, add sequence dimension -------------
        out = self.W_E3(c)                              # (d,)
        return out.unsqueeze(0)                         # (1, d)


if __name__ == "__main__":
    # ============================================================
    # Verification block — Task B-E3
    # Runs on CPU or CUDA. No model download needed.
    #
    # Expected output:
    #   CHECK 1 PASSED  (output shape: torch.Size([1, 4096]))
    #   CHECK 2 PASSED  (attention weights sum to 1 ...)
    #   CHECK 2 INFO    (peak patch shifts between images: ...)
    #   CHECK 3 PASSED  (trainable parameters: 4,199,424)
    #   ALL CHECKS PASSED
    # ============================================================

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N    = 576    # CLIP patch count
    D_V  = 1024   # CLIP hidden dimension
    D    = 4096   # LLM hidden dimension

    torch.manual_seed(0)

    e3  = ExpertE3(d_v=D_V, d=D).to(DEVICE)

    # Two different mock images (different random patch features)
    torch.manual_seed(1)
    Z_V1 = torch.randn(N, D_V, device=DEVICE)
    torch.manual_seed(2)
    Z_V2 = torch.randn(N, D_V, device=DEVICE)

    all_passed = True

    # ----------------------------------------------------------
    # CHECK 1 — Output shape is (1, 4096)
    # ----------------------------------------------------------
    with torch.no_grad():
        V1 = e3(Z_V1)

    expected_shape = torch.Size([1, D])
    if V1.shape == expected_shape:
        print(f"CHECK 1 PASSED  (output shape: {V1.shape})")
    else:
        print(f"CHECK 1 FAILED: shape {V1.shape}, expected {expected_shape}")
        all_passed = False

    # ----------------------------------------------------------
    # CHECK 2 — Attention weights sum to 1; peak patch shifts
    # between different images
    # ----------------------------------------------------------
    with torch.no_grad():
        # Compute attention weights manually (same as forward step 1-2)
        scores1  = Z_V1 @ e3.w / math.sqrt(D_V)   # (N,)
        weights1 = F.softmax(scores1, dim=0)       # (N,)

        scores2  = Z_V2 @ e3.w / math.sqrt(D_V)
        weights2 = F.softmax(scores2, dim=0)

    sum1 = weights1.sum().item()
    sum2 = weights2.sum().item()

    if abs(sum1 - 1.0) < 1e-4 and abs(sum2 - 1.0) < 1e-4:
        print(
            f"CHECK 2 PASSED  "
            f"(attention weights sum to 1: image1={sum1:.6f}, image2={sum2:.6f})"
        )
    else:
        print(
            f"CHECK 2 FAILED: weight sums = {sum1:.6f}, {sum2:.6f} "
            f"(expected both ≈ 1.0)"
        )
        all_passed = False

    peak1 = weights1.argmax().item()
    peak2 = weights2.argmax().item()
    print(
        f"CHECK 2 INFO    "
        f"(peak patch index: image1={peak1}, image2={peak2} "
        f"— shifts={'yes' if peak1 != peak2 else 'no, same peak (coincidental)'})"
    )

    # ----------------------------------------------------------
    # CHECK 3 — Trainable parameter count
    # Expected: d_v + (d_v · d + d)  =  1024 + (1024×4096 + 4096)
    #           = 1024 + 4,198,400 = 4,199,424
    # ----------------------------------------------------------
    expected_params = D_V + (D_V * D + D)    # w + W_E3_weight + W_E3_bias
    actual_params   = sum(p.numel() for p in e3.parameters())

    if actual_params == expected_params:
        print(
            f"CHECK 3 PASSED  "
            f"(trainable parameters: {actual_params:,} "
            f"= {D_V} + {D_V}×{D} + {D})"
        )
    else:
        print(
            f"CHECK 3 FAILED: {actual_params:,} parameters, "
            f"expected {expected_params:,}"
        )
        all_passed = False

    print()
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
