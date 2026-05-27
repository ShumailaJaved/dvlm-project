# ============================================================
# connector/expert_e2.py
# Task B-E2: Q-Former Expert (cross-attention compression to 32 tokens)
# ============================================================
# WHERE TO RUN:
#   Verification (__main__): anywhere — CPU/Mac or Colab.
#   In training:             Google Colab with T4 GPU (Linux).
#
# COMMAND (run verification):
#   python connector/expert_e2.py
#
# USAGE IN TRAINING:
#   from connector.expert_e2 import ExpertE2
#   e2 = ExpertE2(d_v=1024, d=4096, K=32).to(device).half()
#   V = e2(Z_V)   # (576, 1024) → (32, 4096)
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertE2(nn.Module):
    """
    Expert E2: Q-Former cross-attention connector.

    Compresses N=576 CLIP patch tokens to K=32 output tokens via
    cross-attention.  K fixed learnable query vectors Q ∈ R^(K×d)
    attend over all patch positions, producing a K-length summary.

    Formula:
        K_v = Z_V @ W_K^T          (N, d)  — patch keys in LLM space
        V_v = Z_V @ W_V^T          (N, d)  — patch values in LLM space
        φ₂  = softmax(Q K_v^T / √d) V_v    (K, d)

    Output length: L_2 = K = 32  (compressed; spatial info partially lost)
    Question-aware: No  (Q is input-independent)

    Args:
        d_v (int): CLIP hidden dimension.    Default: 1024
        d   (int): LLM  hidden dimension.   Default: 4096
        K   (int): Number of query vectors. Default: 32

    Note:
        Q is a fixed learnable parameter — it is NOT derived from the
        input. Do not confuse it with the question vector q from
        QuestionPooler.
    """

    def __init__(self, d_v: int = 1024, d: int = 4096, K: int = 32):
        """
        Initialise ExpertE2.

        Initialisation:
            Q    — Xavier uniform (appropriate scale for attention)
            W_K  — Kaiming uniform (PyTorch Linear default)
            W_V  — Kaiming uniform (PyTorch Linear default)

        Args:
            d_v (int): CLIP hidden dimension.
            d   (int): LLM  hidden dimension.
            K   (int): Number of learnable query tokens.
        """
        super().__init__()
        self.d   = d
        self.K   = K
        self.d_v = d_v

        # K learnable query vectors: each row is a d-dim query
        # Xavier uniform keeps scale suitable for dot-product attention.
        self.Q = nn.Parameter(torch.empty(K, d))
        nn.init.xavier_uniform_(self.Q)

        # Project CLIP patch tokens to LLM space for keys and values.
        # No bias — projections should be zero-centred at init.
        self.W_K = nn.Linear(d_v, d, bias=False)
        self.W_V = nn.Linear(d_v, d, bias=False)

    def forward(self, Z_V: torch.Tensor) -> torch.Tensor:
        """
        Compress N patch tokens to K output tokens via cross-attention.

        Args:
            Z_V (torch.Tensor): CLIP patch tokens,
                                shape (N, d_v) = (576, 1024).

        Returns:
            torch.Tensor: Compressed visual tokens,
                          shape (K, d) = (32, 4096).
        """
        # Project patches to key and value in LLM embedding space
        K_v = self.W_K(Z_V)   # (N, d)
        V_v = self.W_V(Z_V)   # (N, d)

        # Cross-attention: Q (K queries) attend over N patch positions.
        # scaled_dot_product_attention computes:
        #   softmax(Q @ K.T / sqrt(d)) @ V
        # We add a batch dimension (1) as required by the function.
        q = self.Q.unsqueeze(0)    # (1, K, d)
        k = K_v.unsqueeze(0)       # (1, N, d)
        v = V_v.unsqueeze(0)       # (1, N, d)

        # scale_factor = 1/sqrt(d) is applied internally
        out = F.scaled_dot_product_attention(q, k, v)   # (1, K, d)
        return out.squeeze(0)   # (K, d) = (32, 4096)


if __name__ == "__main__":
    # ============================================================
    # Verification block — Task B-E2
    # Runs on CPU or CUDA. No model download needed.
    #
    # Expected output:
    #   CHECK 1 PASSED  (output shape: torch.Size([32, 4096]))
    #   CHECK 2 PASSED  (max attention weight row-sum deviation: ...)
    #   CHECK 3 PASSED  (trainable parameters: 8,519,680)
    #   ALL CHECKS PASSED
    # ============================================================

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N    = 576    # CLIP patch count
    D_V  = 1024   # CLIP hidden dimension
    D    = 4096   # LLM hidden dimension
    K    = 32     # number of Q-Former query tokens

    torch.manual_seed(0)

    e2  = ExpertE2(d_v=D_V, d=D, K=K).to(DEVICE)
    Z_V = torch.randn(N, D_V, device=DEVICE)

    all_passed = True

    # ----------------------------------------------------------
    # CHECK 1 — Output shape is (32, 4096)
    # ----------------------------------------------------------
    with torch.no_grad():
        V = e2(Z_V)

    expected_shape = torch.Size([K, D])
    if V.shape == expected_shape:
        print(f"CHECK 1 PASSED  (output shape: {V.shape})")
    else:
        print(f"CHECK 1 FAILED: shape {V.shape}, expected {expected_shape}")
        all_passed = False

    # ----------------------------------------------------------
    # CHECK 2 — Attention weights sum to 1 over N=576 for each query
    # Compute attention weights manually using the same formula as
    # scaled_dot_product_attention (before the weighted-sum step).
    # ----------------------------------------------------------
    with torch.no_grad():
        K_v = e2.W_K(Z_V)                              # (N, d)
        # scores: (K, N) = Q @ K_v.T / sqrt(d)
        scores  = (e2.Q @ K_v.T) / math.sqrt(D)       # (K, N)
        weights = F.softmax(scores, dim=-1)            # (K, N), rows sum to 1

    # Each of the K rows should sum to 1.0
    row_sums  = weights.sum(dim=-1)                    # (K,)
    max_dev   = (row_sums - 1.0).abs().max().item()
    tolerance = 1e-4

    if max_dev < tolerance:
        print(
            f"CHECK 2 PASSED  "
            f"(max attention weight row-sum deviation: {max_dev:.2e} < {tolerance})"
        )
    else:
        print(
            f"CHECK 2 FAILED: max row-sum deviation = {max_dev:.6f} "
            f"(expected < {tolerance})"
        )
        all_passed = False

    # ----------------------------------------------------------
    # CHECK 3 — Trainable parameter count
    # Expected: K·d + 2·d_v·d = 32×4096 + 2×1024×4096 = 8,519,680
    # ----------------------------------------------------------
    expected_params = K * D + 2 * D_V * D
    actual_params   = sum(p.numel() for p in e2.parameters())

    if actual_params == expected_params:
        print(
            f"CHECK 3 PASSED  "
            f"(trainable parameters: {actual_params:,} = "
            f"{K}×{D} + 2×{D_V}×{D})"
        )
    else:
        print(
            f"CHECK 3 FAILED: {actual_params:,} parameters, "
            f"expected {expected_params:,}"
        )
        all_passed = False

    print()
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
