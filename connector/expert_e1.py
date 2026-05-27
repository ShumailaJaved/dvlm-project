# ============================================================
# connector/expert_e1.py
# Task B-E1: MLP Expert (Reuse LLaVA-1.5 pretrained projector)
# ============================================================
# WHERE TO RUN:
#   Verification (__main__): anywhere — CPU/Mac or Colab.
#   In training:             Google Colab with T4 GPU (Linux).
#
# COMMAND (run verification):
#   python connector/expert_e1.py
#
# USAGE IN TRAINING:
#   from connector.expert_e1 import ExpertE1
#   e1 = ExpertE1(model.model.mm_projector)
#   V = e1(Z_V)   # (576, 1024) → (576, 4096)
# ============================================================

import torch
import torch.nn as nn


class ExpertE1(nn.Module):
    """
    Expert E1: Two-layer MLP connector (LLaVA-1.5 baseline).

    Wraps LLaVA-1.5's pretrained mm_projector as a frozen expert.
    The same linear + GELU + linear transformation is applied to every
    patch token independently — it is question-blind and spatially
    equivariant.

    Architecture (LLaVA-1.5 mlp2x_gelu):
        Linear(d_v=1024, d=4096) → GELU → Linear(d=4096, d=4096)

    Output length: L_1 = N = 576  (full spatial resolution preserved)
    Question-aware: No

    Args:
        pretrained_mlp (nn.Module): The mm_projector from a loaded
            LLaVA model. In practice: model.model.mm_projector
    """

    def __init__(self, pretrained_mlp: nn.Module):
        """
        Initialise ExpertE1.

        The pretrained MLP weights are frozen immediately so that E1
        introduces zero new trainable parameters — it reuses the
        pretrained CLIP→Vicuna alignment as-is.

        Args:
            pretrained_mlp (nn.Module): LLaVA's pretrained mm_projector.
        """
        super().__init__()
        self.mlp = pretrained_mlp

        # Freeze every parameter — no gradient updates for E1
        for param in self.mlp.parameters():
            param.requires_grad = False

    def forward(self, Z_V: torch.Tensor) -> torch.Tensor:
        """
        Project CLIP patch tokens to LLM embedding space.

        The same MLP is applied identically to every token (no
        interaction across patch positions).

        Args:
            Z_V (torch.Tensor): CLIP patch tokens,
                                shape (N, d_v) = (576, 1024).

        Returns:
            torch.Tensor: Projected tokens, shape (N, d) = (576, 4096).
        """
        return self.mlp(Z_V)


if __name__ == "__main__":
    # ============================================================
    # Verification block — Task B-E1
    # Uses a mock mm_projector matching LLaVA-1.5's mlp2x_gelu
    # architecture. No model download needed.
    #
    # Expected output:
    #   CHECK 1 PASSED  (output shape: torch.Size([576, 4096]))
    #   CHECK 2 PASSED  (0 trainable parameters)
    #   ALL CHECKS PASSED
    # ============================================================

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N    = 576    # CLIP patch count
    D_V  = 1024   # CLIP hidden dimension
    D    = 4096   # LLM hidden dimension

    torch.manual_seed(0)

    # Build a mock mm_projector matching LLaVA-1.5 mlp2x_gelu:
    #   Linear(1024, 4096) → GELU → Linear(4096, 4096)
    mock_mlp = nn.Sequential(
        nn.Linear(D_V, D),
        nn.GELU(),
        nn.Linear(D, D),
    ).to(DEVICE)

    # Wrap with ExpertE1 — parameters should be frozen after this
    e1 = ExpertE1(mock_mlp).to(DEVICE)

    # Random input: one image worth of CLIP patch tokens
    Z_V = torch.randn(N, D_V, device=DEVICE)

    all_passed = True

    # ----------------------------------------------------------
    # CHECK 1 — Output shape is (576, 4096)
    # ----------------------------------------------------------
    with torch.no_grad():
        V = e1(Z_V)

    expected_shape = torch.Size([N, D])
    if V.shape == expected_shape:
        print(f"CHECK 1 PASSED  (output shape: {V.shape})")
    else:
        print(f"CHECK 1 FAILED: shape {V.shape}, expected {expected_shape}")
        all_passed = False

    # ----------------------------------------------------------
    # CHECK 2 — Zero trainable parameters (all frozen)
    # ----------------------------------------------------------
    n_trainable = sum(p.numel() for p in e1.parameters() if p.requires_grad)
    if n_trainable == 0:
        print(f"CHECK 2 PASSED  (trainable parameters: {n_trainable})")
    else:
        print(
            f"CHECK 2 FAILED: {n_trainable} trainable parameters "
            f"(expected 0 — all weights should be frozen)"
        )
        all_passed = False

    print()
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
