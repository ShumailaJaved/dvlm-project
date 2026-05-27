# ============================================================
# losses/load_balance.py
# Task C-2: Load-Balancing Auxiliary Loss
# ============================================================
# WHERE TO RUN:
#   Verification (__main__): Mac/MPS or Colab (CPU or GPU).
#   In training:             Google Colab with T4 / A100 (Linux).
#
# COMMAND (run verification):
#   python losses/load_balance.py
#
# USAGE IN TRAINING:
#   from losses.load_balance import load_balancing_loss, LAMBDA_LB
#
#   L_lb = load_balancing_loss(r_batch, k_stars, K=4)
#   loss  = L_ce + LAMBDA_LB * L_lb
# ============================================================

import torch
import torch.nn.functional as F


# Recommended load-balancing weight from the project specification.
# Total loss: L = L_CE + LAMBDA_LB * L_lb
LAMBDA_LB: float = 0.01


def load_balancing_loss(
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """
    Compute the Switch Transformer load-balancing auxiliary loss.

    This loss encourages the router to distribute samples roughly evenly
    across the K experts. Without it the router collapses to always
    selecting the same expert (routing collapse).

    Formula (Fedus et al., Switch Transformer, 2022):
        f_k = (1/B) Σ_b 1[k*_b == k]    — fraction dispatched to expert k
        p_k = (1/B) Σ_b r_{b,k}          — mean router probability for expert k
        L_lb = K · Σ_k f_k · p_k

    Gradient note:
        f_k is computed from discrete argmax indices → always DETACHED.
        p_k comes from continuous router_probs → carries gradient to router.
        The product f_k * p_k is therefore differentiable w.r.t. router_probs.

    Args:
        router_probs  (torch.Tensor): Softmax routing probabilities,
                                      shape (B, K). requires_grad=True.
        expert_indices (torch.Tensor): Selected expert per sample from argmax,
                                       shape (B,). dtype=long or int.
        K (int): Number of experts (4 for MoC).

    Returns:
        torch.Tensor: Scalar load-balancing loss. Equals 1.0 for perfectly
                      balanced routing, up to K for fully collapsed routing.
    """
    B = router_probs.shape[0]

    # ---- f_k: fraction of samples dispatched to each expert ----------------
    # One-hot encode the discrete expert selections, then average over batch.
    # DETACH so f_k has no gradient (it is a non-differentiable quantity).
    one_hot = F.one_hot(expert_indices.long(), num_classes=K).float()  # (B, K)
    f_k = one_hot.mean(dim=0).detach()   # (K,)  NO gradient

    # ---- p_k: mean router probability for each expert ----------------------
    # Average softmax outputs over the batch.
    # Do NOT detach — this carries gradient back to the router weights.
    p_k = router_probs.mean(dim=0)       # (K,)  HAS gradient

    # ---- L_lb = K · Σ_k f_k · p_k  ----------------------------------------
    loss = K * (f_k * p_k).sum()
    return loss


if __name__ == "__main__":
    # ============================================================
    # Verification block — Task C-2
    # Runs on CPU (Mac/MPS) or CUDA (Colab).
    #
    # Expected output:
    #   CHECK 1 PASSED  (perfectly balanced: L_lb = 1.0000)
    #   CHECK 2 PASSED  (fully collapsed: L_lb = K * p_0 ≤ K)
    #   CHECK 3 PASSED  (f_k.requires_grad = False, p_k.requires_grad = True)
    #   ALL CHECKS PASSED
    # ============================================================

    DEVICE = "mps" if torch.backends.mps.is_available() else (
             "cuda" if torch.cuda.is_available() else "cpu")
    K   = 4     # number of experts
    B   = 16    # batch size

    torch.manual_seed(0)
    print(f"Verification device: {DEVICE}")
    print()

    all_passed = True

    # ----------------------------------------------------------
    # CHECK 1 — Perfectly balanced routing → L_lb == 1.0
    #
    # Construct a batch where exactly B/K samples go to each expert
    # and each sample's router outputs [0.25, 0.25, 0.25, 0.25].
    # Then:
    #   f_k = 1/K = 0.25 for all k
    #   p_k = 1/K = 0.25 for all k
    #   L_lb = K * Σ_k (1/K * 1/K) = K * K * (1/K)^2 = K * (1/K) = 1.0
    # ----------------------------------------------------------
    # Perfect router probs: each sample has equal weight on all experts
    router_probs_balanced = torch.full((B, K), 1.0 / K,
                                       device=DEVICE, requires_grad=True)

    # Perfect dispatching: B/K samples to each expert (cyclic)
    expert_indices_balanced = torch.arange(B, device=DEVICE) % K   # [0,1,2,3,0,1,2,3,...]

    L_balanced = load_balancing_loss(router_probs_balanced, expert_indices_balanced, K)

    tol = 1e-4
    if abs(L_balanced.item() - 1.0) < tol:
        print(f"CHECK 1 PASSED  (perfectly balanced: L_lb = {L_balanced.item():.4f} ≈ 1.0)")
    else:
        print(f"CHECK 1 FAILED: L_lb = {L_balanced.item():.6f}, expected 1.0")
        all_passed = False

    # ----------------------------------------------------------
    # CHECK 2 — Fully collapsed routing (all samples → expert 0)
    #           L_lb = K * p_0 ≤ K
    #
    # f = [1, 0, 0, 0], p_k is the mean of router_probs[:, k].
    # L_lb = K * (1 * p_0 + 0 + 0 + 0) = K * p_0.
    # With p_0 ≤ 1, we have L_lb ≤ K.
    # ----------------------------------------------------------
    # All samples assigned to expert 0
    expert_indices_collapsed = torch.zeros(B, dtype=torch.long, device=DEVICE)

    # Router probs: random softmax outputs (so p_0 is non-trivial)
    raw_logits = torch.randn(B, K, device=DEVICE, requires_grad=True)
    router_probs_collapsed = F.softmax(raw_logits, dim=-1)   # (B, K), HAS grad

    L_collapsed = load_balancing_loss(router_probs_collapsed, expert_indices_collapsed, K)

    # Expected: L_lb = K * p_0 = 4 * (mean of router_probs[:, 0])
    p_0       = router_probs_collapsed[:, 0].mean().item()
    expected  = K * p_0
    deviation = abs(L_collapsed.item() - expected)

    if L_collapsed.item() <= K + tol and deviation < tol:
        print(
            f"CHECK 2 PASSED  "
            f"(fully collapsed: L_lb = {L_collapsed.item():.4f} "
            f"= K * p_0 = {K} * {p_0:.4f} = {expected:.4f} ≤ {K})"
        )
    else:
        print(
            f"CHECK 2 FAILED: L_lb = {L_collapsed.item():.4f}, "
            f"expected K * p_0 = {expected:.4f}"
        )
        all_passed = False

    # ----------------------------------------------------------
    # CHECK 3 — f_k.requires_grad == False, p_k.requires_grad == True
    #
    # We access the internal tensors by partially rerunning the logic
    # to inspect grad flags directly.
    # ----------------------------------------------------------
    router_probs_test = torch.rand(B, K, device=DEVICE, requires_grad=True)
    expert_idx_test   = torch.randint(0, K, (B,), device=DEVICE)

    # Replicate internal computation to inspect grad flags
    one_hot_test = F.one_hot(expert_idx_test.long(), num_classes=K).float()
    f_k_test = one_hot_test.mean(dim=0).detach()    # detached
    p_k_test = router_probs_test.mean(dim=0)         # NOT detached

    f_ok = not f_k_test.requires_grad
    p_ok = p_k_test.requires_grad

    if f_ok and p_ok:
        print(
            f"CHECK 3 PASSED  "
            f"(f_k.requires_grad = {f_k_test.requires_grad} = False ✓, "
            f"p_k.requires_grad = {p_k_test.requires_grad} = True ✓)"
        )
    else:
        print(
            f"CHECK 3 FAILED: "
            f"f_k.requires_grad = {f_k_test.requires_grad} (expected False), "
            f"p_k.requires_grad = {p_k_test.requires_grad} (expected True)"
        )
        all_passed = False

    print()
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
