# ============================================================
# connector/router.py
# Task C-1: Lightweight Question-Conditioned Router (MoCRouter)
# ============================================================
# WHERE TO RUN:
#   Verification (__main__): Mac/MPS or Colab (CPU or GPU).
#   In training:             Google Colab with T4 / A100 (Linux).
#
# COMMAND (run verification):
#   python connector/router.py
#
# USAGE IN TRAINING:
#   from connector.router import MoCRouter
#   router = MoCRouter(d=4096, d_r=64, K=4).to(device)   # keep float32 (no .half())
#   r = router(q)                            # (4,) softmax probs
#   k_star = torch.argmax(r.detach()).item() # STE — no grad through argmax
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCRouter(nn.Module):
    """
    Lightweight two-layer MLP router for Mixture of Connectors.

    Takes a question vector q ∈ R^d and produces a K-way probability
    distribution r over the four expert connectors.

    Architecture:
        h  = GELU(W1 q)     (d_r,)  — hidden representation
        r  = softmax(W2 h)  (K,)    — expert routing probabilities

    Initialization note:
        W2 weight and bias are both initialized to ZEROS.
        This means at the start of training, r = softmax(0) = [1/K, …, 1/K]
        for any input q. This prevents any single expert from dominating early.

    Straight-through estimator (used in the training loop, not here):
        k_star = torch.argmax(r.detach())   # discrete, no gradient
        # Gradients from L_lb flow back through r (not through k_star)

    Args:
        d   (int): LLM hidden dimension.     Default: 4096
        d_r (int): Router hidden dimension.  Default: 64
        K   (int): Number of experts.        Default: 4
    """

    def __init__(self, d: int = 4096, d_r: int = 64, K: int = 4):
        """
        Initialise MoCRouter.

        W1 — Kaiming uniform (PyTorch Linear default)
        W2 — zeros (weight + bias), so r = 1/K at the start of training

        Args:
            d   (int): Input (question vector) dimension.
            d_r (int): Hidden dimension of the router MLP.
            K   (int): Number of output expert classes.
        """
        super().__init__()
        self.d   = d
        self.d_r = d_r
        self.K   = K

        # First layer: project question to router hidden space
        self.W1 = nn.Linear(d, d_r)        # Kaiming uniform (default)

        # Second layer: hidden → expert logits, ZEROS init for uniform start
        self.W2 = nn.Linear(d_r, K)
        nn.init.zeros_(self.W2.weight)      # all-zero weights
        nn.init.zeros_(self.W2.bias)        # all-zero bias → softmax([0,…]) = 1/K

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute the routing probability distribution over K experts.

        Args:
            q (torch.Tensor): Question vector from QuestionPooler,
                              shape (d,). Same device and dtype as
                              the module.

        Returns:
            torch.Tensor: Routing probabilities r of shape (K,).
                          r.sum() == 1.0 and r_k > 0 for all k.
        """
        # Upcast to float32: W1 multiplies q (d=4096) by a (d_r=64, d=4096)
        # weight matrix — the row-wise dot products each sum 4096 float16
        # values and can overflow.  float32 accumulation is safe.
        _orig_dtype = q.dtype
        q32  = q.float()
        W1   = self.W1.weight.float()
        b1   = self.W1.bias.float()
        W2   = self.W2.weight.float()
        b2   = self.W2.bias.float()
        h    = F.gelu(F.linear(q32, W1, b1))        # (d_r,) float32
        r    = F.softmax(F.linear(h, W2, b2), dim=-1)  # (K,) float32
        return r.to(_orig_dtype)


if __name__ == "__main__":
    # ============================================================
    # Verification block — Task C-1
    # Runs on CPU (Mac/MPS) or CUDA (Colab).
    #
    # Expected output:
    #   CHECK 1 PASSED  (r at init ≈ [0.25, 0.25, 0.25, 0.25])
    #   CHECK 2 PASSED  (W2.weight.grad is non-zero after L_lb backward)
    #   CHECK 3 PASSED  (k_star = 0 ∈ {0, 1, 2, 3})
    #   ALL CHECKS PASSED
    # ============================================================

    DEVICE = "mps" if torch.backends.mps.is_available() else (
             "cuda" if torch.cuda.is_available() else "cpu")
    D   = 4096   # LLM hidden dimension
    D_R = 64     # router hidden dimension
    K   = 4      # number of experts
    B   = 8      # batch size for gradient check

    torch.manual_seed(0)
    print(f"Verification device: {DEVICE}")
    print()

    router = MoCRouter(d=D, d_r=D_R, K=K).to(DEVICE)

    all_passed = True

    # ----------------------------------------------------------
    # CHECK 1 — At init, r ≈ [0.25, 0.25, 0.25, 0.25] for any q
    # Since W2.weight = W2.bias = 0, W2(h) = 0 → softmax([0,0,0,0])
    # ----------------------------------------------------------
    q_test = torch.randn(D, device=DEVICE)
    with torch.no_grad():
        r = router(q_test)

    expected = torch.full((K,), 1.0 / K, device=DEVICE)
    max_dev  = (r - expected).abs().max().item()
    tol      = 1e-6

    if max_dev < tol:
        print(f"CHECK 1 PASSED  (r at init ≈ {r.tolist()}, max dev = {max_dev:.2e})")
    else:
        print(f"CHECK 1 FAILED: r = {r.tolist()}, expected all {1/K:.4f} "
              f"(max dev = {max_dev:.2e} ≥ {tol})")
        all_passed = False

    # ----------------------------------------------------------
    # CHECK 2 — After one backward step on L_lb, W2.weight.grad ≠ 0
    #
    # We simulate a batch of B questions, compute r for each, then
    # compute a load-balancing loss L_lb = K * Σ_k f_k * p_k and
    # backprop. The gradient should reach W2 (via the STE path).
    # ----------------------------------------------------------
    router.zero_grad()

    # Compute r for a batch of questions
    q_batch = torch.randn(B, D, device=DEVICE)
    r_batch = torch.stack([router(q) for q in q_batch])   # (B, K)

    # Discrete expert selection (straight-through estimator — detached)
    k_stars = torch.argmax(r_batch.detach(), dim=-1)   # (B,), all 0s at init

    # f_k: fraction dispatched to each expert (no gradient)
    one_hot = F.one_hot(k_stars, num_classes=K).float()   # (B, K)
    f_k = one_hot.mean(dim=0).detach()                    # (K,)

    # p_k: mean router probability (HAS gradient — carries grad to router)
    p_k = r_batch.mean(dim=0)   # (K,)

    # Load-balancing loss (Switch Transformer formula)
    L_lb = K * (f_k * p_k).sum()
    L_lb.backward()

    grad = router.W2.weight.grad
    if grad is not None and grad.abs().sum().item() > 0:
        print(
            f"CHECK 2 PASSED  "
            f"(W2.weight.grad is non-zero; |grad|_sum = {grad.abs().sum().item():.6f})"
        )
    else:
        print(
            f"CHECK 2 FAILED: W2.weight.grad = "
            f"{'None' if grad is None else grad}"
        )
        all_passed = False

    # ----------------------------------------------------------
    # CHECK 3 — k_star is an integer in {0, 1, 2, 3}
    # With W2=0, argmax([0.25, 0.25, 0.25, 0.25]) returns 0 (ties → first)
    # ----------------------------------------------------------
    with torch.no_grad():
        r_check = router(torch.randn(D, device=DEVICE))
    k_star = torch.argmax(r_check.detach()).item()

    if k_star in {0, 1, 2, 3}:
        print(f"CHECK 3 PASSED  (k_star = {k_star} ∈ {{0, 1, 2, 3}})")
    else:
        print(f"CHECK 3 FAILED: k_star = {k_star}, expected value in {{0,1,2,3}}")
        all_passed = False

    print()
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
