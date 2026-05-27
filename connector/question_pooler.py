# ============================================================
# connector/question_pooler.py
# Task A-1: Attention-Pooled Question Vector Extractor
# ============================================================
# WHERE TO RUN:
#   Verification (__main__ block): anywhere — CPU/Mac or Colab.
#                                  No LLaVA model download required.
#   In actual training use:        Google Colab, T4 GPU (Linux).
#
# COMMAND (run verification):
#   python connector/question_pooler.py
#
# USAGE IN TRAINING (Task A-1.2 integration):
#   from connector.question_pooler import QuestionPooler
#
#   pooler = QuestionPooler(d=4096).to(device).half()   # match LLM dtype
#   question_input_ids = tokenizer(question, ...)['input_ids'][0]  # (T,)
#   U = model.model.embed_tokens(question_input_ids)    # (T, 4096)
#   q = pooler(U)                                       # (4096,)
#   # Pass q to MoCRouter and ExpertE4 (QCGP)
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuestionPooler(nn.Module):
    """
    Attention-pooled question vector extractor (Task A-1).

    Collapses a sequence of T LLM token embeddings into a single
    d-dimensional question vector q using a learned attention mechanism.

    The module holds one learnable parameter, q_pool ∈ R^d, which acts
    as an attention query. Each token embedding in U is scored against
    q_pool, the scores are softmax-normalised over T positions, and the
    weighted sum of embeddings gives q.

    Mathematical formula (matches the paper's MoC subsection):
        scores  = (q_pool  @ U.T) / sqrt(d)    shape: (T,)
        weights = softmax(scores, dim=0)         shape: (T,), sums to 1
        q       = weights @ U                    shape: (d,)

    Args:
        d (int): LLM hidden dimension.  d = 4096 for LLaVA-1.5-7B.

    Important:
        Call .half() on this module when pairing with LLaVA-1.5-7B
        so that q_pool and U share the float16 compute dtype.
        Do NOT pass U through any LLM layers — use embed_tokens output
        only, before positional encoding is added.
    """

    def __init__(self, d: int):
        """
        Initialise QuestionPooler.

        q_pool is drawn from N(0, 1/sqrt(d)).  This keeps the initial
        attention scores (q_pool @ u_t / sqrt(d)) near zero for a
        random unit-norm token embedding u_t, so the softmax starts
        close to uniform over all T positions.

        Args:
            d (int): LLM hidden dimension.
        """
        super().__init__()
        self.d = d

        # Learned pooling query vector, shape (d,)
        # Initialise with small normal values so early attention is near-uniform.
        self.q_pool = nn.Parameter(
            torch.empty(d).normal_(mean=0.0, std=1.0 / math.sqrt(d))
        )

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        """
        Compute the attention-pooled question vector q.

        Args:
            U (torch.Tensor): Token embeddings from model.model.embed_tokens,
                              shape (T, d), dtype must match this module
                              (float16 in training, float32 for CPU tests).
                              T is the number of question tokens.

        Returns:
            torch.Tensor: Question vector q of shape (d,).
        """
        # --- Step 1: score each token against the pooling query ---------------
        # U @ q_pool  →  (T, d) @ (d,)  =  (T,)
        # Divide by sqrt(d) to keep scores well-scaled (same as scaled-dot
        # product attention; prevents softmax from saturating early).
        scores = U @ self.q_pool / math.sqrt(self.d)   # (T,)

        # --- Step 2: softmax over T positions to get attention weights --------
        # weights[t] = how much to attend to token t
        weights = F.softmax(scores, dim=0)              # (T,), sums to 1.0

        # --- Step 3: weighted sum of token embeddings -------------------------
        # weights @ U  →  (T,) @ (T, d)  =  (d,)
        q = weights @ U                                 # (d,)

        return q


# ============================================================
# A-1.2 Integration note
# ============================================================
# In the LLaVA forward pass, extract question embeddings BEFORE the
# LLM processes them (no positional encoding at this stage):
#
#   # 1. Tokenise the question text
#   enc = tokenizer(question_text, return_tensors="pt")
#   question_input_ids = enc["input_ids"].squeeze(0).to(device)  # (T,)
#
#   # 2. Look up token embeddings — do NOT call model.model.forward()
#   with torch.no_grad():
#       U = model.model.embed_tokens(question_input_ids)          # (T, 4096)
#
#   # 3. Pool into a single question vector
#   q = pooler(U)                                                  # (4096,)
#
#   # 4. Forward q to both the router and ExpertE4
#   r      = router(q)          # (4,) softmax probabilities
#   k_star = torch.argmax(r.detach()).item()
#   if k_star == 3:             # E4 needs q
#       V = experts[k_star](Z_V, q)
#   else:
#       V = experts[k_star](Z_V)
# ============================================================


if __name__ == "__main__":
    # ============================================================
    # Verification block — Task A-1.3
    # Uses a mock embed_tokens (nn.Embedding) with fixed random weights.
    # No LLaVA download needed; runs on CPU or CUDA.
    #
    # Expected output:
    #   CHECK 1 PASSED
    #   CHECK 2 PASSED
    #   CHECK 3 PASSED  (shape: torch.Size([4096]))
    #   ALL CHECKS PASSED
    # ============================================================

    # -- Setup -------------------------------------------------------
    # Use GPU if available (float16 to match training), else CPU (float32).
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # float16 on CPU has limited op support; use float32 there instead.
    DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32
    D      = 4096    # LLaVA-1.5-7B hidden dimension
    VOCAB  = 32000   # approximate tokenizer vocabulary size

    print(f"Verification device: {DEVICE}, dtype: {DTYPE}")
    print()

    # Fix random seed so results are deterministic across runs.
    torch.manual_seed(42)

    # Mock embed_tokens: simulates model.model.embed_tokens(ids)
    # Each row is a learned d-dimensional embedding for one token.
    embed_tokens = nn.Embedding(VOCAB, D).to(DEVICE).to(DTYPE)

    # Instantiate QuestionPooler in the correct dtype (float16 for GPU).
    pooler = QuestionPooler(d=D).to(DEVICE).to(DTYPE)

    all_passed = True

    # ==============================================================
    # CHECK 1 — Different questions → pooled vectors differ
    # Simulates two distinct questions, e.g.:
    #   Q1: "What is labeled A?"    (7 tokens)
    #   Q2: "What color is the background?"  (8 tokens, different IDs)
    # ==============================================================
    ids_q1 = torch.tensor(
        [1023, 4521, 6789, 234, 1, 9876, 3421], device=DEVICE
    )   # 7 tokens representing question 1
    ids_q2 = torch.tensor(
        [9999, 1111, 3333, 5555, 7777, 2222, 8888, 4444], device=DEVICE
    )   # 8 tokens representing question 2

    with torch.no_grad():
        U1 = embed_tokens(ids_q1)   # (7, 4096)
        U2 = embed_tokens(ids_q2)   # (8, 4096)
        q1 = pooler(U1)             # (4096,)
        q2 = pooler(U2)             # (4096,)

    diff_12 = (q1 - q2).norm().item()

    if diff_12 > 0.0:
        print(f"CHECK 1 PASSED  (|q1 - q2|_2 = {diff_12:.4f} > 0)")
    else:
        print(f"CHECK 1 FAILED: |q1 - q2|_2 = {diff_12:.6f} (expected > 0)")
        all_passed = False

    # ==============================================================
    # CHECK 2 — Same question twice → identical pooled vectors
    # The pooler has no dropout or randomness; same input must give
    # bitwise-identical output (diff norm == exactly 0.0).
    # ==============================================================
    ids_q3 = ids_q1.clone()   # identical token IDs to question 1

    with torch.no_grad():
        U3 = embed_tokens(ids_q3)   # (7, 4096), same as U1
        q3 = pooler(U3)             # (4096,)

    diff_13 = (q1 - q3).norm().item()

    if diff_13 == 0.0:
        print(f"CHECK 2 PASSED  (|q1 - q3|_2 = {diff_13} — deterministic)")
    else:
        print(
            f"CHECK 2 FAILED: |q1 - q3|_2 = {diff_13:.8f} "
            f"(expected 0.0 — computation is non-deterministic)"
        )
        all_passed = False

    # ==============================================================
    # CHECK 3 — Output shape is torch.Size([4096])
    # ==============================================================
    expected_shape = torch.Size([D])

    if q1.shape == expected_shape:
        print(f"CHECK 3 PASSED  (shape: {q1.shape})")
    else:
        print(
            f"CHECK 3 FAILED: shape is {q1.shape}, "
            f"expected {expected_shape}"
        )
        all_passed = False

    # -- Summary -----------------------------------------------------
    print()
    if all_passed:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
