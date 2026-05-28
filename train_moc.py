# ============================================================
# train_moc.py
# Task D-2: Full MoC Training with Load-Balancing
# ============================================================
# WHERE TO RUN: Google Colab with A100 GPU (Linux only).
#
# COMMAND:
#   python train_moc.py
#
# OPTIONAL FLAGS:
#   --epochs    2      (default: 2)
#   --lr_moc    2e-4   (default: 2e-4, learning rate for trainable MoC params)
#   --lr_lora   5e-5   (default: 5e-5, learning rate for LoRA adapters)
#   --batch     4      (default: 4 per device; accum=4 → effective batch=16)
#   --val_every 750    (default: 750 steps between validation evaluations)
#   --n_val     200    (default: 200 validation samples per checkpoint eval)
#   --n_test    500    (default: 500 test samples for final evaluation)
#   --out_dir   results
#
# INSTALL IN COLAB FIRST:
#   !pip install "bitsandbytes>=0.41" "peft>=0.9" accelerate datasets tqdm
#   !pip install -e ./LLaVA
#
# OUTPUT:
#   results/moc_best.json          — best checkpoint metrics
#   results/moc_final.json         — final evaluation metrics
#   results/ckpt_moc_best/         — best PEFT checkpoint (by val accuracy)
#   results/log_moc.csv            — step-level training log
#   results/routing_log_moc.csv    — per-step expert selection counts
# ============================================================

import argparse
import csv
import json
import math
import os
import re
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---- make connector.* and losses.* importable from project root ------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------

MODEL_ID       = "liuhaotian/llava-v1.5-7b"
ANSWER_LETTERS = "ABCDE"
MAX_LEN        = 512
ACCUM_STEPS    = 4      # gradient accumulation; effective_batch = batch * ACCUM_STEPS
WARMUP_FRAC    = 0.03   # fraction of steps used for linear LR warmup
LAMBDA_LB      = 0.1    # load-balancing loss coefficient (L_total = L_CE + λ * L_lb)
K_EXPERTS      = 4      # number of MoC experts


# ============================================================
# Transformers ≥ 4.46 compatibility patch
# ============================================================

def patch_generate_compat(model) -> None:
    """
    Transformers ≥ 4.46 passes 'cache_position' (and sometimes
    'num_logits_to_keep') to model.forward() during generation.
    LLaVA's forward() signature does not accept these kwargs → TypeError.

    Wraps prepare_inputs_for_generation on the model instance to strip
    the unknown keys before they reach forward().
    """
    _orig = model.prepare_inputs_for_generation

    def _compat(*args, **kwargs):
        out = _orig(*args, **kwargs)
        out.pop("cache_position",     None)
        out.pop("num_logits_to_keep", None)
        return out

    model.prepare_inputs_for_generation = _compat


# ============================================================
# Data utilities (identical to train_single.py)
# ============================================================

def build_prompt(sample: dict, with_image: bool = True) -> tuple:
    """
    Format a ScienceQA sample as a LLaVA vicuna_v1 prompt string.

    Args:
        sample:     ScienceQA dataset row.
        with_image: If True, prepend the <image> token.

    Returns:
        (full_prompt_str, answer_letter_str)
    """
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    choices  = sample["choices"]
    options  = "\n".join(f"{ANSWER_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    answer_letter = ANSWER_LETTERS[sample["answer"]]
    img_tok  = f"{DEFAULT_IMAGE_TOKEN}\n" if with_image else ""

    user_msg = (
        f"{img_tok}{sample['question']}\n{options}\n"
        "Answer with the option letter only (A, B, C, D, or E)."
    )
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], answer_letter)
    return conv.get_prompt(), answer_letter


def make_labels(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Build training labels: -100 for the prompt portion, actual IDs for the answer.

    Finds the last " ASSISTANT:" separator and unmasks everything after it.
    """
    sep_ids = tokenizer.encode(" ASSISTANT:", add_special_tokens=False)
    seq     = input_ids.tolist()
    labels  = torch.full_like(input_ids, -100)
    for i in range(len(seq) - len(sep_ids), -1, -1):
        if seq[i : i + len(sep_ids)] == sep_ids:
            labels[i + len(sep_ids) :] = input_ids[i + len(sep_ids) :]
            break
    return labels


def tokenize_single(sample: dict, tokenizer, image_processor, max_len: int) -> dict:
    """Tokenize one ScienceQA sample: text → ids/labels, image → pixel_values."""
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils   import tokenizer_image_token

    prompt, _ = build_prompt(sample, with_image=True)
    ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).squeeze(0)[:max_len]

    return {
        "input_ids":    ids,
        "labels":       make_labels(ids, tokenizer),
        "pixel_values": image_processor.preprocess(
            sample["image"], return_tensors="pt"
        )["pixel_values"],
    }


class ScienceQADataset(Dataset):
    """Lazy-loading ScienceQA dataset; tokenizes on __getitem__."""

    def __init__(self, hf_dataset, tokenizer, image_processor, max_len: int):
        """
        Args:
            hf_dataset:      Image-filtered HuggingFace Dataset.
            tokenizer:       LLaVA tokenizer.
            image_processor: CLIP image preprocessor.
            max_len:         Token truncation length.
        """
        self.dataset         = hf_dataset
        self.tokenizer       = tokenizer
        self.image_processor = image_processor
        self.max_len         = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return tokenize_single(
            self.dataset[idx], self.tokenizer, self.image_processor, self.max_len
        )


def collate_fn(batch: list, pad_id: int) -> dict:
    """
    Pad variable-length sequences and stack pixel_values into a batch.

    Args:
        batch:  List of dicts from ScienceQADataset.__getitem__.
        pad_id: Padding token id.

    Returns:
        Batched dict: input_ids, labels, attention_mask, images.
    """
    max_len = max(b["input_ids"].shape[0] for b in batch)
    ids_list, labs_list, mask_list = [], [], []

    for b in batch:
        n   = b["input_ids"].shape[0]
        pad = max_len - n
        ids_list.append(torch.cat([b["input_ids"],
                                    torch.full((pad,), pad_id, dtype=torch.long)]))
        labs_list.append(torch.cat([b["labels"],
                                     torch.full((pad,), -100, dtype=torch.long)]))
        mask_list.append(torch.cat([torch.ones(n, dtype=torch.long),
                                     torch.zeros(pad, dtype=torch.long)]))

    return {
        "input_ids":      torch.stack(ids_list),
        "labels":         torch.stack(labs_list),
        "attention_mask": torch.stack(mask_list),
        "images":         torch.cat([b["pixel_values"] for b in batch], dim=0),
    }


# ============================================================
# Evaluation
# ============================================================

def extract_answer(text: str) -> str:
    """Extract the answer letter (A-E) from model output using regex."""
    text = text.strip()
    for pat in [r'\b([A-E])\b', r'\(([A-E])\)', r'answer\s+is\s+([A-E])']:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""


@torch.no_grad()
def evaluate(model, hf_dataset, tokenizer, image_processor,
             device: str, n_samples: int) -> dict:
    """
    Compute Acc_image, Acc_text, and Delta_v on n_samples items.

    Runs inference with and without the image for each sample.

    Args:
        model:           Trained PEFT model.
        hf_dataset:      Image-filtered HuggingFace Dataset split.
        tokenizer:       LLaVA tokenizer.
        image_processor: CLIP image preprocessor.
        device:          CUDA device string.
        n_samples:       Number of samples to evaluate.

    Returns:
        Dict with acc_image, acc_text, delta_v (floats 0–1).
    """
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils   import tokenizer_image_token

    model.eval()
    items = list(hf_dataset.select(range(min(n_samples, len(hf_dataset)))))
    correct_img, correct_txt = 0, 0
    print(f"Evaluating on {n_samples} samples (image + text-only = {n_samples*2} passes)...")

    for item in tqdm(items, desc="Evaluating", leave=False):
        gt = ANSWER_LETTERS[item["answer"]]
        for with_img in (True, False):
            prompt, _ = build_prompt(item, with_image=with_img)
            ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).to(device)

            if with_img:
                pix  = image_processor.preprocess(
                    item["image"], return_tensors="pt"
                )["pixel_values"].to(device, dtype=torch.float16)
                imgs = pix
                ids  = ids.unsqueeze(0)   # (T,) → (1, T) required by generate
            else:
                ids  = ids[ids != IMAGE_TOKEN_INDEX].unsqueeze(0)
                imgs = None

            out  = model.generate(inputs=ids, images=imgs,
                                   max_new_tokens=3, do_sample=False)
            pred = extract_answer(tokenizer.decode(out[0], skip_special_tokens=True))

            if with_img:
                correct_img += int(pred == gt)
            else:
                correct_txt += int(pred == gt)

    n = len(items)
    model.train()
    return {
        "acc_image": correct_img / n,
        "acc_text":  correct_txt / n,
        "delta_v":   (correct_img - correct_txt) / n,
    }


# ============================================================
# Main training function
# ============================================================

def train(args):
    """
    Full D-2 training pipeline: MoC with load-balancing auxiliary loss.

    Steps:
      1. Load LLaVA-1.5-7B in 4-bit NF4.
      2. Class surgery: upgrade_to_moc → build_moc → set_moc.
      3. QLoRA: prepare_model_for_kbit_training → moc.half() → get_peft_model.
      4. Load ScienceQA.
      5. Train with L_total = L_CE + LAMBDA_LB * L_lb.
         Log: step-loss, L_lb, per-expert counts, τ_g every 50 steps.
         Evaluate on val every val_every steps; save best checkpoint.
      6. Final test evaluation.
      7. Save metrics JSON and CSV logs.

    Args:
        args: Parsed argparse.Namespace.
    """
    import torch.optim as optim
    from datasets import load_dataset
    from transformers import BitsAndBytesConfig
    from llava.model.builder import load_pretrained_model
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    from connector.moc   import build_moc, upgrade_to_moc
    from losses.load_balance import load_balancing_loss

    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)
    log_path     = os.path.join(args.out_dir, "log_moc.csv")
    routing_path = os.path.join(args.out_dir, "routing_log_moc.csv")
    best_path    = os.path.join(args.out_dir, "moc_best.json")
    final_path   = os.path.join(args.out_dir, "moc_final.json")
    ckpt_dir     = os.path.join(args.out_dir, "ckpt_moc_best")

    print(f"\n{'='*60}")
    print(f"  Full MoC Training")
    print(f"  Epochs: {args.epochs}  |  Batch: {args.batch}×{ACCUM_STEPS}={args.batch*ACCUM_STEPS}")
    print(f"  L_total = L_CE + {LAMBDA_LB} * L_lb")
    print(f"{'='*60}\n")

    # ---- 1. Load base model --------------------------------------------------
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    print("Loading LLaVA-1.5-7B in 4-bit NF4 …")
    tokenizer, base_model, img_proc, _ = load_pretrained_model(
        MODEL_ID, None, "llava-v1.5-7b", quantization_config=bnb_cfg)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 2. MoC setup --------------------------------------------------------
    # CRITICAL ORDER:
    #   upgrade_to_moc   (class surgery, must be BEFORE get_peft_model)
    #   build_moc        (has direct mm_projector access here)
    #   set_moc          (attach)
    #   prepare_model_for_kbit_training
    #   moc.half()       (re-cast AFTER kbit prep, which upcasts fp16 → fp32)
    #   get_peft_model   (wraps in PeftModel — must be LAST)
    print("Setting up MoC …")
    model = upgrade_to_moc(base_model)
    moc   = build_moc(model).to(device)
    model.set_moc(moc)

    # ---- 3. Apply QLoRA ------------------------------------------------------
    print("Applying QLoRA …")
    model = prepare_model_for_kbit_training(model)
    moc.half()   # re-cast after kbit prep upcasted fp16 params to fp32

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # ---- CRITICAL: restore MoC gradients after PEFT freezing ----------------
    # get_peft_model calls mark_only_lora_as_trainable(), which sets
    # requires_grad=False on ALL parameters whose name does not contain
    # "lora_".  This silently freezes the router, pooler, and experts,
    # so r.requires_grad becomes False and L_lb has no gradient path to
    # the router — causing permanent routing collapse.
    # Fix: explicitly re-enable requires_grad for every trainable MoC
    # component.  E1 (experts[0]) wraps the frozen mm_projector and has
    # zero trainable parameters; leave it frozen.
    moc.router.requires_grad_(True)          # MoCRouter (W1, W2)
    moc.pooler.requires_grad_(True)          # QuestionPooler (q_pool)
    moc.experts[1].requires_grad_(True)      # ExpertE2 (Q-Former)
    moc.experts[2].requires_grad_(True)      # ExpertE3 (global token)
    moc.experts[3].requires_grad_(True)      # ExpertE4 (QCGP)
    # moc.experts[0] stays frozen — ExpertE1 wraps the pretrained mm_projector
    # -------------------------------------------------------------------------

    model.print_trainable_parameters()
    patch_generate_compat(model)   # strip cache_position / num_logits_to_keep

    # Convenience reference to the inner MoCLlavaForCausalLM for routing logs
    inner = model.base_model.model

    # ---- 4. Dataset ----------------------------------------------------------
    print("Loading ScienceQA …")
    raw    = load_dataset("derek-thomas/ScienceQA")
    train_hf = raw["train"].filter(lambda x: x["image"] is not None)
    val_hf   = raw["validation"].filter(lambda x: x["image"] is not None)
    test_hf  = raw["test"].filter(lambda x: x["image"] is not None)
    print(f"  Train: {len(train_hf):,}  |  Val: {len(val_hf):,}  |  Test: {len(test_hf):,}")

    train_ds = ScienceQADataset(train_hf, tokenizer, img_proc, MAX_LEN)
    loader   = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=2, pin_memory=True,
    )

    # ---- 5. Optimizer and LR scheduler ---------------------------------------
    # Two parameter groups:
    #   MoC params (router + pooler + E2 + E3 + E4) → higher lr (args.lr_moc)
    #   LoRA adapter params → lower lr (args.lr_lora)
    # Note: we collect only params with requires_grad=True from moc so that
    # E1's frozen mm_projector weights are excluded automatically.
    moc_params  = [p for p in moc.parameters() if p.requires_grad]
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora_" in n]

    optimizer   = optim.AdamW([
        {"params": moc_params,  "lr": args.lr_moc},
        {"params": lora_params, "lr": args.lr_lora},
    ], weight_decay=0.01)

    total_steps  = math.ceil(len(train_ds) / (args.batch * ACCUM_STEPS)) * args.epochs
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))
    print(f"  Total steps: {total_steps}  |  Warmup: {warmup_steps}\n")

    def lr_lambda(step):
        """Linear warmup then cosine decay."""
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- 6. Training loop ----------------------------------------------------
    model.train()
    t0             = time.time()
    global_step    = 0
    accum_ce       = 0.0   # accumulated CE loss for logging
    accum_lb       = 0.0   # accumulated load-balancing loss for logging
    expert_counts  = [0, 0, 0, 0]   # running expert selection totals
    best_val_acc   = -1.0
    best_step      = -1

    # CSV log headers
    train_log_rows   = [["step", "epoch", "loss_ce", "loss_lb", "loss_total",
                          "tau_g", "lr", "elapsed_min"]]
    routing_log_rows = [["step", "epoch", "n_e1", "n_e2", "n_e3", "n_e4",
                          "frac_e1", "frac_e2", "frac_e3", "frac_e4"]]

    print(f"Starting training — {total_steps} effective steps …\n")

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(loader):
            ids    = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask   = batch["attention_mask"].to(device)
            images = batch["images"].to(device, dtype=torch.float16)

            # -- Forward pass (CE loss computed inside the model) --
            out  = model(input_ids=ids, labels=labels,
                         attention_mask=mask, images=images)
            L_CE = out.loss

            # -- Load-balancing loss from stored router outputs --
            # inner._last_router_outputs: list of (r, k_star) per sample in batch
            router_outs = inner._last_router_outputs
            if router_outs:
                r_batch = torch.stack([ro[0] for ro in router_outs])     # (B, K)
                k_stars = torch.tensor([ro[1] for ro in router_outs],
                                       device=device, dtype=torch.long)  # (B,)
                L_lb = load_balancing_loss(r_batch, k_stars, K=K_EXPERTS)

                # Update running expert selection counts
                for k in k_stars.tolist():
                    expert_counts[k] += 1
            else:
                L_lb = torch.tensor(0.0, device=device)

            # -- Total loss with gradient accumulation scaling --
            loss = (L_CE + LAMBDA_LB * L_lb) / ACCUM_STEPS

            # NaN guard: skip batches with non-finite loss before backward.
            # A NaN loss.backward() fires NaN gradients into LoRA adapters,
            # corrupting all subsequent forward passes — much worse than
            # skipping one batch.  Root cause (float16 overflow in pooler/
            # router) is fixed above; this is a defensive fallback.
            if not torch.isfinite(loss):
                print(
                    f"  WARNING: non-finite loss "
                    f"(L_CE={L_CE.item():.4f}  L_lb={L_lb.item():.4f}) "
                    f"at epoch {epoch+1} batch {batch_idx} — skipping batch"
                )
                optimizer.zero_grad()
                continue

            loss.backward()

            accum_ce += L_CE.item()
            accum_lb += L_lb.item()

            # -- Optimizer step every ACCUM_STEPS mini-batches --
            if (batch_idx + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # -- Log every 50 optimizer steps --
                if global_step % 50 == 0:
                    elapsed  = (time.time() - t0) / 60
                    cur_lr   = scheduler.get_last_lr()[0]
                    avg_ce   = accum_ce / (50 * ACCUM_STEPS)
                    avg_lb   = accum_lb / (50 * ACCUM_STEPS)
                    avg_tot  = avg_ce + LAMBDA_LB * avg_lb

                    # τ_g from ExpertE4 (expert index 3)
                    tau_g = inner.moc.experts[3].tau_g.item()

                    total_sel = sum(expert_counts) or 1
                    fracs = [c / total_sel for c in expert_counts]

                    print(
                        f"Ep {epoch+1}/{args.epochs}  "
                        f"Step {global_step}/{total_steps}  "
                        f"L_CE {avg_ce:.4f}  L_lb {avg_lb:.4f}  "
                        f"τ_g {tau_g:.3f}  "
                        f"Experts {expert_counts}  "
                        f"Elapsed {elapsed:.1f} min"
                    )

                    train_log_rows.append([
                        global_step, epoch + 1,
                        round(avg_ce,  5), round(avg_lb, 5), round(avg_tot, 5),
                        round(tau_g, 4), round(cur_lr, 8), round(elapsed, 2),
                    ])
                    routing_log_rows.append([
                        global_step, epoch + 1,
                        *expert_counts,
                        *[round(f, 4) for f in fracs],
                    ])
                    accum_ce, accum_lb = 0.0, 0.0

                # -- Validation every val_every optimizer steps --
                if global_step % args.val_every == 0:
                    print(f"\n  → Validation at step {global_step} …")
                    val_metrics = evaluate(
                        model, val_hf, tokenizer, img_proc, device, args.n_val)
                    val_acc = val_metrics["acc_image"]
                    print(
                        f"  Val Acc_image={val_acc*100:.1f}%  "
                        f"Acc_text={val_metrics['acc_text']*100:.1f}%  "
                        f"Delta_v={val_metrics['delta_v']*100:.1f}%\n"
                    )
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_step    = global_step
                        os.makedirs(ckpt_dir, exist_ok=True)
                        model.save_pretrained(ckpt_dir)
                        val_metrics["step"] = global_step
                        with open(best_path, "w") as f:
                            json.dump(val_metrics, f, indent=2)
                        print(f"  ★ New best ({best_val_acc*100:.1f}%) → {ckpt_dir}\n")

        print(f"  — Epoch {epoch + 1} complete.")

    train_time_h = (time.time() - t0) / 3600
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\nTraining done: {train_time_h:.2f} h  |  Peak VRAM: {peak_vram_gb:.1f} GB")
    print(f"Best val acc: {best_val_acc*100:.1f}% at step {best_step}\n")

    # ---- 7. Save CSV logs ----------------------------------------------------
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerows(train_log_rows)
    with open(routing_path, "w", newline="") as f:
        csv.writer(f).writerows(routing_log_rows)
    print(f"Training log  → {log_path}")
    print(f"Routing log   → {routing_path}")

    # ---- 8. Final test evaluation --------------------------------------------
    print(f"\nFinal evaluation on {args.n_test} test samples …")
    final_metrics = evaluate(
        model, test_hf, tokenizer, img_proc, device, args.n_test)
    final_metrics.update({
        "train_time_h": round(train_time_h, 2),
        "peak_vram_gb": round(peak_vram_gb, 1),
        "best_val_acc": round(best_val_acc, 4),
        "best_step":    best_step,
        "expert_counts_total": expert_counts,
    })

    print(f"\n{'─'*40}")
    print(f"  MoC Final Test Results")
    print(f"  Acc_image : {final_metrics['acc_image']*100:.1f}%")
    print(f"  Acc_text  : {final_metrics['acc_text']*100:.1f}%")
    print(f"  Delta_v   : {final_metrics['delta_v']*100:.1f}%")
    print(f"  Time      : {train_time_h:.2f} h")
    print(f"  Peak VRAM : {peak_vram_gb:.1f} GB")
    print(f"  Expert selection totals: {expert_counts}")
    print(f"{'─'*40}\n")

    with open(final_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Final results → {final_path}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task D-2: Full MoC Training")
    parser.add_argument("--epochs",    type=int,   default=2,
                        help="Number of training epochs.")
    parser.add_argument("--lr_moc",   type=float, default=2e-4,
                        help="Learning rate for all MoC parameters (experts + router + pooler).")
    parser.add_argument("--lr_lora",  type=float, default=5e-5,
                        help="Learning rate for LoRA adapter parameters.")
    parser.add_argument("--batch",    type=int,   default=4,
                        help="Per-device batch size (accum=4 → effective batch=16).")
    parser.add_argument("--val_every", type=int,  default=750,
                        help="Run validation every N optimizer steps.")
    parser.add_argument("--n_val",    type=int,   default=200,
                        help="Number of validation samples to evaluate at each checkpoint.")
    parser.add_argument("--n_test",   type=int,   default=500,
                        help="Number of test samples for final evaluation.")
    parser.add_argument("--out_dir",  type=str,   default="results",
                        help="Output directory for checkpoints and result files.")
    args = parser.parse_args()
    train(args)
