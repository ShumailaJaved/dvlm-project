# ============================================================
# train_single.py
# Task D-1: Single-Expert Ablation Training
# ============================================================
# WHERE TO RUN: Google Colab with A100 GPU (Linux only).
#   QLoRA requires CUDA 11.x+ and Linux.
#
# COMMAND:
#   python train_single.py --expert E1   # MLP (frozen pretrained projector)
#   python train_single.py --expert E2   # Q-Former (32 output tokens)
#   python train_single.py --expert E3   # global attention token (1 token)
#   python train_single.py --expert E4   # QCGP (question-conditioned, 576 tokens)
#
# OPTIONAL FLAGS:
#   --epochs   2      (default: 2)
#   --lr_expert 2e-4  (default: 2e-4, learning rate for expert params)
#   --lr_lora   5e-5  (default: 5e-5, learning rate for LoRA adapters)
#   --batch     4     (default: 4 per device; accum=4 → effective batch=16)
#   --n_eval    500   (default: 500 test samples)
#   --out_dir   results
#
# INSTALL IN COLAB FIRST:
#   !pip install "bitsandbytes>=0.41" "peft>=0.9" accelerate datasets tqdm
#   !pip install -e ./LLaVA
#
# OUTPUT:
#   results/single_expert_{EXPERT}.json  — metrics dict
#   results/ckpt_{EXPERT}/               — PEFT checkpoint
#   results/log_{EXPERT}.csv            — step-level training log
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

# ---- make connector.* importable when run from project root ----------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------

MODEL_ID       = "liuhaotian/llava-v1.5-7b"
ANSWER_LETTERS = "ABCDE"
MAX_LEN        = 512    # max token length; sequences longer than this are truncated
ACCUM_STEPS    = 4      # gradient accumulation; effective_batch = batch * ACCUM_STEPS
WARMUP_FRAC    = 0.03   # fraction of total steps used for linear LR warmup


# ============================================================
# Transformers ≥ 4.46 compatibility patch
# ============================================================

def patch_generate_compat(_model=None) -> None:
    """
    Transformers ≥ 4.46 passes 'cache_position' (and sometimes
    'num_logits_to_keep') to LlavaLlamaForCausalLM.forward() during
    generation, but LLaVA's forward() signature doesn't accept them.

    Root cause: PEFT's generate() overwrites prepare_inputs_for_generation
    on the model instance right before the inner generate loop — so any
    instance-level patch on prepare_inputs_for_generation is clobbered.

    Fix: patch LlavaLlamaForCausalLM.forward() at the CLASS level to drop
    the unknown kwargs before delegating.  Guard flag makes it idempotent.
    """
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

    if getattr(LlavaLlamaForCausalLM, "_transformers_compat_patched", False):
        return

    _orig_forward = LlavaLlamaForCausalLM.forward

    def _forward_compat(self, *args, **kwargs):
        kwargs.pop("cache_position",     None)
        kwargs.pop("num_logits_to_keep", None)
        return _orig_forward(self, *args, **kwargs)

    # Restore the original forward's explicit parameter names (attention_mask,
    # input_ids, …) via __wrapped__, so transformers' _validate_model_kwargs
    # still sees them with inspect.signature.  Without this the wrapper's
    # (*args, **kwargs) signature causes transformers ≥ 4.46 to reject
    # 'attention_mask' as an unknown model kwarg.
    import functools
    functools.update_wrapper(_forward_compat, _orig_forward)

    LlavaLlamaForCausalLM.forward = _forward_compat

    # Belt-and-suspenders: silence _validate_model_kwargs on this class only.
    # In some transformers/PEFT version combos the __wrapped__ trick alone is
    # not enough (the validator checks prepare_inputs_for_generation, not forward).
    LlavaLlamaForCausalLM._validate_model_kwargs = \
        lambda self, model_kwargs: None   # no-op — skip kwarg validation

    LlavaLlamaForCausalLM._transformers_compat_patched = True


# ============================================================
# Data utilities
# ============================================================

def build_prompt(sample: dict, with_image: bool = True,
                 for_eval: bool = False) -> tuple:
    """
    Format a ScienceQA sample as a LLaVA vicuna_v1 prompt string.

    Args:
        sample:     A ScienceQA dataset row with question/choices/answer/image.
        with_image: If True, prepend the <image> token to the question.
        for_eval:   If True, leave the assistant turn empty so the model
                    generates the answer.  If False (training), the answer
                    letter is appended as a supervised target.

    Returns:
        (full_prompt_str, answer_letter_str)
    """
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    question = sample["question"]
    choices  = sample["choices"]
    options  = "\n".join(f"{ANSWER_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    answer_letter = ANSWER_LETTERS[sample["answer"]]

    img_tok  = f"{DEFAULT_IMAGE_TOKEN}\n" if with_image else ""
    user_msg = (
        f"{img_tok}{question}\n{options}\n"
        "Answer with the option letter only (A, B, C, D, or E)."
    )

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], user_msg)
    # During evaluation leave the assistant turn empty so the model generates
    # the answer.  During training fill it in as the supervised target.
    conv.append_message(conv.roles[1], None if for_eval else answer_letter)
    return conv.get_prompt(), answer_letter


def make_labels(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Build training labels from input_ids: -100 for the prompt, keep answer tokens.

    Finds the last " ASSISTANT:" separator token sequence and unmasks
    everything after it so only the answer letter (and EOS) are trained on.

    Args:
        input_ids: 1-D token id tensor for the full prompt + answer.
        tokenizer: LLaVA tokenizer.

    Returns:
        labels tensor, same shape as input_ids.
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
    """
    Tokenize one ScienceQA sample: text → input_ids/labels, image → pixel_values.

    Args:
        sample:          Single ScienceQA dataset row.
        tokenizer:       LLaVA tokenizer.
        image_processor: CLIP image preprocessor.
        max_len:         Truncation length for input_ids.

    Returns:
        Dict with input_ids (LongTensor), labels (LongTensor), pixel_values (FloatTensor).
    """
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils   import tokenizer_image_token

    prompt, _ = build_prompt(sample, with_image=True)
    ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).squeeze(0)[:max_len]

    labels = make_labels(ids, tokenizer)

    pixel_values = image_processor.preprocess(
        sample["image"], return_tensors="pt"
    )["pixel_values"]   # (1, 3, 336, 336)

    return {"input_ids": ids, "labels": labels, "pixel_values": pixel_values}


class ScienceQADataset(Dataset):
    """
    Lazy-loading ScienceQA dataset: tokenizes and processes images on-the-fly.
    Expects the HuggingFace dataset already filtered to image-only samples.
    """

    def __init__(self, hf_dataset, tokenizer, image_processor, max_len: int):
        """
        Args:
            hf_dataset:      HuggingFace Dataset filtered to image-only samples.
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
    Collate a list of samples into a padded batch.

    Pads input_ids and labels to the longest sequence in the batch.
    Stacks pixel_values along the batch dimension.

    Args:
        batch:  List of dicts from ScienceQADataset.__getitem__.
        pad_id: Token id used for padding (usually eos_token_id).

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
# Expert plug-in
# ============================================================

def plug_in_expert(model, expert_name: str, device: str):
    """
    Replace model.model.mm_projector with the specified expert module.

    For E1/E2/E3 (question-blind): directly replaces mm_projector so
    LLaVA's standard encode_images() calls the expert automatically.

    For E4 (question-conditioned): returns the expert + pooler without
    installing; caller must do class surgery via upgrade_to_e4_model().

    Args:
        model:       Loaded LlavaLlamaForCausalLM.
        expert_name: "E1", "E2", "E3", or "E4".
        device:      CUDA device string (e.g. "cuda").

    Returns:
        (model, expert_module, pooler_or_None)
    """
    from connector.expert_e1       import ExpertE1
    from connector.expert_e2       import ExpertE2
    from connector.expert_e3       import ExpertE3
    from connector.expert_e4       import ExpertE4
    from connector.question_pooler import QuestionPooler

    if expert_name == "E1":
        expert = ExpertE1(model.model.mm_projector)
        model.model.mm_projector = expert
        return model, expert, None

    elif expert_name == "E2":
        expert = ExpertE2(d_v=1024, d=4096, K=32)
        model.model.mm_projector = expert
        return model, expert, None

    elif expert_name == "E3":
        expert = ExpertE3(d_v=1024, d=4096)
        model.model.mm_projector = expert
        return model, expert, None

    elif expert_name == "E4":
        # E4 needs the question vector alongside Z_V.
        # Cannot replace mm_projector alone; caller does class surgery.
        expert = ExpertE4(d_v=1024, d=4096, d_k=256)
        pooler = QuestionPooler(d=4096)
        return model, expert, pooler

    else:
        raise ValueError(f"Unknown expert '{expert_name}'. Choose E1, E2, E3, or E4.")


# ============================================================
# SingleE4LlavaForCausalLM — E4 requires question conditioning
# ============================================================

def _llava_available():
    """Return True if the LLaVA package is importable."""
    try:
        import llava  # noqa: F401
        return True
    except ImportError:
        return False


if _llava_available():
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

    class SingleE4LlavaForCausalLM(LlavaLlamaForCausalLM):
        """
        LlavaLlamaForCausalLM subclass that uses ExpertE4 (QCGP) exclusively.

        Overrides prepare_inputs_labels_for_multimodal to:
          1. Encode Z_V via the vision tower (no mm_projector).
          2. Get question embeddings from embed_tokens.
          3. Pool question → q via QuestionPooler.
          4. Run E4(Z_V, q) → V (576, d).
          5. Insert V at IMAGE_TOKEN_INDEX positions.
        """

        def __init__(self, config):
            super().__init__(config)
            self.e4     = None
            self.pooler = None

        def set_e4(self, e4, pooler) -> None:
            """Attach the E4 expert and question pooler."""
            self.e4     = e4
            self.pooler = pooler

        def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask,
            past_key_values, labels, images, image_sizes=None,
        ):
            """Override to route through E4 instead of the MLP projector."""
            vision_tower = self.model.vision_tower

            # Fall back to standard LLaVA logic when E4 is not set or no images
            if (self.e4 is None or vision_tower is None or images is None
                    or input_ids.shape[1] == 1
                    or (not isinstance(images, list) and images.ndim != 4)):
                return super().prepare_inputs_labels_for_multimodal(
                    input_ids, position_ids, attention_mask,
                    past_key_values, labels, images, image_sizes)

            _labels         = labels
            _position_ids   = position_ids
            _attention_mask = attention_mask

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                attention_mask = attention_mask.bool()
            if position_ids is None:
                position_ids = torch.arange(
                    0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            if labels is None:
                labels = torch.full_like(input_ids, IGNORE_INDEX)

            ids_list  = [ids[mask] for ids, mask in zip(input_ids, attention_mask)]
            labs_list = [lab[mask] for lab, mask in zip(labels, attention_mask)]

            # Encode all images once (no mm_projector)
            Z_V_batch = vision_tower(images)   # (B, 576, 1024)

            new_embeds = []
            new_labs   = []

            for b_idx, cur_ids in enumerate(ids_list):
                # Get text embeddings for question conditioning
                text_ids = cur_ids[cur_ids != IMAGE_TOKEN_INDEX]
                U        = self.model.embed_tokens(text_ids)   # (T, d)

                # Match E4 dtype
                _dtype = self.e4.W_E4.weight.dtype
                Z_V = Z_V_batch[b_idx].to(dtype=_dtype)
                U_e = U.to(dtype=_dtype)
                q   = self.pooler(U_e)                # (d,)
                V   = self.e4(Z_V, q).to(dtype=U.dtype)  # (576, d)

                # Text-only sample: no image token
                if (cur_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                    new_embeds.append(self.model.embed_tokens(cur_ids))
                    new_labs.append(labs_list[b_idx])
                    continue

                # Split at image token positions and interleave V
                positions = (
                    [-1]
                    + torch.where(cur_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                    + [cur_ids.shape[0]]
                )
                cur_labs = labs_list[b_idx]
                id_segs, lab_segs = [], []
                for i in range(len(positions) - 1):
                    s, e = positions[i] + 1, positions[i + 1]
                    id_segs.append(cur_ids[s:e])
                    lab_segs.append(cur_labs[s:e])

                all_text_embeds = self.model.embed_tokens(torch.cat(id_segs))
                emb_splits = torch.split(all_text_embeds,
                                         [x.shape[0] for x in lab_segs], dim=0)

                n_img = (cur_ids == IMAGE_TOKEN_INDEX).sum().item()
                out_embeds, out_labs = [], []
                for i in range(n_img + 1):
                    out_embeds.append(emb_splits[i])
                    out_labs.append(lab_segs[i])
                    if i < n_img:
                        out_embeds.append(V.to(self.device))
                        out_labs.append(torch.full(
                            (V.shape[0],), IGNORE_INDEX,
                            device=cur_labs.device, dtype=cur_labs.dtype))

                new_embeds.append(torch.cat([x.to(self.device) for x in out_embeds]))
                new_labs.append(torch.cat(out_labs))

            # Truncate to model max length
            max_seq = getattr(self.config, "tokenizer_model_max_length", None)
            if max_seq:
                new_embeds = [x[:max_seq] for x in new_embeds]
                new_labs   = [x[:max_seq] for x in new_labs]

            # Pad all sequences in the batch to the same length
            L   = max(x.shape[0] for x in new_embeds)
            B   = len(new_embeds)
            dev = input_ids.device

            labs_pad  = torch.full((B, L), IGNORE_INDEX,
                                   dtype=new_labs[0].dtype, device=dev)
            attn_pad  = torch.zeros((B, L), dtype=torch.bool, device=dev)
            pos_pad   = torch.zeros((B, L), dtype=torch.long, device=dev)
            stacked   = []

            for i, (emb, lab) in enumerate(zip(new_embeds, new_labs)):
                n = emb.shape[0]
                z = torch.zeros((L - n, emb.shape[1]), dtype=emb.dtype, device=emb.device)
                stacked.append(torch.cat([emb, z], dim=0))
                labs_pad[i, :n] = lab
                attn_pad[i, :n] = True
                pos_pad[i, :n]  = torch.arange(n, dtype=torch.long, device=dev)

            new_embeds = torch.stack(stacked, dim=0)

            if _labels is None:         labs_pad = None
            if _attention_mask is None: attn_pad = None
            else:                       attn_pad = attn_pad.to(_attention_mask.dtype)
            if _position_ids is None:   pos_pad  = None

            return (None, pos_pad, attn_pad, past_key_values, new_embeds, labs_pad)


def upgrade_to_e4_model(model, e4, pooler):
    """
    Upgrade a LlavaLlamaForCausalLM to SingleE4LlavaForCausalLM via class surgery.

    MUST be called before prepare_model_for_kbit_training and get_peft_model.

    Args:
        model:  Loaded LlavaLlamaForCausalLM.
        e4:     ExpertE4 instance.
        pooler: QuestionPooler instance.

    Returns:
        The same object, now typed as SingleE4LlavaForCausalLM.
    """
    model.__class__ = SingleE4LlavaForCausalLM
    model.e4        = None
    model.pooler    = None
    model.model     = model.model   # preserve nn.Module submodule attribute
    model.set_e4(e4, pooler)
    return model


# ============================================================
# Evaluation
# ============================================================

def extract_answer(text: str) -> str:
    """
    Extract the predicted answer letter (A-E) from model output.

    Tries three patterns in order: standalone letter, parenthesised letter,
    phrase 'the answer is X'.  Falls back to the very first character if it
    is a valid option letter (handles outputs like "A." or "A\n").
    Returns "" if nothing matches.
    """
    text = text.strip()
    for pattern in [r'\b([A-E])\b', r'\(([A-E])\)', r'answer\s+is\s+([A-E])']:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # Fallback: first character (covers "A.", "A\n", etc.)
    if text and text[0].upper() in ANSWER_LETTERS:
        return text[0].upper()
    return ""


@torch.no_grad()
def evaluate(model, hf_dataset, tokenizer, image_processor,
             device: str, n_samples: int) -> dict:
    """
    Compute Acc_image, Acc_text, and Delta_v on n_samples test samples.

    Runs inference twice per sample: once with the image, once without
    (removing the image token from input_ids).

    Args:
        model:           Trained model (possibly PEFT-wrapped).
        hf_dataset:      Image-filtered HuggingFace Dataset split.
        tokenizer:       LLaVA tokenizer.
        image_processor: CLIP image preprocessor.
        device:          CUDA device string.
        n_samples:       Number of samples to evaluate.

    Returns:
        Dict with acc_image, acc_text, delta_v (all as floats 0-1).
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
            # for_eval=True: leave assistant turn empty so model generates answer.
            prompt, _ = build_prompt(item, with_image=with_img, for_eval=True)
            ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).to(device)

            if with_img and item["image"] is not None:
                pix  = image_processor.preprocess(
                    item["image"], return_tensors="pt"
                )["pixel_values"].to(device, dtype=torch.float16)
                imgs = pix
                ids  = ids.unsqueeze(0)   # (T,) → (1, T) required by generate
            else:
                # Strip image placeholder for text-only run
                ids  = ids[ids != IMAGE_TOKEN_INDEX].unsqueeze(0)
                imgs = None

            # attention_mask must be explicit: transformers ≥ 4.46 calls
            # attention_mask.new_ones(...) in _update_model_kwargs_for_generation
            # and crashes if attention_mask is None.
            attn_mask  = torch.ones_like(ids)
            MAX_NEW    = 3
            out  = model.generate(inputs=ids, images=imgs,
                                   attention_mask=attn_mask,
                                   max_new_tokens=MAX_NEW, do_sample=False)
            # LLaVA routes through inputs_embeds so out[0] may not contain
            # input tokens at the front — slice from the tail instead.
            new_tokens = out[0][-MAX_NEW:]
            pred = extract_answer(tokenizer.decode(new_tokens, skip_special_tokens=True))

            if with_img:
                correct_img += int(pred == gt)
            else:
                correct_txt += int(pred == gt)

    n        = len(items)
    acc_img  = correct_img / n
    acc_txt  = correct_txt / n
    delta_v  = acc_img - acc_txt
    model.train()
    return {"acc_image": acc_img, "acc_text": acc_txt, "delta_v": delta_v}


# ============================================================
# Main training function
# ============================================================

def train(args):
    """
    Full D-1 training pipeline for one expert.

    Steps:
      1. Load LLaVA-1.5-7B in 4-bit NF4.
      2. Install the chosen expert (replace mm_projector or class surgery for E4).
      3. Apply QLoRA (prepare_model_for_kbit_training → get_peft_model).
      4. Load and preprocess ScienceQA.
      5. Train for args.epochs with gradient accumulation.
      6. Evaluate on test set: Acc_image, Acc_text, Delta_v.
      7. Save checkpoint and results JSON.

    Args:
        args: Parsed argparse.Namespace.
    """
    import torch.optim as optim
    from datasets import load_dataset
    from transformers import BitsAndBytesConfig
    from llava.model.builder import load_pretrained_model
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)
    result_path = os.path.join(args.out_dir, f"single_expert_{args.expert}.json")
    log_path    = os.path.join(args.out_dir, f"log_{args.expert}.csv")

    print(f"\n{'='*60}")
    print(f"  Single-Expert Training: {args.expert}")
    print(f"  Epochs: {args.epochs}  |  Batch: {args.batch}×{ACCUM_STEPS}={args.batch*ACCUM_STEPS}")
    print(f"{'='*60}\n")

    # ---- 1. Load base model in 4-bit NF4 ------------------------------------
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    print("Loading LLaVA-1.5-7B in 4-bit NF4 …")
    tokenizer, base_model, img_proc, _ = load_pretrained_model(
        MODEL_ID, None, "llava-v1.5-7b", quantization_config=bnb_cfg)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 2. Install expert ---------------------------------------------------
    print(f"Installing expert {args.expert} …")
    model, expert, pooler = plug_in_expert(base_model, args.expert, device)

    if args.expert == "E4":
        # Class surgery must happen BEFORE prepare_model_for_kbit_training
        model = upgrade_to_e4_model(model, expert, pooler)

    # ---- 3. Apply QLoRA ------------------------------------------------------
    # IMPORTANT ORDER: prepare_model_for_kbit_training → re-cast expert → get_peft_model
    print("Applying QLoRA …")
    model = prepare_model_for_kbit_training(model)

    # prepare_model_for_kbit_training upcasts float16 params to float32.
    # Re-cast trainable expert modules back to float16 to match LLaVA's
    # float16 compute pipeline (bnb_4bit_compute_dtype=float16).
    if args.expert in ("E2", "E3", "E4"):
        expert.to(device).half()
    if pooler is not None:
        pooler.to(device).half()

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # ---- CRITICAL: restore expert/pooler gradients after PEFT freezing -------
    # get_peft_model calls mark_only_lora_as_trainable(), which sets
    # requires_grad=False on ALL parameters whose name does not contain
    # "lora_".  This silently freezes the expert and pooler modules, so
    # their parameters never receive gradient updates — same root cause as
    # the router collapse bug in train_moc.py.
    # Fix: explicitly re-enable requires_grad for trainable modules.
    # E1 stays frozen — it wraps the pretrained mm_projector intentionally.
    if args.expert in ("E2", "E3", "E4"):
        expert.requires_grad_(True)
    if pooler is not None:
        pooler.requires_grad_(True)
    # -------------------------------------------------------------------------

    model.print_trainable_parameters()
    patch_generate_compat(model)   # strip cache_position / num_logits_to_keep

    # ---- 4. Dataset ----------------------------------------------------------
    print("Loading ScienceQA …")
    raw      = load_dataset("derek-thomas/ScienceQA")
    train_hf = raw["train"].filter(lambda x: x["image"] is not None)
    test_hf  = raw["test"].filter(lambda x: x["image"] is not None)
    print(f"  Train: {len(train_hf):,}  |  Test: {len(test_hf):,}")

    train_ds = ScienceQADataset(train_hf, tokenizer, img_proc, MAX_LEN)
    loader   = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=2, pin_memory=True,
    )

    # ---- 5. Optimizer and LR scheduler ---------------------------------------
    # Two groups: expert params at higher lr (args.lr_expert),
    #             LoRA adapter params at lower lr (args.lr_lora).
    expert_params = []
    if args.expert in ("E2", "E3", "E4"):
        expert_params += [p for p in expert.parameters() if p.requires_grad]
    if pooler is not None:
        expert_params += [p for p in pooler.parameters() if p.requires_grad]

    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora_" in n]

    param_groups = []
    if expert_params:
        param_groups.append({"params": expert_params, "lr": args.lr_expert})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args.lr_lora})

    optimizer    = optim.AdamW(param_groups, weight_decay=0.01)
    total_steps  = math.ceil(len(train_ds) / (args.batch * ACCUM_STEPS)) * args.epochs
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))

    def lr_lambda(step):
        """Linear warmup then cosine decay."""
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- 6. Training loop ----------------------------------------------------
    print(f"\nStarting training — {total_steps} effective steps …\n")
    model.train()
    t0          = time.time()
    global_step = 0
    accum_loss  = 0.0

    log_rows = [["step", "epoch", "loss", "lr", "elapsed_min"]]

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(loader):
            ids    = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask   = batch["attention_mask"].to(device)
            images = batch["images"].to(device, dtype=torch.float16)

            out  = model(input_ids=ids, labels=labels,
                         attention_mask=mask, images=images)
            loss = out.loss / ACCUM_STEPS
            loss.backward()
            accum_loss += loss.item()

            # Optimizer step every ACCUM_STEPS mini-batches
            if (batch_idx + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    elapsed = (time.time() - t0) / 60
                    cur_lr  = scheduler.get_last_lr()[0]
                    avg_loss = accum_loss / 50
                    print(
                        f"Ep {epoch+1}/{args.epochs}  "
                        f"Step {global_step}/{total_steps}  "
                        f"Loss {avg_loss:.4f}  "
                        f"LR {cur_lr:.2e}  "
                        f"Elapsed {elapsed:.1f} min"
                    )
                    log_rows.append([global_step, epoch + 1,
                                     round(avg_loss, 5), round(cur_lr, 8),
                                     round(elapsed, 2)])
                    accum_loss = 0.0

        print(f"  — Epoch {epoch + 1} complete.")

    train_time_h = (time.time() - t0) / 3600
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\nTraining done: {train_time_h:.2f} h  |  Peak VRAM: {peak_vram_gb:.1f} GB\n")

    # ---- 7. Save training log ------------------------------------------------
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerows(log_rows)
    print(f"Training log → {log_path}")

    # ---- 8. Save checkpoint --------------------------------------------------
    ckpt_dir = os.path.join(args.out_dir, f"ckpt_{args.expert}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    print(f"Checkpoint → {ckpt_dir}")

    # ---- 9. Evaluate on test set ---------------------------------------------
    print(f"\nEvaluating on {args.n_eval} test samples …")
    metrics = evaluate(model, test_hf, tokenizer, img_proc, device, args.n_eval)
    metrics.update({
        "expert":        args.expert,
        "train_time_h":  round(train_time_h, 2),
        "peak_vram_gb":  round(peak_vram_gb, 1),
    })

    print(f"\n{'─'*40}")
    print(f"  Expert      : {args.expert}")
    print(f"  Acc_image   : {metrics['acc_image']*100:.1f}%")
    print(f"  Acc_text    : {metrics['acc_text']*100:.1f}%")
    print(f"  Delta_v     : {metrics['delta_v']*100:.1f}%")
    print(f"  Time        : {metrics['train_time_h']:.2f} h")
    print(f"  Peak VRAM   : {metrics['peak_vram_gb']:.1f} GB")
    print(f"{'─'*40}\n")

    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results → {result_path}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task D-1: Single-Expert Training")
    parser.add_argument(
        "--expert", required=True, choices=["E1", "E2", "E3", "E4"],
        help="Expert to train: E1 (MLP), E2 (Q-Former), E3 (global token), E4 (QCGP).",
    )
    parser.add_argument("--epochs",     type=int,   default=2,
                        help="Number of training epochs. Default: 2.")
    parser.add_argument("--lr_expert",  type=float, default=2e-4,
                        help="Learning rate for expert module parameters.")
    parser.add_argument("--lr_lora",    type=float, default=5e-5,
                        help="Learning rate for LoRA adapter parameters.")
    parser.add_argument("--batch",      type=int,   default=4,
                        help="Per-device batch size (accum=4 → effective=16).")
    parser.add_argument("--n_eval",     type=int,   default=500,
                        help="Number of test samples to evaluate on.")
    parser.add_argument("--out_dir",    type=str,   default="results",
                        help="Output directory for checkpoints and result files.")
    args = parser.parse_args()
    train(args)
