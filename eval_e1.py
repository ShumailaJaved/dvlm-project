# ============================================================
# eval_e1.py
# Standalone evaluation for the E1 (MLP) single-expert checkpoint.
# Loads the saved PEFT checkpoint and runs Acc_image / Acc_text /
# Delta_v on the ScienceQA test split.
#
# WHERE TO RUN: Google Colab (same environment as training).
#
# COMMAND:
#   python eval_e1.py
#   python eval_e1.py --n_eval 500       # more samples
#   python eval_e1.py --ckpt_dir results/single/ckpt_E1
#
# OUTPUT:
#   results/single/single_expert_E1.json  — metrics dict
#   Prints Acc_image / Acc_text / Delta_v to stdout.
# ============================================================

import argparse
import json
import os
import re
import sys

import torch
from tqdm import tqdm

# ---- make connector.* importable from project root -------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------

MODEL_ID       = "liuhaotian/llava-v1.5-7b"
ANSWER_LETTERS = "ABCDE"


# ============================================================
# Transformers ≥ 4.46 compatibility patch
# ============================================================

def patch_generate_compat(_model=None) -> None:
    """
    Transformers ≥ 4.46 passes 'cache_position' (and sometimes
    'num_logits_to_keep') to LlavaLlamaForCausalLM.forward() during
    generation, but LLaVA's forward() signature doesn't accept them.

    Root cause: PEFT's PeftModelForCausalLM.generate() overwrites
    model.prepare_inputs_for_generation with its own version just before
    calling the inner generate loop — so patching prepare_inputs_for_generation
    on the PeftModel instance is always clobbered.

    Fix: patch LlavaLlamaForCausalLM.forward() at the CLASS level to accept
    and silently drop the unknown kwargs before delegating to the original
    forward.  A guard flag ensures the patch is applied only once.
    """
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

    if getattr(LlavaLlamaForCausalLM, "_transformers_compat_patched", False):
        return   # already patched — idempotent

    _orig_forward = LlavaLlamaForCausalLM.forward

    def _forward_compat(self, *args, **kwargs):
        kwargs.pop("cache_position",     None)   # added in transformers ≥ 4.46
        kwargs.pop("num_logits_to_keep", None)   # added in transformers ≥ 4.45
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
# Helpers (duplicated from train_single.py to keep script self-contained)
# ============================================================

def build_prompt(sample: dict, with_image: bool = True, for_eval: bool = False) -> tuple:
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
    if for_eval:
        conv.append_message(conv.roles[1], None)  # let the model generate
    else:
        conv.append_message(conv.roles[1], answer_letter)
    return conv.get_prompt(), answer_letter


def extract_answer(text: str) -> str:
    text = text.strip()
    for pat in [r'\b([A-E])\b', r'\(([A-E])\)', r'answer\s+is\s+([A-E])']:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # fallback: first character if it's a valid option letter
    if text and text[0].upper() in ANSWER_LETTERS:
        return text[0].upper()
    return ""


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, hf_dataset, tokenizer, image_processor,
             device: str, n_samples: int) -> dict:
    """
    Compute Acc_image, Acc_text, Delta_v on n_samples test samples.
    Runs two inference passes per sample: with image and text-only.
    """
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils   import tokenizer_image_token

    model.eval()
    items = list(hf_dataset.select(range(min(n_samples, len(hf_dataset)))))
    correct_img, correct_txt = 0, 0
    print(f"Evaluating on {len(items)} samples "
          f"(image + text-only = {len(items)*2} passes)...")

    for item in tqdm(items, desc="Evaluating"):
        gt = ANSWER_LETTERS[item["answer"]]

        for with_img in (True, False):
            prompt, _ = build_prompt(item, with_image=with_img, for_eval=True)
            ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).to(device)

            if with_img and item["image"] is not None:
                pix  = image_processor.preprocess(
                    item["image"], return_tensors="pt"
                )["pixel_values"].to(device, dtype=torch.float16)
                imgs = pix
                ids  = ids.unsqueeze(0)   # (T,) → (1, T)
            else:
                ids  = ids[ids != IMAGE_TOKEN_INDEX].unsqueeze(0)
                imgs = None

            # attention_mask must be explicit: transformers ≥ 4.46 calls
            # attention_mask.new_ones(...) in _update_model_kwargs_for_generation
            # and crashes if attention_mask is None.
            attn_mask = torch.ones_like(ids)
            MAX_NEW = 3
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

    n = len(items)
    return {
        "acc_image": correct_img / n,
        "acc_text":  correct_txt / n,
        "delta_v":   (correct_img - correct_txt) / n,
    }


# ============================================================
# Main
# ============================================================

def main(args):
    from datasets import load_dataset
    from transformers import BitsAndBytesConfig
    from llava.model.builder import load_pretrained_model
    from peft import PeftModel
    from connector.expert_e1 import ExpertE1

    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)
    result_path = os.path.join(args.out_dir, "single_expert_E1.json")

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

    # ---- 2. Install E1 expert (same as training) ----------------------------
    print("Installing expert E1 …")
    expert = ExpertE1(base_model.model.mm_projector)
    base_model.model.mm_projector = expert

    # ---- 3. Load PEFT checkpoint --------------------------------------------
    print(f"Loading PEFT checkpoint from {args.ckpt_dir} …")
    if not os.path.isdir(args.ckpt_dir):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {args.ckpt_dir}\n"
            "Run train_single.py --expert E1 first."
        )
    model = PeftModel.from_pretrained(base_model, args.ckpt_dir)
    model.eval()
    patch_generate_compat(model)   # strip cache_position / num_logits_to_keep
    print("Checkpoint loaded.")

    # ---- 4. Load ScienceQA test split ---------------------------------------
    print("Loading ScienceQA test split …")
    raw     = load_dataset("derek-thomas/ScienceQA")
    test_hf = raw["test"].filter(lambda x: x["image"] is not None)
    print(f"  Test samples (image-only): {len(test_hf):,}")

    # ---- 5. Evaluate ---------------------------------------------------------
    print(f"\nRunning evaluation on {args.n_eval} samples …")
    metrics = evaluate(model, test_hf, tokenizer, img_proc, device, args.n_eval)
    metrics["expert"]    = "E1"
    metrics["n_eval"]    = args.n_eval
    metrics["ckpt_dir"]  = args.ckpt_dir

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    metrics["peak_vram_gb"] = round(peak_vram_gb, 1)

    # ---- 6. Print results ----------------------------------------------------
    print(f"\n{'─'*40}")
    print(f"  Expert      : E1 (MLP — pretrained projector)")
    print(f"  Acc_image   : {metrics['acc_image']*100:.1f}%")
    print(f"  Acc_text    : {metrics['acc_text']*100:.1f}%")
    print(f"  Delta_v     : {metrics['delta_v']*100:.1f}%")
    print(f"  Samples     : {args.n_eval}")
    print(f"  Peak VRAM   : {peak_vram_gb:.1f} GB")
    print(f"{'─'*40}\n")

    # ---- 7. Save JSON --------------------------------------------------------
    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved → {result_path}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone evaluation for the E1 single-expert checkpoint."
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="results/single/ckpt_E1",
        help="Path to the saved PEFT checkpoint directory.",
    )
    parser.add_argument(
        "--n_eval", type=int, default=200,
        help="Number of test samples to evaluate on (default: 200).",
    )
    parser.add_argument(
        "--out_dir", type=str, default="results/single",
        help="Directory to write the results JSON.",
    )
    args = parser.parse_args()
    main(args)
