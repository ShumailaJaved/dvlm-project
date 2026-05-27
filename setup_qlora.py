# ============================================================
# setup_qlora.py
# Task A-0: QLoRA Environment Migration
# ============================================================
# WHERE TO RUN: Google Colab with T4 GPU (Linux only).
#   bitsandbytes 4-bit quantization requires CUDA 11.x+ on Linux.
#   Do NOT run on Windows or macOS.
#
# COMMAND:
#   python setup_qlora.py
#
# RUN THIS IN A COLAB CELL FIRST (install dependencies):
#   !pip install "bitsandbytes>=0.41" "peft>=0.9" "accelerate" \
#       "transformers>=4.37" datasets tqdm
#   !pip install -e ./LLaVA           # install local LLaVA package
#
# IMPORTANT ORDER: prepare_model_for_kbit_training MUST be called
#   BEFORE get_peft_model — wrong order causes gradient issues.
# ============================================================

import math
import sys

import torch


# ============================================================
# A-0.1  BitsAndBytes Configuration
# ============================================================

def get_bnb_config():
    """
    Return a BitsAndBytesConfig for 4-bit NF4 quantization.

    NF4 (NormalFloat4) is the recommended quantization data type for
    QLoRA (Dettmers et al., 2023).  Double quantization further reduces
    the memory footprint of the quantization constants themselves.

    Returns:
        BitsAndBytesConfig with:
            load_in_4bit         = True
            bnb_4bit_quant_type  = "nf4"
            bnb_4bit_compute_dtype = torch.float16
            bnb_4bit_use_double_quant = True
    """
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 — best for LLMs
        bnb_4bit_compute_dtype=torch.float16,   # intermediate compute dtype
        bnb_4bit_use_double_quant=True,     # quantize the quant constants too
    )
    return bnb_config


# ============================================================
# A-0.1  Model Loading
# ============================================================

def load_llava_4bit(model_id: str = "liuhaotian/llava-v1.5-7b"):
    """
    Load LLaVA-1.5-7B in 4-bit NF4 QLoRA mode.

    Builds an explicit BitsAndBytesConfig via get_bnb_config() and passes
    it as quantization_config to LLaVA's own loader (load_pretrained_model).
    This is preferred over the convenience flag load_4bit=True because it
    gives full control over quant settings and is compatible with newer
    versions of transformers / bitsandbytes.

    Steps performed:
        1. Print GPU memory before loading.
        2. Build BitsAndBytesConfig (NF4, double quant, fp16 compute).
        3. Download / load LLaVA-1.5-7B with that config.
        4. Print GPU memory after loading.

    Args:
        model_id (str): HuggingFace model path.
                        Default: "liuhaotian/llava-v1.5-7b".

    Returns:
        tuple: (tokenizer, model, image_processor, context_len)
            tokenizer       — LLaVA tokenizer
            model           — LLaVA model in 4-bit NF4
            image_processor — CLIP image preprocessor
            context_len     — maximum context length
    """
    from llava.model.builder import load_pretrained_model

    # Print GPU memory BEFORE loading
    mem_before = torch.cuda.memory_allocated() / (1024 ** 3)   # bytes → GB
    print(f"GPU memory before loading model : {mem_before:.2f} GB")

    # Build the quantization config explicitly instead of using load_4bit=True.
    # Passing quantization_config directly is more explicit and avoids relying
    # on LLaVA's internal flag handling, which varies across versions.
    bnb_config = get_bnb_config()

    print(f"Loading {model_id} in 4-bit NF4 …")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_id,
        model_base=None,
        model_name="llava-v1.5-7b",
        quantization_config=bnb_config,   # explicit NF4 config (not load_4bit)
    )

    # Print GPU memory AFTER loading
    mem_after = torch.cuda.memory_allocated() / (1024 ** 3)   # bytes → GB
    print(f"GPU memory after loading model  : {mem_after:.2f} GB")
    print(f"Model memory footprint          : {mem_after - mem_before:.2f} GB")

    return tokenizer, model, image_processor, context_len


# ============================================================
# A-0.1  QLoRA Adapter Application
# ============================================================

def apply_qlora(model):
    """
    Prepare the 4-bit model for training and attach LoRA adapters.

    CRITICAL ORDER:
        prepare_model_for_kbit_training(model)   ← MUST come first
        get_peft_model(model, lora_config)        ← MUST come second

    Calling get_peft_model before prepare_model_for_kbit_training causes
    gradient checkpointing to be incompatible with the quantized weights,
    producing zero or NaN gradients.

    LoRA targets: q_proj, k_proj, v_proj, o_proj (attention projections only).
    The CLIP encoder and the MLP projector remain frozen.
    The 4-bit base weights remain frozen; only LoRA adapters are trainable.

    Hyperparameters (from project specification):
        r       = 16
        alpha   = 32
        dropout = 0.05

    Args:
        model: LLaVA model loaded with load_llava_4bit().

    Returns:
        model: Model with LoRA adapters attached and ready for training.
               Call model.print_trainable_parameters() to verify.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Step 1: Enable gradient checkpointing and mark certain layers trainable.
    # This MUST happen before LoRA is attached.
    print("Calling prepare_model_for_kbit_training …")
    model = prepare_model_for_kbit_training(model)

    # Step 2: Configure LoRA adapters.
    lora_config = LoraConfig(
        r=16,                   # rank of the low-rank decomposition
        lora_alpha=32,          # scaling factor (effective LR scale = alpha/r)
        lora_dropout=0.05,      # dropout applied inside LoRA layers
        target_modules=[        # which linear layers get LoRA adapters
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        bias="none",            # do not adapt bias terms
        task_type="CAUSAL_LM",  # causal language modelling objective
    )

    # Step 3: Wrap model with LoRA adapters.
    print("Attaching LoRA adapters …")
    model = get_peft_model(model, lora_config)

    # Print a summary of trainable parameters.
    model.print_trainable_parameters()

    return model


# ============================================================
# A-0.2  Full Dataset Pipeline
# ============================================================

def load_scienceqa_full():
    """
    Load all ScienceQA splits and filter to image-only questions.

    Applies the same image-only filter used in the midterm, but now
    loads ALL 6 218 training samples (previously capped at 2 000).

    Dataset: derek-thomas/ScienceQA on HuggingFace Hub.

    Filter rule: keep a sample only if its "image" field is not None.
    Actual image-only split sizes (confirmed from HuggingFace Hub):
        Train      : 6,218
        Validation : 2,097
        Test       : 2,017

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
            Each is a HuggingFace Dataset object filtered to image-only.
    """
    from datasets import load_dataset

    print("Loading ScienceQA from HuggingFace Hub …")
    raw = load_dataset("derek-thomas/ScienceQA")

    def has_image(example):
        """Return True if this sample contains an image."""
        return example["image"] is not None

    # Filter all three splits to keep only image-containing questions.
    print("Filtering splits to image-only samples …")
    train_ds = raw["train"].filter(has_image)
    val_ds   = raw["validation"].filter(has_image)
    test_ds  = raw["test"].filter(has_image)

    print(f"  Train : {len(train_ds):,} samples")
    print(f"  Val   : {len(val_ds):,} samples")
    print(f"  Test  : {len(test_ds):,} samples")

    return train_ds, val_ds, test_ds


# ============================================================
# A-0.2  Gradient Step Calculation
# ============================================================

def compute_gradient_steps(
    train_size: int,
    effective_batch_size: int,
    epochs: int,
) -> int:
    """
    Compute the total number of gradient update steps for training.

    With gradient accumulation, one optimizer step happens every
    effective_batch_size samples.  If the dataset doesn't divide evenly,
    the last partial batch still produces one step (ceil).

    Formula:
        steps_per_epoch = ceil(train_size / effective_batch_size)
        total_steps     = steps_per_epoch * epochs

    For this project: train_size=6218, effective_batch_size=16, epochs=2
        steps_per_epoch = ceil(6218 / 16) = ceil(388.625) = 389
        total_steps     = 389 * 2 = 778

    Args:
        train_size (int):           Number of training samples.
        effective_batch_size (int): Batch size after gradient accumulation.
        epochs (int):               Number of training epochs.

    Returns:
        int: Total number of gradient update steps.
    """
    steps_per_epoch = math.ceil(train_size / effective_batch_size)
    total_steps     = steps_per_epoch * epochs
    return total_steps


# ============================================================
# Verification block — Task A-0
# ============================================================

if __name__ == "__main__":
    # ============================================================
    # Verification for Task A-0.
    #
    # CHECK 1 (CUDA required): GPU memory after loading < 15 GB
    # CHECK 2: Train split size == 6,218
    # CHECK 3: Val   split size == 2,097
    # CHECK 4: Test  split size == 2,017
    # CHECK 5: Gradient steps  ==   778  (2 epochs, batch 16, 6218 samples)
    #
    # Check  1 runs only on CUDA (Colab T4).
    # Checks 2–4 require internet access to HuggingFace Hub.
    # Check  5 is pure arithmetic — always runs.
    # ============================================================

    all_passed = True

    # ---------------------------------------------------------------
    # CHECK 1 — GPU memory after 4-bit loading < 15 GB (CUDA only)
    # ---------------------------------------------------------------
    if not torch.cuda.is_available():
        print("CHECK 1 SKIPPED: CUDA not available — run on Colab T4 GPU")
    else:
        print("CHECK 1: Loading LLaVA-1.5-7B in 4-bit NF4 …")
        try:
            tokenizer, model, image_processor, context_len = load_llava_4bit()
            mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            limit_gb = 15.0

            if mem_gb < limit_gb:
                print(
                    f"CHECK 1 PASSED  "
                    f"(GPU memory = {mem_gb:.2f} GB < {limit_gb:.1f} GB limit)"
                )
            else:
                print(
                    f"CHECK 1 FAILED: GPU memory = {mem_gb:.2f} GB "
                    f">= {limit_gb:.1f} GB limit — model does not fit"
                )
                all_passed = False

        except Exception as exc:
            print(f"CHECK 1 FAILED: model loading raised an exception: {exc}")
            all_passed = False

    # ---------------------------------------------------------------
    # CHECK 2–4 — Dataset split sizes
    # ---------------------------------------------------------------
    try:
        train_ds, val_ds, test_ds = load_scienceqa_full()

        # CHECK 2: Train size
        if len(train_ds) == 6218:
            print(f"CHECK 2 PASSED  (train size = {len(train_ds):,})")
        else:
            print(
                f"CHECK 2 FAILED: train size = {len(train_ds):,}, "
                f"expected 6,218"
            )
            all_passed = False

        # CHECK 3: Val size
        if len(val_ds) == 2097:
            print(f"CHECK 3 PASSED  (val size = {len(val_ds):,})")
        else:
            print(
                f"CHECK 3 FAILED: val size = {len(val_ds):,}, "
                f"expected 2,097"
            )
            all_passed = False

        # CHECK 4: Test size
        if len(test_ds) == 2017:
            print(f"CHECK 4 PASSED  (test size = {len(test_ds):,})")
        else:
            print(
                f"CHECK 4 FAILED: test size = {len(test_ds):,}, "
                f"expected 2,017"
            )
            all_passed = False

    except ImportError:
        print(
            "CHECK 2–4 SKIPPED: 'datasets' library not installed. "
            "Install with: pip install datasets"
        )
    except Exception as exc:
        print(f"CHECK 2–4 FAILED: dataset loading raised an exception: {exc}")
        all_passed = False

    # ---------------------------------------------------------------
    # CHECK 5 — Gradient step count (pure arithmetic, always runs)
    # ---------------------------------------------------------------
    TRAIN_SIZE    = 6218
    BATCH_SIZE    = 16
    EPOCHS        = 2
    EXPECTED_STEPS = 778   # ceil(6218/16) * 2 = 389 * 2

    computed_steps = compute_gradient_steps(TRAIN_SIZE, BATCH_SIZE, EPOCHS)

    if computed_steps == EXPECTED_STEPS:
        print(
            f"CHECK 5 PASSED  "
            f"({TRAIN_SIZE} samples / batch {BATCH_SIZE} × {EPOCHS} epochs "
            f"= {computed_steps} gradient steps)"
        )
    else:
        print(
            f"CHECK 5 FAILED: computed {computed_steps} steps, "
            f"expected {EXPECTED_STEPS}"
        )
        all_passed = False

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print()
    if all_passed:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
