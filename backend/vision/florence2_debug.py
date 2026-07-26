"""
Florence-2 Debug Script — zero exception handling, full traceback.

Run from project root:
    python -m backend.vision.florence2_debug <path_to_image>
"""
from __future__ import annotations

import sys
from pathlib import Path


def run(file_path: str) -> None:
    import transformers
    print(f"\ntransformers version : {transformers.__version__}")
    print(f"Python version       : {sys.version}")

    from transformers import AutoProcessor, AutoModelForCausalLM
    import torch

    MODEL = "microsoft/Florence-2-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\nLoading processor...")
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    print("Processor loaded OK")

    print(f"\nLoading model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    print("Model loaded OK")

    # ── Inspect config before patching ───────────────────────────────────────
    lang_model = getattr(model, "language_model", None)
    print(f"\nmodel.language_model type : {type(lang_model)}")
    if lang_model:
        cfg = getattr(lang_model, "config", None)
        print(f"language_model.config type : {type(cfg)}")
        print(f"has forced_bos_token_id    : {hasattr(cfg, 'forced_bos_token_id')}")
        print(f"has forced_eos_token_id    : {hasattr(cfg, 'forced_eos_token_id')}")

    # ── Apply patch directly here — verify it sticks ──────────────────────────
    print("\nApplying patch...")
    cfg = model.language_model.config
    cfg.forced_bos_token_id  = None
    cfg.forced_eos_token_id  = None
    cfg.suppress_tokens      = None
    cfg.begin_suppress_tokens= None
    print(f"After patch — forced_bos_token_id : {cfg.forced_bos_token_id}")
    print("Patch applied OK")

    # ── Load image ─────────────────────────────────────────────────────────────
    from PIL import Image
    path = Path(file_path)
    print(f"\nLoading image: {path.name}")
    img = Image.open(path).convert("RGB")
    print(f"Image size: {img.size}")

    # ── Prepare inputs ─────────────────────────────────────────────────────────
    task  = "<CAPTION>"
    print(f"\nPreparing inputs for task: {task}")
    inputs = processor(
        text=          task,
        images=        img,
        return_tensors="pt",
    ).to(device, dtype)
    print(f"input_ids shape   : {inputs['input_ids'].shape}")
    print(f"pixel_values shape: {inputs['pixel_values'].shape}")

    # ── Generate — NO try/except — raw traceback on failure ───────────────────
    print("\nCalling model.generate()...")
    generated_ids = model.generate(
        input_ids=    inputs["input_ids"],
        pixel_values= inputs["pixel_values"],
        max_new_tokens=128,
        do_sample=    False,
        num_beams=    3,
    )
    print(f"generated_ids shape: {generated_ids.shape}")

    # ── Decode ─────────────────────────────────────────────────────────────────
    print("\nDecoding...")
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]
    print(f"Raw output: {generated_text[:200]}")

    # ── Post-process ──────────────────────────────────────────────────────────
    print("\nPost-processing...")
    result = processor.post_process_generation(
        generated_text,
        task=       task,
        image_size= (img.width, img.height),
    )
    print(f"\nFinal result: {result}")
    print("\n✓ Florence-2 works correctly with this transformers version.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backend.vision.florence2_debug <image_path>")
        sys.exit(1)
    run(sys.argv[1])