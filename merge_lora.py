# -*- coding: utf-8 -*-
"""
Merge LoRA/QLoRA adapter into base model -> standalone merged model.

Usage:
python .\merge_lora.py --base_model_id "Qwen/Qwen2.5-3B-Instruct" --adapter_dir "C:\Users\USER\mse\server\RAG_base\outputs_train\checkpoint-30000" --out_dir "C:\Users\USER\mse\server\RAG_base\merged_qwen2p5_3b_jobfit" --dtype auto --device_map auto

Notes:
- This script does NOT need bitsandbytes/triton.
- It loads base model in fp16/bf16 (NOT 4bit) then merges LoRA weights.
"""

import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def pick_dtype(dtype_str: str):
    dtype_str = (dtype_str or "auto").lower()
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "fp32":
        return torch.float32

    # auto
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_id", type=str, required=True)
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--device_map", type=str, default="auto", help="auto | cpu | balanced | etc.")
    ap.add_argument("--max_shard_size", type=str, default="2GB")
    ap.add_argument("--trust_remote_code", action="store_true", help="Enable trust_remote_code")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dtype = pick_dtype(args.dtype)
    trust = bool(args.trust_remote_code)

    print(f"[merge] base_model_id   = {args.base_model_id}")
    print(f"[merge] adapter_dir     = {args.adapter_dir}")
    print(f"[merge] out_dir         = {args.out_dir}")
    print(f"[merge] dtype           = {dtype}")
    print(f"[torch.cuda.is_available={torch.cuda.is_available()}]")

    # tokenizer: prefer adapter_dir if it contains tokenizer files (your checkpoints do)
    try:
        tok = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=trust)
        print("[merge] tokenizer loaded from adapter_dir")
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=trust)
        print("[merge] tokenizer loaded from base_model_id")

    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # base model in fp16/bf16 (NOT quantized)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=dtype,
        device_map=args.device_map if torch.cuda.is_available() else None,
        trust_remote_code=trust,
    )

    # load adapter
    peft_model = PeftModel.from_pretrained(base, args.adapter_dir)
    print("[merge] adapter loaded -> merging...")

    merged = peft_model.merge_and_unload()
    merged.eval()

    print("[merge] saving merged model...")
    merged.save_pretrained(
        args.out_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tok.save_pretrained(args.out_dir)

    manifest = {
        "base_model_id": args.base_model_id,
        "adapter_dir": args.adapter_dir,
        "dtype": str(dtype),
        "device_map": args.device_map,
    }
    with open(os.path.join(args.out_dir, "merge_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[merge] DONE -> {args.out_dir}")


if __name__ == "__main__":
    main()
