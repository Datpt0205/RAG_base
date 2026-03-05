# -*- coding: utf-8 -*-
"""
Estimate trained tokens from trainer_state.json (no dataset required).

Reads the actual cumulative `num_tokens` from log_history and handles
counter resets that may occur across epochs or dataloader restarts.
Also cross-validates via total_flos if model config is available.

Usage:
  python estimate_trained_tokens_no_dataset.py --checkpoint_dir ./outputs_train/checkpoint-30000
  python estimate_trained_tokens_no_dataset.py --checkpoint_dir ./outputs_train/checkpoint-30000 --all_checkpoints
"""

import os
import json
import glob
import argparse
from typing import List, Tuple


def human(n: float) -> str:
    units = ["", "K", "M", "B", "T"]
    x = float(n)
    for u in units:
        if abs(x) < 1000:
            return f"{x:.2f}{u}"
        x /= 1000.0
    return f"{x:.2f}P"


def load_trainer_state(ckpt_dir: str) -> dict:
    p = os.path.join(ckpt_dir, "trainer_state.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def load_model_config(ckpt_dir: str) -> dict | None:
    for name in ["config.json", "../config.json"]:
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def estimate_params_from_config(config: dict) -> int | None:
    """Rough parameter estimation for transformer models."""
    try:
        hidden = config.get("hidden_size", 0)
        layers = config.get("num_hidden_layers", 0)
        vocab = config.get("vocab_size", 0)
        intermediate = config.get("intermediate_size", hidden * 4)

        if not all([hidden, layers, vocab]):
            return None

        # Embedding
        params = vocab * hidden
        # Each transformer layer: attention + MLP
        # Attention: 4 * hidden^2 (Q, K, V, O projections)
        # MLP: 2 * hidden * intermediate (up + down, or 3x for SwiGLU)
        num_kv_heads = config.get("num_key_value_heads", config.get("num_attention_heads", 0))
        num_heads = config.get("num_attention_heads", 0)
        head_dim = hidden // num_heads if num_heads else 0

        if num_kv_heads and head_dim:
            attn_params = (num_heads * head_dim * hidden) + (2 * num_kv_heads * head_dim * hidden) + (hidden * hidden)
        else:
            attn_params = 4 * hidden * hidden

        mlp_params = 2 * hidden * intermediate
        if config.get("hidden_act") in ["silu", "swiglu", "gelu"]:
            mlp_params = 3 * hidden * intermediate

        params += layers * (attn_params + mlp_params)
        # LM head (often tied with embeddings)
        if not config.get("tie_word_embeddings", True):
            params += vocab * hidden

        return int(params)
    except Exception:
        return None


def extract_training_tokens(log_history: list) -> Tuple[int, List[dict]]:
    """
    Extract total trained tokens from log_history.

    The `num_tokens` field in HuggingFace trainer log_history is cumulative
    but may RESET when the dataloader restarts (e.g., at epoch boundaries
    in some TRL versions, or when training is resumed from checkpoint).

    Strategy: detect resets (when num_tokens decreases) and sum all segments.

    Returns:
        (total_tokens, segment_info_list)
    """
    train_entries = [
        e for e in log_history
        if "num_tokens" in e and "eval_loss" not in e and "eval_entropy" not in e
    ]

    if not train_entries:
        return 0, []

    segments = []
    seg_start_step = train_entries[0].get("step", 0)
    seg_start_tokens = train_entries[0]["num_tokens"]
    prev_tokens = seg_start_tokens
    peak_tokens = seg_start_tokens

    for i in range(1, len(train_entries)):
        curr_tokens = train_entries[i]["num_tokens"]
        curr_step = train_entries[i].get("step", 0)

        if curr_tokens < prev_tokens * 0.5:
            # Reset detected
            segments.append({
                "start_step": seg_start_step,
                "end_step": train_entries[i - 1].get("step", 0),
                "peak_tokens": peak_tokens,
            })
            seg_start_step = curr_step
            seg_start_tokens = curr_tokens
            peak_tokens = curr_tokens
        else:
            peak_tokens = max(peak_tokens, curr_tokens)

        prev_tokens = curr_tokens

    # Last segment
    segments.append({
        "start_step": seg_start_step,
        "end_step": train_entries[-1].get("step", 0),
        "peak_tokens": peak_tokens,
    })

    total_tokens = sum(s["peak_tokens"] for s in segments)
    return int(total_tokens), segments


def estimate_tokens_from_flos(total_flos: float, num_params: int) -> int:
    """
    Estimate tokens from FLOPs using the Chinchilla scaling law approximation:
        FLOPs ≈ 6 × num_params × num_tokens
    """
    if total_flos and num_params:
        return int(total_flos / (6 * num_params))
    return 0


def process_checkpoint(ckpt_dir: str, verbose: bool = True):
    st = load_trainer_state(ckpt_dir)
    global_step = int(st.get("global_step", 0) or 0)
    total_flos = float(st.get("total_flos", 0) or 0)
    max_steps = st.get("max_steps", "N/A")
    num_epochs = st.get("num_train_epochs", "N/A")
    epoch = st.get("epoch", 0)
    train_batch_size = st.get("train_batch_size", "N/A")
    log_history = st.get("log_history", [])

    total_tokens, segments = extract_training_tokens(log_history)

    if verbose:
        print("=" * 60)
        print(f"CHECKPOINT: {ckpt_dir}")
        print("=" * 60)
        print(f"  global_step      : {global_step}")
        print(f"  epoch            : {epoch}")
        print(f"  max_steps        : {max_steps}")
        print(f"  num_train_epochs : {num_epochs}")
        print(f"  train_batch_size : {train_batch_size}")
        print(f"  total_flos       : {total_flos:.4e}")

        print(f"\n--- Token Segments (from num_tokens in log_history) ---")
        if len(segments) > 1:
            for i, seg in enumerate(segments):
                print(f"  Segment {i + 1}: steps {seg['start_step']:>6d} → {seg['end_step']:>6d} | peak = {human(seg['peak_tokens']):>10s} ({seg['peak_tokens']:,})")
            print(f"  {'':>60s}")
        elif segments:
            seg = segments[0]
            print(f"  Single segment: steps {seg['start_step']} → {seg['end_step']} | peak = {human(seg['peak_tokens'])} ({seg['peak_tokens']:,})")

        print(f"\n{'=' * 60}")
        print(f"  ✅ TOTAL TRAINED TOKENS: {human(total_tokens)} ({total_tokens:,})")
        print(f"{'=' * 60}")

        # Cross-validate with FLOPs
        config = load_model_config(ckpt_dir)
        if config:
            num_params = estimate_params_from_config(config)
            if num_params:
                flos_estimate = estimate_tokens_from_flos(total_flos, num_params)
                print(f"\n--- Cross-validation via total_flos ---")
                print(f"  Estimated model params : {human(num_params)} ({num_params:,})")
                print(f"  FLOPs-based estimate   : {human(flos_estimate)} ({flos_estimate:,})")
                ratio = flos_estimate / total_tokens if total_tokens else 0
                print(f"  Ratio (flos/actual)    : {ratio:.2f}x")
                if 0.8 <= ratio <= 1.2:
                    print(f"  ✅ Estimates are consistent (within 20%)")
                else:
                    print(f"  ⚠️  Estimates differ significantly — flos estimate is rough")

        # Avg tokens per step
        if global_step > 0 and total_tokens > 0:
            avg_per_step = total_tokens / global_step
            print(f"\n--- Additional Info ---")
            print(f"  Avg tokens/step : {avg_per_step:,.0f}")
            print(f"  Avg seq length  : ~{avg_per_step / max(int(train_batch_size) if str(train_batch_size).isdigit() else 1, 1):,.0f} (assuming batch_size={train_batch_size})")

    return {
        "checkpoint": ckpt_dir,
        "global_step": global_step,
        "total_tokens": total_tokens,
        "segments": segments,
        "total_flos": total_flos,
    }


def main():
    ap = argparse.ArgumentParser(description="Estimate trained tokens from trainer_state.json")
    ap.add_argument("--checkpoint_dir", required=True, help="Path to a checkpoint directory")
    ap.add_argument("--all_checkpoints", action="store_true",
                    help="Scan all checkpoint-* dirs in the parent of checkpoint_dir")
    args = ap.parse_args()

    if args.all_checkpoints:
        parent = os.path.dirname(args.checkpoint_dir.rstrip("/\\"))
        ckpts = sorted(glob.glob(os.path.join(parent, "checkpoint-*")),
                       key=lambda x: int(x.rsplit("-", 1)[-1]))
        print(f"Found {len(ckpts)} checkpoints in {parent}\n")
        results = []
        for ckpt in ckpts:
            try:
                r = process_checkpoint(ckpt, verbose=False)
                results.append(r)
            except FileNotFoundError:
                continue

        print(f"{'Step':>8s} | {'Tokens':>14s} | {'Human':>10s} | {'Segments':>8s}")
        print("-" * 50)
        for r in results:
            print(f"{r['global_step']:>8d} | {r['total_tokens']:>14,d} | {human(r['total_tokens']):>10s} | {len(r['segments']):>8d}")

        if results:
            last = results[-1]
            print(f"\n✅ Latest checkpoint (step {last['global_step']}): {human(last['total_tokens'])} tokens")
    else:
        process_checkpoint(args.checkpoint_dir)


if __name__ == "__main__":
    main()
