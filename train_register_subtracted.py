"""Train the Register-Subtracted TopK SAE on Qwen2.5-0.5B layer 9.

Pipeline:
1. Compute the position-0 register vector by running ~256 inputs through
   the base model and averaging the layer-9 position-0 residual.
2. Train a vanilla TopK SAE on the full sequence (no exclude_first_n),
   with the register subtracted from the position-0 activations before
   encoding.
3. Save the SAE checkpoint plus the register vector so fair_eval can
   load both.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from legible_weights.data.activations import collect_activations
from legible_weights.data.adapters import GPT2, PYTHIA, QWEN_LLAMA
from legible_weights.sae.register_subtracted import (
    RegisterSubtractedSAEConfig,
    RegisterSubtractedTopKSAE,
)
from legible_weights.sae.unified_train import UnifiedTrainConfig, train_unified


# Multi-model dispatch — pick the right adapter per architecture
BASE_SPECS = {
    "qwen2.5-0.5b": {"hf": "Qwen/Qwen2.5-0.5B", "adapter": QWEN_LLAMA},
    "gpt2-small":   {"hf": "openai-community/gpt2", "adapter": GPT2},
    "pythia-1.4b":  {"hf": "EleutherAI/pythia-1.4b", "adapter": PYTHIA},
}


@torch.no_grad()
def compute_register_vector(model, tok, device, layer: int, n_sequences: int = 256, seq_len: int = 64):
    """Mean position-0 residual at the target layer, averaged over n_sequences."""
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                       split="train", streaming=True)
    texts = [row["text"] for _, row in zip(range(n_sequences), ds)]
    accum = None
    count = 0
    for i in range(0, n_sequences, 8):
        batch = texts[i:i + 8]
        enc = tok(batch, return_tensors="pt", padding="max_length",
                   truncation=True, max_length=seq_len).to(device)
        out = model(**enc, output_hidden_states=True)
        reg = out.hidden_states[layer + 1][:, 0, :].float().cpu()
        if accum is None:
            accum = reg.sum(dim=0)
        else:
            accum += reg.sum(dim=0)
        count += reg.shape[0]
    return accum / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=list(BASE_SPECS), default="qwen2.5-0.5b",
                    help="Model family key — picks the right adapter")
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--n-tokens", type=int, default=5_000_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--expansion", type=int, default=16)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    spec = BASE_SPECS[args.base]
    model_name = spec["hf"]
    adapter = spec["adapter"]

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device).eval()
    d_model = model.config.hidden_size

    # 1) Register vector
    print("[setup] computing position-0 register vector...")
    register = compute_register_vector(model, tok, device, args.layer)
    print(f"[setup] register vector: norm={register.norm().item():.2f}, d_model={d_model}")
    torch.save(register, args.out / "register_vector.pt")

    # 2) Collect activations (full sequence, with positions)
    print("[data] collecting activations with positions")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                       split="train", streaming=True)
    t0 = time.time()
    activations, positions = collect_activations(
        model=model, tokenizer=tok,
        texts=(row["text"] for row in ds),
        layer_idx=args.layer, n_tokens=args.n_tokens, seq_len=args.seq_len,
        batch_size=8, device=device, exclude_first_n=0,
        adapter=adapter, shuffle=True, return_positions=True,
    )
    print(f"[data] {tuple(activations.shape)} in {time.time() - t0:.1f}s")

    del model
    torch.cuda.empty_cache()

    # 3) Build SAE, install register, train
    cfg = RegisterSubtractedSAEConfig(d_in=d_model, d_hidden=d_model * args.expansion, k=args.k)
    sae = RegisterSubtractedTopKSAE(cfg)
    sae.set_register(register)
    sae.to(device)
    print(f"[sae] d_in={cfg.d_in} d_hidden={cfg.d_hidden} k={cfg.k}")

    train_cfg = UnifiedTrainConfig(
        batch_size=args.batch_size, lr=args.lr,
        n_epochs=args.epochs, device=device,
    )

    t0 = time.time()
    history = train_unified(sae, activations, train_cfg, positions=positions)
    wall = time.time() - t0
    print(f"[train] {len(history)} log points in {wall:.1f}s")

    # 4) Save
    torch.save(sae.state_dict(), args.out / "sae.pt")
    (args.out / "config.json").write_text(json.dumps({
        "base_model": model_name,
        "base_key": args.base,
        "adapter": adapter.name,
        "layer": args.layer,
        "arch": "RegisterSubtractedTopK",
        "sae": {"d_in": cfg.d_in, "d_hidden": cfg.d_hidden, "k": cfg.k},
        "training": {
            "n_tokens": args.n_tokens, "seq_len": args.seq_len,
            "batch_size": args.batch_size, "lr": args.lr,
            "n_epochs": args.epochs, "exclude_first_n": 0,
            "dataset": "HuggingFaceFW/fineweb-edu:sample-10BT",
        },
        "register_norm": float(register.norm().item()),
    }, indent=2))
    (args.out / "history.json").write_text(json.dumps(history, indent=2))

    if history:
        last = history[-1]
        print(f"[final] step={last['step']} mse={last['mse']:.4f} "
              f"ev={last['explained_var']:.3f} l0={last['l0']:.1f}")


if __name__ == "__main__":
    main()
