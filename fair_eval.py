"""Fair head-to-head eval of TopK vs PositionAware on positions >= 4.

The headline EV from the benchmark is on each architecture's own training
distribution: TopK on positions >= 4, PositionAware on all positions. To
compare them apples-to-apples we re-evaluate PositionAware on positions >= 4
only (where TopK is also evaluated) and ask whether PositionAware matches /
beats TopK at the regular-position end of the sequence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from legible_weights.data.activations import collect_activations
from legible_weights.data.adapters import QWEN_LLAMA
from legible_weights.sae.model import SAEConfig, TopKSAE
from legible_weights.sae.position_aware import (
    PositionAwareSAEConfig,
    PositionAwareTopKSAE,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path,
                    default=Path("experiments/small_sae_bench/qwen2.5-0.5b-l9"))
    ap.add_argument("--n-tokens", type=int, default=500_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--dataset-offset", type=int, default=100_000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = json.loads((args.bench_dir / "report.json").read_text())
    layer = report["layer"]
    d_in = report["d_in"]
    d_hidden = report["d_hidden"]
    k = report["k"]

    # Load TopK SAE
    topk = TopKSAE(SAEConfig(d_in=d_in, d_hidden=d_hidden, k=k))
    topk.load_state_dict(torch.load(args.bench_dir / "topk" / "sae.pt",
                                     map_location="cpu", weights_only=True))
    topk.to(device).eval()

    # Load PositionAware SAE
    pa = PositionAwareTopKSAE(PositionAwareSAEConfig(
        d_in=d_in, d_hidden=d_hidden, k=k, max_pos=16,
    ))
    pa.load_state_dict(torch.load(args.bench_dir / "position_aware" / "sae.pt",
                                  map_location="cpu", weights_only=True))
    pa.to(device).eval()

    # Collect held-out activations with positions
    print(f"[load] base=Qwen/Qwen2.5-0.5B layer={layer}")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    ds = ds.skip(args.dataset_offset)
    acts, pos = collect_activations(
        model=model, tokenizer=tok,
        texts=(row["text"] for row in ds),
        layer_idx=layer, n_tokens=args.n_tokens, seq_len=args.seq_len,
        batch_size=8, device=device, exclude_first_n=0,
        adapter=QWEN_LLAMA, shuffle=False, return_positions=True,
    )
    print(f"[data] held-out: {tuple(acts.shape)}, pos in [{pos.min()},{pos.max()}]")
    del model
    torch.cuda.empty_cache()

    # Evaluate each architecture on:
    #   A) positions >= 4   — what TopK was trained for; fair head-to-head
    #   B) positions 0..3   — where PositionAware claims its unique capability
    #   C) all positions    — the headline number reported in the benchmark
    out = {}
    pos_buckets = {
        "positions_ge_4": pos >= 4,
        "positions_0_to_3": pos < 4,
        "all_positions": torch.ones_like(pos, dtype=torch.bool),
    }
    for bucket_name, mask in pos_buckets.items():
        a_cpu = acts[mask]                # fp16 on CPU
        p_cpu = pos[mask]
        n = a_cpu.shape[0]
        chunk = 4096

        def chunked_eval(sae, needs_pos):
            sq_err_sum = 0.0
            count = 0
            l0_sum = 0.0
            n_seen = 0
            x_var_acc = torch.zeros(a_cpu.shape[1], dtype=torch.float64)
            x_mean_acc = torch.zeros(a_cpu.shape[1], dtype=torch.float64)
            with torch.no_grad():
                # First pass: mean
                for i in range(0, n, chunk):
                    x = a_cpu[i:i + chunk].to(device, dtype=torch.float32)
                    x_mean_acc += x.sum(dim=0).double().cpu()
                    n_seen += x.shape[0]
                x_mean = (x_mean_acc / n_seen).to(device, dtype=torch.float32)
                # Second pass: per-batch SAE forward + variance accumulators
                sq_diff_x = 0.0
                sq_diff_resid = 0.0
                for i in range(0, n, chunk):
                    x = a_cpu[i:i + chunk].to(device, dtype=torch.float32)
                    if needs_pos:
                        pp = p_cpu[i:i + chunk].to(device).long()
                        recon, acts_ = sae(x, pp)
                    else:
                        recon, acts_ = sae(x)
                    sq_err_sum += (recon - x).pow(2).sum().item()
                    count += x.numel()
                    sq_diff_x += (x - x_mean).pow(2).sum().item()
                    sq_diff_resid += (x - recon).pow(2).sum().item()
                    l0_sum += (acts_ > 0).float().sum().item()
            mse = sq_err_sum / count
            ev = 1.0 - sq_diff_resid / sq_diff_x if sq_diff_x > 0 else float("nan")
            l0 = l0_sum / n
            return {"mse": mse, "ev": ev, "l0": l0}

        out[bucket_name] = {
            "n_tokens": int(mask.sum().item()),
            "topk": chunked_eval(topk, needs_pos=False),
            "position_aware": chunked_eval(pa, needs_pos=True),
        }
        print(f"\n[{bucket_name}] n={int(mask.sum().item())}")
        print(f"  topk:           mse={out[bucket_name]['topk']['mse']:.4f}  "
              f"ev={out[bucket_name]['topk']['ev']:.4f}  "
              f"L0={out[bucket_name]['topk']['l0']:.1f}")
        print(f"  position_aware: mse={out[bucket_name]['position_aware']['mse']:.4f}  "
              f"ev={out[bucket_name]['position_aware']['ev']:.4f}  "
              f"L0={out[bucket_name]['position_aware']['l0']:.1f}")

    (args.bench_dir / "fair_eval.json").write_text(json.dumps(out, indent=2))
    print(f"\n[save] wrote {args.bench_dir}/fair_eval.json")


if __name__ == "__main__":
    main()
