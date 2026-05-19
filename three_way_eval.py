"""Three-way fair head-to-head: TopK vs Position-Aware vs Register-Subtracted.

All three SAEs evaluated on the same held-out Qwen2.5-0.5B layer-9
activations, with the same three position buckets (>=4, 0-3, all) and
CE recovery via splice intervention.
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
from legible_weights.eval.recovery import ce_recovery
from legible_weights.sae.model import SAEConfig, TopKSAE
from legible_weights.sae.position_aware import (
    PositionAwareSAEConfig, PositionAwareTopKSAE,
)
from legible_weights.sae.register_subtracted import (
    RegisterSubtractedSAEConfig, RegisterSubtractedTopKSAE,
)


def _bench_cfg(ckpt_dir):
    # TopK/PositionAware live under results/l9/{topk,position_aware}/sae.pt;
    # the architecture metadata is in the parent's report.json.
    report = json.loads((ckpt_dir.parent / "report.json").read_text())
    return report["d_in"], report["d_hidden"], report["k"]


def load_topk(ckpt_dir):
    d_in, d_hidden, k = _bench_cfg(ckpt_dir)
    sae = TopKSAE(SAEConfig(d_in=d_in, d_hidden=d_hidden, k=k))
    sae.load_state_dict(torch.load(ckpt_dir / "sae.pt", map_location="cpu", weights_only=True))
    return sae


def load_position_aware(ckpt_dir):
    d_in, d_hidden, k = _bench_cfg(ckpt_dir)
    sae = PositionAwareTopKSAE(PositionAwareSAEConfig(
        d_in=d_in, d_hidden=d_hidden, k=k, max_pos=16,
    ))
    sae.load_state_dict(torch.load(ckpt_dir / "sae.pt", map_location="cpu", weights_only=True))
    return sae


def load_register_subtracted(ckpt_dir):
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    sae = RegisterSubtractedTopKSAE(RegisterSubtractedSAEConfig(
        d_in=cfg["sae"]["d_in"], d_hidden=cfg["sae"]["d_hidden"], k=cfg["sae"]["k"],
    ))
    sae.load_state_dict(torch.load(ckpt_dir / "sae.pt", map_location="cpu", weights_only=True))
    return sae


def chunked_eval(sae, acts_cpu, pos_cpu, kind, device, chunk=4096):
    """Returns dict of mse / ev / l0 for the given activations."""
    n = acts_cpu.shape[0]
    sq_err_sum = 0.0
    count = 0
    l0_sum = 0.0
    sum_x = torch.zeros(acts_cpu.shape[1], dtype=torch.float64)
    n_seen = 0
    with torch.no_grad():
        # First pass: mean
        for i in range(0, n, chunk):
            x = acts_cpu[i:i + chunk].to(device, dtype=torch.float32)
            sum_x += x.sum(dim=0).double().cpu()
            n_seen += x.shape[0]
        x_mean = (sum_x / n_seen).to(device, dtype=torch.float32)
        # Second pass: SAE + variance
        sq_diff_x = 0.0
        sq_diff_resid = 0.0
        for i in range(0, n, chunk):
            x = acts_cpu[i:i + chunk].to(device, dtype=torch.float32)
            if kind == "topk":
                recon, acts = sae(x)
            else:
                pp = pos_cpu[i:i + chunk].to(device).long()
                recon, acts = sae(x, pp)
            sq_err_sum += (recon - x).pow(2).sum().item()
            count += x.numel()
            sq_diff_x += (x - x_mean).pow(2).sum().item()
            sq_diff_resid += (x - recon).pow(2).sum().item()
            l0_sum += (acts > 0).float().sum().item()
    mse = sq_err_sum / count
    ev = 1.0 - sq_diff_resid / sq_diff_x if sq_diff_x > 0 else float("nan")
    l0 = l0_sum / n
    return {"mse": mse, "ev": ev, "l0": l0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk-dir", type=Path, default=Path("results/l9/topk"))
    ap.add_argument("--position-aware-dir", type=Path, default=Path("results/l9/position_aware"))
    ap.add_argument("--register-subtracted-dir", type=Path, default=Path("results/register_subtracted-l9"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--n-tokens", type=int, default=500_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--dataset-offset", type=int, default=100_000)
    ap.add_argument("--ce-n-batches", type=int, default=8)
    ap.add_argument("--ce-batch-size", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    topk = load_topk(args.topk_dir).to(device).eval()
    pa = load_position_aware(args.position_aware_dir).to(device).eval()
    rs = load_register_subtracted(args.register_subtracted_dir).to(device).eval()
    print(f"[load] all 3 SAEs loaded")
    print(f"[load] register_subtracted register norm: {rs.register_vector.norm().item():.2f}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
    ).to(device).eval()

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                       split="train", streaming=True)
    ds = ds.skip(args.dataset_offset)
    acts, pos = collect_activations(
        model=model, tokenizer=tok,
        texts=(row["text"] for row in ds),
        layer_idx=args.layer, n_tokens=args.n_tokens, seq_len=args.seq_len,
        batch_size=8, device=device, exclude_first_n=0,
        adapter=QWEN_LLAMA, shuffle=False, return_positions=True,
    )
    print(f"[data] held-out: {tuple(acts.shape)}")

    out = {}
    buckets = {
        "positions_ge_4": pos >= 4,
        "positions_0_to_3": pos < 4,
        "all_positions": torch.ones_like(pos, dtype=torch.bool),
    }
    for bucket_name, mask in buckets.items():
        a_cpu = acts[mask]
        p_cpu = pos[mask]
        out[bucket_name] = {
            "n_tokens": int(mask.sum().item()),
            "topk": chunked_eval(topk, a_cpu, p_cpu, "topk", device),
            "position_aware": chunked_eval(pa, a_cpu, p_cpu, "pa", device),
            "register_subtracted": chunked_eval(rs, a_cpu, p_cpu, "rs", device),
        }
        print(f"\n[{bucket_name}] n={int(mask.sum().item())}")
        for k in ["topk", "position_aware", "register_subtracted"]:
            r = out[bucket_name][k]
            print(f"  {k:>20s}: mse={r['mse']:.4f}  ev={r['ev']:.4f}  L0={r['l0']:.1f}")

    # CE recovery: TopK with exclude_first_n=4, PA with full sequence, RS with full sequence
    print("\n[ce] computing CE recovery for all 3")
    out["ce_recovery"] = {}
    for name, sae, exclude_n, position_aware_flag in [
        ("topk", topk, 4, False),
        ("position_aware", pa, 0, True),
        ("register_subtracted", rs, 0, True),
    ]:
        ds_ce = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                              split="train", streaming=True)
        ds_ce = ds_ce.skip(args.dataset_offset + 20_000)
        ce_texts = (row["text"] for row in ds_ce)
        rec = ce_recovery(
            model=model, tokenizer=tok, sae=sae, texts=ce_texts,
            layer_idx=args.layer, n_batches=args.ce_n_batches,
            batch_size=args.ce_batch_size, seq_len=args.seq_len,
            device=device, exclude_first_n=exclude_n,
            position_aware=position_aware_flag,
        )
        out["ce_recovery"][name] = {
            "ce_clean": rec.ce_clean, "ce_recon": rec.ce_recon,
            "ce_zero": rec.ce_zero, "recovered": rec.recovered,
        }
        print(f"  {name:>20s}: clean={rec.ce_clean:.3f} recon={rec.ce_recon:.3f} "
              f"recovered={rec.recovered:.4f}")

    (args.out / "three_way_eval.json").write_text(json.dumps(out, indent=2))
    print(f"\n[save] wrote {args.out}/three_way_eval.json")


if __name__ == "__main__":
    main()
