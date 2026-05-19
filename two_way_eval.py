"""Two-way head-to-head: TopK vs Register-Subtracted.

For cross-model verification on GPT-2 and Pythia where we don't have a
matched Position-Aware checkpoint. Same three position buckets + CE
recovery as three_way_eval.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from legible_weights.data.activations import collect_activations
from legible_weights.data.adapters import GPT2, PYTHIA, QWEN_LLAMA
from legible_weights.eval.recovery import ce_recovery
from legible_weights.sae.model import SAEConfig, TopKSAE
from legible_weights.sae.register_subtracted import (
    RegisterSubtractedSAEConfig, RegisterSubtractedTopKSAE,
)


BASE_SPECS = {
    "qwen2.5-0.5b": {"hf": "Qwen/Qwen2.5-0.5B", "adapter": QWEN_LLAMA},
    "gpt2-small":   {"hf": "openai-community/gpt2", "adapter": GPT2},
    "pythia-1.4b":  {"hf": "EleutherAI/pythia-1.4b", "adapter": PYTHIA},
}


def load_topk(ckpt_dir):
    # Architecture metadata in parent's report.json
    report = json.loads((ckpt_dir.parent / "report.json").read_text())
    sae = TopKSAE(SAEConfig(d_in=report["d_in"], d_hidden=report["d_hidden"], k=report["k"]))
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
    n = acts_cpu.shape[0]
    sum_x = torch.zeros(acts_cpu.shape[1], dtype=torch.float64)
    n_seen = 0
    with torch.no_grad():
        for i in range(0, n, chunk):
            x = acts_cpu[i:i + chunk].to(device, dtype=torch.float32)
            sum_x += x.sum(dim=0).double().cpu()
            n_seen += x.shape[0]
        x_mean = (sum_x / n_seen).to(device, dtype=torch.float32)
        sq_err_sum = 0.0
        count = 0
        l0_sum = 0.0
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
    return {
        "mse": sq_err_sum / count,
        "ev": 1.0 - sq_diff_resid / sq_diff_x if sq_diff_x > 0 else float("nan"),
        "l0": l0_sum / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=list(BASE_SPECS), required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--topk-dir", type=Path, required=True)
    ap.add_argument("--register-subtracted-dir", type=Path, required=True)
    ap.add_argument("--n-tokens", type=int, default=500_000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--dataset-offset", type=int, default=100_000)
    ap.add_argument("--ce-n-batches", type=int, default=8)
    ap.add_argument("--ce-batch-size", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    spec = BASE_SPECS[args.base]
    model_name = spec["hf"]
    adapter = spec["adapter"]

    topk = load_topk(args.topk_dir).to(device).eval()
    rs = load_register_subtracted(args.register_subtracted_dir).to(device).eval()
    print(f"[load] TopK + RS loaded; register norm: {rs.register_vector.norm().item():.2f}")

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
    ).to(device).eval()

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                       split="train", streaming=True).skip(args.dataset_offset)
    acts, pos = collect_activations(
        model=model, tokenizer=tok,
        texts=(row["text"] for row in ds),
        layer_idx=args.layer, n_tokens=args.n_tokens, seq_len=args.seq_len,
        batch_size=8, device=device, exclude_first_n=0,
        adapter=adapter, shuffle=False, return_positions=True,
    )
    print(f"[data] held-out: {tuple(acts.shape)}")

    out = {"base": args.base, "layer": args.layer}
    buckets = {
        "positions_ge_4": pos >= 4,
        "positions_0_to_3": pos < 4,
        "all_positions": torch.ones_like(pos, dtype=torch.bool),
    }
    for name, mask in buckets.items():
        a_cpu = acts[mask]
        p_cpu = pos[mask]
        out[name] = {
            "n_tokens": int(mask.sum().item()),
            "topk": chunked_eval(topk, a_cpu, p_cpu, "topk", device),
            "register_subtracted": chunked_eval(rs, a_cpu, p_cpu, "rs", device),
        }
        print(f"\n[{name}] n={int(mask.sum().item())}")
        for k in ["topk", "register_subtracted"]:
            r = out[name][k]
            print(f"  {k:>22s}: mse={r['mse']:.4f}  ev={r['ev']:.4f}  L0={r['l0']:.1f}")

    print("\n[ce] CE recovery (splice intervention)")
    out["ce_recovery"] = {}
    for arch_name, sae, ex_n, pa_flag in [
        ("topk", topk, 4, False),
        ("register_subtracted", rs, 0, True),
    ]:
        ds_ce = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                              split="train", streaming=True).skip(args.dataset_offset + 20_000)
        ce_texts = (row["text"] for row in ds_ce)
        rec = ce_recovery(
            model=model, tokenizer=tok, sae=sae, texts=ce_texts,
            layer_idx=args.layer, n_batches=args.ce_n_batches,
            batch_size=args.ce_batch_size, seq_len=args.seq_len,
            device=device, exclude_first_n=ex_n, position_aware=pa_flag,
        )
        out["ce_recovery"][arch_name] = {
            "ce_clean": rec.ce_clean, "ce_recon": rec.ce_recon,
            "ce_zero": rec.ce_zero, "recovered": rec.recovered,
        }
        print(f"  {arch_name:>22s}: clean={rec.ce_clean:.3f}  recon={rec.ce_recon:.3f}  "
              f"recovered={rec.recovered:.4f}")

    (args.out / "two_way_eval.json").write_text(json.dumps(out, indent=2))
    print(f"\n[save] wrote {args.out}/two_way_eval.json")


if __name__ == "__main__":
    main()
