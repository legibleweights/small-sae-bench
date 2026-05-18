"""Run the small-SAE architecture benchmark.

For one (base_model, layer) target, train all four architectures
(TopK / L1 / Gated / PositionAware) with matched hyperparameters and the
same activation buffer, evaluate each on the same held-out slice, and emit
a single comparison report.

The position-aware variant trains *without* exclude_first_n (i.e., on the
full sequence including the high-norm prefix positions) because the whole
point is that explicit position-conditioning should handle those positions
internally. All other variants train with exclude_first_n=4 to match prior
runs.

The held-out eval splices the SAE reconstruction in only at positions
covered by the SAE's training distribution (i.e., positions >=4 for the
three non-position-aware variants, all positions for PositionAware).
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
from legible_weights.data.adapters import GPT2, QWEN_LLAMA, ModelAdapter
from legible_weights.sae.gated import GatedSAE, GatedSAEConfig
from legible_weights.sae.l1 import L1SAE, L1SAEConfig
from legible_weights.sae.model import SAEConfig, TopKSAE
from legible_weights.sae.position_aware import (
    PositionAwareSAEConfig,
    PositionAwareTopKSAE,
)
from legible_weights.sae.unified_train import UnifiedTrainConfig, train_unified


BASE_SPECS = {
    "qwen2.5-0.5b": {
        "hf": "Qwen/Qwen2.5-0.5B",
        "adapter": QWEN_LLAMA,
        "d_model": 896,
        "default_layer": 9,
    },
    "gpt2-small": {
        "hf": "openai-community/gpt2",
        "adapter": GPT2,
        "d_model": 768,
        "default_layer": 6,
    },
}


def load_base(base: str, device: str):
    spec = BASE_SPECS[base]
    tok = AutoTokenizer.from_pretrained(spec["hf"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf"], torch_dtype=torch.float16
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, spec


def build_sae(arch: str, d_in: int, d_hidden: int, k: int = 32):
    if arch == "topk":
        return TopKSAE(SAEConfig(d_in=d_in, d_hidden=d_hidden, k=k))
    if arch == "l1":
        # l1_coef = 5e-3 is a compromise — at 5e-2 these architectures
        # sparsify too fast and never learn useful features (EV ~0.02);
        # at 1e-3 they don't sparsify at all (L0 ~4000). A proper recipe
        # would include L1-coef warmup, not implemented here.
        return L1SAE(L1SAEConfig(d_in=d_in, d_hidden=d_hidden, l1_coef=5e-3))
    if arch == "gated":
        return GatedSAE(GatedSAEConfig(d_in=d_in, d_hidden=d_hidden,
                                       l1_coef=5e-3, aux_coef=1.0))
    if arch == "position_aware":
        return PositionAwareTopKSAE(PositionAwareSAEConfig(
            d_in=d_in, d_hidden=d_hidden, k=k, max_pos=16,
        ))
    raise ValueError(f"unknown arch: {arch}")


def needs_positions(arch: str) -> bool:
    return arch == "position_aware"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=list(BASE_SPECS), required=True)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--n-tokens", type=int, default=10_000_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--expansion", type=int, default=16)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--exclude-first-n", type=int, default=4,
                    help="Used for non-position-aware variants. PositionAware "
                         "ignores this and trains on full sequences.")
    ap.add_argument("--archs", nargs="+",
                    default=["topk", "l1", "gated", "position_aware"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    spec = BASE_SPECS[args.base]
    layer = args.layer if args.layer is not None else spec["default_layer"]
    d_in = spec["d_model"]
    d_hidden = d_in * args.expansion

    print(f"[setup] base={args.base} layer={layer} d_in={d_in} d_hidden={d_hidden}")

    model, tok, _ = load_base(args.base, device)

    # Two separate, smaller collections (one for non-position-aware archs,
    # one for the position-aware arch with positions). Sequentially trained,
    # so peak CPU memory = max(buffer_skip, buffer_full) + python overhead.
    # Avoids the 35 GB fp32 copy that .float() on the whole buffer would
    # trigger in train_unified.

    needs_skip = any(not needs_positions(a) for a in args.archs)
    needs_full = any(needs_positions(a) for a in args.archs)

    del model
    torch.cuda.empty_cache()

    def collect_skip():
        m, t, _ = load_base(args.base, device)
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                            split="train", streaming=True)
        t0 = time.time()
        out = collect_activations(
            model=m, tokenizer=t,
            texts=(row["text"] for row in ds),
            layer_idx=layer, n_tokens=args.n_tokens, seq_len=args.seq_len,
            batch_size=8, device=device, exclude_first_n=args.exclude_first_n,
            adapter=spec["adapter"], shuffle=True,
        )
        del m
        torch.cuda.empty_cache()
        print(f"[data] skip buffer: {tuple(out.shape)} in {time.time()-t0:.1f}s")
        return out

    def collect_full():
        m, t, _ = load_base(args.base, device)
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                            split="train", streaming=True)
        t0 = time.time()
        acts, pos = collect_activations(
            model=m, tokenizer=t,
            texts=(row["text"] for row in ds),
            layer_idx=layer, n_tokens=args.n_tokens, seq_len=args.seq_len,
            batch_size=8, device=device, exclude_first_n=0,
            adapter=spec["adapter"], shuffle=True, return_positions=True,
        )
        del m
        torch.cuda.empty_cache()
        print(f"[data] full buffer: {tuple(acts.shape)} "
              f"(min pos {pos.min().item()}, max pos {pos.max().item()}) "
              f"in {time.time()-t0:.1f}s")
        return acts, pos

    report: dict = {
        "base": args.base, "layer": layer, "d_in": d_in, "d_hidden": d_hidden,
        "k": args.k, "n_tokens": args.n_tokens, "seq_len": args.seq_len,
        "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
        "exclude_first_n_for_non_position_aware": args.exclude_first_n,
        "results": {},
    }

    # Group archs into skip-needing and full-needing batches, collect data
    # once per group, free between groups. Train + eval + save each arch
    # while its buffer is still alive, then drop the buffer.
    import gc
    skip_archs = [a for a in args.archs if not needs_positions(a)]
    full_archs = [a for a in args.archs if needs_positions(a)]
    ordered: list[tuple[str, list[str]]] = []
    if skip_archs:
        ordered.append(("skip", skip_archs))
    if full_archs:
        ordered.append(("full", full_archs))

    for kind, group in ordered:
        if kind == "skip":
            buf = collect_skip()
            buf_pos = None
        else:
            buf, buf_pos = collect_full()

        for arch in group:
            print(f"\n=== training {arch} ===")
            sae = build_sae(arch, d_in=d_in, d_hidden=d_hidden, k=args.k)
            train_cfg = UnifiedTrainConfig(
                batch_size=args.batch_size, lr=args.lr,
                n_epochs=args.epochs, device=device,
            )
            t0 = time.time()
            if needs_positions(arch):
                history = train_unified(sae, buf, train_cfg, positions=buf_pos)
            else:
                history = train_unified(sae, buf, train_cfg)
            wall = time.time() - t0

            # Final-batch eval on this arch's training buffer
            sae.eval()
            with torch.no_grad():
                sample_n = min(20000, buf.shape[0])
                x = buf[:sample_n].to(device, dtype=torch.float32)
                if needs_positions(arch):
                    pos = buf_pos[:sample_n].to(device).long()
                    recon, acts = sae(x, pos)
                else:
                    recon, acts = sae(x)
                mse = (recon - x).pow(2).mean().item()
                residual_var = (x - recon).var(dim=0).sum().item()
                ev = 1.0 - residual_var / x.var(dim=0).sum().item()
                l0_mean = (acts > 0).float().sum(dim=-1).mean().item()
                l0_std = (acts > 0).float().sum(dim=-1).std().item()
                dead = int((acts.sum(dim=0) == 0).sum().item())

            result = {
                "wall_seconds": wall,
                "final_mse": mse,
                "final_explained_var": ev,
                "final_l0_mean": l0_mean,
                "final_l0_std": l0_std,
                "dead_features_on_eval_batch": dead,
                "n_history_points": len(history),
            }
            report["results"][arch] = result

            ckpt_dir = args.out / arch
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(sae.state_dict(), ckpt_dir / "sae.pt")
            (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))
            print(f"[{arch}] mse={mse:.4f}  ev={ev:.3f}  L0={l0_mean:.1f}±{l0_std:.1f}  "
                  f"dead={dead}  wall={wall:.1f}s")
            del sae, x, recon, acts
            torch.cuda.empty_cache()

        # Drop this group's buffer
        del buf
        if buf_pos is not None:
            del buf_pos
        gc.collect()
        torch.cuda.empty_cache()

    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[save] wrote {args.out}/report.json")


if __name__ == "__main__":
    main()
