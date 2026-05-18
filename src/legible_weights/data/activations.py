"""Collect residual-stream activations from a HuggingFace causal LM.

The model is run in eval mode with no grad; activations are captured by a
forward hook on the target decoder layer and accumulated into a buffer of
shape (n_tokens, d_model). The buffer is shuffled once before being returned
so downstream training sees IID batches.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from legible_weights.data.adapters import QWEN_LLAMA, ModelAdapter


def load_model(
    model_name: str,
    device: str | torch.device,
    dtype: torch.dtype = torch.float16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def _infer_d_model(model: torch.nn.Module) -> int:
    cfg = model.config
    for attr in ("hidden_size", "hidden_dim", "n_embd", "d_model"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise ValueError(f"Could not infer hidden size from config: {cfg}")


@torch.no_grad()
def collect_activations(
    model: torch.nn.Module,
    tokenizer,
    texts: Iterable[str],
    layer_idx: int,
    n_tokens: int,
    seq_len: int = 512,
    batch_size: int = 8,
    device: str | torch.device = "cuda",
    exclude_first_n: int = 0,
    adapter: ModelAdapter = QWEN_LLAMA,
    shuffle: bool = True,
    return_positions: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Stream `texts` through the model, capturing layer_idx residual stream.

    Returns a tensor of shape (n_tokens, d_model), fp16, on CPU. Caller is
    responsible for moving batches to GPU during SAE training.

    If `exclude_first_n > 0`, the first N token positions of each sequence are
    treated as padding (not collected). This is the standard workaround for
    transformer attention-sink / outlier-position effects, where the first few
    positions have anomalously high residual-stream norm.

    `adapter` controls model-family-specific behavior (layer access path,
    output unwrapping, forward signature). Defaults to the Qwen/Llama/Gemma
    layout for backward compatibility with prior experiments.

    If `return_positions=True`, also returns a (n_tokens,) int tensor giving
    each token's original position within its sequence. Required for
    position-aware SAEs that condition on position.
    """
    d_model = _infer_d_model(model)
    buf = torch.empty((n_tokens, d_model), dtype=torch.float16)
    pos_buf = torch.empty((n_tokens,), dtype=torch.int32) if return_positions else None
    filled = 0

    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, outputs):
        hs = adapter.output_to_hidden(outputs)
        captured.append(hs.detach().to(torch.float16).cpu())

    layer = adapter.get_layer(model, layer_idx)
    handle = layer.register_forward_hook(hook)

    try:
        batch: list[str] = []
        pbar = tqdm(total=n_tokens, desc=f"collect L{layer_idx} [{adapter.name}]", unit="tok")
        for text in texts:
            if filled >= n_tokens:
                break
            batch.append(text)
            if len(batch) < batch_size:
                continue

            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=seq_len,
            ).to(device)
            captured.clear()
            adapter.forward(model, dict(enc))
            hs = captured[0]  # (B, L, D)
            mask = enc.attention_mask.cpu().bool().clone()
            if exclude_first_n > 0:
                mask[:, :exclude_first_n] = False
            valid = hs[mask]  # (n_valid_tokens, D)
            take = min(valid.shape[0], n_tokens - filled)
            buf[filled : filled + take] = valid[:take]
            if pos_buf is not None:
                # Per-position index broadcast across batch, then masked the same way
                pos_grid = torch.arange(hs.size(1)).unsqueeze(0).expand(hs.size(0), -1)
                valid_pos = pos_grid[mask]
                pos_buf[filled : filled + take] = valid_pos[:take].to(torch.int32)
            filled += take
            pbar.update(take)
            batch = []
        pbar.close()
    finally:
        handle.remove()

    buf = buf[:filled]
    if pos_buf is not None:
        pos_buf = pos_buf[:filled]
    if shuffle:
        perm = torch.randperm(buf.shape[0])
        buf = buf[perm]
        if pos_buf is not None:
            pos_buf = pos_buf[perm]
    if pos_buf is not None:
        return buf, pos_buf
    return buf
