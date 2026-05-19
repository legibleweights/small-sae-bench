"""Model adapters — a thin uniform interface over AR and diffusion LMs.

The existing `collect_activations` assumed an autoregressive HF model with the
Qwen/Llama/Gemma layout (`model.model.layers[i]`, decoder block outputs are
tuples, forward takes `**tokenizer_output`). The cross-paradigm comparison
project adds GPT-2 and MDLM, both of which deviate from that layout:

- GPT-2 layers live at `model.transformer.h[i]`.
- MDLM is a diffusion LM. Its blocks live at `model.backbone.blocks[i]` and
  return a tensor directly (not a tuple). Its forward signature is
  `(input_ids, timesteps)` not `(**enc)`.

A `ModelAdapter` encapsulates these differences so the activation-collection
loop is identical across model families.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class ModelAdapter:
    """Uniform interface for activation collection across architectures."""

    name: str
    get_layer: Callable[[Any, int], torch.nn.Module]
    output_to_hidden: Callable[[Any], torch.Tensor]
    forward: Callable[[Any, dict[str, torch.Tensor]], Any]
    # Whether tokenizer output should be passed through model(**enc)
    # vs. forward() handles the unpacking itself.


def _ar_get_qwen_llama_layer(model, idx: int) -> torch.nn.Module:
    return model.model.layers[idx]


def _ar_get_gpt2_layer(model, idx: int) -> torch.nn.Module:
    return model.transformer.h[idx]


def _ar_get_pythia_layer(model, idx: int) -> torch.nn.Module:
    return model.gpt_neox.layers[idx]


def _ar_output_to_hidden(outputs):
    # Most AR decoder blocks return (hidden_states, ...)
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def _ar_forward(model, enc):
    return model(**enc)


def _mdlm_get_block(model, idx: int) -> torch.nn.Module:
    return model.backbone.blocks[idx]


def _mdlm_output_to_hidden(outputs):
    # DDiTBlock.forward returns x directly (not a tuple)
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def _mdlm_forward(model, enc):
    """MDLM expects (input_ids, timesteps). We pass t=0 (clean text).

    The shipped MDLM-OWT checkpoint has time_conditioning=False, so the
    specific value of timesteps does not affect computation — but the
    model still expects the argument.
    """
    input_ids = enc["input_ids"]
    timesteps = torch.zeros(input_ids.shape[0], device=input_ids.device)
    return model(input_ids=input_ids, timesteps=timesteps)


QWEN_LLAMA = ModelAdapter(
    name="qwen_llama",
    get_layer=_ar_get_qwen_llama_layer,
    output_to_hidden=_ar_output_to_hidden,
    forward=_ar_forward,
)

GPT2 = ModelAdapter(
    name="gpt2",
    get_layer=_ar_get_gpt2_layer,
    output_to_hidden=_ar_output_to_hidden,
    forward=_ar_forward,
)

PYTHIA = ModelAdapter(
    name="pythia",
    get_layer=_ar_get_pythia_layer,
    output_to_hidden=_ar_output_to_hidden,
    forward=_ar_forward,
)

MDLM = ModelAdapter(
    name="mdlm",
    get_layer=_mdlm_get_block,
    output_to_hidden=_mdlm_output_to_hidden,
    forward=_mdlm_forward,
)
