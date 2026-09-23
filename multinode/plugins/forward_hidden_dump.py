"""Megatron side: dump per-layer hidden states (hyper-connection stream after each decoder layer) for ONE microbatch.

Companion of plugins/forward_logits_dump.py (same hook point: first train_step -> forward_only -> exit). Registers
forward hooks on language_model.embedding, decoder.layers[i], decoder.hyper_connection_mixer and decoder.final_layernorm.
Under sequence parallelism every layer output is sequence-sharded across TP, so each captured tensor is all-gathered
along the sequence dim (every TP rank participates) and TP rank 0 saves FWD_HIDDEN_DIR/layers.pt with keys
"embed", "layer_{i}" (0-based), "mixer", "final_norm", "logits_argmax" as fp32 [T, dim] CPU tensors.
Only the first microbatch is captured; global_batch_size must be 1 for this config.
"""

from __future__ import annotations

import os
import sys
import time
from functools import partial

import torch
from megatron.core import mpu
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region, gather_from_tensor_model_parallel_region
from swift.megatron.trainers.base import BaseMegatronTrainer

OUT = os.environ.get("FWD_HIDDEN_DIR")
VOCAB = int(os.environ.get("FWD_VOCAB_SIZE", "248320"))
_PATCH_ATTR = "_forward_hidden_dump"
_store: dict[str, torch.Tensor] = {}
_hooks = []


def _rank() -> str:
    return os.environ.get("RANK", "?")


def _log(msg: str) -> None:
    print(f"[mghid] rank={_rank()} {msg}", flush=True)


def _gather_seq(t: torch.Tensor) -> torch.Tensor:
    """[s_local, b, d] (SP-sharded) -> [T, d] fp32 on CPU, full sequence."""
    if t.dim() == 3:
        full = gather_from_sequence_parallel_region(t.contiguous(), tensor_parallel_output_grad=False)
        return full[:, 0, :].detach().float().cpu()
    full = gather_from_sequence_parallel_region(t.contiguous(), tensor_parallel_output_grad=False)
    return full.detach().float().cpu()


def _keep(key):
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if key in _store:  # only the first microbatch
            return
        _store[key] = _gather_seq(t)
    return hook


def _install_hooks(models) -> dict:
    picked = {}
    for m in models:
        inner = m
        while hasattr(inner, "module"):
            inner = inner.module
        lm = getattr(inner, "language_model", inner)
        emb = getattr(lm, "embedding", None)
        if emb is not None:
            picked["embed"] = "language_model.embedding"; _hooks.append(emb.register_forward_hook(_keep("embed")))
        dec = lm.decoder
        for i, layer in enumerate(dec.layers):
            key = f"layer_{layer.layer_number - 1}"  # mcore layer_number is 1-based -> 0-based like HF
            picked[key] = f"decoder.layers[{i}]"
            _hooks.append(layer.register_forward_hook(_keep(key)))
        mixer = getattr(dec, "hyper_connection_mixer", None)
        if mixer is not None:
            picked["mixer"] = "decoder.hyper_connection_mixer"; _hooks.append(mixer.register_forward_hook(_keep("mixer")))
        fn = getattr(dec, "final_layernorm", None)
        if fn is not None:
            picked["final_norm"] = "decoder.final_layernorm"; _hooks.append(fn.register_forward_hook(_keep("final_norm")))
    return picked


def _loss_func(output_tensor, non_loss_data: bool = False, *, data):
    if non_loss_data:
        logits = output_tensor[0]  # [T, V_local]
        full = gather_from_tensor_model_parallel_region(logits.contiguous())[:, :VOCAB]
        _store["logits_argmax"] = full.float().argmax(-1).cpu()
        _store["logits_row0_16"] = full[:16].float().cpu()
        return {}
    return torch.zeros(1, device=output_tensor.device), 1, {}


def _forward_step(data_iterator, model, *, trainer):
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    data = trainer.get_batch(data_iterator, getattr(inner, "vp_stage", None))
    data.pop("loss_scale", None); data.pop("channel", None); data.pop("labels", None)
    _store["input_ids"] = data["input_ids"].reshape(-1).cpu()
    return model(**data), partial(_loss_func, data=data)


def _dump_then_exit(self, train_data_iterator):
    if OUT is None:
        raise RuntimeError("FWD_HIDDEN_DIR is not set")
    args = self.args
    picked = _install_hooks(self.wrapped_models)
    for m in self.wrapped_models:
        m.eval()
    _log(f"hidden dump: layers hooked={sum(k.startswith('layer_') for k in picked)} others={[k for k in picked if not k.startswith('layer_')]}")
    started = time.perf_counter()
    with torch.no_grad():
        get_forward_backward_func()(
            forward_step_func=partial(_forward_step, trainer=self), data_iterator=self._replace_data_iterator(train_data_iterator),
            model=self.wrapped_models, num_microbatches=args.num_microbatches, seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size, forward_only=True, collect_non_loss_data=True)
    torch.cuda.synchronize()
    for h in _hooks:
        h.remove()
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_data_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        os.makedirs(OUT, exist_ok=True)
        torch.save({"modules": picked, "tp": mpu.get_tensor_model_parallel_world_size(), **_store}, os.path.join(OUT, "layers.pt"))
        _log(f"saved {OUT}/layers.pt keys={len(_store)} in {time.perf_counter() - started:.1f}s")
    torch.distributed.barrier()
    if _rank() == "0":
        open(os.path.join(OUT, "DONE"), "w").write("ok\n")
    sys.stdout.flush()
    sys.exit(0)


if not getattr(BaseMegatronTrainer.train_step, _PATCH_ATTR, False):
    setattr(_dump_then_exit, _PATCH_ATTR, True)
    BaseMegatronTrainer.train_step = _dump_then_exit
