"""Megatron side: TEACHER-FORCED per-layer isolation test (fp32).

Same as plugins/forward_hidden_dump.py, but before every decoder layer i the hyper-connection stream is
overwritten with the transformers fp32 output of layer i-1 (FWD_TEACHER_FILE = hf layers.pt), the mixer input
with HF layer_47 and final_layernorm input with HF mixer. Pad rows (>= FWD_TEACHER_NREAL) keep Megatron's own
values. Every layer therefore sees the identical input as transformers, so the captured output error of layer i
is that layer's OWN numerical error (no cross-layer amplification). Saves FWD_HIDDEN_DIR/layers.pt like the dump.
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
TEACHER = os.environ.get("FWD_TEACHER_FILE")
NREAL = int(os.environ.get("FWD_TEACHER_NREAL", "0"))
VOCAB = int(os.environ.get("FWD_VOCAB_SIZE", "248320"))
_PATCH_ATTR = "_forward_hidden_teacher"
_store: dict[str, torch.Tensor] = {}
_hooks = []
_stats = {"forced": 0}


def _rank() -> str:
    return os.environ.get("RANK", "?")


def _log(msg: str) -> None:
    print(f"[mgteach] rank={_rank()} {msg}", flush=True)


def _gather_seq(t: torch.Tensor) -> torch.Tensor:
    full = gather_from_sequence_parallel_region(t.contiguous(), tensor_parallel_output_grad=False)
    return (full[:, 0, :] if t.dim() == 3 else full).detach().float().cpu()


def _keep(key):
    def hook(mod, inp, out):
        if key in _store:
            return
        _store[key] = _gather_seq(out[0] if isinstance(out, (tuple, list)) else out)
    return hook


def _force(teacher_cpu: torch.Tensor, key: str):
    def pre_hook(mod, args, kwargs):
        if key in _store:  # only first microbatch
            return None
        if args:
            h = args[0]
        else:
            h = kwargs["hidden_states"]
        if h.dim() != 3 or h.shape[1] != 1 or h.shape[2] != teacher_cpu.shape[1]:
            raise RuntimeError(f"{key}: unexpected input shape {tuple(h.shape)} vs teacher {tuple(teacher_cpu.shape)}")
        s_local = h.shape[0]
        g0 = mpu.get_tensor_model_parallel_rank() * s_local
        n = max(0, min(s_local, NREAL - g0))
        new = h.clone()
        if n > 0:
            new[:n, 0, :] = teacher_cpu[g0:g0 + n].to(device=h.device, dtype=h.dtype)
        _stats["forced"] += 1
        if args:
            return (new,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs); kwargs["hidden_states"] = new
        return args, kwargs
    return pre_hook


def _install_hooks(models, teacher: dict) -> dict:
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
            n = layer.layer_number - 1
            key = f"layer_{n}"
            picked[key] = f"decoder.layers[{i}]"
            t = teacher["embed"].float().repeat(1, 4) if n == 0 else teacher[f"layer_{n-1}"].float()
            _hooks.append(layer.register_forward_pre_hook(_force(t, key), with_kwargs=True))
            _hooks.append(layer.register_forward_hook(_keep(key)))
        mixer = getattr(dec, "hyper_connection_mixer", None)
        if mixer is not None:
            picked["mixer"] = "decoder.hyper_connection_mixer"
            _hooks.append(mixer.register_forward_pre_hook(_force(teacher["layer_47"].float(), "mixer"), with_kwargs=True))
            _hooks.append(mixer.register_forward_hook(_keep("mixer")))
        fn = getattr(dec, "final_layernorm", None)
        if fn is not None:
            picked["final_norm"] = "decoder.final_layernorm"
            _hooks.append(fn.register_forward_pre_hook(_force(teacher["mixer"].float(), "final_norm"), with_kwargs=True))
            _hooks.append(fn.register_forward_hook(_keep("final_norm")))
    return picked


def _loss_func(output_tensor, non_loss_data: bool = False, *, data):
    if non_loss_data:
        full = gather_from_tensor_model_parallel_region(output_tensor[0].contiguous())[:, :VOCAB]
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
    if OUT is None or TEACHER is None or NREAL <= 0:
        raise RuntimeError("FWD_HIDDEN_DIR / FWD_TEACHER_FILE / FWD_TEACHER_NREAL must be set")
    args = self.args
    teacher = torch.load(TEACHER, map_location="cpu")
    picked = _install_hooks(self.wrapped_models, teacher)
    for m in self.wrapped_models:
        m.eval()
    _log(f"teacher-forced dump: teacher={TEACHER} nreal={NREAL} layers hooked={sum(k.startswith('layer_') for k in picked)} others={[k for k in picked if not k.startswith('layer_')]}")
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
        torch.save({"modules": picked, "teacher": TEACHER, "nreal": NREAL, "tp": mpu.get_tensor_model_parallel_world_size(), **_store}, os.path.join(OUT, "layers.pt"))
        _log(f"saved {OUT}/layers.pt keys={len(_store)} forced_inputs={_stats['forced']} in {time.perf_counter() - started:.1f}s")
    torch.distributed.barrier()
    if _rank() == "0":
        open(os.path.join(OUT, "DONE"), "w").write("ok\n")
    sys.stdout.flush()
    sys.exit(0)


if not getattr(BaseMegatronTrainer.train_step, _PATCH_ATTR, False):
    setattr(_dump_then_exit, _PATCH_ATTR, True)
    BaseMegatronTrainer.train_step = _dump_then_exit
