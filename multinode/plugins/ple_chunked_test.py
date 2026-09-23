"""Exactness test: chunked vs original PLE compute (forward + grads). Single GPU."""
import os, sys, types, math, torch
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29777")
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1")
torch.distributed.init_process_group("gloo", rank=0, world_size=1)
from megatron.core import parallel_state
parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed; model_parallel_cuda_manual_seed(0)
from megatron.core.model_parallel_config import ModelParallelConfig
from mcore_bridge.model.modules.ple import Qwen4ExpTextPLELayer
import ple_chunked  # applies the patch
orig = ple_chunked._original_compute
dev = "cuda"
cfg = types.SimpleNamespace(
    hidden_size=256, hc_count=4, ple_embed_dim=128, ple_conv_kernel_size=4, ngram_size=3,
    layernorm_epsilon=1e-6, params_dtype=torch.bfloat16, sequence_parallel=False,
    eos_token_id=7, ple_seed=1234, split_ngram_parts=4, make_ngram_vocab_size_divisible_by=128,
    heads_per_ngram=8, ngram_vocab_size_base=20000, vocab_size=5000, padded_vocab_size=5120, ple_layer_ids=[2], hc_lowrank=32, rms_norm_eps=1e-6,
    tensor_model_parallel_size=1, context_parallel_size=1, perform_initialization=True,
    use_cpu_initialization=False, init_method=torch.nn.init.normal_, hidden_dropout=0.0,
)
# fill any missing ModelParallelConfig fields with defaults
mpc = ModelParallelConfig()
for k, v in vars(mpc).items():
    if not hasattr(cfg, k): setattr(cfg, k, v)
cfg.params_dtype = torch.bfloat16
torch.manual_seed(0)
layer = Qwen4ExpTextPLELayer(cfg, 2).to(dev)
for p in layer.parameters():
    if p.dim() >= 1: torch.nn.init.normal_(p, std=0.05)
L = 5003
ids = torch.randint(0, 5000, (1, L), device=dev); ids[0, 777] = 7; ids[0, 2500] = 7; ids[0, 2501] = 7
h0 = torch.randn(1, L, cfg.hc_count * cfg.hidden_size, device=dev, dtype=torch.bfloat16)
def run(fn, chunk):
    ple_chunked.CHUNK = chunk
    h = h0.clone().requires_grad_(True)
    layer.zero_grad(set_to_none=True)
    out = fn(layer, h, ids)
    out.float().pow(2).mean().backward()
    g = {n: p.grad.detach().clone() for n, p in layer.named_parameters() if p.grad is not None}
    return out.detach().clone(), h.grad.detach().clone(), g
o_ref, hg_ref, g_ref = run(orig, 0)
for chunk in (512, 1000, 4096):
    o, hg, g = run(Qwen4ExpTextPLELayer.compute, chunk)
    d_out = (o.float() - o_ref.float()).abs().max().item()
    d_hg = (hg.float() - hg_ref.float()).abs().max().item()
    d_g = max((g[k].float() - g_ref[k].float()).abs().max().item() for k in g_ref)
    print(f"chunk={chunk}: max|dout|={d_out:.3e} max|dgrad_in|={d_hg:.3e} max|dgrad_param|={d_g:.3e} "
          f"bitwise_out={torch.equal(o, o_ref)}")
print("ok")
