"""Lightweight checks of the actual loaded execution configuration.

No diagnostic model forward is inserted into training startup. Packed attention
boundaries were independently verified against masked equal-shaped CP5 forwards.
BF16 forwards with different GEMM shapes are not a training acceptance gate.
"""
import json,os,time
from pathlib import Path
import torch

def verify_execution_configuration(module,rank):
    from search_g1.full_state_restore import publish
    assert module.config._attn_implementation == 'flash_attention_2'
    layers=[m for m in module.modules() if m.__class__.__name__=='Qwen2FlashAttention2']
    configured_layers=module.config.num_hidden_layers
    assert type(configured_layers) is int and configured_layers>0
    assert len(layers)==configured_layers
    assert {p.dtype for p in module.parameters()} == {torch.float32}
    assert module.mixed_precision.param_dtype == torch.bfloat16
    proof=dict(rank=rank,actual_loaded_model_forward=False,
        attention_backend=module.config._attn_implementation,
        actual_qwen_flash_attention_layers=len(layers),
        configured_transformer_layers=configured_layers,
        compute_parameter_dtype='torch.bfloat16',master_parameter_dtype='torch.float32',
        checked_unix=time.time(),
        scope='Actual loaded configuration only; no startup numeric or forward gate; original saved RNG restored next')
    publish(Path(os.environ['SEARCH_G1_BASE'])/'logs'/f'actual-varlen-execution-rank-{rank}.json',proof)
