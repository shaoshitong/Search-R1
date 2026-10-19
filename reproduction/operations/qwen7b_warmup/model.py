"""Official 7B block27 on frozen native layer26 question+first5 features.

The sampler retains20 policy features, but the warmup head sees only the first5.
Raw FP32 parameters and native RoPE are preserved under BF16 compute autocast.
"""
import inspect
import json
from pathlib import Path
import torch
from torch import nn


class Probe(nn.Module):
    def __init__(self, model_dir, question_max_tokens, prefix_tokens=5):
        super().__init__()
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding
        from safetensors import safe_open
        root = Path(model_dir)
        config = Qwen2Config.from_pretrained(str(root),local_files_only=True)
        assert (config.num_hidden_layers,config.hidden_size,config.num_attention_heads,
            config.num_key_value_heads,config.intermediate_size)==(28,3584,28,4,18944)
        assert prefix_tokens == 5
        config._attn_implementation = 'flash_attention_2'
        self.qmax,self.prefix = question_max_tokens,prefix_tokens
        self.width = question_max_tokens+prefix_tokens
        self.block = Qwen2DecoderLayer(config,layer_idx=27).float()
        self.rotary = Qwen2RotaryEmbedding(config=config).float()
        self.output = nn.Linear(self.width*3584,1,dtype=torch.float32)
        mapping = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
        keys = {k:v for k,v in mapping.items() if k.startswith('model.layers.27.')}
        state={}
        for filename in sorted(set(keys.values())):
            with safe_open(str(root/filename),framework='pt',device='cpu') as handle:
                for key,owner in keys.items():
                    if owner==filename:state[key[len('model.layers.27.'):]]=handle.get_tensor(key).float()
        self.block.load_state_dict(state,strict=True)
        assert all(torch.equal(v,state[k]) for k,v in self.block.state_dict().items())
        assert 'FlashAttention2' in type(self.block.self_attn).__name__
        self.cache_position = 'cache_position' in inspect.signature(self.block.forward).parameters

    def forward(self, batch):
        # A prefix20 capture is retained remotely; features after prefix5 never
        # enter this model or its loss graph.
        features=batch['features'][:,:self.width].detach().contiguous()
        mask=batch['attention_mask'][:,:self.width].contiguous()
        positions=batch['position_ids'][:,:self.width].contiguous()
        assert features.shape[1:]==(self.width,3584) and mask.dtype==torch.bool
        assert bool(mask[:,0].all()) and not features.requires_grad
        kwargs=dict(attention_mask=mask,position_ids=positions,
            position_embeddings=self.rotary(features,positions),use_cache=False,output_attentions=False)
        if self.cache_position:kwargs['cache_position']=torch.arange(self.width,device=features.device)
        hidden=self.block(features,**kwargs)
        if isinstance(hidden,tuple):hidden=hidden[0]
        return self.output(hidden.masked_fill(~mask.unsqueeze(-1),0).flatten(1)).squeeze(-1).float()


def native_buffers(model):
    persistent=model.state_dict()
    return {name:value.detach().cpu().clone() for name,value in model.named_buffers() if name not in persistent}


def restore_buffers(model, buffers):
    targets={name:value for name,value in model.named_buffers() if name not in model.state_dict()}
    assert set(targets)==set(buffers)
    with torch.no_grad():
        for name,value in targets.items():
            assert value.shape==buffers[name].shape
            value.copy_(buffers[name].to(value.device))
