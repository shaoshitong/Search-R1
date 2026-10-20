"""Fresh official7B block21/24/27 + equally seeded Q273/Prefix20 flatten head.

No E1 model/Adam inheritance. Captured HF replay features are frozen; all
variants execute the full causal question+prefix block before the readout.
"""
import torch
from torch import nn
from .probe_ablation_model import NativeBlockProbe,ProbeAblationSpec,CAPTURE_PAIRS


class FlattenReadout(nn.Module):
    def __init__(self,hidden_size=3584,question_max_tokens=273,prefix_tokens=20,seed=1729):
        super().__init__();self.width=question_max_tokens+prefix_tokens;self.hidden_size=hidden_size
        if type(seed) is not int or min(hidden_size,question_max_tokens,prefix_tokens)<1:
            raise ValueError('Positive explicit flatten geometry and integer seed required')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.output=nn.Linear(self.width*hidden_size,1,dtype=torch.float32)

    def forward(self,hidden,attention_mask,return_weights=False):
        if (hidden.ndim!=3 or hidden.shape[1:]!=(self.width,self.hidden_size) or
                attention_mask.shape!=hidden.shape[:2] or attention_mask.dtype!=torch.bool):
            raise ValueError('Full QMAX+Prefix flatten geometry/mask required')
        value=hidden.masked_fill(~attention_mask.unsqueeze(-1),0)
        if not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite unmasked flatten hidden')
        logits=self.output(value.flatten(1)).squeeze(-1).float()
        return dict(logits=logits) if return_weights else logits


class ReplayLayerProbe(NativeBlockProbe):
    def __init__(self,model_dir,block_layer,head_seed=1729):
        if type(block_layer) is not int or block_layer not in CAPTURE_PAIRS:
            raise ValueError('Only explicit zero-based official blocks21/24/27')
        spec=ProbeAblationSpec(block_layer,CAPTURE_PAIRS[block_layer],readout='mean',scope='question_prefix')
        super().__init__(model_dir,spec)
        # Replace the temporary small readout from the shared official block
        # loader, never its block. The final identical-shape fresh heads use
        # their own common RNG seed and do not consume the training RNG clock.
        self.readout=FlattenReadout(seed=head_seed)
        self.initialization_proof.update(readout='flatten',scope='question_prefix',
            classifier_shape=list(self.readout.output.weight.shape),head_seed=head_seed,
            block_source='official_initial_Qwen2.5-7B-Instruct',E1_inherited=False,
            optimizer_initial_step=0,EMA_initialization='exact_copy_of_fresh_raw',
            feature_backend='huggingface_frozen_causal_prefill',native_bitexact_claim=False)
