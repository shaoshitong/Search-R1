"""The same native block35 with Q273+policy20 and zero-added head columns."""
import torch
from torch import nn

from reproduction.prefix_selection200.prefix5_model import OnlineQuestionProbe as Probe5
from reproduction.prefix_selection200.prefix5_model import PARAMETER_SHAPES,validate_model_state as validate5
from reproduction.prefix20_online.model import validate_prefix_positions
from .data import QMAX,PREFIX,WIDTH,DIM


def lift_state(state):
    validate5(state)
    result=dict(state)
    weight=torch.zeros((1,WIDTH*DIM),dtype=torch.float32)
    weight[:,:278*DIM].copy_(state['output.weight'])
    result['output.weight']=weight
    assert torch.equal(result['output.weight'][:,:278*DIM],state['output.weight'])
    assert torch.count_nonzero(result['output.weight'][:,278*DIM:])==0
    return result


def validate_model_state(state):
    assert set(state)==set(PARAMETER_SHAPES) and len(state)==14
    for key,shape in PARAMETER_SHAPES.items():
        expected=(1,WIDTH*DIM) if key=='output.weight' else shape
        value=state[key]
        assert tuple(value.shape)==expected and value.dtype==torch.float32 and value.device.type=='cpu'
        assert torch.isfinite(value).all()


class Probe20(Probe5):
    def __init__(self,model_dir):
        super().__init__(model_dir)
        self.policy_tokens=PREFIX;self.max_tokens=WIDTH
        self.output=nn.Linear(WIDTH*DIM,1,dtype=torch.float32)

    def forward(self,batch):
        features=batch['features'].detach();mask=batch['attention_mask'];positions=batch['position_ids']
        question=batch['question_mask'];policy=batch['policy_mask'];lengths=batch['question_lengths']
        assert features.shape[1:]==(WIDTH,DIM) and features.shape[:2]==mask.shape==positions.shape
        assert mask.dtype==torch.bool and positions.dtype==torch.long
        slots=torch.arange(WIDTH,device=mask.device).unsqueeze(0)
        assert torch.equal(question,slots<lengths[:,None]) and not policy[:,:QMAX].any()
        assert torch.equal(mask,question|policy) and torch.equal(positions[question],slots.expand_as(mask)[question])
        assert torch.equal(policy[:,QMAX:].sum(-1),batch['policy_lengths'])
        for i,bounds in enumerate(batch['policy_segments']):
            if bounds:validate_prefix_positions(policy[i:i+1,QMAX:],positions[i:i+1,QMAX:],[bounds],self.max_positions)
            else:assert not policy[i].any()
        kwargs=dict(attention_mask=mask,position_ids=positions,
                    position_embeddings=self.rotary(features,positions),use_cache=False,output_attentions=False)
        if self._cache_position:kwargs['cache_position']=torch.arange(WIDTH,device=features.device)
        hidden=self.block(features,**kwargs)
        if isinstance(hidden,tuple):hidden=hidden[0]
        return self.output(hidden.masked_fill(~mask.unsqueeze(-1),0).flatten(1)).squeeze(-1).float()
