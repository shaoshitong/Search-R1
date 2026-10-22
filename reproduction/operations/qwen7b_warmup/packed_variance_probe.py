"""One-question, variable-own-prefix packed variance regression, no BCE."""
import hashlib,json,random
import torch
from torch import nn


def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def view(group,*,seed,epoch,fixed_m=None):
    # M, identities and trajectory order are drawn before any EM is accessed.
    rng=random.Random(int(digest(['packed-population-variance',seed,epoch,group['qid']]),16))
    m=rng.randint(3,10)if fixed_m is None else fixed_m
    indices=rng.sample(range(len(group['trajectory_ids'])),m)
    tids=[group['trajectory_ids'][i]for i in indices]
    em=[group['labels'][i]for i in indices];mean=sum(em)/m;variance=mean*(1-mean)
    return dict(qid=group['qid'],trajectory_ids=tids,labels=em,M=m,target_variance=variance,target4variance=4*variance,true_mixed=0<sum(em)<m)


def plan(groups,epoch,seed=2026102227):
    rows=[view(groups[q],seed=seed,epoch=epoch)for q in sorted(groups)]
    random.Random(int(digest(['packed-question-order',seed,epoch]),16)).shuffle(rows)
    assert len(rows)==len({v['qid']for v in rows})==len(groups)
    return rows


def pack_group(v,records,separator,pad_token=151643,prefix=20):
    own=[records[t]for t in v['trajectory_ids']];prompt=own[0]['prompt']
    assert prompt and all(r['prompt']==prompt for r in own)
    tokens=list(prompt);valid=[True]*len(prompt);types=[0]*len(prompt)
    for r in own:
        policy=r['policy20'];assert 0<len(policy)<=prefix
        tokens+=separator+policy+[pad_token]*(prefix-len(policy))
        valid +=[True]*len(separator)+[True]*len(policy)+[False]*(prefix-len(policy))
        types +=[2]*len(separator)+[1]*prefix
    # Strict EM and targets do not enter tokens, masks or positions.
    pos=[];counter=0
    for active in valid:pos.append(counter if active else 0);counter+=int(active)
    return dict(input_ids=tokens,attention_mask=valid,position_ids=pos,token_types=types)


class PackedVarianceHead(nn.Module):
    def __init__(self,hidden=3584,dimension=256,max_positions=1024):
        super().__init__();self.projection=nn.Linear(hidden,dimension);self.types=nn.Embedding(3,dimension);self.positions=nn.Embedding(max_positions,dimension)
        self.cls=nn.Parameter(torch.empty(1,1,dimension));nn.init.normal_(self.cls,std=.02)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(dimension,8,4*dimension,dropout=0.,activation='gelu',batch_first=True,norm_first=True)for _ in range(2)])
        self.norm=nn.LayerNorm(dimension);self.output=nn.Linear(dimension,1)
    def forward(self,hidden,mask,positions,types):
        x=self.projection(hidden.masked_fill(~mask[...,None],0))+self.types(types)+self.positions(positions)
        x=torch.cat((self.cls.expand(len(x),-1,-1),x),1);valid=torch.cat((torch.ones((len(mask),1),dtype=torch.bool,device=mask.device),mask),1)
        for layer in self.layers:x=layer(x,src_key_padding_mask=~valid)
        return self.output(self.norm(x[:,0])).squeeze(-1).float()


class PackedVarianceProbe(nn.Module):
    def __init__(self,block,rotary,cache_position,head):
        super().__init__();self.block=block;self.rotary=rotary;self.cache_position=cache_position;self.group_head=head
    def forward(self,batch):
        hidden=batch['features'].detach();mask=batch['attention_mask'];positions=batch['position_ids']
        kwargs=dict(attention_mask=mask,position_ids=positions,position_embeddings=self.rotary(hidden,positions),use_cache=False,output_attentions=False)
        if self.cache_position:kwargs['cache_position']=torch.arange(hidden.shape[1],device=hidden.device)
        value=self.block(hidden,**kwargs);value=value[0]if isinstance(value,tuple)else value
        return self.group_head(value,mask,positions,batch['token_types'])
