"""Official native block27 + direct, permutation-invariant real-G5 Mixed head.

Production starts from the official a09a block, with both Adam optimizers fresh.
Supply the existing official Probe loader's module and verified inventory/hash.
No checkpoint or actor tensor files are opened by this adapter. The legacy
loader's unused scalar output is never registered or run.
The new head reads every retained policy token; only shared question *slots*
are averaged across trajectories (equal in the ideal causal computation).
This average also avoids making harmless HF numerical differences depend on
which trajectory happens to be first. It never pools the100 policy tokens.
"""
import copy,hashlib,json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
OFFICIAL_REVISION='a09a35458c702b33eeacc393d103063234e8bc28'


def block_state_hash(block):
    h=hashlib.sha256()
    for name,value in block.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class MixedTokenHead(nn.Module):
    def __init__(self,hidden_size=3584,qmax=273,prefix_tokens=20,dimension=256,layers=2,heads=8):
        super().__init__()
        if prefix_tokens!=20 or dimension%heads or layers<1:raise ValueError('P20 and a valid noncausal attention geometry required')
        self.hidden_size,self.qmax,self.prefix,self.dimension=hidden_size,qmax,prefix_tokens,dimension
        self.projection=nn.Linear(hidden_size,dimension)
        self.types=nn.Embedding(2,dimension)
        self.question_slots=nn.Embedding(qmax,dimension)
        self.policy_slots=nn.Embedding(prefix_tokens,dimension)
        self.cls=nn.Parameter(torch.zeros(1,1,dimension));nn.init.normal_(self.cls,std=.02)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(dimension,heads,4*dimension,dropout=0.,activation='gelu',batch_first=True,norm_first=True)for _ in range(layers)])
        self.norm=nn.LayerNorm(dimension);self.classifier=nn.Linear(dimension,1)

    def forward(self,hidden,mask,positions):
        width=self.qmax+self.prefix
        if hidden.ndim!=4 or hidden.shape[1:]!=(5,width,self.hidden_size)or mask.shape!=hidden.shape[:-1]or mask.dtype!=torch.bool or positions.shape!=mask.shape:
            raise ValueError('Exactly Q complete G5 native Q273+P20 outputs and their masks/positions required')
        qm=mask[:,:,:self.qmax];qp=positions[:,:,:self.qmax]
        if not torch.equal(qm,qm[:,:1].expand_as(qm))or not torch.equal(qp,qp[:,:1].expand_as(qp)):
            raise ValueError('Five trajectories must share question mask and causal positions')
        clean=hidden.masked_fill(~mask[...,None],0)
        if not bool(torch.isfinite(clean).all()):raise ValueError('Nonfinite real native hidden state')
        # Question slots are shared; all5*20 policy slots remain separate.
        question=clean[:,:,:self.qmax].mean(1)
        policy=clean[:,:,self.qmax:].reshape(hidden.shape[0],5*self.prefix,self.hidden_size)
        q=self.projection(question)+self.types.weight[0]+self.question_slots.weight
        p=self.projection(policy)+self.types.weight[1]+self.policy_slots.weight.repeat(5,1)
        x=torch.cat((self.cls.expand(hidden.shape[0],-1,-1),q,p),1)
        valid=torch.cat((torch.ones((hidden.shape[0],1),dtype=torch.bool,device=mask.device),qm[:,0],mask[:,:,self.qmax:].reshape(hidden.shape[0],5*self.prefix)),1)
        for layer in self.layers:x=layer(x,src_key_padding_mask=~valid)
        return self.classifier(self.norm(x[:,0])).squeeze(-1).float()


class NativeMixedGroupProbe(nn.Module):
    """The original native-forward contract, plus a label-free direct group head."""
    def __init__(self,block,rotary,cache_position,head):
        super().__init__();self.block=block;self.rotary=rotary;self.cache_position=cache_position;self.group_head=head
        self.qmax,self.prefix=head.qmax,head.prefix;self.width=self.qmax+self.prefix

    @classmethod
    def _from_verified_legacy(cls,legacy,*,expected_block_hash,head_seed,dimension=256,layers=2,heads=8,hidden_size=3584):
        if legacy.prefix!=20 or block_state_hash(legacy.block)!=expected_block_hash:
            raise ValueError('Actual loaded source native block hash or P20 geometry differs')
        # Seed ONLY the CPU default generator; preserve source CPU and CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(head_seed)
            head=MixedTokenHead(hidden_size,legacy.qmax,20,dimension,layers,heads)
        return cls(legacy.block,legacy.rotary,legacy.cache_position,head)

    @classmethod
    def from_official_probe(cls,legacy,*,official_inventory,expected_block_hash,head_seed,official_revision=OFFICIAL_REVISION,dimension=256,layers=2,heads=8,hidden_size=3584):
        """Production API: only the official block; zero checkpoint/state IO.

        The existing Probe loader verifies the official28/3584/28/4/18944
        config and loads only block27 safetensors. The caller binds that exact
        official inventory and block hash; this adapter verifies the small
        metadata file and the in-memory loaded block, not the whole7B model.
        """
        if official_revision!=OFFICIAL_REVISION:raise ValueError('Exact official Qwen2.5-7B-Instruct revision required')
        p=Path(official_inventory['path'])
        if not p.is_absolute()or not 0<p.stat().st_size<=16*1024**2:raise ValueError('Small absolute official inventory metadata required')
        raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=official_inventory['sha256']or('bytes'in official_inventory and len(raw)!=official_inventory['bytes']):raise ValueError('Official inventory SHA/bytes differs')
        if not isinstance(json.loads(raw),dict):raise ValueError('Official inventory must be metadata JSON')
        if hidden_size==3584:
            shape={n:tuple(v.shape)for n,v in legacy.block.state_dict().items()}
            if legacy.qmax!=273 or shape.get('input_layernorm.weight')!=(3584,)or shape.get('mlp.down_proj.weight')!=(3584,18944)or shape.get('self_attn.q_proj.weight')!=(3584,3584):raise ValueError('Official block27 native7B geometry differs')
        model=cls._from_verified_legacy(legacy,expected_block_hash=expected_block_hash,head_seed=head_seed,dimension=dimension,layers=layers,heads=heads,hidden_size=hidden_size)
        model.initialization_source='official_initial_Qwen2.5-7B-Instruct_block27'
        model.official_revision=official_revision;model.official_inventory_sha256=official_inventory['sha256']
        return model

    @classmethod
    def from_restored_probe(cls,legacy,*,expected_block_hash,source_checkpoint,head_seed,dimension=256,layers=2,heads=8,hidden_size=3584):
        """Historical nonproduction adapter; the new production uses official0."""
        return cls._from_verified_legacy(legacy,expected_block_hash=expected_block_hash,head_seed=head_seed,dimension=dimension,layers=layers,heads=heads,hidden_size=hidden_size)

    def encode_trajectories(self,batch):
        features=batch['features'][:,:self.width].detach().contiguous();mask=batch['attention_mask'][:,:self.width].contiguous();positions=batch['position_ids'][:,:self.width].contiguous()
        if features.shape[1:]!=(self.width,self.group_head.hidden_size)or mask.dtype!=torch.bool or mask.shape!=features.shape[:2]or positions.shape!=mask.shape or not bool(mask[:,0].all()):
            raise ValueError('Unchanged native causal feature geometry/mask/positions required')
        kwargs=dict(attention_mask=mask,position_ids=positions,position_embeddings=self.rotary(features,positions),use_cache=False,output_attentions=False)
        if self.cache_position:kwargs['cache_position']=torch.arange(self.width,device=features.device)
        hidden=self.block(features,**kwargs)
        if isinstance(hidden,tuple):hidden=hidden[0]
        return dict(hidden=hidden,attention_mask=mask,position_ids=positions)

    def group_logits(self,encoded):
        return self.group_head(encoded['hidden'],encoded['attention_mask'],encoded['position_ids'])

    def forward(self,batch):
        encoded=self.encode_trajectories(batch);n=encoded['hidden'].shape[0]
        if n%5:raise ValueError('Flat input order must contain complete own G5 groups')
        return self.group_logits({key:value.reshape(n//5,5,*value.shape[1:])for key,value in encoded.items()})


def mixed_only_bce(logits,own_em5,question_weights=None):
    """Single binary group objective. Labels are never passed to model.forward."""
    if own_em5.ndim!=2 or own_em5.shape!=(logits.numel(),5)or logits.ndim!=1 or not bool(((own_em5==0)|(own_em5==1)).all()):
        raise ValueError('One direct logit and five authentic binary EM labels per group required')
    count=own_em5.sum(-1);target=((count>0)&(count<5)).to(logits.dtype)
    per=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    if question_weights is None:return per.mean()
    if question_weights.shape!=logits.shape or not bool(torch.isfinite(question_weights).all())or not bool((question_weights>0).all()):raise ValueError('Positive question weights required')
    return (per*question_weights).sum()/question_weights.sum()


def fresh_official_training_state(official_probe,*,official_inventory,expected_block_hash,head_seed,official_revision=OFFICIAL_REVISION,learning_rate=5e-6,**head_kwargs):
    """Production raw/EMA plus nativeAdam0/headAdam0; no prior probe state."""
    if learning_rate!=5e-6:raise ValueError('Official-start branch LR is fixed5e-6')
    model=NativeMixedGroupProbe.from_official_probe(official_probe,official_inventory=official_inventory,expected_block_hash=expected_block_hash,head_seed=head_seed,official_revision=official_revision,**head_kwargs)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    block_adam=torch.optim.Adam(model.block.parameters(),lr=learning_rate)
    head_adam=torch.optim.Adam(model.group_head.parameters(),lr=learning_rate)
    assert not block_adam.state and not head_adam.state
    return dict(model=model,ema=ema,block_adam=block_adam,head_adam=head_adam,provenance=dict(initialization_source=model.initialization_source,official_revision=official_revision,official_inventory=dict(official_inventory),initial_native_block_sha256=expected_block_hash,native_Adam_steps=0,group_head_Adam_steps=0,head_seed=head_seed,lr=learning_rate,ema_decay=.99,EMA_initial_full_raw_copy=True,posttraining_probe_checkpoint_read=False,source_four_rank_optimizer_or_RNG_inherited=False,whole_model_lossless_resume_claim=False))


def optimizer_parameter_names(model,optimizer):
    """Names in the actual serialized optimizer order, never state_dict order."""
    names={id(p):name for name,p in model.named_parameters()};ordered=[]
    for group in optimizer.param_groups:
        for p in group['params']:
            if id(p)not in names:raise ValueError('Unknown source optimizer parameter')
            ordered.append(names[id(p)])
    if len(ordered)!=len(set(ordered)):raise ValueError('Repeated source optimizer parameter')
    return ordered


def map_block_adam_state(source_state,source_names,expected_block_names):
    """Name map the inherited block Adam; return retired readout state archive.

    source_names MUST be optimizer_parameter_names(restored_legacy,oldAdam).
    Target parameter groups retain their original options and relative order.
    No moments or steps are reset, and no source objects are modified.
    """
    old_ids=[p for g in source_state['param_groups']for p in g['params']]
    if len(old_ids)!=len(source_names)or len(set(old_ids))!=len(old_ids)or len(set(source_names))!=len(source_names):raise ValueError('Ambiguous source Adam name/ID mapping')
    names=dict(zip(old_ids,source_names));wanted=set(expected_block_names)
    if not wanted or any(not n.startswith('block.')for n in wanted)or {n for n in source_names if n.startswith('block.')}!=wanted:raise ValueError('Exact full inherited block parameter names required')
    result=dict(state={},param_groups=[]);retired={};target_names=[];clocks=set();next_id=0
    for group in source_state['param_groups']:
        target=copy.deepcopy({k:v for k,v in group.items()if k!='params'});target['params']=[]
        for old_id in group['params']:
            name=names[old_id];saved=source_state['state'].get(old_id)
            if name in wanted:
                if not saved or 'step'not in saved or 'exp_avg'not in saved or 'exp_avg_sq'not in saved:raise ValueError('Inherited block Adam moments/step absent')
                clocks.add(int(saved['step'].item()if torch.is_tensor(saved['step'])else saved['step']));target['params'].append(next_id);result['state'][next_id]=copy.deepcopy(saved);target_names.append(name);next_id+=1
            else:retired[name]=copy.deepcopy(saved)
        if target['params']:result['param_groups'].append(target)
    if len(clocks)!=1:raise ValueError('Inherited block Adam clocks differ')
    return result,dict(inherited_block_names=target_names,inherited_steps=clocks.pop(),retired_optimizer_state_by_name=retired,source_parameter_names=list(source_names),structure_changed=True,entire_model_lossless_resume_claim=False)


def inherit_block_adam(restored_legacy,source_adam,new_model,learning_rate=5e-6):
    """Historical only; official production MUST use fresh_official_training_state."""
    source_names=optimizer_parameter_names(restored_legacy,source_adam);wanted=['block.'+n for n,_ in new_model.block.named_parameters()]
    if set(dict(new_model.block.named_parameters()))!=set(dict(restored_legacy.block.named_parameters()))or any(p is not dict(restored_legacy.block.named_parameters())[n]for n,p in new_model.block.named_parameters()):raise ValueError('Native block must be the actual restored legacy module')
    mapped,archive=map_block_adam_state(source_adam.state_dict(),source_names,wanted)
    if learning_rate!=5e-6:raise ValueError('Initial Mixed-only branch LR is fixed5e-6')
    lookup=dict(new_model.named_parameters());names=iter(archive['inherited_block_names']);groups=[]
    for g in mapped['param_groups']:
        group=copy.deepcopy({k:v for k,v in g.items()if k!='params'});group['params']=[lookup[next(names)]for _ in g['params']];groups.append(group)
    optimizer=torch.optim.Adam(groups);optimizer.load_state_dict(copy.deepcopy(mapped))
    archive['source_learning_rates']=[g['lr']for g in optimizer.param_groups]
    for g in optimizer.param_groups:g['lr']=learning_rate
    if len({float(g['lr'])for g in source_adam.param_groups})!=1:raise ValueError('Explicit separate head LR required for multi-LR source')
    head_adam=torch.optim.Adam(new_model.group_head.parameters(),lr=learning_rate,betas=source_adam.param_groups[0]['betas'],eps=source_adam.param_groups[0]['eps'],weight_decay=source_adam.param_groups[0]['weight_decay'],amsgrad=source_adam.param_groups[0]['amsgrad'])
    archive['retired_parameters']={n:p.detach().cpu().clone()for n,p in restored_legacy.named_parameters()if not n.startswith('block.')}
    archive['new_head_steps']=0
    return optimizer,head_adam,archive
