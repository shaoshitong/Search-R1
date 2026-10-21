"""Native causal block27 + direct, permutation-invariant real-G5 Mixed head.

No actor files are opened here. Supply a checkpoint-restored legacy Probe and
its verified block hash. Its retired scalar output is never registered or run.
The new head reads every retained policy token; only shared question *slots*
are averaged across trajectories (equal in the ideal causal computation).
This average also avoids making harmless HF numerical differences depend on
which trajectory happens to be first. It never pools the100 policy tokens.
"""
import copy,hashlib
import torch
from torch import nn
import torch.nn.functional as F
E4_SOURCE_SHA='e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8'


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
    def from_restored_probe(cls,legacy,*,expected_block_hash,source_checkpoint,head_seed,dimension=256,layers=2,heads=8,hidden_size=3584):
        if source_checkpoint.get('sha256')!=E4_SOURCE_SHA or source_checkpoint.get('steps')!=14418:
            raise ValueError('New Mixed-only branch must inherit the pre-round E4 checkpoint/Adam14418')
        if legacy.prefix!=20 or block_state_hash(legacy.block)!=expected_block_hash:
            raise ValueError('Restore the actual source checkpoint block first; actor initialization is not accepted')
        # Seed ONLY the CPU default generator; preserve source CPU and CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(head_seed)
            head=MixedTokenHead(hidden_size,legacy.qmax,20,dimension,layers,heads)
        return cls(legacy.block,legacy.rotary,legacy.cache_position,head)

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
    """Return inherited block Adam + fresh separate head Adam + archived head."""
    source_names=optimizer_parameter_names(restored_legacy,source_adam);wanted=['block.'+n for n,_ in new_model.block.named_parameters()]
    if set(dict(new_model.block.named_parameters()))!=set(dict(restored_legacy.block.named_parameters()))or any(p is not dict(restored_legacy.block.named_parameters())[n]for n,p in new_model.block.named_parameters()):raise ValueError('Native block must be the actual restored legacy module')
    mapped,archive=map_block_adam_state(source_adam.state_dict(),source_names,wanted)
    if archive['inherited_steps']!=14418:raise ValueError('Only the original E4 Adam14418 source is accepted')
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
