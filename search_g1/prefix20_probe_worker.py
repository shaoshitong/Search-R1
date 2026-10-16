"""Allocated four-rank probe fitting on authentic, short trajectory contexts.

The frozen official encoder and pretrained Linear20 architecture are retained.
Only detached policy-position features enter the trainable decoder block.
"""
import hashlib
import json
import copy
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

from search_g1.prefix20_contract import CONTRACT


class CapturedLayer(Exception):
    pass


def rng_state(device):
    import random,numpy as np
    return dict(python=random.getstate(),numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state(device))


def restore_rng(value,device):
    import random,numpy as np
    random.setstate(value['python']);np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch_cpu']);torch.cuda.set_rng_state(value['torch_cuda'],device)


@contextmanager
def probe_rng(head):
    actor=rng_state(head.device)
    try:
        restore_rng(head.rng,head.device)
        yield
    finally:
        head.rng=rng_state(head.device)
        restore_rng(actor,head.device)


def validate_rows(rows, *, labels_required):
    seen=set()
    for row in rows:
        tid=row['trajectory_id']
        if not isinstance(tid,str) or not tid or tid in seen:
            raise ValueError('PREFIX20_DUPLICATE_OR_INVALID_TRAJECTORY')
        seen.add(tid)
        if labels_required and (type(row.get('strict_em')) is not int or row['strict_em'] not in (0,1)):
            raise ValueError('PREFIX20_TRAINING_LABEL_MISSING')
        if not labels_required and 'strict_em' in row:
            raise ValueError('PREFIX20_ACQUISITION_MUST_NOT_RECEIVE_FINAL_LABELS')
        tokens=row['policy_token_ids'];positions=row['policy_position_ids'];segments=row['prefix_segments']
        if not 1<=len(tokens)<=20 or len(positions)!=len(tokens):
            raise ValueError('PREFIX20_INVALID_ACTUAL_PREFIX_LENGTH')
        if any(type(t) is not int or t<0 for t in tokens):
            raise ValueError('PREFIX20_INVALID_ACTUAL_TOKEN_IDS')
        from reproduction.prefix20_online.model import expected_policy_positions
        bounds=[{k:s[k] for k in ('turn_index','policy_start','policy_length','position_start')} for s in segments]
        if expected_policy_positions(bounds)!=positions:
            raise ValueError('PREFIX20_ACTUAL_ROPE_POSITION_MISMATCH')
        for s in segments:
            start,n,request=s['policy_start'],s['policy_length'],s['request_length']
            context=s['context_tokens']
            if (request!=s['position_start'] or len(context)!=request+n or
                    context[request:]!=tokens[start:start+n]):
                raise ValueError('PREFIX20_TOKENS_DIFFER_FROM_OWN_CAUSAL_REQUEST')
        if segments[0]['context_tokens'][:segments[0]['request_length']]!=row['initial_prompt_token_ids']:
            raise ValueError('PREFIX20_INITIAL_QUESTION_CONTEXT_CHANGED')
    return rows


def validate_global_partition(shards):
    if len(shards)!=4 or any(len(ids)!=640 for ids in shards):
        raise ValueError('PREFIX20_FOUR_LOCAL640_SHARDS_REQUIRED')
    ids=[tid for shard in shards for tid in shard]
    if len(set(ids))!=2560:
        raise ValueError('PREFIX20_GLOBAL_PASS_REPEATS_OR_OMITS_TRAJECTORIES')
    return ids


class RankProbe:
    def __init__(self,payload,control_group):
        if dist.get_world_size()!=4 or not torch.cuda.is_available():
            raise ValueError('PREFIX20_PROBE_REQUIRES_FOUR_ALLOCATED_RANKS')
        from transformers import AutoModel
        from safetensors.torch import load_file
        from reproduction.prefix20_online.model import PrefixCorrectnessProbe
        self.device=torch.device('cuda',torch.cuda.current_device())
        checkpoint=Path(payload['head_safetensors'])
        from reproduction.prefix20_joint4.assets import sha
        if sha(checkpoint)!='535c7bfee07374dde88b7d7f9e4f08ac874d46088ef961f15ea6ab7dd849a560':
            raise ValueError('PREFIX20_PRETRAINED_HEAD_SHA_MISMATCH')
        config_path=Path(payload['probe_config'])
        if sha(config_path)!='be7456474ad62f3c874a4fb176d9858fa8cef385077e96247a3e8c795e3347d7':
            raise ValueError('PREFIX20_EXPORTED_CONFIG_SHA_MISMATCH')
        config=json.loads(config_path.read_text())
        self.exported_config=config
        head=PrefixCorrectnessProbe(payload['model_dir'],max_tokens=20)
        head.load_state_dict(load_file(str(checkpoint),device='cpu'),strict=True)
        head=head.to(self.device,dtype=torch.float32)
        self.model=head
        # A CPU control group lets all ranks report validation/encoding/OOM
        # failures before any gradient collective. Gradients are then averaged
        # explicitly over the four equal local64 losses (global256 BCE).
        self.control_group=control_group
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-6,weight_decay=0)
        self.encoder=AutoModel.from_pretrained(payload['model_dir'],torch_dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True,trust_remote_code=False,low_cpu_mem_usage=True)
        self.encoder.requires_grad_(False).eval()
        if len(self.encoder.layers)!=36 or self.encoder.config.hidden_size!=2048:
            raise ValueError('PREFIX20_OFFICIAL_ENCODER_ARCHITECTURE_MISMATCH')
        self.captured=[]
        self.hook=self.encoder.layers[27].register_forward_hook(self.capture_hook)
        self.completed_outer=0;self.actual_steps=0;self.pending_outer=None
        self.encoder_micro=16
        self.rng=rng_state(self.device)

    def capture_hook(self,module,inputs,value):
        self.captured.append(value[0] if isinstance(value,tuple) else value)
        raise CapturedLayer()

    def local_phase(self,function):
        result=None;failure=None
        try:result=function()
        except Exception as error:failure=type(error).__name__
        failures=[None]*4
        dist.all_gather_object(failures,failure,group=self.control_group)
        if any(value is not None for value in failures):
            raise RuntimeError('PREFIX20_COLLECTIVE_LOCAL_PHASE_FAILED: '+str(failures))
        return result

    def encode(self,rows):
        features=torch.zeros((len(rows),20,2048),dtype=torch.bfloat16,device=self.device)
        masks=torch.zeros((len(rows),20),dtype=torch.bool,device=self.device)
        positions=torch.zeros((len(rows),20),dtype=torch.long,device=self.device)
        bounds=[];work=[]
        for i,row in enumerate(rows):
            n=len(row['policy_token_ids']);masks[i,:n]=True
            positions[i,:n]=torch.tensor(row['policy_position_ids'],dtype=torch.long,device=self.device)
            bounds.append([{k:s[k] for k in ('turn_index','policy_start','policy_length','position_start')}
                           for s in row['prefix_segments']])
            work.extend((i,s) for s in row['prefix_segments'])
        work.sort(key=lambda x:len(x[1]['context_tokens']))
        with torch.no_grad():
            for start in range(0,len(work),self.encoder_micro):
                selected=work[start:start+self.encoder_micro]
                width=max(len(s['context_tokens']) for _,s in selected)
                ids=torch.zeros((len(selected),width),dtype=torch.long,device=self.device)
                mask=torch.zeros_like(ids);rope=torch.zeros_like(ids)
                for slot,(_,s) in enumerate(selected):
                    n=len(s['context_tokens']);ids[slot,:n]=torch.tensor(s['context_tokens'],device=self.device)
                    mask[slot,:n]=1;rope[slot,:n]=torch.arange(n,device=self.device)
                self.captured.clear()
                try:self.encoder(input_ids=ids,attention_mask=mask,position_ids=rope,use_cache=False,return_dict=True)
                except CapturedLayer:pass
                if len(self.captured)!=1:raise RuntimeError('PREFIX20_LAYER27_CAPTURE_MISSING')
                hidden=self.captured.pop().detach()
                for slot,(i,s) in enumerate(selected):
                    lo,n,req=s['policy_start'],s['policy_length'],s['request_length']
                    features[i,lo:lo+n]=hidden[slot,req:req+n]
        return dict(features=features.detach(),attention_mask=masks,position_ids=positions,policy_segments=bounds)

    def predict(self,rows):
        validate_rows(rows,labels_required=False)
        self.encoder.to(self.device);self.model.eval();result=[]
        try:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for start in range(0,len(rows),64):
                    batch=rows[start:start+64]
                    probabilities=torch.sigmoid(self.model(self.encode(batch))).cpu().tolist()
                    result.extend(dict(trajectory_id=row['trajectory_id'],em_probability=p) for row,p in zip(batch,probabilities))
        finally:self.encoder.to('cpu');torch.cuda.empty_cache()
        return dict(rank=dist.get_rank(),completed_outer=self.completed_outer,actual_steps=self.actual_steps,predictions=result)

    def update(self,payload):
        outer=payload.get('outer');batches=payload.get('batches',[])
        def precheck():
            if type(outer) is not int or outer!=self.completed_outer+1 or not 1<=outer<=200 or self.pending_outer is not None:
                raise ValueError('PREFIX20_PROBE_OUTER_BOUNDARY_INVALID')
            if len(batches)!=10 or any(len(batch)!=64 for batch in batches):
                raise ValueError('PREFIX20_REQUIRES_TEN_LOCAL64_BATCHES')
            return validate_rows([r for batch in batches for r in batch],labels_required=True)
        rows=self.local_phase(precheck)
        shards=[None]*4
        dist.all_gather_object(shards,[r['trajectory_id'] for r in rows],group=self.control_group)
        validate_global_partition(shards)
        self.pending_outer=outer
        self.local_phase(lambda:self.encoder.to(self.device));self.model.train()
        loss_sum=correct=total=0
        try:
            for batch in batches:
                self.optimizer.zero_grad(set_to_none=True)
                def backward():
                    labels=torch.tensor([r['strict_em'] for r in batch],dtype=torch.float32,device=self.device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=self.model(self.encode(batch))
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('PREFIX20_NONFINITE_PROBE_LOSS')
                    loss.backward()
                    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()):
                        raise FloatingPointError('PREFIX20_NONFINITE_OR_MISSING_PROBE_GRADIENT')
                    return loss.detach(),logits.detach(),labels
                loss,logits,labels=self.local_phase(backward)
                for parameter in self.model.parameters():
                    parameter.grad.div_(4)
                    dist.all_reduce(parameter.grad,op=dist.ReduceOp.SUM)
                def gradient_check():
                    if any(not bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()):
                        raise FloatingPointError('PREFIX20_NONFINITE_REDUCED_PROBE_GRADIENT')
                self.local_phase(gradient_check)
                self.local_phase(self.optimizer.step);self.actual_steps+=1
                loss_sum+=float(loss.detach())*64;correct+=int(((logits.detach()>=0)==labels.bool()).sum());total+=64
            counts=[None]*4;dist.all_gather_object(counts,self.actual_steps,group=self.control_group)
            if counts!=[outer*10]*4:
                raise RuntimeError('PREFIX20_PROBE_RANK_STEP_BOUNDARIES_DIVERGED')
            self.completed_outer=outer;self.pending_outer=None
            return dict(rank=dist.get_rank(),completed_outer=outer,actual_steps=self.actual_steps,
                        local_trajectories=total,global_batch=256,learning_rate=1e-6,
                        own_trajectory_em_bce=loss_sum/total,training_binary_accuracy=correct/total)
        finally:self.encoder.to('cpu');torch.cuda.empty_cache()


def worker_operation(worker,operation,payload):
    worker._assert_generation_phase_inactive()
    if not worker._is_actor or dist.get_world_size()!=4:raise ValueError('PREFIX20_OWN_FOUR_ACTOR_WORKERS_REQUIRED')
    if operation=='init':
        if getattr(worker,'_prefix20_probe',None) is not None:raise ValueError('PREFIX20_PROBE_ALREADY_INITIALIZED')
        device=torch.device('cuda',torch.cuda.current_device());actor_rng=rng_state(device)
        control_group=dist.new_group(backend='gloo');head=None;failure=None
        try:
            import random,numpy as np
            seed=1729+dist.get_rank();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            try:head=RankProbe(payload,control_group)
            except Exception as error:failure=type(error).__name__
            failures=[None]*4;dist.all_gather_object(failures,failure,group=control_group)
            if any(value is not None for value in failures):
                raise RuntimeError('PREFIX20_COLLECTIVE_INITIALIZATION_FAILED: '+str(failures))
            worker._prefix20_probe=head
        finally:restore_rng(actor_rng,device)
        return dict(rank=dist.get_rank(),world_size=4,completed_outer=0,actual_steps=0,
                    pretrained_head_loaded=True,extra_warmup_updates=0)
    head=getattr(worker,'_prefix20_probe',None)
    if head is None:raise ValueError('PREFIX20_PROBE_NOT_INITIALIZED')
    if operation=='predict':
        with probe_rng(head):return head.predict(payload['rows'])
    if operation=='update':
        with probe_rng(head):return head.update(payload)
    if operation=='state_dict':
        from search_g1.training_state import cpu_tree
        if head.pending_outer is not None:raise ValueError('PREFIX20_CANNOT_CHECKPOINT_PARTIAL_PROBE_OUTER')
        return dict(rank=dist.get_rank(),completed_outer=head.completed_outer,actual_steps=head.actual_steps,
            model=cpu_tree(head.model.state_dict()),optimizer=cpu_tree(head.optimizer.state_dict()),
            rng=cpu_tree(head.rng))
    if operation=='load_state_dict':
        saved=payload['state']
        if saved['rank']!=dist.get_rank() or not 0<=saved['completed_outer']<=200 or saved['actual_steps']!=10*saved['completed_outer']:
            raise ValueError('PREFIX20_RECOVERY_RANK_OR_OUTER_MISMATCH')
        head.model.load_state_dict(saved['model'],strict=True)
        head.optimizer.load_state_dict(saved['optimizer'])
        if any(float(group['lr'])!=1e-6 for group in head.optimizer.param_groups):
            raise ValueError('PREFIX20_RECOVERY_LEARNING_RATE_CHANGED')
        head.rng=copy.deepcopy(saved['rng']);head.completed_outer=saved['completed_outer'];head.actual_steps=saved['actual_steps'];head.pending_outer=None
        return dict(rank=dist.get_rank(),completed_outer=head.completed_outer,actual_steps=head.actual_steps,
                    model_optimizer_rng_restored=True)
    raise ValueError('PREFIX20_UNKNOWN_PROBE_OPERATION')
