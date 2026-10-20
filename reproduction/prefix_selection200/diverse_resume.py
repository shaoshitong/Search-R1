"""Qualified B CP50 fork: exact Adam 4->8 split, no new optimization step.

Only shapes whose true unpadded parameter counts are independently identified
from the saved FP32 HF model are admitted. Source rank r is split into target
2r/2r+1 without interleaving. Four parent RNG states are retained and mapped to
eight explicit streams; new odd streams are derived, not called exact resume.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

ADAPTER='b_diverse_4to8_cp50_v1'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())


def fork_rng(value,target_rank):
    import numpy as np
    import torch
    result=copy.deepcopy(value)
    if target_rank%2==0:return result
    h=hashlib.sha256(value['torch_cpu'].numpy().tobytes()+value['torch_cuda'].numpy().tobytes()+str(target_rank).encode()).digest()
    seed=int.from_bytes(h[:8],'little')%(2**63-1)
    result['python']=random.Random(seed).getstate();result['numpy']=np.random.RandomState(seed%(2**32)).get_state()
    result['torch_cpu']=torch.Generator().manual_seed(seed).get_state()
    if result['torch_cuda'].numel()!=16:raise ValueError('Require the actual saved Philox seed/offset state')
    result['torch_cuda'][:8]=torch.tensor(list(seed.to_bytes(8,'little')),dtype=torch.uint8)
    return result


def flat_numels(model):
    """Saved safetensors omit tied aliases; grouped root/layer counts are exact."""
    from safetensors import safe_open
    import torch
    tied=read(Path(model)/'config.json').get('tie_word_embeddings') is True
    counts=[0]*37;aliases={}
    for f in sorted(Path(model).glob('*.safetensors')):
        with safe_open(str(f),framework='pt',device='cpu')as tensors:
            for key in tensors.keys():
                if key in ('model.embed_tokens.weight','lm_head.weight') and tied:aliases[key]=(str(f),key)
                if key=='lm_head.weight' and tied:continue
                size=1
                for n in tensors.get_slice(key).get_shape():size*=n
                index=int(key.split('.')[2])+1 if key.startswith('model.layers.')else 0
                counts[index]+=size
    if tied and 'lm_head.weight' in aliases:
        with safe_open(aliases['model.embed_tokens.weight'][0],framework='pt',device='cpu')as left, safe_open(aliases['lm_head.weight'][0],framework='pt',device='cpu')as right:
            if not torch.equal(left.get_tensor('model.embed_tokens.weight'),right.get_tensor('lm_head.weight')):raise ValueError('Declared tied saved HF aliases differ')
    if any(n<=0 or n%8 for n in counts):raise ValueError('Unpadded native root/36-layer counts must divide eight')
    return counts


def split_optimizer(saved,half,global_counts):
    import torch
    result=copy.deepcopy(saved['optimizer']);layouts=copy.deepcopy(saved['optimizer_parameter_layout'])
    params=[p for g in result['param_groups']for p in g['params']]
    flat=[p for group in layouts for p in group]
    if len(flat)!=37 or len(params)!=37:raise ValueError('Original FSDP root+36 layer layout required')
    comparisons=[]
    for i,(pid,layout)in enumerate(zip(params,flat)):
        original=saved['optimizer']['state'][pid];state=result['state'][pid];n=global_counts[i]//8
        if layout!={'shape':[2*n],'numel':2*n,'dtype':'torch.float32'}:raise ValueError('Source four-way shard has padding or different wrap/order')
        for name in ('exp_avg','exp_avg_sq'):
            value=original[name]
            if value.dtype!=torch.float32 or list(value.shape)!=[2*n]or not torch.isfinite(value).all():raise ValueError('Nonfinite or non-FP32 original Adam moment')
            state[name]=value[half*n:(half+1)*n].clone()
            if not torch.equal(state[name],value[half*n:(half+1)*n]):raise ValueError('Adam moment changed during exact shard split')
        if int(original['step'].item())!=1000:raise ValueError('Source actor Adam is not complete CP50/1000')
        layout.update(shape=[n],numel=n)
        comparisons.append(dict(flat_parameter=i,source_numel=2*n,target_numel=n,bitwise_moments_preserved=True))
    return result,layouts,comparisons


def imported_actor_boundary(actor,outer):
    """Accept an imported state, never synthesize eight historical write journals."""
    try:
        ref=actor['imported_4to8_cp50'];p=read(ref['path'])
        source=read(p['source_joint_path'])
        return (outer==50 and sha(ref['path'])==ref['sha256'] and p['status']=='CP50_4TO8_EXACT_ADAM_RESHARD_COMPLETE'
            and sha(p['source_joint_path'])==p['source_joint_sha256'] and p['source_cpu_qualified'] is True
            and p['new_optimizer_steps']==0 and p['all_moments_bitwise_preserved'] is True
            and actor['rank_states']==p['rank_states'] and actor['rank_receipts']==source['actor']['rank_receipts']
            and len(actor['rank_receipts'])==4 and actor['actual_adam_steps_per_rank']==[1000]*8
            and source['actor']['actual_adam_steps_per_rank']==[1000]*4 and len(actor['rank_states'])==8)
    except(KeyError,ValueError,OSError,TypeError):return False


def validate_imported(base,config):
    if config.get('resume_joint',{}).get('recovery_mode')=='diverse_eight_to_four_cp181_v1':
        from .diverse_four_resume import validate_imported as four_validate
        return four_validate(base,config)
    if config.get('resume_joint',{}).get('recovery_mode')in ('diverse_same_eight_latest_joint_v1','diverse_same_four_latest_joint_v1'):
        from .diverse_native_resume import validate_imported as native_validate
        return native_validate(base,config)
    b=Path(base);ref=read(b/'preparation/resume-joint-receipt.json');j=read(b/'latest-joint.json')
    if (ref['adapter']!=ADAPTER or ref['completed_outer']!=50 or config['world_size']!=8
            or sha(ref['source_joint_path'])!=ref['source_joint_sha256']or not imported_actor_boundary(j['actor'],50)
            or j['actor']!=read(b/'actor/latest.json')or j['probe']!=read(b/'probe/latest.json')):
        raise ValueError('Imported exact CP50 boundary differs')
    return 50


def import_complete(base):
    import gc
    import torch
    from .actor import _state_binding
    from search_g1.training_state import atomic_save
    from .b_progressive import expand
    b=Path(base);c=read(b/'preparation/launch-config.json');spec=c['resume_joint']
    receipt=b/'preparation/resume-joint-receipt.json'
    if receipt.exists():validate_imported(b,c);return read(receipt)
    intent=b/'preparation/resume-joint-intent.json'
    if intent.exists():raise ValueError('Existing CP50 import requires reconciliation, not blind replay')
    source=Path(spec['source_base']);jp=source/'outers/outer-000050/complete.json';j=read(jp)
    proof_path=source/'preparation/ab-joint-0050-independent-full-sha-cpu.json';qualified=read(proof_path)
    if (spec['adapter']!=ADAPTER or spec['completed_outer']!=50 or j['run_id']!=c['run_id']
            or j['actor']['actual_adam_steps_per_rank']!=[1000]*4 or j['probe']['actual_adam_steps_per_rank']!=[700]*4
            or qualified['status']!='AB_JOINT_FULL_SHA_CPU_PASS'or qualified['outer']!=50
            or not qualified['actual_cpu_deserialization']or sha(jp)!=spec['source_joint_sha256']):
        raise ValueError('Genuine parent B CP50 full-state proof required')
    seals={row['path']:row for row in qualified['files']}
    for item in j['actor']['rank_states']:
        seal=seals[item['path']];st=Path(item['path']).stat()
        if (st.st_size,st.st_ino,st.st_mtime_ns)!=(seal['bytes'],seal['inode'],seal['mtime_ns']):raise ValueError('Qualified parent Adam file changed')
    once(intent,dict(adapter=ADAPTER,source_joint_path=str(jp),source_joint_sha256=sha(jp),created_unix=time.time()))
    target=b/'actor/imported-cp50-eight';target.mkdir(parents=True)
    counts=flat_numels(j['actor']['hf_model_dir']);rank_states=[];splits=[]
    binding=_state_binding(dict(c,grounding_state=j['actor']['grounding_state']))
    for parent,item in enumerate(j['actor']['rank_states']):
        if sha(item['path'])!=item['sha256']:raise ValueError('Parent Adam full SHA differs')
        saved=torch.load(item['path'],map_location='cpu',mmap=True,weights_only=False)
        if saved['rank']!=parent or saved['actual_adam_steps']!=1000 or saved['scheduler']['last_epoch']!=50:raise ValueError('Source rank or scheduler boundary differs')
        for half in (0,1):
            rank=2*parent+half;optimizer,layout,comparison=split_optimizer(saved,half,counts)
            out={k:v for k,v in saved.items()if k not in ('optimizer','optimizer_parameter_layout','rng')}
            out.update(rank=rank,binding=binding,optimizer=optimizer,optimizer_parameter_layout=layout,
                optimizer_layout='FSDP_FULL_SHARD_same_Qwen2DecoderLayer_wrap_world8',rng=fork_rng(saved['rng'],rank))
            rank_states.append(atomic_save(target/('rank-%d.pt'%rank),out))
            splits.append(dict(rank=rank,source_rank=parent,source_half=half,parameters=comparison))
            once(target/('rank-%d-reshard-complete.json'%rank),dict(rank=rank,actual_adam_steps=1000,new_optimizer_steps=0,file=rank_states[-1]))
            del out,optimizer,layout;gc.collect()
        del saved;gc.collect()
    migration=dict(status='CP50_4TO8_EXACT_ADAM_RESHARD_COMPLETE',adapter=ADAPTER,source_joint_path=str(jp),
        source_joint_sha256=sha(jp),source_cpu_qualified=True,source_cpu_proof_path=str(proof_path),source_cpu_proof_sha256=sha(proof_path),
        rank_states=rank_states,splits=splits,new_optimizer_steps=0,all_moments_bitwise_preserved=True,
        rng_scope='even child inherits parent; odd child derives independent deterministic stream; not exact executor resume',completed_unix=time.time())
    migration_path=b/'preparation/actor-cp50-4to8-reshard.json';once(migration_path,migration)
    actor=copy.deepcopy(j['actor']);actor.update(binding=binding,source_commit=c['source_commit'],rank_states=rank_states,
        state_dir=str(target),state_path=str(target),actual_adam_steps_per_rank=[1000]*8,new_adam_steps_per_rank=[0]*8,
        skipped_update_opportunities=[0]*8,imported_4to8_cp50=dict(path=str(migration_path),sha256=sha(migration_path)),
        historical_rank_receipts_world=4,new_eight_rank_updates_at_import=0)
    (b/'actor').mkdir(exist_ok=True);once(b/'actor/latest.json',actor)
    probe=torch.load(j['probe_snapshot'],map_location='cpu',mmap=True,weights_only=False)
    if sha(j['probe_snapshot'])!=j['probe']['sha256']:raise ValueError('Source probe snapshot SHA differs')
    probe=expand(probe,20)
    probe['binding'].update(world_size=8,rank_questions_per_period=320,rank_trajectories_per_period=1600)
    probe['binding_sha256']=digest(probe['binding']);probe['actual_adam_steps_per_rank']=[700]*8
    original_rng=probe['rank_rng'];new_rng=[]
    for rank in range(8):
        parent=original_rng[rank//2];row=dict(parent,**fork_rng(parent,rank));row['rank']=rank;new_rng.append(row)
    probe['rank_rng']=new_rng
    from .probe import atomic_checkpoint
    (b/'probe').mkdir();probe_receipt=atomic_checkpoint(b/'probe',probe,50,dimension_migration=True)
    os.link(b/'probe/latest.pt',b/'probe/head-imported-cp50.pt')
    for n in range(1,51):
        src=source/'outers'/('outer-%06d'%n);dst=b/'outers'/src.name;dst.mkdir(parents=True)
        for name in ('complete.json','pool-state-at-completion.json','grounding-state-at-completion.json','grounding-refresh.json'):
            if (src/name).exists()and not(dst/name).exists():(dst/name).write_bytes((src/name).read_bytes())
    (b/'question-pool').mkdir()
    for name in ('binding.json','provenance.json'):
        if name=='binding.json' or (source/'question-pool'/name).exists():
            (b/'question-pool'/name).write_bytes((source/'question-pool'/name).read_bytes())
    (b/'question-pool/state.json').write_bytes(Path(j['pool_snapshot']).read_bytes())
    fork=copy.deepcopy(j);fork.update(actor=actor,probe=probe_receipt,probe_snapshot=str(b/'probe/head-imported-cp50.pt'),
        imported_boundary=True,source_cp50_joint_sha256=sha(jp))
    # Historical parent joint stays immutable. The converted state has a new
    # receipt; it is not represented as eight new historical Adam journals.
    (b/'outers/outer-000050/parent-complete.json').write_bytes(jp.read_bytes())
    (b/'outers/outer-000050/complete.json').write_text(json.dumps(fork,sort_keys=True))
    once(b/'latest-joint.json',fork)
    result=dict(status='B_CP50_DIVERSE_EIGHT_IMPORTED',adapter=ADAPTER,completed_outer=50,source_joint_path=str(jp),
        source_joint_sha256=sha(jp),actor_migration=dict(path=str(migration_path),sha256=sha(migration_path)),
        imported_actor_adam=[1000]*8,imported_probe_adam=[700]*8,optimizer_reset=False,
        exact_model_and_adam_preserved=True,exact_executor_rng_resume=False,new_optimizer_steps=0)
    once(receipt,result);validate_imported(b,c);return result


def main():
    import argparse,traceback
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    terminal=Path(args.base)/'preparation/import-worker-terminal.json'
    try:
        result=import_complete(args.base)
    except BaseException:
        once(terminal,dict(status='FAILED',traceback=traceback.format_exc(),finished_unix=time.time()))
        raise
    once(terminal,dict(status='COMPLETE',receipt=result,finished_unix=time.time()))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
