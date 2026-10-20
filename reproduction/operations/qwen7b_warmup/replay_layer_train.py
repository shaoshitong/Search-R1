"""Matched fresh21/24/27 HF-replay probes: four epochs,512 actual updates.

Allfit/diagnostic/selection use ONE same-layer ReplayFeatureReader per process.
It verifies full files once at startup and every reconstructed row on use;
epoch evaluation never constructs another reader or repeats a bulk SHA scan.
No Slurm submission, actor update or untouched-final evaluation occurs here.
"""
import argparse,copy,fcntl,json,os,random,shlex,signal,time,traceback
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from . import probe_ablation_train as u
from .probe_ablation_loss import probe_ablation_loss
from .probe_ablation_model import CAPTURE_PAIRS,REVISION

STOP=False
LOSS=dict(trajectory_weight=1.,three_event_weight=1.)
TARGET='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/multilayer-four-epoch-v1'


def build_plan(reference,seed=1729):
    cohort=reference['cohort_qids']
    if len(cohort)!=len(set(cohort)) or len(cohort)!=6528:raise ValueError('Exactly6528 distinct matched questions required')
    if len(reference['batches'])!=128 or any(len(b)!=51 for b in reference['batches']) or sorted(q for b in reference['batches'] for q in b)!=sorted(cohort):
        raise ValueError('Source must be the accepted128/B51 once-per-question plan')
    batches=[];epochs=[]
    for epoch in range(1,5):
        own=list(cohort);random.Random(int(u.digest(['fresh-multilayer4epoch-v1',seed,epoch,cohort]),16)).shuffle(own)
        batches.extend(own[i:i+51] for i in range(0,len(own),51));epochs.extend([epoch]*128)
    return dict(cohort_qids=list(cohort),batches=batches,batch_epochs=epochs,seed=seed,
        epochs=4,updates_per_epoch=128,global_groups=51,update_budget=512,labels_used=False,
        source_reference_plan_sha256=u.digest(reference),same_plan_required_for_all_layers=True)


def adam_steps(optimizer):
    if not optimizer.state:return 0
    parameters=[p for g in optimizer.param_groups for p in g['params']]
    if set(parameters)!=set(optimizer.state):raise ValueError('Partially populated fresh Adam')
    return u.steps(optimizer)


def seed_training(seed,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed(seed)


def initialize(factory,training_seed,device):
    model=factory().to(device);ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-6,weight_decay=0)
    seed_training(training_seed,device)
    if adam_steps(optimizer)!=0 or u.model_hash(model)!=u.model_hash(ema):raise ValueError('Fresh Adam0/EMA0 contract differs')
    return model,ema,optimizer


def restore(blob,model,ema,optimizer,device):
    model.load_state_dict(blob['raw'],strict=True);ema.load_state_dict(blob['ema'],strict=True)
    u.restore_buffers(model,blob['raw_buffers']);u.restore_buffers(ema,blob['ema_buffers'])
    optimizer.load_state_dict(copy.deepcopy(blob['optimizer']))
    for state in optimizer.state.values():
        for key,value in state.items():
            if torch.is_tensor(value) and key!='step':state[key]=value.to(device)
    u.restore_rng(blob['rng'],device)
    if adam_steps(optimizer)!=blob['steps'] or u.state_hashes(model,ema,optimizer,device)!=blob['accepted_hashes']:
        raise ValueError('Fresh-branch fullstate restore mismatch')


def phase(cursor):
    return 'EPOCH_TRAINED_EVALUATION_PENDING' if cursor and cursor%128==0 else 'NEXT_TRAIN_UPDATE_PENDING'


def checkpoint_blob(model,ema,optimizer,device,cursor,binding):
    if adam_steps(optimizer)!=cursor:raise ValueError('Fresh Adam count must equal accepted updates')
    return dict(role='FRESH_OFFICIAL_BLOCK_UNIFORM_HF_REPLAY_LAYER_PROBE',cursor=cursor,steps=cursor,
        raw=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
        raw_buffers=u.buffers(model),ema_buffers=u.buffers(ema),rng=u.rng_state(device),
        accepted_hashes=u.state_hashes(model,ema,optimizer,device),next_batch_index=cursor,
        completed_training_epochs=cursor//128,within_epoch_update=cursor%128,phase=phase(cursor),
        scheduler=None,learning_rate_schedule='constant5e-6',actor_updates=0,
        E1_checkpoint_inherited=False,optimizer_initial_step=0,within_new_branch_full_resume=True,**binding)


def save(base,blob):
    directory=base/'checkpoints';directory.mkdir(exist_ok=True);dest=directory/('cursor-%06d.pt'%blob['cursor'])
    if dest.exists():raise ValueError('Do not overwrite immutable fresh-layer checkpoint')
    temporary=dest.with_suffix('.tmp')
    with temporary.open('wb') as f:torch.save(blob,f);f.flush();os.fsync(f.fileno())
    temporary.replace(dest);check=torch.load(dest,map_location='cpu',mmap=True,weights_only=False)
    if check['cursor']!=check['steps'] or check['cursor']!=blob['cursor']:raise ValueError('Saved cursor/Adam differ')
    if any(not bool(torch.isfinite(t).all()) for field in ('raw','ema') for t in check[field].values()):raise ValueError('Nonfinite saved weights')
    states=check['optimizer']['state']
    if (check['cursor']==0 and states) or (check['cursor']>0 and len(states)!=sum(len(g['params']) for g in check['optimizer']['param_groups'])):
        raise ValueError('Invalid complete fresh Adam population')
    for state in states.values():
        if int(state['step'])!=check['cursor'] or any(not bool(torch.isfinite(state[k]).all()) for k in ('exp_avg','exp_avg_sq')):raise ValueError('Invalid saved Adam moments/count')
    actual=dict(raw=u.model_mapping_hash(check['raw']),ema=u.model_mapping_hash(check['ema']),adam=u.optimizer_blob_hash(check['optimizer']),
        raw_buffers=u.tensor_hash(check['raw_buffers']),ema_buffers=u.tensor_hash(check['ema_buffers']),rng=u.rng_hash(check['rng']))
    if actual!=check['accepted_hashes']:raise ValueError('Saved fullstate component hashes differ')
    receipt=dict(path=str(dest),sha256=u.sha(dest),bytes=dest.stat().st_size,cursor=check['cursor'],steps=check['steps'],
        phase=check['phase'],accepted_hashes=actual,config_sha256=blob['config_sha256'],epoch_time=time.time())
    u.atomic(base/'latest.json',receipt);return receipt


def update(model,ema,optimizer,reader,lookup,qids,config,device):
    if any(g['lr']!=5e-6 or g['weight_decay']!=0 for g in optimizer.param_groups):raise ValueError('Actual Adam hyperparameters must remain fixed')
    before=adam_steps(optimizer);model.train();optimizer.zero_grad(set_to_none=True);parts=[];selected=[lookup[q] for q in qids];started=time.monotonic()
    for offset in range(0,len(selected),config['micro_groups']):
        own=selected[offset:offset+config['micro_groups']];data=reader.get([tid for g in own for tid in g['trajectory_ids']],device)
        with torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext():
            parts.append(model(data).reshape(len(own),5))
    logits=torch.cat(parts);labels=torch.tensor([g['labels'] for g in selected],device=device,dtype=logits.dtype)
    result=probe_ablation_loss(logits,labels,**LOSS)
    if not bool(torch.isfinite(result['loss'])):raise ValueError('Nonfinite matched layer loss')
    result['loss'].backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise ValueError('Missing/nonfinite block/head gradient')
    optimizer.step()
    if adam_steps(optimizer)!=before+1:raise ValueError('Fresh optimizer did not advance once')
    with torch.no_grad():
        for target,value in zip(ema.parameters(),model.parameters()):target.mul_(.99).add_(value,alpha=.01)
    called=result['mixed_probability'].detach()>=.5;truth=result['mixed_label'].detach().bool()
    tp=int((called&truth).sum());fn=int((~called&truth).sum());fp=int((called&~truth).sum());tn=int((~called&~truth).sum())
    return dict(loss=float(result['loss'].detach()),trajectory_bce=float(result['trajectory_bce'].detach()),
        group_event_nll=float(result['group_event_nll'].detach()),seconds=time.monotonic()-started,
        mixed_confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),actor_updates=0)


def write_progress(base,cursor,metrics):
    row=dict(cursor=cursor,steps=cursor,epoch=(cursor-1)//128+1,within_epoch_update=(cursor-1)%128+1,epoch_time=time.time(),**metrics)
    u.atomic(base/'progress.json',row)
    with (base/'curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
    print(json.dumps(row),flush=True)


def evaluate_epoch(base,epoch,model,ema,optimizer,reader,diagnostic,selection,device,binding):
    cursor=128*epoch;path=base/('evaluation-epoch-%04d.json'%epoch);receipt=u.read(base/'latest.json')
    if path.exists():
        previous=u.read(path)
        if previous['config_sha256']!=binding['config_sha256'] or previous['cursor']!=cursor:raise ValueError('Existing epoch evaluation binding differs')
        for own in previous['metrics'].values():u.checked(own['predictions'])
        return previous
    if receipt['cursor']!=cursor or receipt['steps']!=cursor:raise ValueError('Durable fullstate must precede epoch evaluation')
    before=u.state_hashes(model,ema,optimizer,device);native=u.buffers(ema);rng=u.rng_state(device);(base/'predictions').mkdir(exist_ok=True)
    try:
        metrics={name:u.evaluate(ema,reader,groups,device,base/'predictions'/('epoch-%04d-%s.jsonl.gz'%(epoch,name)))
            for name,groups in [('diagnostic512',diagnostic),('selection2048',selection)]}
    finally:u.restore_rng(rng,device);u.restore_buffers(ema,native)
    if u.state_hashes(model,ema,optimizer,device)!=before:raise ValueError('Evaluation changed training fullstate')
    result=dict(status='EPOCH_FULLSTATE_SAVED_THEN_FIXED_POOLS_EVALUATED',epoch=epoch,cursor=cursor,
        checkpoint=receipt,metrics=metrics,config_sha256=binding['config_sha256'],feature_backend='huggingface_frozen_causal_prefill',
        native_bitexact_claim=False,threshold_fitted=False,untouched_final_read=False,epoch_time=time.time())
    u.atomic(path,result);return result


def verify_inputs(c,repo):
    if (c['epochs'],c['update_budget'],c['global_groups'],c['micro_groups'],c['lr'],c['ema_decay'])!=(4,512,51,2,5e-6,.99):
        raise ValueError('Fix matched4epoch/512/B51/micro2/LR5e-6/EMA.99')
    if c['loss_kwargs']!=LOSS or c['head_seed']!=1729 or c['training_seed']!=1729 or c['save_every']!=16:
        raise ValueError('Fixed objective/seeds/save cadence required')
    layer=c['block_layer']
    if type(layer) is not int or layer not in CAPTURE_PAIRS:raise ValueError('Block layer must be21/24/27')
    required=('source_matched_config','reference_plan','plan','bank','selection_bank','replay_bank_complete','replay_features','model_config','model_inventory','model_index')
    paths={k:u.checked(c[k]) for k in required}
    code_required={'reproduction/operations/qwen7b_warmup/'+n for n in ('replay_layer_train.py','replay_layer_probe.py','replay_feature_reader.py','replay_probe_layers.py','probe_ablation_train.py','probe_ablation_loss.py','probe_ablation_model.py','runtime.sh')}
    if not code_required<=set(c['code_hashes']):raise ValueError('Complete immutable actual runtime bindings required')
    for name,want in c['code_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or u.sha(repo/name)!=want:raise ValueError('Code SHA differs: '+name)
    source=u.read(paths['source_matched_config'])
    for k in ('bank','selection_bank'):
        if any(c[k][field]!=source[k][field] for field in ('path','sha256')):raise ValueError('Source label bank differs from accepted matched config')
    if any(c['reference_plan'][field]!=source['plan'][field] for field in ('path','sha256')) or u.read(paths['plan'])!=build_plan(u.read(paths['reference_plan'])):raise ValueError('Common four-epoch shuffle plan differs')
    complete=u.read(paths['replay_bank_complete']);cap=str(CAPTURE_PAIRS[layer])
    if (complete['status']!='ALL_FOUR_MATCHED9088Q_MULTILAYER_HF_REPLAY_BANKS_SHA_VERIFIED' or
            complete['questions']!=9088 or complete['trajectories']!=45440 or
            complete['source_runtime']!='huggingface_frozen_causal_prefill' or
            complete['native_bitexact_equivalence_claim'] is not False or complete['full_replay_shared_file_SHA_verified'] is not True or
            complete['actor_updates']!=0 or complete['new_responses_generated']!=0 or complete['features_by_capture_layer'][cap]!=c['replay_features']):
        raise ValueError('All four complete uniform-HF replay shards required for this exact layer')
    if Path(c['model_dir'])/'config.json'!=paths['model_config'] or Path(c['model_dir'])/'official-inventory.json'!=paths['model_inventory'] or Path(c['model_dir'])/'model.safetensors.index.json'!=paths['model_index']:
        raise ValueError('Official model metadata path binding differs')
    inventory=u.read(paths['model_inventory'])
    if inventory['sha']!=REVISION:raise ValueError('Official initial7B revision differs')
    mapping=u.read(paths['model_index'])['weight_map'];prefix='model.layers.%d.'%layer;names={v for k,v in mapping.items() if k.startswith(prefix)}
    if not names or set(c['official_block_shards'])!=names:raise ValueError('Official source block/shard binding differs')
    # Called only AFTER one-H100 guard: actual official shard SHA and model
    # tensors are read on the allocated GPU node, never on the login node.
    for name,item in c['official_block_shards'].items():
        if Path(name).name!=name or Path(item['path'])!=Path(c['model_dir'])/name:raise ValueError('Official shard locator differs')
        u.checked(item)
    return paths,complete


def run(base):
    c=u.read(base/'config.json');config_sha=u.sha(base/'config.json');repo=Path(__file__).resolve().parents[3]
    if int(os.environ.get('WORLD_SIZE','1'))!=1 or torch.cuda.device_count()!=1:raise ValueError('Exactly one allocated GPU per fresh layer branch')
    device=torch.device('cuda',0);props=torch.cuda.get_device_properties(device)
    if 'H100' not in props.name or props.total_memory<75*1024**3:raise ValueError('Dedicated >=75GiB H100 required')
    if os.environ.get('FLASH_ATTENTION_DETERMINISTIC')!='1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Deterministic startup flags required')
    torch.cuda.set_device(device);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    paths,complete=verify_inputs(c,repo)
    os.environ['QWEN7B_QUESTION_MAX_TOKENS']='273';os.environ['SEARCH_G1_B_PREFIX_TOKENS']='20'
    bank=u.read(paths['bank']);diagnostic=bank['diagnostic'];selection=u.read(paths['selection_bank'])['selection_validation'];plan=u.read(paths['plan'])
    if len(diagnostic)!=512 or len(selection)!=2048:raise ValueError('Exact512/2048 viewed validation pools required')
    lookup=u.validate_plan(plan,bank['train'],512,51,[g['qid'] for g in diagnostic+selection]);fit=[lookup[q] for q in plan['cohort_qids']]
    groups=fit+diagnostic+selection
    if len({g['question_hash'] for g in groups})!=9088:raise ValueError('Question hash overlap across fit/evaluation')
    identities=u.read(u.checked(complete['group_identities']))['groups'];by={g['qid']:g for g in identities}
    if len(by)!=9088:raise ValueError('Merged replay question identity count differs')
    for role,rows in [('fit',fit),('diagnostic512',diagnostic),('selection2048',selection)]:
        for g in rows:
            own=by[g['qid']]
            if own['role']!=role or own['trajectory_ids']!=g['trajectory_ids'] or own['question_hash']!=g['question_hash']:raise ValueError('HF replay/source own-G5 identity differs')
    u.atomic(base/'controller.json',dict(status='FRESH_LAYER_INPUTS_BOUND_READER_SHA_STARTING',pid=os.getpid(),job_id=os.environ.get('SLURM_JOB_ID'),
        block_layer=c['block_layer'],capture_layer=CAPTURE_PAIRS[c['block_layer']],config_sha256=config_sha,epoch_time=time.time()))
    from .replay_feature_reader import ReplayFeatureReader
    reader=ReplayFeatureReader(u.read(paths['replay_features']),CAPTURE_PAIRS[c['block_layer']])
    try:
        expected={tid for g in groups for tid in g['trajectory_ids']}
        if set(reader.records)!=expected:raise ValueError('Replay reader must cover exactly all fit/diag/selection real G5')
        for g in groups:
            if any(str(reader.records[tid][1]['qid'])!=g['qid'] for tid in g['trajectory_ids']):raise ValueError('Replay row question ID differs')
        from .replay_layer_probe import ReplayLayerProbe
        model,ema,optimizer=initialize(lambda:ReplayLayerProbe(c['model_dir'],c['block_layer'],c['head_seed']),c['training_seed'],device)
        official_block_sha=u.model_hash(model.block)
        binding=dict(config_sha256=config_sha,plan_sha256=c['plan']['sha256'],replay_bank_complete=c['replay_bank_complete'],
            replay_features=c['replay_features'],block_layer=c['block_layer'],capture_layer=CAPTURE_PAIRS[c['block_layer']],
            official_initial_block_sha256=official_block_sha,official_block_shards=c['official_block_shards'],
            common_head_initial_sha256=u.model_hash(model.readout),initialization=model.initialization_proof,
            feature_backend='huggingface_frozen_causal_prefill',native_bitexact_claim=False)
        for sig in (signal.SIGTERM,signal.SIGUSR1,signal.SIGINT):signal.signal(sig,request_stop)
        cursor=0
        if (base/'latest.json').exists():
            receipt=u.read(base/'latest.json');saved=torch.load(u.checked(receipt),map_location='cpu',mmap=True,weights_only=False);cursor=saved['cursor']
            if saved['config_sha256']!=config_sha or not 0<=cursor<=512 or saved['steps']!=cursor:raise ValueError('Exact fresh branch cursor/config required')
            for name,key in [('progress.json','cursor'),('update-intent.json','next_cursor')]:
                if (base/name).exists() and u.read(base/name)[key]>cursor:raise ValueError('Unresolved later update: no silent rollback')
            restore(saved,model,ema,optimizer,device);del saved
        else:
            if any((base/n).exists() for n in ('progress.json','update-intent.json')):raise ValueError('Existing progress without accepted checkpoint')
            save(base,checkpoint_blob(model,ema,optimizer,device,0,binding))
        u.atomic(base/'restore-acceptance.json',dict(cursor=cursor,steps=adam_steps(optimizer),accepted_hashes=u.state_hashes(model,ema,optimizer,device),
            fresh_official_branch=True,E1_resume_claim=False,common_head_initial_sha256=binding['common_head_initial_sha256'],
            official_initial_block_sha256=official_block_sha,epoch_time=time.time()))
        # If evaluation failed after its durable boundary CP, finish it before
        # advancing another training update. Existing paired results are checked.
        if cursor and cursor%128==0 and not STOP:evaluate_epoch(base,cursor//128,model,ema,optimizer,reader,diagnostic,selection,device,binding)
        if cursor<512 and not STOP:
            u.atomic(base/'update-intent.json',dict(next_cursor=cursor+1,phase='REAL_NEXT_UPDATE_DOUBLE_REPLAY'))
            start=torch.load(u.checked(u.read(base/'latest.json')),map_location='cpu',mmap=True,weights_only=False)
            update(model,ema,optimizer,reader,lookup,plan['batches'][cursor],c,device);expected_hashes=u.state_hashes(model,ema,optimizer,device)
            restore(start,model,ema,optimizer,device);metrics=update(model,ema,optimizer,reader,lookup,plan['batches'][cursor],c,device)
            actual=u.state_hashes(model,ema,optimizer,device);proof=dict(passed=actual==expected_hashes,expected=expected_hashes,actual=actual,source_cursor=cursor,steps=adam_steps(optimizer))
            u.atomic(base/('next-update-replay-cursor%06d.json'%cursor),proof)
            if actual!=expected_hashes:raise ValueError('Real fresh-layer next-update replay mismatch')
            cursor+=1;save(base,checkpoint_blob(model,ema,optimizer,device,cursor,binding));write_progress(base,cursor,metrics);del start
            if cursor%128==0 and not STOP:evaluate_epoch(base,cursor//128,model,ema,optimizer,reader,diagnostic,selection,device,binding)
        while cursor<512 and not STOP:
            u.atomic(base/'update-intent.json',dict(next_cursor=cursor+1,phase='ACTUAL_GLOBAL_OPTIMIZER_UPDATE'))
            metrics=update(model,ema,optimizer,reader,lookup,plan['batches'][cursor],c,device);cursor+=1;write_progress(base,cursor,metrics)
            if cursor%16==0 or STOP or cursor==512:save(base,checkpoint_blob(model,ema,optimizer,device,cursor,binding))
            if cursor%128==0 and not STOP:evaluate_epoch(base,cursor//128,model,ema,optimizer,reader,diagnostic,selection,device,binding)
        if STOP:
            if u.read(base/'latest.json')['cursor']!=cursor:save(base,checkpoint_blob(model,ema,optimizer,device,cursor,binding))
            u.atomic(base/'interrupted.json',dict(cursor=cursor,steps=adam_steps(optimizer),phase=phase(cursor)));return
        results=[u.read(base/('evaluation-epoch-%04d.json'%epoch)) for epoch in range(1,5)]
        if any(r['cursor']!=128*epoch or r['config_sha256']!=config_sha for epoch,r in enumerate(results,1)):
            raise ValueError('All four epoch evaluations must match this branch and budget')
        u.atomic(base/'complete.json',dict(status='FRESH_OFFICIAL_MULTILAYER_HF_REPLAY_FOUR_EPOCHS_COMPLETE',cursor=cursor,steps=adam_steps(optimizer),
            epochs=4,epoch_evaluations=results,checkpoint=u.read(base/'latest.json'),actor_updates=0,**binding))
    finally:
        reader.close()
        if torch.cuda.is_initialized():u.atomic(base/'gpu-memory.json',dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),epoch_time=time.time()))


def render_sbatch(base,code_root,block_layer):
    if block_layer not in CAPTURE_PAIRS or Path(base).parent!=Path(TARGET):raise ValueError('Independent explicit layer branch required')
    for p in (base,code_root):
        if not Path(p).is_absolute() or any(x.isspace() for x in p) or any(x in p for x in '\r\n;`$'):raise ValueError('Safe absolute deployment path required')
    return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-hflayer-@@LAYER@@
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=@@BASE@@/logs/train-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.replay_layer_train --base @@BASE_Q@@
'''.replace('@@LAYER@@',str(block_layer)).replace('@@BASE@@',base).replace('@@BASE_Q@@',shlex.quote(base)).replace('@@CODE@@',shlex.quote(code_root))


def request_stop(_sig,_frame):
    global STOP
    STOP=True


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();base=a.base.resolve()
    if str(base.parent)!=TARGET:raise ValueError('Use independent remote four-epoch layer branch')
    base.mkdir(parents=True,exist_ok=True)
    with (base/'fresh-layer.lock').open('a+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(base)
        except BaseException as error:
            u.atomic(base/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),epoch_time=time.time()));raise


if __name__=='__main__':main()
