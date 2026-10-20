"""Independent E4 raw/EMA/Adam continuation: targeted versus random G5 data.

Both branches inherit source Adam14418 and canonical rank0 RNG, archive all
four source ranks, and take477 complete-global-B48 updates of one identical
loss. Neither a source-E1 reset nor a four-rank bit-exact continuation claim.
"""
import argparse,copy,fcntl,json,os,platform,signal,time,traceback
from pathlib import Path
os.environ['QWEN7B_QUESTION_MAX_TOKENS']='273'
os.environ['SEARCH_G1_B_PREFIX_TOKENS']='20'
import torch
from . import probe_ablation_train as u
from . import evaluate_frozen_validation as ev

ROLE='E4_PREFIX20_TRAIN_ONLY_HARD_VS_RANDOM_CONTINUATION'
SOURCE_SHA='e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8'
SOURCE_STEPS=14418
LOSS=dict(trajectory_weight=1.,three_event_weight=1.,inter_rank_weight=1.)
STOP=False


def validate_config(c):
    if (c['role']!=ROLE or c['variant'] not in ('targeted','random_control') or
            c['source_checkpoint']['sha256']!=SOURCE_SHA or c['source_checkpoint']['steps']!=SOURCE_STEPS or
            c['source_checkpoint']['epoch']!=4 or c['global_groups']!=48 or c['update_budget']!=477 or
            c['fit_questions']!=22896 or c['prefix_tokens']!=20 or c['question_max_tokens']!=273 or
            c['recipe']['loss_kwargs']!=LOSS or c['lr']!=5e-6 or c['ema_decay']!=.99 or
            c['weight_decay']!=0 or c['actor_updates']!=0 or c['four_rank_bitexact_resume_claim'] is not False or
            c['E1_reset'] is not False or c['source_raw_ema_adam_RNG_preserved'] is not True or
            c['canonical_source_rng_rank']!=0 or c['source_world_size']!=4 or c['current_world_size']!=1 or
            type(c['micro_groups']) is not int or not 1<=c['micro_groups']<=4 or
            type(c['save_every']) is not int or c['save_every'] not in (16,32)):
        raise ValueError('Pinned E4/B48/477/22896/P20/common-loss continuation required')


def selected_binding(c,selection):
    if (selection['status']!='TRAIN_ONLY_HARD_AND_RANDOM22896Q_CLASS_MATCHED_PLANS_COMPLETE' or
            selection['outputs'][c['variant']]['groups']!=c['bank'] or
            selection['outputs'][c['variant']]['plan']!=c['plan'] or
            selection['source_candidates']!=c['candidate_plan'] or
            selection['source_scoring_receipt']!=c['scoring_complete'] or
            selection['source_bank']!=c['source_bank'] or selection['features']!=c['features'] or
            c['diagnostic_bank']!=c['source_bank'] or selection['checkpoint']['sha256']!=SOURCE_SHA):
        raise ValueError('Runtime branch differs from actual accepted E4 CPU selection')


def require_final_checkpoint(receipt):
    if receipt['cursor']!=477 or receipt['steps']!=14895:raise ValueError('Final full checkpoint must precede validation')


def validate_plan(plan,train,forbidden_qids=(),forbidden_hashes=(),synthetic_fixture=False):
    total,batch,budget=(len(train),plan['global_groups'],plan['update_budget']) if synthetic_fixture else (22896,48,477)
    qids=plan['cohort_qids'];flat=[q for row in plan['batches'] for q in row]
    if (len(train)!=total or len(qids)!=total or len(set(qids))!=total or flat!=qids or
            len(plan['batches'])!=budget or any(len(row)!=batch or len(set(row))!=batch for row in plan['batches']) or
            plan.get('each_question_once') is not True or plan.get('epochs')!=1 or
            plan.get('global_groups')!=batch or plan.get('update_budget')!=budget or
            plan.get('ordered_qid_sha256')!=u.digest(qids)):
        raise ValueError('Exactly one fixed own-question pass with477 complete B48 updates required')
    lookup={g['qid']:g for g in train};seen_tids=set();seen_hashes=set()
    if set(lookup)!=set(qids) or set(qids)&set(forbidden_qids):raise ValueError('Train/plan/external qids differ or overlap')
    for group in train:
        tids=group['trajectory_ids'];ys=group['labels'];h=group['question_hash']
        if (group.get('split')!='train' or h in seen_hashes or h in forbidden_hashes or
                len(tids)!=5 or len(set(tids))!=5 or seen_tids.intersection(tids) or
                len(ys)!=5 or any(type(y) is not int or y not in (0,1) for y in ys)):
            raise ValueError('Distinct real train question/hash/own G5 IDs and own strict EM required')
        seen_tids.update(tids);seen_hashes.add(h)
    return lookup


def verify_source(c,receipt,source_config,saved):
    descriptor=c['source_checkpoint']
    ev.validate_checkpoint_metadata(descriptor,receipt,source_config,saved)
    if (saved['bank_sha256']!=c['source_bank']['sha256'] or saved['steps']!=SOURCE_STEPS or
            source_config['fit_lr']!=5e-6 or source_config['fit_ema']!=.99 or
            any(row['steps']!=SOURCE_STEPS for row in saved['rank_states'])):
        raise ValueError('E4 own bank/Adam14418/source LR/EMA lineage mismatch')
    return copy.deepcopy(saved['rank_states'])


def verify_restored_source(saved,model,ema,optimizer,device):
    current=u.state_hashes(model,ema,optimizer,device)
    for key,field in (('raw','model_sha256'),('ema','ema_sha256'),('adam','optimizer_sha256')):
        if any(row[field]!=current[key] for row in saved['rank_states']):raise ValueError('Restored E4 '+key+' differs')
    for key in ('raw_buffers','ema_buffers'):
        if current[key]!=u.tensor_hash(saved[key]):raise ValueError('Restored E4 native RoPE differs')
    rank0=next(r for r in saved['rank_states'] if r['rank']==0)
    if current['rng']!=u.rng_hash(rank0['rng']):raise ValueError('E4 canonical rank0 RNG reset')
    if any(g['lr']!=5e-6 or g['weight_decay']!=0 for g in optimizer.param_groups):raise ValueError('Inherited Adam LR/weight decay changed')
    return current


def replay_next(blob,model,ema,optimizer,reader,lookup,batch,c,device):
    """Two real optimizer updates from exactly the same accepted full state."""
    u.run_update(model,ema,optimizer,reader,lookup,batch,c,device)
    expected=u.state_hashes(model,ema,optimizer,device)
    u.restore_blob(blob,model,ema,optimizer,device)
    metrics=u.run_update(model,ema,optimizer,reader,lookup,batch,c,device)
    actual=u.state_hashes(model,ema,optimizer,device)
    if actual!=expected:raise ValueError('Actual complete B48 next-update replay mismatch')
    return metrics,dict(passed=True,expected=expected,actual=actual,
        source_steps=blob['steps'],accepted_steps=u.steps(optimizer),
        next_update_replayed_twice=True,four_rank_bitexact_resume_claim=False)


def blob(model,ema,optimizer,device,cursor,binding,ranks):
    value=u.checkpoint_blob(model,ema,optimizer,device,cursor,binding,ranks)
    value.update(role=ROLE,source_steps=SOURCE_STEPS,stage='evaluation_pending' if cursor==477 else 'training')
    return value


def check_resume(saved,cursor,c,binding,progress=None,intent=None):
    if (saved['role']!=ROLE or saved['cursor']!=cursor or not 1<=cursor<=477 or
            saved['steps']!=SOURCE_STEPS+cursor or saved['source_steps']!=SOURCE_STEPS or
            saved['config_sha256']!=binding['config_sha256'] or
            saved['matched_binding_sha256']!=binding['matched_binding_sha256'] or
            saved['stage']!=('evaluation_pending' if cursor==477 else 'training')):
        raise ValueError('Accepted E4 continuation position/binding mismatch')
    if progress and progress['cursor']>cursor:raise ValueError('Unsaved accepted progress: do not silently roll back')
    if intent and intent['next_cursor']>cursor:raise ValueError('Unresolved later optimizer transaction: do not silently roll back')


def render_sbatch(variant,base,code_root):
    if variant not in ('targeted','random_control'):raise ValueError('Unknown paired variant')
    for value in (base,code_root):
        p=Path(value)
        if not p.is_absolute() or any(x.isspace() for x in str(p)) or '..' in p.parts:raise ValueError('Safe absolute source/run paths required')
    return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-hard477-@@VAR@@
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=@@BASE@@/logs/fit-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun --kill-on-bad-exit=1 bash @@CODE@@/reproduction/operations/qwen7b_warmup/runtime.sh -u -m reproduction.operations.qwen7b_warmup.hard_enrichment_train --base @@BASE@@
'''.replace('@@VAR@@',variant).replace('@@BASE@@',str(base)).replace('@@CODE@@',str(code_root))


def stop(_sig,_frame):
    global STOP
    STOP=True


def run(base):
    if platform.system()!='Linux' or not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Real checkpoints/features load only inside authorized remote Slurm GPU jobs')
    c=u.read(base/'config.json');validate_config(c);repo=Path(__file__).resolve().parents[3]
    required=('source_checkpoint','source_receipt','source_config','source_bank','model_config','bank','features',
        'plan','selection_complete','candidate_plan','scoring_complete','diagnostic_bank','validation_complete','feature_manifest_refs')
    paths={key:ev.checked(c[key]) for key in required}
    selected_binding(c,u.read(paths['selection_complete']))
    if Path(c['model_dir'])/'config.json'!=paths['model_config']:raise ValueError('Official model config/path binding differs')
    needed={'reproduction/operations/qwen7b_warmup/'+name for name in
        ('hard_enrichment_train.py','probe_ablation_train.py','probe_ablation_loss.py','evaluate_frozen_validation.py',
         'model.py','packed_features.py','native_capture.py','download_model.py')}
    if not needed<=set(c['code_hashes']):raise ValueError('Missing full immutable training import bindings')
    for name,wanted in c['code_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or u.sha(repo/name)!=wanted:raise ValueError('Source drift '+name)
    if int(os.environ.get('WORLD_SIZE','1'))!=1 or torch.cuda.device_count()!=1:raise ValueError('One GPU per paired branch required')
    device=torch.device('cuda',0);prop=torch.cuda.get_device_properties(device)
    if 'H100' not in prop.name or prop.total_memory<75*1024**3:raise ValueError('Full-global-B48 graph requires >=75GiB H100')
    if os.environ.get('FLASH_ATTENTION_DETERMINISTIC')!='1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Deterministic FA/CUBLAS flags required')
    torch.cuda.set_device(device);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    from .model import Probe
    from .score_training_candidates import candidate_reader
    train=u.read(paths['bank'])['train'];original=u.read(paths['diagnostic_bank']);diagnostic=original['diagnostic'];plan=u.read(paths['plan'])
    source_groups={g['qid']:g for g in original['train']}
    for g in train:
        if g['qid'] not in source_groups or any(g[k]!=source_groups[g['qid']][k] for k in ('question_hash','trajectory_ids','labels')):
            raise ValueError('Own training qid/hash/IDs/EM differ from actual E4 source bank')
    del source_groups,original
    validation,validation_reader,validation_proof=ev.accepted_validation(c['validation_complete'])
    reader=None
    try:
        if len(diagnostic)!=512:raise ValueError('Original fixed diagnostic512 required')
        forbidden={g['qid'] for g in diagnostic+validation}
        hashes={g['question_hash'] for g in diagnostic}
        vbank=u.read(ev.checked(validation_proof['consumer_inputs']['groups']))['validation']
        hashes.update(g['question_hash'] for g in vbank)
        if len(forbidden)!=2560:raise ValueError('Diagnostic and fixed validation qids overlap')
        lookup=validate_plan(plan,train,forbidden,hashes)
        source=torch.load(paths['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
        source_ranks=verify_source(c,u.read(paths['source_receipt']),u.read(paths['source_config']),source)
        assets={key:c[key] for key in required};matched=dict(source_assets=assets,loss=LOSS,global_groups=48,
            update_budget=477,micro_groups=c['micro_groups'],lr=5e-6,ema_decay=.99,source_steps=SOURCE_STEPS,
            code_hashes=c['code_hashes'],variant=c['variant'],seed=c['seed'])
        binding=dict(config_sha256=u.sha(base/'config.json'),matched_binding_sha256=u.digest(matched),
            matched_binding=matched,source_checkpoint=c['source_checkpoint'],recipe=c['recipe'],plan_sha256=c['plan']['sha256'])
        u.atomic(base/'controller.json',dict(status='E4_PAIRED_CONTINUATION_BOUND',pid=os.getpid(),
            start_ticks=Path('/proc/self/stat').read_text().split()[21],
            job_id=os.environ['SLURM_JOB_ID'],variant=c['variant'],device=prop.name,source_steps=SOURCE_STEPS,
            source_all_ranks_archived=True,canonical_source_rng_rank=0,four_rank_bitexact_resume_claim=False,**binding))
        reader=candidate_reader(u.read(paths['features']),train+diagnostic,u.read(paths['feature_manifest_refs']))
        model=Probe(c['model_dir'],273,20).to(device);ema=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=torch.optim.Adam(model.parameters(),lr=5e-6,weight_decay=0)
        for sig in (signal.SIGTERM,signal.SIGUSR1,signal.SIGINT):signal.signal(sig,stop)
        if (base/'latest.json').exists():
            receipt=u.read(base/'latest.json');saved=torch.load(ev.checked(receipt),map_location='cpu',mmap=True,weights_only=False);cursor=saved['cursor']
            check_resume(saved,cursor,c,binding,u.read(base/'progress.json') if (base/'progress.json').exists() else None,
                u.read(base/'update-intent.json') if (base/'update-intent.json').exists() else None)
            u.restore_blob(saved,model,ema,optimizer,device)
            if u.state_hashes(model,ema,optimizer,device)!=saved['accepted_hashes']:raise ValueError('Restored complete state hash mismatch')
            source_ranks=saved['source_all_rank_states'];resume_blob=saved
        else:
            if any((base/name).exists() for name in ('progress.json','update-intent.json')):raise ValueError('Prior unsaved transaction/progress, refuse reset')
            u.restore_blob(source,model,ema,optimizer,device);verify_restored_source(source,model,ema,optimizer,device)
            cursor=0;resume_blob=source
        if any(g['lr']!=5e-6 or g['weight_decay']!=0 for g in optimizer.param_groups):raise ValueError('Restored inherited Adam recipe changed')
        if cursor<477:
            u.atomic(base/'update-intent.json',dict(next_cursor=cursor+1,state='ACTUAL_NEXT_UPDATE_REPLAY_IN_PROGRESS',epoch_time=time.time()))
            metrics,proof=replay_next(resume_blob,model,ema,optimizer,reader,lookup,plan['batches'][cursor],c,device)
            u.atomic(base/('next-update-replay-cursor%06d.json'%cursor),proof);cursor+=1
            u.save_checkpoint(base,blob(model,ema,optimizer,device,cursor,binding,source_ranks))
            row=dict(cursor=cursor,steps=u.steps(optimizer),budget=477,epoch_time=time.time(),**metrics)
            u.atomic(base/'progress.json',row)
            with (base/'curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
            print(json.dumps(row),flush=True)
        del resume_blob,source
        if 'saved' in locals():del saved
        while cursor<477 and not STOP:
            u.atomic(base/'update-intent.json',dict(next_cursor=cursor+1,state='OPTIMIZER_TRANSACTION_IN_PROGRESS',epoch_time=time.time()))
            metrics=u.run_update(model,ema,optimizer,reader,lookup,plan['batches'][cursor],c,device);cursor+=1
            row=dict(cursor=cursor,steps=u.steps(optimizer),budget=477,epoch_time=time.time(),
                max_gpu_allocated=torch.cuda.max_memory_allocated(device),**metrics)
            u.atomic(base/'progress.json',row)
            with (base/'curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
            print(json.dumps(row),flush=True)
            if cursor%c['save_every']==0 or STOP or cursor==477:u.save_checkpoint(base,blob(model,ema,optimizer,device,cursor,binding,source_ranks))
        if STOP:
            if u.read(base/'latest.json')['cursor']!=cursor:u.save_checkpoint(base,blob(model,ema,optimizer,device,cursor,binding,source_ranks))
            u.atomic(base/'interrupted.json',dict(status='FULL_ACCEPTED_CURSOR_SAVED',cursor=cursor,steps=u.steps(optimizer)));return
        accepted=u.read(base/'latest.json')
        require_final_checkpoint(accepted)
        before=u.state_hashes(model,ema,optimizer,device);rng=u.rng_state(device);buffers=u.buffers(ema)
        (base/'predictions').mkdir(exist_ok=True)
        try:
            metrics={name:ev.evaluate(ema,own_reader,groups,device,base/'predictions'/(name+'.jsonl.gz'),
                base/'evaluation-progress.json',name,micro_groups=2)
                for name,own_reader,groups in [('diagnostic512',reader,diagnostic),('fixed_validation_G5',validation_reader,validation)]}
        finally:u.restore_rng(rng,device);u.restore_buffers(ema,buffers)
        if u.state_hashes(model,ema,optimizer,device)!=before:raise ValueError('Evaluation altered accepted model/Adam/RNG')
        ev.checked(accepted)
        u.atomic(base/'metrics.json',dict(metrics=metrics,cursor=477,steps=14895,checkpoint=accepted,**binding))
        u.atomic(base/'complete.json',dict(status='E4_HARD_RANDOM477_UPDATES_AND_FIXED_G5_EVALUATION_COMPLETE',
            cursor=477,steps=14895,checkpoint=accepted,metrics=metrics,actor_updates=0,**binding))
    finally:
        validation_reader.close()
        if reader is not None:reader.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();base=ev.remote(a.base)
    with (base/'hard-fit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(base)
        except BaseException as error:
            u.atomic(base/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),epoch_time=time.time()));raise


if __name__=='__main__':main()
