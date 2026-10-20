"""Isolated E1→938 joint hidden-count experiment; no active runner patches.

Only explicit parent GO may submit the separately rendered job. Preparation
validates metadata, not a real GPU restore. Every actual save/evaluation keeps
both Adam clocks, both EMA models and the historical four-rank archive.
"""
import argparse,copy,fcntl,gzip,json,os,platform,shlex,signal,subprocess,sys,time,traceback
from contextlib import nullcontext
from pathlib import Path
from . import peer_append_train as a,direct_hidden_count_prepare as prep

TARGET=a.A/'direct-hidden-count30016-train-v1'
STOP=False
MILESTONES=(256,512,768,938)


def configured(prepared,publication,preparation_ref):
    c=copy.deepcopy(prepared)
    if (c.get('source_cursor')!=0 or c.get('source_adam_steps')!=10464
            or c.get('update_budget')!=938 or c.get('global_questions')!=32
            or c.get('fit_questions')!=30016 or c.get('fit_real_trajectories')!=300160
            or c.get('evaluation_cumulative_cursors')!=list(MILESTONES)
            or c.get('grouping')!=dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise')
            or c.get('recipe')!=dict(id='hidden-direct-count6-from-E1',trajectory_BCE_weight=1.,count_NLL_weight=1.,product_group_loss_weight=0.)):
        raise ValueError('Exact preregistered E1/B32×938/four-view direct count recipe required')
    c.update(source_commit=publication['commit'],code_hashes={n:r['sha256'] for n,r in publication['files'].items()},
        prepare_only=False,preparation_reference=preparation_ref,
        GPU_submission_requires_explicit_parent_GO=True,first_checkpoint_independent_CPU_gate=True,
        checkpoint_components=9,first_new_head_Adam=0,final_new_head_Adam=938)
    return c


def actual_inputs(c):
    from . import probe_g10_train as t
    groups,_,plan,bank=a.accepted_inputs(c)
    evaluations=a.validate_evaluations(c,groups)
    lookup=t.validate_plan(plan,groups,c)
    refs=a.read(a.checked(bank['outputs']['features-layer26.json']))
    if len(plan['batches'])!=938 or any(len(x)!=32 for x in plan['batches']):
        raise ValueError('Same full938 batches required; reference historical256 are not completed here')
    return groups,refs,plan,evaluations,lookup


def render(code,sha):
    return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-direct-count938
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
#SBATCH --output=@@BASE@@/logs/train-%j.log
#SBATCH --chdir=@@BASE@@
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.direct_hidden_count_train --action train --config-sha256 @@SHA@@
'''.replace('@@BASE@@',str(TARGET)).replace('@@CODE@@',shlex.quote(str(code))).replace('@@SHA@@',sha)


def prepare():
    prior=a.read(prep.TARGET/'preparation.json')
    if prior['status']!='HIDDEN_DIRECT_COUNT_PREPARED_ONLY_NO_GPU_TRAINING_SUBMITTED':raise ValueError('Actual CPU preparation required')
    qualified=a.read(prep.TARGET/'cpu-qualification.json')
    if qualified['tests']!=10 or qualified['GPU_used'] is not False:raise ValueError('Actual focused CPU API qualification required')
    pub=a.read(TARGET/'source-publication.json');root=Path(pub['code_root'])
    for n,v in pub['files'].items():a.checked(dict(path=str(root/n),**v))
    c=configured(a.read(a.checked(prior['config'])),pub,a.proof(prep.TARGET/'preparation.json'))
    groups,refs,plan,evaluations,_=actual_inputs(c)
    a.immutable(TARGET/'config.json',c);path=TARGET/'training.sbatch';text=render(root,a.proof(TARGET/'config.json')['sha256'])
    if path.exists() and path.read_text()!=text:raise ValueError('Immutable sbatch differs')
    if not path.exists():path.write_text(text)
    result=dict(status='JOINT_HIDDEN_COUNT938_ENTRY_PREPARED_NOT_GPU_QUALIFIED_OR_SUBMITTED',
        config=a.proof(TARGET/'config.json'),sbatch=a.proof(path),actual_questions=len(groups),actual_G10=300160,
        batches=len(plan['batches']),feature_manifests=len(refs),legacy_Adam=10464,new_head_Adam=0,
        final_legacy_Adam=11402,final_new_head_Adam=938,evaluation_cursors=list(MILESTONES),
        actual_E1_tensor_loaded=False,actual_GPU_next_update_still_required=True,final_pool_read=False)
    a.immutable(TARGET/'preparation.json',result);return result


def joint_update(legacy,ema,adam,head,head_ema,head_adam,reader,lookup,qids,c,device,epoch):
    from .direct_hidden_count_update import bounded_update
    grouping=dict(c['grouping']);grouping['epoch']=epoch
    return bounded_update(legacy,ema,adam,head,head_ema,head_adam,reader,[lookup[q] for q in qids],device,
        grouping_kwargs=grouping,micro_trajectories=c['micro_trajectories'],ema_decay=c['new_head']['EMA_decay'])


def score_fixed(legacy,head,reader,groups,device,destination,micro_questions=4,forward_adapter=None):
    """Label-blind count scoring; labels join only after q has been computed."""
    import torch
    from . import probe_ablation_train as u
    from .direct_hidden_count import native_hidden_forward
    forward_adapter=forward_adapter or native_hidden_forward
    legacy.eval();head.eval();rows=[]
    with torch.no_grad():
        for offset in range(0,len(groups),micro_questions):
            part=groups[offset:offset+micro_questions]
            if any(len(g['trajectory_ids'])!=5 or len(set(g['trajectory_ids']))!=5 for g in part):
                raise ValueError('Exactly the precommitted five own trajectories required')
            data=reader.get([tid for g in part for tid in g['trajectory_ids']],device)
            context=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
            with context:z,h=forward_adapter(legacy,data,head)
            # Same float32 small count-MLP arithmetic used by training loss.
            result=head.probabilities(h.reshape(len(part),5,head.embedding_dimension))
            counts=result['count_probabilities'].cpu().tolist();mixed=result['mixed_probability'].cpu().tolist()
            ps=z.reshape(len(part),5).sigmoid().cpu().tolist()
            for g,p,q,count in zip(part,ps,mixed,counts):
                if len(count)!=6 or abs(sum(count)-1)>1e-5 or abs(q-sum(count[1:5]))>1e-6:
                    raise ValueError('Direct count probability normalization differs')
                rows.append(dict(qid=g['qid'],trajectory_ids=g['trajectory_ids'],p_j=p,
                    count_probabilities=count,predicted_mixed=q,strict_em=g['labels'],score_kind='hidden_direct_count6'))
    if len(rows)!=len(groups):raise ValueError('Fixed evaluation coverage differs')
    destination=Path(destination);temporary=destination.with_suffix('.tmp')
    if destination.exists():raise ValueError('Refuse overwrite of fixed evaluation predictions')
    with gzip.open(temporary,'wt') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    temporary.replace(destination)
    chosen=sorted(rows,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512]
    return dict(pool=u.prediction_metrics(rows),selected512=u.prediction_metrics(chosen),
        predictions=a.proof(destination),score_kind='hidden_direct_count6',labels_used_for_scoring=False,
        evaluation_only=True,calibration_fitted=False)


def evaluate_at(cursor,legacy,ema,adam,head,head_ema,head_adam,device,evaluations,binding):
    from . import direct_hidden_checkpoint as cp,probe_ablation_train as u
    from .replay_feature_reader import ReplayFeatureReader
    base=TARGET/'evaluations'/('cursor-%06d'%cursor);base.mkdir(parents=True,exist_ok=True)
    path=base/'complete.json'
    if path.exists():
        done=a.read(path)
        if done['cursor']!=cursor or done['config_sha256']!=binding['config']['sha256'] or done['training_nine_components_unchanged'] is not True:
            raise ValueError('Existing fixed count evaluation lineage differs')
        a.checked(done['checkpoint'])
        for v in done['predictions'].values():a.checked(v)
        return done['metrics']
    checkpoint=a.read(TARGET/'latest.json')
    if checkpoint['cursor']!=cursor:raise ValueError('Current durable joint checkpoint must precede validation')
    before=cp.state_hashes(legacy,ema,adam,head,head_ema,head_adam,device)
    metrics={};predictions={}
    for e,frozen in evaluations:
        reader=ReplayFeatureReader(a.read(a.checked(e['features'])),26)
        rng=u.rng_state(device);buffers=u.buffers(ema);head_buffers=u.buffers(head_ema)
        try:metrics[e['name']]=score_fixed(ema,head_ema,reader,frozen,device,base/(e['name']+'.jsonl.gz'))
        finally:u.restore_rng(rng,device);u.restore_buffers(ema,buffers);u.restore_buffers(head_ema,head_buffers);reader.close()
        predictions[e['name']]=a.proof(base/(e['name']+'.jsonl.gz'))
    after=cp.state_hashes(legacy,ema,adam,head,head_ema,head_adam,device)
    if before!=after:raise ValueError('Evaluation changed joint raw/EMA/twoAdam/nativebuffers/RNG')
    a.immutable(path,dict(status='JOINT_COUNT_CHECKPOINT_FIRST_FIXED_HF_EVALUATION_ACCEPTED',cursor=cursor,
        legacy_steps=10464+cursor,new_head_steps=cursor,config_sha256=binding['config']['sha256'],
        checkpoint=checkpoint,metrics=metrics,predictions=predictions,before_nine_component_hashes=before,
        after_nine_component_hashes=after,training_nine_components_unchanged=True,
        actor_updates=0,extra_optimizer_updates=0,final_pool_read=False))
    return metrics


def audit_first():
    import torch
    from . import direct_hidden_checkpoint as cp
    deadline=time.time()+7000
    while not (TARGET/'progress.json').exists() or a.read(TARGET/'progress.json')['cursor']<1:
        if (TARGET/'failure.json').exists():raise ValueError('Joint trainer failed before CP1')
        if time.time()>deadline:raise ValueError('Bounded first joint checkpoint wait expired')
        time.sleep(1)
    p=TARGET/'checkpoints/cursor-000001.pt';saved=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
    replay=a.read(TARGET/'next-update-replay.json');verified=cp.verify_joint_file(p,expected_cursor=1);hashes=verified['accepted_hashes']
    if (saved['cursor']!=1 or saved['legacy']['steps']!=10465 or saved['head']['steps']!=1
            or saved['binding']['config']['sha256']!=a.proof(TARGET/'config.json')['sha256']
            or replay['status']!='PASSED' or replay['actual']!=replay['expected'] or hashes!=replay['actual']
            or hashes!=saved['accepted_hashes']):raise ValueError('Independent first joint fullstate reload differs')
    result=dict(status='JOINT_CP1_INDEPENDENT_FULL_SHA_NINE_COMPONENT_CPU_RELOAD_PASSED',checkpoint=a.proof(p),
        cursor=1,legacy_Adam=10465,new_head_Adam=1,component_hashes=hashes,CUDA_used=False)
    a.immutable(TARGET/'first-cp-independent-audit.json',result);return result


def run(expected_sha):
    import torch
    from . import probe_ablation_train as u,direct_hidden_checkpoint as cp
    from .direct_hidden_count import initialize_head,new_head_state,restore_new_head
    from .replay_feature_reader import ReplayFeatureReader
    from .model import Probe
    c=a.read(TARGET/'config.json');pub=a.read(TARGET/'source-publication.json')
    if a.proof(TARGET/'config.json')['sha256']!=expected_sha or c['prepare_only'] is not False:
        raise ValueError('Exact independent joint production config required')
    for n,v in pub['files'].items():a.checked(dict(path=str(Path(pub['code_root'])/n),**v))
    if (int(os.environ.get('WORLD_SIZE','1'))!=1 or torch.cuda.device_count()!=1
            or os.environ.get('FLASH_ATTENTION_DETERMINISTIC')!='1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'):
        raise ValueError('Owned deterministic single-GPU allocation required')
    os.environ['QWEN7B_QUESTION_MAX_TOKENS']='273';os.environ['SEARCH_G1_B_PREFIX_TOKENS']='20'
    device=torch.device('cuda',0);torch.cuda.set_device(device);props=torch.cuda.get_device_properties(device)
    if 'H100' not in props.name or props.total_memory<75*1024**3:raise ValueError('Dedicated >=75GiB H100 required')
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    if Path(c['model_dir'])/'config.json'!=a.checked(c['model_config']):raise ValueError('Official model config binding differs')
    groups,refs,plan,evaluations,lookup=actual_inputs(c)
    source=torch.load(a.checked(c['source_checkpoint']),map_location='cpu',mmap=True,weights_only=False)
    if (c['source_checkpoint']['sha256']!=u.SOURCE_SHA or source['steps']!=10464 or source['epoch']!=1
            or source['completed_epoch'] is not True or source['cursor']!=0 or source['actor_updates']!=0
            or source['config_sha256']!=c['source_config']['sha256'] or source['bank_sha256']!=c['source_bank']['sha256']
            or len(source['rank_states'])!=4 or {s['rank'] for s in source['rank_states']}!={0,1,2,3}):
        raise ValueError('Exact original E1 raw/EMA/Adam10464/four-rank archive required')
    archive=copy.deepcopy(source['rank_states']);archive_hashes=[u.rng_hash(r['rng']) for r in archive]
    legacy=Probe(c['model_dir'],273,20).to(device);ema=copy.deepcopy(legacy).eval().requires_grad_(False)
    adam=torch.optim.Adam(legacy.parameters(),lr=5e-6,weight_decay=0)
    u.restore_blob(source,legacy,ema,adam,device)
    for key,current in [('model_sha256',u.model_hash(legacy)),('ema_sha256',u.model_hash(ema)),('optimizer_sha256',u.optimizer_hash(adam))]:
        if any(s[key]!=current for s in archive):raise ValueError('Inherited E1 state differs: '+key)
    before=u.state_hashes(legacy,ema,adam,device)
    head,head_ema,head_adam=initialize_head(seed=c['new_head']['initialization_seed'],device=device)
    if before!=u.state_hashes(legacy,ema,adam,device):raise ValueError('New head initialization changed inherited state/RNG')
    initial_head=copy.deepcopy(new_head_state(head,head_ema,head_adam))
    binding=dict(config=a.proof(TARGET/'config.json'),source_checkpoint=c['source_checkpoint'],source_config=c['source_config'],
        source_bank=c['source_bank'],g10_complete=c['g10_complete'],plan=c['plan'],plan_sha256=c['plan']['sha256'],recipe=c['recipe'])
    reader=ReplayFeatureReader(refs,26);needed={tid for g in groups for tid in g['trajectory_ids']}
    if set(reader.records)!=needed:raise ValueError('Actual 30016×10 HF feature coverage differs')
    a.atomic(TARGET/'controller.json',dict(pid=os.getpid(),job_id=os.environ.get('SLURM_JOB_ID'),status='JOINT_E1_INPUTS_ACCEPTED',
        source_legacy_Adam=10464,source_new_head_Adam=0,device=props.name,source_world_size=4,current_world_size=1))
    for sig in (signal.SIGTERM,signal.SIGUSR1,signal.SIGINT):signal.signal(sig,request_stop)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    log=(TARGET/'first-cp-independent-audit.log').open('ab',0)
    child=subprocess.Popen([sys.executable,'-u','-m','reproduction.operations.qwen7b_warmup.direct_hidden_count_train','--action','audit-first'],stdout=log,stderr=subprocess.STDOUT,env=env)
    a.atomic(TARGET/'first-cp-audit-controller.json',dict(pid=child.pid,job_id=os.environ.get('SLURM_JOB_ID'),inside_allocated_job=True))
    cursor=0
    def hashes():return cp.state_hashes(legacy,ema,adam,head,head_ema,head_adam,device)
    def save():
        return cp.save_joint(TARGET,cp.joint_blob(legacy,ema,adam,head,head_ema,head_adam,device,cursor,binding,archive,
            dict(cursor=cursor,plan_sha256=c['plan']['sha256'])))
    def update():return joint_update(legacy,ema,adam,head,head_ema,head_adam,reader,lookup,plan['batches'][cursor],c,device,plan['batch_epochs'][cursor])
    def accept(metrics):
        row=dict(cursor=cursor,steps=u.steps(adam),head_steps=cursor,time=time.time(),**metrics);a.atomic(TARGET/'progress.json',row)
        with (TARGET/'curve.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
        print(json.dumps(row,allow_nan=False),flush=True)
    try:
        if (TARGET/'latest.json').exists():
            saved=torch.load(a.checked(a.read(TARGET/'latest.json')),map_location='cpu',mmap=True,weights_only=False);cursor=saved['cursor']
            if any(a.read(TARGET/n)[k]>cursor for n,k in [('progress.json','cursor'),('update-intent.json','next_cursor')] if (TARGET/n).exists()):
                raise ValueError('Unresolved later joint update; no silent rollback')
            cp.restore_joint(saved,legacy,ema,adam,head,head_ema,head_adam,device,expected_binding=binding,expected_cursor=cursor)
            archive=saved['legacy']['source_all_rank_states'];del saved
        else:
            if any((TARGET/n).exists() for n in ('progress.json','update-intent.json')):raise ValueError('Unresolved joint state; no reset')
            a.atomic(TARGET/'update-intent.json',dict(next_cursor=1,state='FIRST_JOINT_UPDATE_DOUBLE_REPLAY'))
            update();expected=hashes();u.restore_blob(source,legacy,ema,adam,device);restore_new_head(initial_head,head,head_ema,head_adam,device)
            metrics=update();actual=hashes()
            a.atomic(TARGET/'next-update-replay.json',dict(status='PASSED' if expected==actual else 'FAILED',expected=expected,actual=actual,
                source_legacy_Adam=10464,source_new_head_Adam=0,candidate_legacy_Adam=10465,candidate_new_head_Adam=1))
            if expected!=actual:raise ValueError('Actual joint GPU first next-update replay differs')
            cursor=1;save();accept(metrics)
        del source,initial_head
        if [u.rng_hash(r['rng']) for r in archive]!=archive_hashes:raise ValueError('Original four-rank RNG archive changed')
        if child.wait(timeout=180)!=0 or not (TARGET/'first-cp-independent-audit.json').exists():raise ValueError('Independent joint CP1 hardgate failed')
        for milestone in MILESTONES:
            if milestone<cursor and not (TARGET/'evaluations'/('cursor-%06d'%milestone)/'complete.json').exists():
                raise ValueError('Earlier required fixed evaluation missing; refuse silent skip')
            if milestone==cursor:evaluate_at(cursor,legacy,ema,adam,head,head_ema,head_adam,device,evaluations,binding)
        while cursor<938 and not STOP:
            a.atomic(TARGET/'update-intent.json',dict(next_cursor=cursor+1,state='JOINT_TWO_ADAM_TRANSACTION_IN_PROGRESS'))
            metrics=update();cursor+=1;accept(metrics)
            if cursor%c['save_every']==0 or STOP or cursor in MILESTONES:save()
            if cursor in MILESTONES:evaluate_at(cursor,legacy,ema,adam,head,head_ema,head_adam,device,evaluations,binding)
        if STOP:
            if a.read(TARGET/'latest.json')['cursor']!=cursor:save()
            a.atomic(TARGET/'interrupted.json',dict(status='EXACT_JOINT_ACCEPTED_CURSOR_SAVED',cursor=cursor));return
        if cursor!=938 or u.steps(adam)!=11402:raise ValueError('Exact938 joint budget required')
        a.immutable(TARGET/'complete.json',dict(status='JOINT_HIDDEN_DIRECT_COUNT938_AND_FIXED_HF_VALIDATION_COMPLETE',cursor=938,
            legacy_steps=11402,new_head_steps=938,checkpoint=a.read(TARGET/'latest.json'),
            evaluation_receipts={str(i):a.proof(TARGET/'evaluations'/('cursor-%06d'%i)/'complete.json') for i in MILESTONES},
            config_sha256=expected_sha,actor_updates=0,legacy_optimizer_reset=False,final_pool_read=False))
    finally:
        reader.close();log.close()
        if child.poll() is None:child.terminate();child.wait(timeout=30)


def request_stop(_sig,_frame):
    global STOP
    STOP=True


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','train','audit-first'),required=True);p.add_argument('--config-sha256');args=p.parse_args()
    if platform.system()!='Linux':p.error('Actual actions remote only')
    if args.action=='train' and (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES')):p.error('Owned GPU allocation required')
    TARGET.mkdir(parents=True,exist_ok=True);(TARGET/'logs').mkdir(exist_ok=True)
    with (TARGET/('first-cp-audit.lock' if args.action=='audit-first' else 'joint-training.lock')).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:print(json.dumps(prepare() if args.action=='prepare' else audit_first() if args.action=='audit-first' else run(args.config_sha256)),flush=True)
        except BaseException as error:
            a.atomic(TARGET/('first-cp-audit-failure.json' if args.action=='audit-first' else 'failure.json'),
                dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),job_id=os.environ.get('SLURM_JOB_ID'),time=time.time()));raise

if __name__=='__main__':main()
