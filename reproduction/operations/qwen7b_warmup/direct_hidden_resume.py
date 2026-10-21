"""Independent exact-state continuation after a normal saved USR1 boundary.

The live501 runner and its config are unchanged. A new allocation first accepts
one actual jointly replayed update from the full interrupted state, then the
same501 runner continues its existing resume branch to the original938 budget.
"""
import copy,fcntl,json,os,shlex,subprocess,sys,time,traceback
from pathlib import Path
from . import peer_append_train as a,direct_hidden_count_train as r

BASE=a.A/'direct-hidden-count30016-train-v2'
ATTEMPT=BASE/'continuations/attempt-0001'
SOURCE_JOB='743619'
RUNTIME='501cf80478b77f68c3e13a96d4659e3da0971b2e'
CONFIG_SHA='3e910b12c0ff52eac70e61566c6135e2ec188bb47ec64ad79524abbd7a657526'


def binding(c,config):
    return dict(config=config,source_checkpoint=c['source_checkpoint'],source_config=c['source_config'],
        source_bank=c['source_bank'],g10_complete=c['g10_complete'],plan=c['plan'],plan_sha256=c['plan']['sha256'],recipe=c['recipe'])


def validate_handoff(config,interrupted,latest,progress,intent,verified,terminal,complete_exists=False):
    if complete_exists:raise ValueError('Full938 already complete; no continuation permitted')
    if not any(s in terminal for s in ('|COMPLETED|0:0|','|TIMEOUT|')):raise ValueError('Original writer must be terminal before continuation')
    cursor=latest['cursor']
    if (type(cursor) is not int or not 1<=cursor<938 or config['source_cursor']!=0
            or config['source_adam_steps']!=10464 or config['update_budget']!=938):raise ValueError('Exact original E1 branch budget and incomplete accepted cursor required')
    if (interrupted.get('status')!='EXACT_JOINT_ACCEPTED_CURSOR_SAVED' or interrupted['cursor']!=cursor
            or progress['cursor']!=cursor or progress['steps']!=10464+cursor or progress['head_steps']!=cursor
            or intent['next_cursor']>cursor):raise ValueError('Missing full current state or unresolved later transaction; never roll back')
    if (verified['cursor']!=cursor or verified['legacy_Adam_steps']!=10464+cursor or verified['new_head_Adam_steps']!=cursor
            or verified['data_cursor']!=dict(cursor=cursor,plan_sha256=config['plan']['sha256'])
            or verified['accepted_hashes']!=latest['accepted_hashes']
            or verified['source_all_four_RNG_preserved'] is not True):raise ValueError('Full nine-state clocks/data cursor/archive differ')
    return dict(source_cursor=cursor,source_legacy_Adam=10464+cursor,source_new_head_Adam=cursor,
        remaining_updates=938-cursor,target_cursor=938,target_legacy_Adam=11402,target_new_head_Adam=938,
        current_source_world_size=1,current_world_size=1,hardware_topology_changed=False,
        original_branch_source_world_size=4,original_four_rank_archive_preserved=True,
        old_optimizer_reset=False,new_head_optimizer_reset=False,actor_updates=0,final_pool_read=False)


def render(code,operator,operator_sha,source_sha):
    worker="import importlib.util,pathlib,hashlib; p=pathlib.Path(%r); assert hashlib.sha256(p.read_bytes()).hexdigest()==%r; s=importlib.util.spec_from_file_location('reproduction.operations.qwen7b_warmup.external_joint_resume',str(p)); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.launch(%r)"%(str(operator),operator_sha,source_sha)
    return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-direct-count938-r1
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=@@BASE@@/logs/resume-%j.log
#SBATCH --chdir=@@BASE@@
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -c @@WORKER@@
'''.replace('@@BASE@@',str(BASE)).replace('@@CODE@@',shlex.quote(str(code))).replace('@@WORKER@@',shlex.quote(worker))


def prepare(operator_path,operator_sha):
    from . import direct_hidden_checkpoint as cp
    if a.proof(BASE/'config.json')['sha256']!=CONFIG_SHA:raise ValueError('Original501 config changed')
    pub=a.read(BASE/'source-publication.json')
    if pub['commit']!=RUNTIME:raise ValueError('Original immutable501 source changed')
    with (BASE/'joint-training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (BASE/'failure.json').exists() or (BASE/'first-cp-audit-failure.json').exists():raise ValueError('Inspect actual failure before continuation')
        terminal=subprocess.check_output(['sacct','-j',SOURCE_JOB,'-X','-n','-P','--format=JobIDRaw,State,ExitCode,Elapsed']).decode()
        # Original old job identity remains historical and is never overwritten.
        if (BASE/'job.id').read_text().strip()!=SOURCE_JOB:raise ValueError('Original source job identity changed')
        source_queue=subprocess.check_output(['squeue','-j',SOURCE_JOB,'-h','-o','%i|%T']).decode()
        if source_queue.strip():raise ValueError('Original writer is still active; no duplicate allocation')
        c=a.read(BASE/'config.json');config=a.proof(BASE/'config.json');latest=a.read(BASE/'latest.json')
        checkpoint=a.checked(latest)
        verified=cp.verify_joint_file(checkpoint,expected_binding=binding(c,config),expected_cursor=latest['cursor'],expected_sha256=latest['sha256'])
        summary=validate_handoff(c,a.read(BASE/'interrupted.json'),latest,a.read(BASE/'progress.json'),a.read(BASE/'update-intent.json'),verified,terminal,(BASE/'complete.json').exists())
        for milestone in r.MILESTONES:
            if milestone<=summary['source_cursor']:
                done=a.read(BASE/'evaluations'/('cursor-%06d'%milestone)/'complete.json')
                if done['cursor']!=milestone or done['config_sha256']!=CONFIG_SHA or not done['training_nine_components_unchanged']:
                    raise ValueError('A completed prefix evaluation is missing or differs')
                for ref in done['predictions'].values():a.checked(ref)
        # Every original accepted transaction must remain present, ordered once.
        curve=[json.loads(line) for line in (BASE/'curve.jsonl').read_text().splitlines() if line]
        if ([v['cursor'] for v in curve]!=list(range(1,summary['source_cursor']+1))
                or any(v['steps']!=10464+v['cursor'] or v['head_steps']!=v['cursor'] for v in curve)):
            raise ValueError('Accepted prefix history has gaps/repetitions; do not invent missing iterations')
        ATTEMPT.mkdir(parents=True,exist_ok=True)
        script=render(pub['code_root'],operator_path,operator_sha,latest['sha256']);path=ATTEMPT/'resume.sbatch'
        if path.exists() and path.read_text()!=script:raise ValueError('Frozen continuation sbatch changed')
        if not path.exists():path.write_text(script)
        result=dict(status='ACTUAL_INTERRUPTED_JOINT_STATE_VERIFIED_CONTINUATION_PREPARED_NOT_SUBMITTED',
            original_job_id=SOURCE_JOB,original_terminal=terminal,source_checkpoint=verified,config=config,
            runtime_commit=RUNTIME,code_root=pub['code_root'],operator=dict(path=str(operator_path),sha256=operator_sha),
            sbatch=a.proof(path),original_interrupted=a.proof(BASE/'interrupted.json'),
            highest_accepted_progress=a.proof(BASE/'progress.json'),highest_intent=a.proof(BASE/'update-intent.json'),**summary)
        a.immutable(ATTEMPT/'preparation.json',result);return result


def resume_training(source_sha):
    """One accepted replay-qualified continuation transaction, then501 loop."""
    import torch
    from . import direct_hidden_checkpoint as cp,probe_ablation_train as u
    from .direct_hidden_count import initialize_head
    from .replay_feature_reader import ReplayFeatureReader
    from .model import Probe
    if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('WORLD_SIZE','1'))!=1 or torch.cuda.device_count()!=1:
        raise ValueError('Owned one-GPU continuation allocation required')
    if os.environ.get('FLASH_ATTENTION_DETERMINISTIC')!='1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise ValueError('Original deterministic runtime flags required')
    with (BASE/'joint-training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepared=a.read(ATTEMPT/'preparation.json');latest=a.read(BASE/'latest.json')
        if latest['sha256']!=source_sha or prepared['source_checkpoint']['sha256']!=source_sha:raise ValueError('Exact accepted handoff checkpoint changed')
        if a.proof(BASE/'config.json')['sha256']!=CONFIG_SHA:raise ValueError('Original scientific config changed')
        c=a.read(BASE/'config.json');device=torch.device('cuda',0);torch.cuda.set_device(device);props=torch.cuda.get_device_properties(device)
        if 'H100' not in props.name or props.total_memory<75*1024**3:raise ValueError('Original H100 hardware class required')
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
        os.environ['QWEN7B_QUESTION_MAX_TOKENS']='273';os.environ['SEARCH_G1_B_PREFIX_TOKENS']='20'
        groups,refs,plan,_,lookup=r.actual_inputs(c);cursor=prepared['source_cursor']
        source=torch.load(a.checked(latest),map_location='cpu',mmap=True,weights_only=False)
        legacy=Probe(c['model_dir'],273,20).to(device);ema=copy.deepcopy(legacy).eval().requires_grad_(False)
        adam=torch.optim.Adam(legacy.parameters(),lr=5e-6,weight_decay=0)
        head,head_ema,head_adam=initialize_head(seed=c['new_head']['initialization_seed'],device=device)
        expected_binding=binding(c,a.proof(BASE/'config.json'))
        cp.restore_joint(source,legacy,ema,adam,head,head_ema,head_adam,device,expected_binding=expected_binding,expected_cursor=cursor)
        reader=ReplayFeatureReader(refs,26)
        a.atomic(ATTEMPT/'controller.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],phase='FULL_JOINT_RESUME_FIRST_UPDATE_REPLAY',source_cursor=cursor,source_checkpoint=prepared['source_checkpoint'],device=props.name))
        try:
            def update():return r.joint_update(legacy,ema,adam,head,head_ema,head_adam,reader,lookup,plan['batches'][cursor],c,device,plan['batch_epochs'][cursor])
            a.atomic(BASE/'update-intent.json',dict(next_cursor=cursor+1,state='FULL_JOINT_CONTINUATION_NEXT_UPDATE_DOUBLE_REPLAY'))
            update();expected=cp.state_hashes(legacy,ema,adam,head,head_ema,head_adam,device)
            cp.restore_joint(source,legacy,ema,adam,head,head_ema,head_adam,device,expected_binding=expected_binding,expected_cursor=cursor)
            metrics=update();actual=cp.state_hashes(legacy,ema,adam,head,head_ema,head_adam,device)
            replay=dict(status='PASSED' if actual==expected else 'FAILED',source_cursor=cursor,candidate_cursor=cursor+1,source_checkpoint=prepared['source_checkpoint'],expected=expected,actual=actual,current_source_world_size=1,current_world_size=1,hardware_topology_changed=False)
            a.immutable(ATTEMPT/'next-update-replay.json',replay)
            if actual!=expected:raise ValueError('Actual continuation next-update nine-state replay differs')
            cursor+=1
            row=dict(cursor=cursor,steps=u.steps(adam),head_steps=cursor,time=time.time(),**metrics)
            blob=cp.joint_blob(legacy,ema,adam,head,head_ema,head_adam,device,cursor,expected_binding,source['legacy']['source_all_rank_states'],dict(cursor=cursor,plan_sha256=c['plan']['sha256']))
            blob['continuation_first_transaction']=row;blob['continuation_source_checkpoint']=prepared['source_checkpoint']
            receipt=cp.save_joint(BASE,blob)
            # An independent CPU process must read the actual persisted9state.
            code="from reproduction.operations.qwen7b_warmup import direct_hidden_checkpoint as cp; import json; v=cp.verify_joint_file(%r,expected_cursor=%d,expected_sha256=%r); print(json.dumps(v))"%(receipt['path'],cursor,receipt['sha256'])
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            check=subprocess.run([sys.executable,'-u','-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,timeout=180)
            if check.returncode:raise ValueError('Independent continuation firstCP CPU process failed: '+check.stderr.decode()[-1000:])
            cpu=json.loads(check.stdout)
            if cpu['accepted_hashes']!=actual:raise ValueError('Independent continuation firstCP nine-state reload differs')
            a.immutable(ATTEMPT/'first-cp-independent-audit.json',cpu)
            a.atomic(BASE/'progress.json',row)
            with (BASE/'curve.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
            a.immutable(ATTEMPT/'handoff-complete.json',dict(status='CONTINUATION_FIRST_NEW_UPDATE_REPLAY_AND_INDEPENDENT_FULLCP_ACCEPTED',source_cursor=prepared['source_cursor'],accepted_cursor=cursor,checkpoint=receipt,independent_CPU_audit=a.proof(ATTEMPT/'first-cp-independent-audit.json'),actual_optimizer_updates_added=1,legacy_Adam=u.steps(adam),new_head_Adam=cursor))
        finally:reader.close()
        del source,legacy,ema,adam,head,head_ema,head_adam,blob,reader,groups,refs,plan,lookup
        import gc
        gc.collect();torch.cuda.empty_cache()
        # The exact unchanged runner now loads accepted latest cursor+1. It
        # performs no optimizer update while constructing/restoring its modules.
        r.run(CONFIG_SHA)


def launch(source_sha):
    try:return resume_training(source_sha)
    except BaseException as error:
        a.atomic(ATTEMPT/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            job_id=os.environ.get('SLURM_JOB_ID'),source_checkpoint_sha256=source_sha,time=time.time()))
        raise
