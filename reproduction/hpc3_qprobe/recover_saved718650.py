"""Once-only native4 continuation of actual saved actor/head5; never reset0."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from reproduction.hpc3_qprobe.control import publish,prepare,link
from reproduction.hpc3_qprobe.formal_audit_failure718650 import ROOT,BASE,JOB,RUN,SOURCE,CHECKPOINT,metadata
from search_g1.qprobe_features import sha_file
from search_g1.qprobe_saved_checkpoint import build_source,read_source,SAVED_STATUS,SAVED_KIND

TARGET=ROOT.parent/'qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014'
SOURCE_RECEIPT=BASE/'failed-job-718650/saved-checkpoint-source.json'
CONTROL=ROOT/'recovery718650-control'


def need(ok,message):
    if not ok:raise ValueError('RECOVERY718650_'+message)


def verified_source(commit):
    need(os.getuid()==14090 and len(commit)==40 and all(c in '0123456789abcdef' for c in commit),
         'ACTUAL_OWNER_SOURCE_REQUIRED')
    source=ROOT/'source'/commit
    proof=json.loads((ROOT/'preparation'/('source-'+commit+'-verification.json')).read_text())
    need(proof['status']=='PUBLISHED_SOURCE_FULL_SHA_PASS' and
         proof['commit']==proof['published_remote_sha']==commit,'LATEST_REPAIR_NOT_PUBLISHED')
    for name,sha in proof['files'].items():need(sha_file(source/name)==sha,'IMMUTABLE_SOURCE_CHANGED_'+name)
    return source,proof


def existing_source():
    need(SOURCE_RECEIPT.is_file(),'SAVED_SOURCE_RECEIPT_MISSING')
    _,saved,cp,base,parent=read_source(SOURCE_RECEIPT)
    need(saved['status']==SAVED_STATUS and saved['step']==5 and saved['world_size']==4 and
         saved['run_id']==RUN and saved['source_base']==str(BASE) and saved['checkpoint']==str(CHECKPOINT) and
         base==BASE and cp==CHECKPOINT and parent['source_commit']==SOURCE,
         'SOURCE_COUNTERS_OR_TYPE_DIFFER')
    return saved


def fresh(commit):
    actual=metadata();source,proof=verified_source(commit)
    intent=ROOT/'preparation/recovery718650-prepare-intent.json'
    need(not intent.exists() and not TARGET.exists(),'PREPARATION_ALREADY_ATTEMPTED_INSPECT')
    if SOURCE_RECEIPT.exists():existing_source()
    else:build_source(SOURCE_RECEIPT)
    os.umask(0o007)
    publish(intent,dict(commit=commit,target=str(TARGET),failed_job=JOB,
        source_receipt=str(SOURCE_RECEIPT),source_receipt_sha256=sha_file(SOURCE_RECEIPT),
        actual_failure=actual,checked_unix=time.time()))
    TARGET.mkdir(mode=0o770);prepare(TARGET)
    link(TARGET/'repo',source);(TARGET/'source').mkdir();link(TARGET/'source'/commit,source)
    link(TARGET/'question-features',ROOT/'question-features')
    for name in ('features-ready.json','official-model-manifest.json'):
        link(TARGET/'preparation'/name,ROOT/'preparation'/name)
    policy=json.loads((BASE/'preparation/personal-storage-authorization.json').read_text())
    need(policy['budget_bytes']==2*10**12 and policy['scope']=='jxu957_native4_qprobe_only',
         'ACTUAL_STORAGE_AUTHORIZATION_DIFFERS')
    publish(TARGET/'preparation/personal-storage-authorization.json',policy)
    publish(TARGET/'preparation/checkpoint-restoration.json',dict(target=str(TARGET),
        source_receipt=str(SOURCE_RECEIPT),source_receipt_sha256=sha_file(SOURCE_RECEIPT),
        kind=existing_source()['kind'],completed_outer=5,actual_adam_writes_per_rank=100,
        native4_readback_required=True,no_teacher_regeneration=True,no_prefix_retraining=True,
        historical_fixed5_reused=True,post_evaluation_rng_available=False,
        bitwise_continuation_from_failed_post_evaluation=False))
    publish(TARGET/'preparation/recovery-source-binding.json',dict(source_commit=commit,source=str(source),
        source_files=len(proof['files']),failed_job=JOB,completed_prefix_outer=5,
        next_training_outer=6,total_training_outers=200,training_world_size=4,
        source_receipt_sha256=sha_file(SOURCE_RECEIPT),feature_cache_reused=True,
        frozen_teacher_questions_reused=5120,frozen_teacher_trajectories_reused=25600,
        native_restore_accepted=False,checked_unix=time.time()))
    return dict(status='ACTUAL_SAVED5_NATIVE4_RECOVERY_PREPARED',target=str(TARGET),commit=commit,
        source_receipt=str(SOURCE_RECEIPT),actual_cuda_restore_accepted=False)


def preflight(commit):
    """Resolve the actual four-rank recipe without creating the live manifest."""
    import copy
    from reproduction.common8.source_files import runtime_files
    from reproduction.common8.configuration import resolved
    from search_g1.qprobe_checkpoint import bind_restore
    from search_g1.full_state_restore import from_config
    source,_=verified_source(commit);existing_source()
    parent=json.loads((BASE/'logs/run-manifest.json').read_text())
    with tempfile.TemporaryDirectory(prefix='config-preflight-',dir=str(TARGET/'tmp')) as temporary:
        base=Path(temporary).resolve();(base/'logs').mkdir()
        run=copy.deepcopy(parent);run['question_probe']=None
        run.update(source_commit=commit,source_sha256={str(p.relative_to(source)):sha_file(p) for p in runtime_files(source)},
                   checkpoint_dir=str(base/'checkpoints/search-g1-common8'))
        bind_restore(run,SOURCE_RECEIPT)
        (base/'logs/run-manifest.json').write_text(json.dumps(run))
        _,cfg,args=resolved(base);actual_source=from_config(cfg)
        need(actual_source['kind']==SAVED_KIND and actual_source['run_id']==RUN and
             cfg.trainer.full_state_restore_receipt==str(SOURCE_RECEIPT),'RESOLVED_RESTORE_DISPATCH_DIFFERS')
        need(cfg.trainer.n_gpus_per_node==4 and cfg.data.train_batch_size==512 and
             cfg.actor_rollout_ref.rollout.n_agent==5 and cfg.actor_rollout_ref.actor.ppo_micro_batch_size==16 and
             cfg.actor_rollout_ref.actor.ppo_mini_batch_size==128 and cfg.trainer.total_training_steps==201 and
             cfg.trainer.val_before_train is False and run['full_state_restore']['completed_outer']==5 and
             run['question_probe']['teacher_warmup_cache']==parent['question_probe']['teacher_warmup_cache'] and
             all(run['question_probe'][key]==value for key,value in (
                 ('candidate_questions',5120),('exploration_questions',64),('selected_questions',512),('responses_per_question',5))),
             'ACTUAL_HYDRA_NATIVE4_OR_TEACHER_BINDING_DIFFERS')
        result=dict(status='ACTUAL_REMOTE_HYDRA_SAVED5_NATIVE4_CONFIGURATION_PASS',source_commit=commit,
            training_world_size=4,actor_local_micro=4,actor_global_micro=16,mini128=True,
            candidate5120_explore64_select512_g5=True,completed_outer=5,next_outer=6,
            teacher_warmup_cache=run['question_probe']['teacher_warmup_cache'],
            source_receipt_sha256=sha_file(SOURCE_RECEIPT),argument_count=len(args),
            live_manifest_created=False,model_reload=False,gpu_allocation=False,checked_unix=time.time())
    publish(TARGET/'preparation/configuration-preflight.json',result)
    return result


def start_prepare(commit,recover_failed=False):
    metadata();source,_=verified_source(commit)
    prior=None
    if recover_failed:
        original=ROOT/'recovery718650-control'
        prior_claim=json.loads((original/'claim.json').read_text())
        prior_result=json.loads((original/'result.json').read_text())
        process=json.loads((original/'process.json').read_text())
        need(prior_claim['commit']=='08faa0f81eb2141280355e33983a3aa45372fde2' and
             prior_claim['failed_job']==JOB and prior_result['exit_code']==1 and
             prior_result['error']=="ValueError('QPROBE_SAVED_CHECKPOINT_ORIGINAL_COMPLETE_OR_SAVED_STATE_BINDING_DIFFERS')" and
             all(prior_result['birth'][key]==process['birth'][key] for key in ('pid','uid','sid','start_ticks')),
             'GENUINE_FAILED_PREPARATION_REQUIRED')
        from reproduction.a800.owned_processes import identity
        try:actual=identity(process['pid'])
        except (FileNotFoundError,ProcessLookupError):actual=None
        need(actual is None or actual['state']=='Z' or
             any(actual[key]!=process['birth'][key] for key in ('pid','uid','sid','start_ticks')),
             'PRIOR_PREPARATION_STILL_LIVE')
        existing_source()
        prior=dict(control=str(original),result_sha256=sha_file(original/'result.json'),
                   process_sha256=sha_file(original/'process.json'),exit_code=1)
    need(not CONTROL.exists() and not TARGET.exists() and
         not (ROOT/'preparation/recovery718650-prepare-intent.json').exists(),'EXISTING_PREPARATION_OUTCOME_INSPECT')
    CONTROL.mkdir(mode=0o770)
    command=[str(BASE/'env/bin/python'),'-m','reproduction.hpc3_qprobe.recover_saved718650',
             'supervise-prepare','--commit',commit]
    if recover_failed:command.append('--recover-failed-preparation')
    publish(CONTROL/'claim.json',dict(commit=commit,command=command,cwd=str(source),target=str(TARGET),
        failed_job=JOB,recovery_of_actual_cpu_failure=prior,checked_unix=time.time(),training_submission=False))
    env=dict(os.environ,QPROBE_SOURCE=str(source),CUDA_VISIBLE_DEVICES='',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    with (CONTROL/'output.log').open('xb') as stream:
        child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=stream,
            stderr=subprocess.STDOUT,start_new_session=True)
    from reproduction.a800.owned_processes import identity
    birth=identity(child.pid);need(birth['uid']==14090 and birth['pid']==birth['sid'],'CPU_OWNER_BIRTH_DIFFERS')
    result=dict(pid=child.pid,birth=birth,commit=commit,checked_unix=time.time(),control=str(CONTROL))
    publish(CONTROL/'process.json',result);return result


def supervise_prepare(commit):
    from reproduction.a800.owned_processes import identity
    claim=json.loads((CONTROL/'claim.json').read_text())
    need(claim['commit']==commit and Path(claim['cwd']).resolve()==Path.cwd().resolve(),'PREPARE_WORKER_SOURCE_DIFFERS')
    birth=identity(os.getpid());code=0;result=None;error=None
    try:result=fresh(commit)
    except Exception as exc:
        import traceback
        traceback.print_exc();code=1;error=repr(exc)
    publish(CONTROL/'result.json',dict(exit_code=code,result=result,error=error,birth=birth,checked_unix=time.time()))
    return code


def submit(commit):
    actual=metadata();source,proof=verified_source(commit);saved=existing_source()
    binding=json.loads((TARGET/'preparation/recovery-source-binding.json').read_text())
    need(binding['source_commit']==commit and binding['source_receipt_sha256']==sha_file(SOURCE_RECEIPT) and
         (TARGET/'repo').resolve()==source and (TARGET/'source'/commit).resolve()==source,
         'PREPARED_SOURCE_BINDING_CHANGED')
    preflight_receipt=json.loads((TARGET/'preparation/configuration-preflight.json').read_text())
    need(preflight_receipt['status']=='ACTUAL_REMOTE_HYDRA_SAVED5_NATIVE4_CONFIGURATION_PASS' and
         preflight_receipt['source_commit']==commit and
         preflight_receipt['source_receipt_sha256']==sha_file(SOURCE_RECEIPT),'REMOTE_CONFIGURATION_PREFLIGHT_REQUIRED')
    claim=TARGET/'preparation/training-submit-claim.json';receipt=TARGET/'preparation/training-submission.json'
    need(not claim.exists() and not receipt.exists(),'SUBMISSION_ALREADY_ATTEMPTED_INSPECT_DO_NOT_REPLAY')
    jobs=subprocess.check_output(['squeue','-h','-u','jxu957','-o','%i|%j|%T'],text=True,timeout=30)
    need(not any('qprobe' in line for line in jobs.splitlines()),'EXISTING_OWN_QPROBE_JOB_REQUIRES_INSPECTION')
    command=['sbatch','--parsable','--exclude=ACD1-31,ACD1-67','--account=jxu957','--qos=formal-user',
        '--partition=emergency_acd','--nodes=1','--ntasks=1','--gres=gpu:4','--cpus-per-task=48',
        '--mem=512G','--time=7-00:00:00','--job-name=qprobe-warmup200','--export=NONE',
        '--chdir='+str(TARGET),'--output='+str(TARGET/'logs/training-%j.log'),
        '--error='+str(TARGET/'logs/training-%j.log'),
        str(source/'reproduction/hpc3_qprobe/train.sh'),str(TARGET),commit]
    test=subprocess.run(command[:1]+['--test-only']+command[1:],capture_output=True,text=True,timeout=30)
    publish(TARGET/'preparation/training-test-only.json',dict(command=command,exit=test.returncode,
        stdout=test.stdout,stderr=test.stderr,checked_unix=time.time()))
    need(test.returncode==0,'ACTUAL_SCHEDULER_TEST_FAILED')
    publish(claim,dict(commit=commit,command=command,recovery_of_failed_job=JOB,
        source_receipt_sha256=sha_file(SOURCE_RECEIPT),actual_failure=actual,checked_unix=time.time()))
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    value=dict(commit=commit,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
        recovery_of_failed_job=JOB,checked_unix=time.time())
    if result.returncode==0:
        value['job']=result.stdout.strip().split(';')[0];need(value['job'].isdecimal(),'UNKNOWN_SUBMISSION_OUTCOME')
    publish(receipt,value);need(result.returncode==0,'ACTUAL_SUBMISSION_FAILED_INSPECT_RECEIPT')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','start-prepare','supervise-prepare','preflight','submit'));p.add_argument('--commit',required=True)
    p.add_argument('--recover-failed-preparation',action='store_true')
    a=p.parse_args()
    if a.recover_failed_preparation:
        need(a.mode in ('start-prepare','supervise-prepare'),'FAILED_PREPARATION_MODE_DIFFERS')
        CONTROL=ROOT/'recovery718650-control-repair1'
    if a.mode=='supervise-prepare':sys.exit(supervise_prepare(a.commit))
    if a.mode=='start-prepare':
        print(json.dumps(start_prepare(a.commit,a.recover_failed_preparation)),flush=True);sys.exit(0)
    result={'prepare':fresh,'start-prepare':start_prepare,'preflight':preflight,'submit':submit}[a.mode](a.commit)
    print(json.dumps(result),flush=True)
