"""Bind fresh actor plus pretrained prefix20 head, without a feature/warmup job."""
import argparse
import json
import os
from pathlib import Path
import time
import subprocess


def prepare(base,commit):
    from reproduction.prefix20_joint4.control import own_base,MODEL,bind_repo,assets_ready,storage_admission,publish
    base=own_base(base);bind_repo(base,commit)
    if Path(__file__).resolve()!=base/'repo/reproduction/prefix20_joint4/prepare_training.py':
        raise ValueError('PREFIX20_PREPARATION_MUST_LOAD_BOUND_RUNTIME_REPO')
    if os.environ.get('SEARCH_G1_TRAINING_WORLD_SIZE')!='4' or not os.environ.get('SLURM_JOB_ID','').isdecimal():
        raise ValueError('PREFIX20_ACTUAL_FOUR_GPU_SLURM_ENTRY_REQUIRED')
    from reproduction.prefix20_joint4.control import validate_slurm_job
    from reproduction.common8.native_allocation import fields
    detail=subprocess.check_output(['scontrol','show','job','-d',os.environ['SLURM_JOB_ID']],text=True,timeout=30)
    submission=base/'preparation/training-submission.json'
    deadline=time.monotonic()+30
    while not submission.exists() and time.monotonic()<deadline:time.sleep(.2)
    if not submission.is_file():raise ValueError('PREFIX20_SUBMISSION_RECEIPT_NOT_PUBLISHED_INSPECT_CLAIM')
    submitted=json.loads(submission.read_text())
    if submitted.get('job')!=os.environ['SLURM_JOB_ID']:raise ValueError('PREFIX20_SUBMITTED_JOB_ID_DIFFERS')
    actual_job=validate_slurm_job(base,commit,submitted,fields(detail))
    assets=assets_ready(base);storage=storage_admission(base)
    manifest=base/'preparation/official-model-manifest.json'
    from reproduction.hpc3_qprobe.model_manifest import create
    if not manifest.exists():create(Path(MODEL),manifest)
    from reproduction.common8.prepare_run import prepare as prepare_actor
    run=prepare_actor(base,Path(MODEL),manifest,commit,rollout_memory_utilization=.1,
        local_retrieval_assets='/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/source/assets-root/assets/wiki',
        execution_site='hpc3')
    from search_g1.prefix20_runtime import validate_config
    from search_g1.rollout_phase_contract import DEFAULT_ROLLOUT_PHASE,validate
    root=base/'assets/prefix20'
    run.update(data_selection_mode='prefix20_probe',mixed_first_selection=False,
        startup_admission_mode='actual_first_five',
        question_probe=validate_config(dict(model_dir=MODEL,head_safetensors=str(root/'head.safetensors'),
            probe_config=str(root/'probe-config.json'),eligible_pool_path=str(root/'eligible-question-pool.json'),
            output_dir=str(base/'artifacts/question-probe'),seed=1729)),
        infra_rollout=validate(dict(DEFAULT_ROLLOUT_PHASE),4))
    from reproduction.common8.stage_retrieval import ASSETS
    from search_g1.qprobe_features import sha_json
    run['retriever_version']=sha_json(ASSETS)
    original=(base/'logs/run-manifest.json').read_bytes()
    with (base/'preparation/prefix20-pre-bind.json').open('xb') as out:out.write(original);out.flush();os.fsync(out.fileno())
    stage=base/'logs/run-manifest.prefix20.partial'
    with stage.open('x') as out:json.dump(run,out,indent=2);out.flush();os.fsync(out.fileno())
    stage.replace(base/'logs/run-manifest.json')
    from reproduction.common8.configuration import resolved
    _,config,_=resolved(base)
    proof=dict(stage='prefix20_joint4_configuration_prepared',run_id=run['run_id'],source_commit=commit,
        training_world_size=4,completed_outers=0,target_outers=200,teacher_warmup_updates=0,
        initial_head_epoch=68,initial_head_optimizer='fresh Adam',probe_local_batch=64,probe_global_batch=256,
        probe_steps_per_outer=10,actual_gpu_acceptance_pending=True,assets=assets,storage=storage,checked_unix=time.time())
    proof['actual_slurm_job_binding']=actual_job
    publish(base/'logs/prefix20-configuration-prepared.json',proof)
    return proof


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    print(json.dumps(prepare(a.base,a.commit)),flush=True)


if __name__=='__main__':main()
