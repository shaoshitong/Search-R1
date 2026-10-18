"""One isolated four-H100 CP61 two-stage selection continuation on HPC3."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from .runtime import shared_devices,visible_allocation,sha,once
from reproduction.prefix_selection200.progressive_prefix import MODE

SOURCE=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r3')
TARGET=SOURCE.with_name('dynrpo-prefix5-thread5-20261017-r4')


def prepare(base):
    from reproduction.prefix_selection200.i1_i6_joint_resume import source_manifest
    from reproduction.prefix_selection200.progressive_resume import writers_gone
    from reproduction.prefix_selection200.two_stage_selection import specification
    base=Path(base).resolve(strict=True)
    if base!=TARGET:raise ValueError('Use the isolated two-stage CP61 namespace')
    p=base/'preparation';template=p/'resume-config-template.json'
    if template.exists():return dict(status='EXISTING_PREPARED_TWO_STAGE',path=str(template))
    original=json.loads((SOURCE/'preparation/launch-config.json').read_text());writers_gone(SOURCE,original)
    joint_path=SOURCE/'outers/outer-000061/complete.json';joint=json.loads(joint_path.read_text())
    if joint['actor']['actual_adam_steps_per_rank']!=[1220]*4 or joint['probe']['actual_adam_steps_per_rank']!=[550]*4:
        raise ValueError('Exact latest complete61 actor1220/probe550 required')
    publication=json.loads((p/'source-publication.json').read_text())
    parent=source_manifest(SOURCE/'repo',original['source_commit'],science_source_commit=original['science_source_commit'])
    target=source_manifest(base/'repo',publication['commit'],science_source_commit=original['science_source_commit'])
    intent=p/'two-stage-human-authorization.json';spec=specification(62)
    authorization=json.loads(intent.read_text())
    if authorization['selection']!=spec or authorization['resume_outer']!=61:raise ValueError('Bind the latest actual human two-stage instruction')
    proof=dict(schema_version=1,mode=MODE,adapter='i1_i6_joint_v1',parent_publication=parent,target_publication=target,
        parent_runtime_source_commit=original['source_commit'],runtime_source_commit=publication['commit'],
        science_source_commit=original['science_source_commit'],authorized_prefix_delta=dict(
            prefix_after_boundaries={'50':20,'100':40,'150':80},epochs=3,updates_per_period=150,candidate_limits=[512,5120],block_reset=False),
        authorized_two_stage_selection=spec,human_intent_sha256=sha(intent),
        controller_changes={name:dict(parent_sha256=parent['files_sha256'].get(name),target_sha256=target['files_sha256'].get(name))
            for name in sorted(set(parent['files_sha256'])|set(target['files_sha256'])) if parent['files_sha256'].get(name)!=target['files_sha256'].get(name)})
    compat=p/'source-compatibility.json';once(compat,proof)
    qualification=SOURCE/'preparation/stopped-joint-000061-v1-full-sha-cpu.json'
    c=dict(original,base=str(base),execution_id=uuid.uuid4().hex,source_commit=publication['commit'],
        source_compatibility=str(compat),source_compatibility_sha256=sha(compat),
        environment_root=str(base/'env'),retriever_gpu_lib=str(base/'retriever-gpu-lib'),
        pool_state_path=str(base/'question-pool/state.json'),prefix_probe_receipt=str(base/'probe/latest.json'),
        two_stage_selection=spec,two_stage_authorization=str(intent),two_stage_authorization_sha256=sha(intent),
        resume_joint=dict(adapter='i1_i6_joint_v1',source_base=str(SOURCE),completed_outer=61,
            source_commit=original['source_commit'],source_execution_id=original['execution_id'],
            source_config_sha256=sha(SOURCE/'preparation/launch-config.json'),source_joint_sha256=sha(joint_path),
            qualification_proof=str(qualification),qualification_proof_sha256=sha(qualification)))
    c.pop('resume_pending_probe',None);c.pop('resume_import',None)
    from reproduction.prefix_selection200.two_stage_resume import plan
    plan(base,c)
    once(template,c)
    return dict(status='CP61_TWO_STAGE_PREPARED_NO_GPU',source_commit=publication['commit'],actor_adam=[1220]*4,probe_adam=[550]*4,selection=spec)


def run(base):
    from reproduction.prefix_selection200.i1_i6_joint_resume import import_complete
    from reproduction.prefix_selection200.launch import run as launch,validate_config
    base=Path(base).resolve(strict=True)
    if base!=TARGET or not os.environ.get('SLURM_JOB_ID'):raise ValueError('Use the exact new Slurm execution')
    p=base/'preparation'
    if (p/'submit-intent.json').exists() or (base/'logs/process-ledger.json').exists():
        raise RuntimeError('Unknown prior execution must be inspected, never started twice')
    c=json.loads((p/'resume-config-template.json').read_text())
    publication=json.loads((p/'source-publication.json').read_text())
    if c['source_commit']!=publication['commit'] or publication['remote_sha']!=publication['commit']:
        raise ValueError('Run only the verified pushed publication')
    for name,digest in publication['files'].items():
        if sha(base/'repo'/name)!=digest:raise ValueError('Deployed source SHA differs: '+name)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');assigned=os.environ.get('SLURM_JOB_GPUS','').split(',')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    by_index={int(row[0]):row[1].strip() for row in csv.reader(io.StringIO(raw))}
    selected=visible_allocation(visible,assigned,by_index)
    c.update(shared_devices(selected,by_index),slurm_job=os.environ['SLURM_JOB_ID'],slurm_node=os.uname().nodename)
    validate_config(c,base)
    once(p/'launch-config.json',c)
    once(p/'submit-intent.json',dict(config_sha256=sha(p/'launch-config.json'),run_id=c['run_id'],
        source_commit=c['source_commit'],job=os.environ['SLURM_JOB_ID']))
    once(base/'logs/slurm-allocation.json',dict(job=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,
        uid=os.getuid(),training=selected,auxiliary=selected,assigned_uuids=[by_index[n] for n in selected],
        total_physical_gpus=4,gpu_resource_layout=c['gpu_resource_layout']))
    print(json.dumps(import_complete(base)),flush=True)
    return launch(base)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    print(json.dumps(prepare(args.base) if args.prepare else run(args.base)),flush=True)
