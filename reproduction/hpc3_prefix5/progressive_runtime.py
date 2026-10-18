"""One isolated four-H100 CP50 progressive-prefix continuation on HPC3."""
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

SOURCE=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r2')
TARGET=SOURCE.with_name('dynrpo-prefix5-thread5-20261017-r3')


def prepare(base):
    from reproduction.prefix_selection200.i1_i6_joint_resume import source_manifest
    from reproduction.prefix_selection200.progressive_resume import writers_gone
    base=Path(base).resolve(strict=True)
    if base!=TARGET:raise ValueError('Use the distinct authorized CP50 progressive namespace')
    p=base/'preparation';template=p/'resume-config-template.json'
    if template.exists():return dict(status='EXISTING_PREPARED_PROGRESSIVE_CP50',path=str(template))
    original=json.loads((SOURCE/'preparation/launch-config.json').read_text());writers_gone(SOURCE,original)
    joint_path=SOURCE/'outers/outer-000050/complete.json';joint=json.loads(joint_path.read_text())
    if joint['actor']['actual_adam_steps_per_rank']!=[1000]*4 or joint['probe']['actual_adam_steps_per_rank']!=[250]*4:
        raise ValueError('Exact CP50 actor1000/probe250 is required')
    publication=json.loads((p/'source-publication.json').read_text())
    parent=source_manifest(SOURCE/'repo',original['source_commit'],science_source_commit=original['science_source_commit'])
    target=source_manifest(base/'repo',publication['commit'],science_source_commit=original['science_source_commit'])
    protected=('reproduction/prefix_selection200/actor.py',
        'reproduction/prefix_selection200/actor_safety.py','reproduction/prefix_selection200/interaction_protocol.py',
        'reproduction/prefix_selection200/grounding.py','reproduction/prefix20_group_joint100/loss.py')
    if any(parent['files_sha256'][name]!=target['files_sha256'][name] for name in protected):
        raise ValueError('Actor/I1-I6 mathematics changed outside authorized prefix/fit scheduling')
    for name in ['stable_top512','mixed_probabilities','join_own_em','inference_logits']:
        if parent['critical_ast_sha256'][name]!=target['critical_ast_sha256'][name]:
            raise ValueError('Strict own-EM/G5 selection and Mixed loss probability must stay unchanged')
    authorization=p/'progressive-human-authorization.json'
    intent=json.loads(authorization.read_text())
    if intent['mode']!=MODE or intent['resume_outer']!=50:raise ValueError('Require the explicit latest CP50 authorization')
    proof=dict(schema_version=1,mode=MODE,adapter='i1_i6_joint_v1',
        parent_publication=parent,target_publication=target,parent_runtime_source_commit=original['source_commit'],
        runtime_source_commit=publication['commit'],science_source_commit=original['science_source_commit'],
        actor_file_sha_unchanged=True,protected_actor_and_loss_files=list(protected),
        authorized_prefix_delta=dict(prefix_after_boundaries={'50':20,'100':40,'150':80},
            epochs=3,updates_per_period=150,candidate_limits=[512,5120],block_reset=False),
        controller_changes={name:dict(parent_sha256=parent['files_sha256'].get(name),target_sha256=target['files_sha256'].get(name))
            for name in sorted(set(parent['files_sha256'])|set(target['files_sha256']))
            if parent['files_sha256'].get(name)!=target['files_sha256'].get(name)})
    compat=p/'source-compatibility.json';once(compat,proof)
    qualification=SOURCE/'preparation/live-milestone-000050-v1-full-sha-cpu.json'
    c=dict(original,base=str(base),execution_id=uuid.uuid4().hex,source_commit=publication['commit'],
        source_compatibility=str(compat),source_compatibility_sha256=sha(compat),
        environment_root=str(base/'env'),retriever_gpu_lib=str(base/'retriever-gpu-lib'),
        pool_state_path=str(base/'question-pool/state.json'),prefix_probe_receipt=str(base/'probe/latest.json'),
        prefix_policy_tokens=20,hidden_capture_contract='sample21_keep20_postblock27_native_residual_sum_v1',
        probe_epochs_per_update=3,probe_updates_per_period=150,
        progressive_prefix=dict(mode=MODE,authorization=str(authorization),authorization_sha256=sha(authorization),
            actor_stage_ranges=[[51,100,20],[101,150,40],[151,200,80]],
            post_actor_refit_boundaries=[50,100,150],short_policy_padding='zero_feature_mask_false_no_fake_tokens'),
        resume_joint=dict(adapter='i1_i6_joint_v1',source_base=str(SOURCE),completed_outer=50,
            source_commit=original['source_commit'],source_execution_id=original['execution_id'],
            source_config_sha256=sha(SOURCE/'preparation/launch-config.json'),source_joint_sha256=sha(joint_path),
            qualification_proof=str(qualification),qualification_proof_sha256=sha(qualification)))
    c['adaptive_candidates']=dict(original['adaptive_candidates'],maximum_questions=5120)
    c['candidate_schedule']=dict(original['candidate_schedule'],effective_from_outer=51)
    c.pop('resume_pending_probe',None);c.pop('resume_import',None)
    once(template,c)
    return dict(status='CP50_PROGRESSIVE_PREPARED_NO_GPU',source_commit=publication['commit'],
                actor_adam=[1000]*4,probe_before=[250]*4,probe_after_bootstrap=[400]*4)


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
