"""Explicitly authorized CP50 fork; actor/math bytes remain immutable.

Reuses the genuine immutable50 full-SHA/CPU proof only while every sealed file
retains its inode/size/mtime/filesystem identity. It does not pretend a stat is
a fresh SHA or repeat the50GB payload scan. Slurm writers must actually be gone.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from .progressive_prefix import MODE


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected(config):return config.get('progressive_prefix',{}).get('mode')==MODE


def verify(config):
    if config.get('two_stage_selection'):
        from .two_stage_resume import verify as suffix_verify
        return suffix_verify(config)
    return _verify_progressive(config)


def _verify_progressive(config):
    proof=read(config['source_compatibility'])
    if sha(config['source_compatibility'])!=config['source_compatibility_sha256'] or proof.get('mode')!=MODE:
        raise ValueError('Progressive continuation needs its exact explicit source delta proof')
    from .i1_i6_joint_resume import _code_inventory
    inventory=_code_inventory(Path(config['base'])/'repo')
    if inventory!=proof['target_publication']['files_sha256'] or config['source_commit']!=proof['runtime_source_commit']:
        raise ValueError('Published progressive runtime differs from actual deployed code')
    parent=proof['parent_publication']['files_sha256']
    protected=('reproduction/prefix_selection200/actor.py',
               'reproduction/prefix_selection200/actor_safety.py',
               'reproduction/prefix_selection200/interaction_protocol.py',
               'reproduction/prefix20_group_joint100/loss.py',
               'reproduction/prefix_selection200/grounding.py')
    if any(inventory.get(path)!=parent.get(path) for path in protected):
        raise ValueError('Progressive probe changes cannot alter actor loss/guard/interaction/grounding math')
    authorization=read(config['progressive_prefix']['authorization'])
    if (sha(config['progressive_prefix']['authorization'])!=config['progressive_prefix']['authorization_sha256']
            or authorization['mode']!=MODE or authorization['resume_outer']!=50
            or authorization['prefix_after_boundaries']!={'50':20,'100':40,'150':80}
            or authorization['probe_epochs']!=3 or authorization['candidate_limits']!=[512,5120]):
        raise ValueError('Progressive dimensions/epochs/candidate cap lack their exact human authorization')
    if (config.get('probe_epochs_per_update')!=3 or config.get('probe_updates_per_period')!=150
            or config.get('adaptive_candidates',{}).get('maximum_questions')!=5120
            or config.get('adaptive_candidates',{}).get('minimum_questions')!=512
            or config.get('probe_learning_rate')!=5e-6):
        raise ValueError('Actual progressive execution differs from its authorized budgets')
    return proof


def writers_gone(source,config):
    from reproduction.hpc3_prefix5.terminal import writers_present
    original=read(Path(source)/'preparation/launch-config.json')
    terminal=dict(execution_backend='slurm_hpc3',slurm_job=str(original['slurm_job']),uid=os.getuid())
    if writers_present(terminal,original):raise RuntimeError('Source Slurm cgroup is still active')


def plan(base,config=None):
    config=config or read(Path(base)/'preparation/launch-config.json')
    if config.get('two_stage_selection'):
        from .two_stage_resume import plan as suffix_plan
        return suffix_plan(base,config)
    base=Path(base);config=config or read(base/'preparation/launch-config.json');verify(config)
    spec=config['resume_joint'];source=Path(spec['source_base'])
    if base==source or base.parent!=source.parent or spec['completed_outer']!=50:
        raise ValueError('Initial progressive execution is a distinct explicitly authorized CP50 fork')
    original=read(source/'preparation/launch-config.json')
    if config['run_id']!=original['run_id'] or config['execution_id']==original['execution_id']:
        raise ValueError('Keep scientific run identity but use a distinct execution')
    writers_gone(source,config)
    joint_path=source/'outers/outer-000050/complete.json';joint=read(joint_path)
    proof=read(spec['qualification_proof'])
    if sha(spec['qualification_proof'])!=spec['qualification_proof_sha256']:
        raise ValueError('Existing CP50 qualification proof SHA changed')
    if (proof['status']!='LIVE_IMMUTABLE_MILESTONE_FULL_SHA_CPU_READBACK_PASS'
            or proof['outer']!=50 or not proof['actual_cpu_deserialization']
            or proof['source_joint_sha256']!=sha(joint_path)
            or proof['actor_actual_adam_steps_per_rank']!=[1000]*4
            or proof['probe']['actual_adam_steps_per_rank']!=[250]*4):
        raise ValueError('CP50 must have its genuine full-state model/Adam/RNG/SHA CPU evidence')
    for row in proof['verified_files']:
        path=Path(row['path']);st=path.stat()
        if (st.st_ino,st.st_size,st.st_mtime_ns)!=(row['inode'],row['bytes'],row['mtime_ns']):
            raise ValueError('Previously fully qualified CP50 payload changed: '+str(path))
        fsid=row.get('shared_filesystem_fsid')
        if st.st_dev!=row['dev'] and (fsid is None or str(os.statvfs(path).f_fsid)!=str(fsid)):
            raise ValueError('Qualified CP50 filesystem identity changed')
    return dict(status='QUALIFIED_STOPPED_CP50_PROGRESSIVE_IMPORT_READY',adapter='i1_i6_joint_v1',
        run_id=config['run_id'],execution_id=config['execution_id'],completed_outer=50,next_outer=51,
        source_base=str(source),source_joint_sha256=sha(joint_path),
        reused_full_cpu_proof=str(spec['qualification_proof']),
        reused_full_cpu_proof_sha256=sha(spec['qualification_proof']),sealed_files=len(proof['verified_files']),
        full_payload_sha_repeated=False,actor_actual_adam=[1000]*4,probe_actual_adam=[250]*4,
        phase_reuse={},imported_prefix_outer=None)


def import_complete(base):
    if read(Path(base)/'preparation/launch-config.json').get('two_stage_selection'):
        from .two_stage_resume import import_complete as suffix_import
        return suffix_import(base)
    base=Path(base);config=read(base/'preparation/launch-config.json');result=plan(base,config)
    receipt=base/'preparation/resume-joint-receipt.json'
    if receipt.exists():validate_imported_boundary(base,config);return read(receipt)
    source=Path(result['source_base']);joint=read(source/'outers/outer-000050/complete.json')
    # Refuse a half-import rather than overwriting it after an unknown outcome.
    intent=base/'preparation/progressive-import-intent.json'
    if intent.exists():raise RuntimeError('Reconcile the recorded partial import; do not duplicate it')
    intent.write_text(json.dumps(result,indent=2));writers_gone(source,config)
    def copy_json(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as handle:handle.write(Path(src).read_bytes())
    for outer in range(1,51):
        root=base/'outers'/f'outer-{outer:06d}';root.mkdir(parents=True,exist_ok=True)
        prior=source/'outers'/root.name
        for name in ['candidate-budget.json','pool-state-at-completion.json']:
            if (prior/name).exists():copy_json(prior/name,root/name)
        copy_json(prior/'complete.json',root/('inherited-complete.json' if outer==50 else 'complete.json'))
    copy_json(joint['pool_snapshot'],base/'question-pool/state.json')
    copy_json(source/'question-pool/binding.json',base/'question-pool/binding.json')
    (base/'actor').mkdir(exist_ok=True);(base/'probe').mkdir(exist_ok=True)
    (base/'actor/latest.json').write_text(json.dumps(joint['actor'],indent=2))
    (base/'probe/latest.json').write_text(json.dumps(joint['probe'],indent=2))
    os.link(joint['probe_snapshot'],base/'probe/latest.pt')
    (base/'latest-joint.json').write_text(json.dumps(joint,indent=2))
    result.update(status='I1_I6_FULL_JOINT_IMPORTED',progressive_bootstrap_pending=True)
    receipt.write_text(json.dumps(result,indent=2));validate_imported_boundary(base,config)
    return result


def validate_imported_boundary(base,config):
    if config.get('two_stage_selection'):
        from .two_stage_resume import validate_imported_boundary as suffix_validate
        return suffix_validate(base,config)
    base=Path(base);receipt=read(base/'preparation/resume-joint-receipt.json');current=plan(base,config)
    if receipt['source_joint_sha256']!=current['source_joint_sha256']:
        raise ValueError('Imported CP50 provenance changed')
    actor=read(base/'actor/latest.json');probe=read(base/'probe/latest.json')
    if actor['completed_outer']!=50 or actor['actual_adam_steps_per_rank']!=[1000]*4:
        raise ValueError('Imported actor must retain actual CP50 optimizer and scheduler')
    if probe['actual_adam_steps_per_rank'] not in ([250]*4,[400]*4):
        raise ValueError('Only retained CP50 probe250 or completed Prefix20 bootstrap400 are admissible')
    return 50
