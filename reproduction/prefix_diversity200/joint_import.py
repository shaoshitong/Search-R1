"""Resume after an actual joint commit without copying rotating history aliases."""
import os
from pathlib import Path
import subprocess
import time
from .actor import read, sha, atomic_json


def import_joint(base):
    from reproduction.prefix_only_correctness_ab.launch import birth_states
    base=Path(base);c=read(base/'preparation/launch-config.json');spec=c['resume_joint']
    source=Path(spec['source_base']);n=spec['completed_outer']
    if source==base or source.parent!=base.parent or not 1<=n<200:
        raise ValueError('Require an isolated continuation of a real incomplete200 run')
    destination=base/'preparation/joint-resume-receipt.json'
    if destination.exists():return read(destination)
    if (base/'preparation/joint-resume-intent.json').exists():
        raise RuntimeError('Reconcile the existing import intent, never replay blindly')
    previous=read(source/'preparation/launch-config.json')
    science=('run_id','model_revision','model_files_sha256','candidate_questions','selected_questions',
        'actor_selected_questions','responses_per_question','prefix_policy_tokens','probe_updates_per_outer',
        'probe_learning_rate','probe_epochs_per_outer','probe_global_batch_average','actor_global_mini_batch',
        'actor_ppo_epochs','seed','official_model_dir','fixed_test_manifest_sha256','canonical_train_sha256',
        'grounding_probe_version')
    if any(c[k]!=previous[k] for k in science):raise ValueError('Scientific protocol changed')
    job=read(source/'preparation/submit-receipt.json')['job_id']
    native=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=15)
    if native.returncode:
        queue=subprocess.check_output(['squeue','-h','-o','%i|%j'],text=True,timeout=15)
        name=read(source/'preparation/submit-intent.json')['job_name']
        if ('Invalid job id specified' not in native.stderr
                or any(line.split('|')[-1]==name for line in queue.splitlines())
                or 'Rotating actor checkpoint does not match its own complete producer'
                    not in (source/'logs'/('native-'+job+'.log')).read_text()):
            raise ValueError('No authoritative terminal evidence for the source')
    elif 'JobState=FAILED' not in native.stdout or 'WorkDir='+str(source) not in native.stdout:
        raise ValueError('Source native allocation is not terminal')
    ledger=read(source/'logs/process-ledger.json')
    births=[b for part in ledger['jobs'].values() for b in part['ledger'].values()]
    if birth_states(births) or not read(source/'logs/cleanup-status.json')['owned_births_all_gone']:
        raise ValueError('Source task processes have not exited')
    joint=read(source/'latest-joint.json');actor=read(source/'actor/latest.json');probe=read(source/'probe/latest.json')
    root=source/'outers'/('outer-%06d'%n)
    if (joint!=read(root/'complete.json') or joint['outer']!=n or joint['run_id']!=c['run_id']
            or joint['actor']!=actor or joint['probe']!=probe or actor['completed_outer']!=n
            or probe['completed_outer']!=n or actor['actual_adam_steps_per_rank']!=[20*n]*8
            or probe['actual_adam_steps_per_rank']!=[20*n]*8 or actor['scheduler_completed_outer']!=n):
        raise ValueError('Actual actor/probe/scheduler are not the same complete joint boundary')
    pool=read(source/'question-pool/state.json')
    if (pool['next_outer']!=n+1 or pool['consumed_candidates']!=5120*n
            or sha(root/'pool-state-at-completion.json')!=joint['pool_sha256']
            or pool!=read(root/'pool-state-at-completion.json')):
        raise ValueError('Actual candidate cursor is not exactly the completed boundary')
    proof=dict(source_base=str(source),source_job=job,completed_outer=n,
        source_config_sha256=sha(source/'preparation/launch-config.json'),joint_sha256=sha(root/'complete.json'),
        native_query=dict(returncode=native.returncode,stdout=native.stdout,stderr=native.stderr),
        original_logs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in (source/'logs').glob('*.log')},
        actor_steps=[20*n]*8,probe_steps=[20*n]*8,scheduler_completed_outer=n,
        candidate_cursor=pool['consumed_candidates'],owned_births_present=[],checked_unix=time.time())
    atomic_json(base/'preparation/joint-source-terminal-proof.json',proof)
    atomic_json(base/'preparation/joint-resume-intent.json',dict(source_base=str(source),completed_outer=n,run_id=c['run_id']))
    assets=[]
    def link(s,d):
        if not s.is_file() or s.is_symlink() or s.resolve()!=s:raise ValueError('Noncanonical source file')
        d.parent.mkdir(parents=True,exist_ok=True);os.link(s,d);a,b=s.stat(),d.stat()
        if (a.st_dev,a.st_ino,a.st_size)!=(b.st_dev,b.st_ino,b.st_size):raise ValueError('State link differs')
        assets.append(dict(source=str(s),target=str(d),bytes=a.st_size,inode=a.st_ino))
    # Actor receipts already reference immutable original HF/rank files. Copying
    # old actor directories would create aliases the rotating pruner cannot own.
    link(source/'actor/latest.json',base/'actor/latest.json')
    for directory in ('probe','question-pool'):
        for p in sorted((source/directory).rglob('*')):
            if p.is_symlink():raise ValueError('No source tree symlinks')
            if p.is_file():link(p,base/directory/p.relative_to(source/directory))
    link(source/'latest-joint.json',base/'latest-joint.json')
    result=dict(status='COMPLETE_JOINT_BOUNDARY_IMPORTED_NO_RECOMPUTATION',run_id=c['run_id'],
        source_base=str(source),completed_outer=n,next_outer=n+1,actor_steps=[20*n]*8,probe_steps=[20*n]*8,
        actor_recomputed=False,probe_recomputed=False,source_originals_modified=False,assets=assets,
        source_actor_files_referenced_by_original_receipt=True,rotating_history_directories_not_aliased=True,
        future_actor8rank_full_optimizer_scheduler_rng_readback_required=True,
        native_executor_exact_resume_claimed=False,large_assets_via_Mac=False,
        terminal_proof_sha256=sha(base/'preparation/joint-source-terminal-proof.json'))
    atomic_json(destination,result)
    return {k:v for k,v in result.items() if k!='assets'}
