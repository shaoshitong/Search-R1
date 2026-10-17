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
    planned=None
    if spec.get('planned_calibration_handoff'):
        planned=read(source/'preparation/calibration-handoff-v1/terminal.json')
        plan=read(source/'preparation/calibration-handoff-v1/plan.json')
        if (sha(source/'preparation/calibration-handoff-v1/plan.json')!=spec['planned_handoff_plan_sha256']
                or planned['status']!='PLANNED_CALIBRATION_HANDOFF_COMPLETE'
                or planned['plan_sha256']!=spec['planned_handoff_plan_sha256']
                or planned['run_id']!=c['run_id'] or planned['job_id']!=str(spec['source_job'])
                or planned['completed_outer']!=n or planned['owned_births_present']
                or planned.get('native_job_cgroup_members') or not planned.get('controller_dead')
                or planned['successor_job']!=os.environ.get('SLURM_JOB_ID')
                or plan['source_base']!=str(source) or plan['target_complete_outer']!=n
                or plan['source_config_sha256']!=sha(source/'preparation/launch-config.json')):
            raise ValueError('Planned source handoff terminal proof differs')
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
        terminal_path = source/'preparation'/('terminal%s-original-inode-full-sha.json' % job)
        terminal = read(terminal_path) if terminal_path.is_file() else {}
        verified_terminal = (spec.get('terminal_proof_sha256') == sha(terminal_path)
            if terminal_path.is_file() else False)
        planned_native=planned['native'] if planned else {}
        planned_terminal=(planned_native.get('JobId')==job and planned_native.get('WorkDir')==str(source)
            and planned_native.get('JobState') in ('CANCELLED','FAILED','COMPLETED'))
        if ('Invalid job id specified' not in native.stderr
                or any(line.split('|')[-1]==name for line in queue.splitlines())
                or (not planned_terminal and (not verified_terminal or terminal.get('job_id') != job
                    or terminal.get('base') != str(source) or terminal.get('native_terminal_state') != 'FAILED'
                    or terminal.get('actual_births_present') or terminal.get('native_job_cgroup_paths')
                    or not terminal.get('owned_gpu_processes_gone')))):
            raise ValueError('No authoritative terminal evidence for the source')
    else:
        fields=dict(part.split('=',1) for part in native.stdout.split() if '=' in part)
        allowed_states=('FAILED','CANCELLED','COMPLETED') if planned else ('FAILED',)
        if (fields.get('JobId')!=job or fields.get('JobState') not in allowed_states
                or fields.get('WorkDir')!=str(source)):
            raise ValueError('Source native allocation is not terminal')
    ledger=read(source/'logs/process-ledger.json')
    births=[b for part in ledger['jobs'].values() for b in part['ledger'].values()]
    cleanup=source/'logs/cleanup-status.json'
    if birth_states(births) or not ((cleanup.exists() and read(cleanup)['owned_births_all_gone']) or planned):
        raise ValueError('Source task processes have not exited')
    joint=read(source/'latest-joint.json');actor=read(source/'actor/latest.json');probe=read(source/'probe/latest.json')
    root=source/'outers'/('outer-%06d'%n)
    if (joint!=read(root/'complete.json') or joint['outer']!=n or joint['run_id']!=c['run_id']
            or joint['actor']!=actor or joint['probe']!=probe or actor['completed_outer']!=n
            or probe['completed_outer']!=n or actor['actual_adam_steps_per_rank']!=[20*n]*8
            or probe['actual_adam_steps_per_rank']!=[20*n]*8 or actor['scheduler_completed_outer']!=n):
        raise ValueError('Actual actor/probe/scheduler are not the same complete joint boundary')
    snapshot=root/'pool-state-at-completion.json'
    pool=read(source/'question-pool/state.json');completed_pool=read(snapshot)
    replay=None
    if pool!=completed_pool and spec.get('replay_uncommitted_prefix_outer')==n+1:
        from .pool import reserve_indices,state_sha
        failed_root=source/'outers'/('outer-%06d'%(n+1))
        reservation=read(failed_root/'pool-reservation.json')
        indices,cycles,after=reserve_indices(completed_pool,n+1)
        stage=read(source/'logs/pipeline-status.json')
        prefix=failed_root/'prefix'
        replay_stage=stage.get('stage')
        allowed_stage=('outer-%06d-prefix'%(n+1),'CANDIDATE_POOL_RESERVATION') if planned else ('outer-%06d-prefix'%(n+1),)
        no_optimization=(not any((source/'actor/outers').glob('outer-%04d-*'%(n+1)))
            and not any((failed_root/'probe-update').glob('rank-*-updates.jsonl')))
        sampled_prefix=any(prefix.glob('rank-*-progress.json')) or any(prefix.glob('rank-*.jsonl*'))
        if (stage.get('outer')!=n+1 or replay_stage not in allowed_stage
                or reservation['outer']!=n+1 or reservation['before_sha256']!=state_sha(completed_pool)
                or reservation['after_sha256']!=state_sha(after) or reservation['after']!=after
                or reservation['indices']!=indices or reservation['cycles']!=cycles or pool!=after
                or (failed_root/'complete.json').exists()
                or (not planned and (prefix/'complete.json').exists())
                or (not planned and (sampled_prefix or any(prefix.glob('rank-*-complete.json'))))
                or not no_optimization
                or any((source/'actor/outers').glob('outer-%04d-*'%(n+1)))
                or (source/'probe'/('head-outer-%04d.pt'%(n+1))).exists()):
            raise ValueError('Cannot prove one original candidate reservation with zero actor/probe updates')
        replay=dict(outer=n+1,original_reservation_sha256=sha(failed_root/'pool-reservation.json'),
                    source_active_pool_sha256=sha(source/'question-pool/state.json'),
                    restored_pool_snapshot_sha256=sha(snapshot),actor_or_probe_updates_discarded=0,
                    prefix_sampling_recomputed=True,same_candidate_indices_verified=True,
                    original_prefix_sampling_started=bool(sampled_prefix),
                    original_prefix_directory_preserved=str(prefix))
        pool=completed_pool
    if (pool['next_outer']!=n+1 or pool['consumed_candidates']!=5120*n
            or sha(snapshot)!=joint['pool_sha256'] or pool!=completed_pool):
        raise ValueError('Actual candidate cursor is not exactly the completed boundary')
    reward_history=[];next_reward=None
    if c.get('grounding_refit_rounds'):
        from .cost_snapshot import grounding_binding
        for key in ('grounding_initial_probe_version','grounding_refit_rounds','grounding_manifest',
                    'grounding_manifest_sha256','interaction_contract','policy_monitor'):
            if c[key]!=previous[key]:raise ValueError('I1-I6 scientific binding changed: '+key)
        used_round=min(150,((n-1)//50)*50)
        used=grounding_binding(source,previous,used_round)
        if (actor['reward']['version']!=used['version']
                or any(row['version']!=used['version'] for row in actor['reward']['ranks'])):
            raise ValueError('Saved actor did not actually use its required reward version')
        next_round=min(150,(n//50)*50)
        for ordinal in (50,100,150):
            pointer=source/'grounding'/('round-%03d-complete.json'%ordinal)
            if ordinal<=next_round and pointer.exists():
                row=grounding_binding(source,previous,ordinal,require_adoption=ordinal<=used_round)
                manifest=read(row['manifest'])
                for name,key in (('commit_probe.pkl','commit'),('boundary_probe.pkl','boundary')):
                    if sha(manifest[key])!=manifest['head_sha256'][name]:
                        raise ValueError('Retained actual fitted reward head changed')
                reward_history.append(row)
        next_reward=next((row for row in reward_history if row['round']==next_round),used)
        # At an exact milestone a missing next fit is recomputed from that same
        # completed actor by launch. After a milestone, missing used versions
        # were rejected above; never silently fall back to official round0.
    proof=dict(source_base=str(source),source_job=job,completed_outer=n,
        source_config_sha256=sha(source/'preparation/launch-config.json'),joint_sha256=sha(root/'complete.json'),
        native_query=dict(returncode=native.returncode,stdout=native.stdout,stderr=native.stderr),
        original_logs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in (source/'logs').glob('*.log')},
        actor_steps=[20*n]*8,probe_steps=[20*n]*8,scheduler_completed_outer=n,
        candidate_cursor=pool['consumed_candidates'],uncommitted_prefix_replay=replay,
        reward_version_for_next_outer=next_reward, reward_version_history=reward_history,
        owned_births_present=[],checked_unix=time.time())
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
            if p.is_file():
                if directory=='question-pool' and p.name=='state.json' and replay:
                    link(snapshot,base/directory/'state.json')
                else:link(p,base/directory/p.relative_to(source/directory))
    link(source/'latest-joint.json',base/'latest-joint.json')
    cached=source/'retrieval-cache';corpus=read(cached/'corpus.json') if (cached/'corpus.json').exists() else None
    if corpus is not None:
        archive=Path(c['retrieval_corpus']);info=archive.stat();content=cached/'wiki_dump.jsonl'
        if (corpus['source']!=str(archive.resolve()) or corpus['source_size']!=info.st_size
                or corpus['source_mtime_ns']!=info.st_mtime_ns or content.stat().st_size!=corpus['content_bytes']):
            raise ValueError('Reusable corpus cache no longer matches original same-node archive')
        link(content,base/'retrieval-cache/wiki_dump.jsonl')
        link(cached/'corpus.json',base/'retrieval-cache/corpus.json')
    if next_reward is not None:
        grounding=base/'grounding';grounding.mkdir()
        initial_adoption=source/'grounding/adopted-round-000.json'
        if initial_adoption.exists():link(initial_adoption,grounding/initial_adoption.name)
        for row in reward_history:
            pointer=Path(row['completed_receipt']);link(pointer,grounding/pointer.name)
            if row['adoption_receipt']:
                adopted=Path(row['adoption_receipt']);link(adopted,grounding/adopted.name)
        live=(read(next_reward['completed_receipt']) if next_reward['completed_receipt'] else
            dict(run_id=c['run_id'],round=0,version=next_reward['version'],
                manifest=next_reward['manifest'],manifest_sha256=next_reward['manifest_sha256']))
        atomic_json(grounding/'live.json',dict(live,restored_from_source=str(source)))
    result=dict(status='COMPLETE_JOINT_BOUNDARY_IMPORTED_NO_RECOMPUTATION',run_id=c['run_id'],
        source_base=str(source),completed_outer=n,next_outer=n+1,actor_steps=[20*n]*8,probe_steps=[20*n]*8,
        actor_recomputed=False,probe_recomputed=False,source_originals_modified=False,assets=assets,
        source_actor_files_referenced_by_original_receipt=True,rotating_history_directories_not_aliased=True,
        future_actor8rank_full_optimizer_scheduler_rng_readback_required=True,
        native_executor_exact_resume_claimed=False,large_assets_via_Mac=False,
        reward_version_for_next_outer=next_reward,reward_version_history=reward_history,
        reward_heads_referenced_by_original_immutable_manifest=next_reward is not None,
        recovered_reward_live_sha256=sha(base/'grounding/live.json') if next_reward else None,
        uncommitted_prefix_replay=replay,
        terminal_proof_sha256=sha(base/'preparation/joint-source-terminal-proof.json'))
    atomic_json(destination,result)
    return {k:v for k,v in result.items() if k!='assets'}


def native_import(base):
    """One pending native job imports only after its exact source is terminal."""
    from .launch import config
    base=Path(base);c=config(base);spec=c['resume_joint']
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Native resume import requires its actual Slurm allocation')
    path=Path(spec['source_base'])/'preparation/calibration-handoff-v1/terminal.json'
    deadline=time.monotonic()+300
    while not path.exists():
        if time.monotonic()>deadline:
            raise TimeoutError('Original handoff terminal proof not yet present; no training launched')
        time.sleep(1)
    return import_joint(base)


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
    print(json.dumps(native_import(a.base),allow_nan=False))
