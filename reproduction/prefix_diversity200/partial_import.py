"""Preserve a committed actor when a later probe diagnostic failed.

Only this logical run's actually completed phases are linked into an isolated
repair directory. Failed probe writes remain original cost evidence; unsaved
probe state cannot be reconstructed and is explicitly recomputed from its last
durable boundary. No optimizer/RNG/executor exact-resume claim is made.
"""
import json
import os
from pathlib import Path
import subprocess
import time
from .actor import read, sha, atomic_json


def import_partial(base):
    from reproduction.prefix_only_correctness_ab.launch import birth_states
    base=Path(base);c=read(base/'preparation/launch-config.json');spec=c['partial_outer_resume']
    source=Path(spec['source_base']);outer=spec['outer'];root=source/'outers'/('outer-%06d'%outer)
    if base==source or source.parent!=base.parent or outer!=1:
        raise ValueError('This importer preserves the actually interrupted first joint outer')
    target=base/'preparation/resume-import-receipt.json'
    if target.exists(): return read(target)
    if (base/'preparation/resume-import-intent.json').exists():
        raise RuntimeError('Reconcile partial import intent; never replay blindly')
    original=read(source/'preparation/launch-config.json')
    science=('run_id','model_revision','model_files_sha256','candidate_questions','selected_questions',
        'actor_selected_questions','responses_per_question','prefix_policy_tokens','probe_updates_per_outer',
        'probe_learning_rate','probe_epochs_per_outer','probe_global_batch_average','actor_global_mini_batch',
        'actor_ppo_epochs','seed','official_model_dir','fixed_test_manifest_sha256','canonical_train_sha256',
        'grounding_probe_version')
    if any(c[k]!=original[k] for k in science):
        raise ValueError('Scientific configuration changed across phase recovery')
    receipt=read(source/'preparation/submit-receipt.json');job=receipt['job_id']
    raw=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True,timeout=15)
    if 'JobState=FAILED' not in raw or ('WorkDir='+str(source)) not in raw:
        raise ValueError('Source native instance is not actually terminal')
    ledger=read(source/'logs/process-ledger.json')
    births=[b for part in ledger['jobs'].values() for b in part['ledger'].values()]
    if birth_states(births) or not read(source/'logs/cleanup-status.json')['owned_births_all_gone']:
        raise ValueError('Source task still has live owned processes')
    actor,probe=read(source/'actor/latest.json'),read(source/'probe/latest.json')
    if (actor['run_id']!=c['run_id'] or actor['completed_outer']!=1
            or actor['actual_adam_steps_per_rank']!=[20]*8 or actor['scheduler_completed_outer']!=1
            or probe['completed_outer']!=0 or probe['actual_adam_steps_per_rank']!=[0]*8
            or actor['original_rollout_sha256']!=sha(root/'actor-selection/complete.json')):
        raise ValueError('Source committed actor20/probe0 boundaries differ')
    physical_probe=[];actor_ledgers=[]
    for rank in range(8):
        p=root/'probe-update'/('rank-%d-updates.jsonl'%rank)
        rows=[json.loads(line) for line in p.read_text().splitlines()]
        if [row['actual_adam_steps'] for row in rows]!=list(range(1,21)):
            raise ValueError('Interrupted probe actual writes are not the observed20')
        physical_probe.append(dict(rank=rank,actual_writes=20,path=str(p),sha256=sha(p)))
        p=Path(actor['state_dir'])/('rank-%d-updates.jsonl'%rank)
        rows=[json.loads(line) for line in p.read_text().splitlines()]
        writes=[r for r in rows if r.get('event')=='ACTUAL_ADAM_WRITE']
        if [r['actual_adam_steps'] for r in writes]!=list(range(1,21)):
            raise ValueError('Committed actor does not have20 direct writes on every rank')
        actor_ledgers.append(dict(rank=rank,actual_writes=20,path=str(p),sha256=sha(p)))
    phases={name:sha(root/name/'complete.json') for name in ('prefix','probe-score','full','actor-selection')}
    logs={p.name:dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in (source/'logs').glob('*.log')}
    proof=dict(source_base=str(source),job_id=job,native=raw,source_config_sha256=sha(source/'preparation/launch-config.json'),
        actor_receipt_sha256=sha(source/'actor/latest.json'),probe_receipt_sha256=sha(source/'probe/latest.json'),
        phase_hashes=phases,actor_direct_ledgers=actor_ledgers,failed_probe_direct_ledgers=physical_probe,
        failed_probe_physical_writes_per_rank=[20]*8,failed_probe_saved_state_steps_per_rank=[0]*8,
        unsaved_probe20_recompute_required=True,actor_recompute_required=False,
        full_original_logs=logs,checked_unix=time.time(),owned_births_present=[])
    atomic_json(base/'preparation/source-terminal-and-cost-proof.json',proof)
    atomic_json(base/'preparation/resume-import-intent.json',dict(run_id=c['run_id'],source_base=str(source),outer=outer))
    assets=[]
    def link_file(s,d):
        if not s.is_file() or s.is_symlink() or s.resolve()!=s:
            raise ValueError('Source import must be a real owned regular file')
        d.parent.mkdir(parents=True,exist_ok=True);os.link(s,d)
        a,b=s.stat(),d.stat()
        if (a.st_dev,a.st_ino,a.st_size)!=(b.st_dev,b.st_ino,b.st_size):
            raise ValueError('Linked state identity differs')
        assets.append(dict(source=str(s),target=str(d),bytes=a.st_size,inode=a.st_ino))
    def link_tree(s,d):
        for p in sorted(s.rglob('*')):
            if p.is_symlink():raise ValueError('No symlink tree imports')
            if p.is_file():link_file(p,d/p.relative_to(s))
    for name in ('actor','probe','question-pool'):link_tree(source/name,base/name)
    dst=base/'outers'/('outer-%06d'%outer)
    for name in ('pool-reservation.json','selection.json','questions.json','actor-binding.json'):link_file(root/name,dst/name)
    for name in phases:link_tree(root/name,dst/name)
    timings={k:dict(v,inherited_source_base=str(source)) for k,v in read(source/'logs/phase-timings.json').items()
        if k.startswith('outer-000001-') and v['exit_code']==0}
    result=dict(status='ACTUAL_ACTOR20_AND_SAME_PREFIX_FULL_ROLLOUT_IMPORTED_PROBE0_RECOMPUTE20',
        run_id=c['run_id'],source_base=str(source),outer=outer,phase_hashes=phases,
        inherited_actor_receipt_sha256=sha(base/'actor/latest.json'),inherited_successful_phase_timings=timings,
        source_originals_modified=False,assets=assets,terminal_and_cost_proof=str(base/'preparation/source-terminal-and-cost-proof.json'),
        terminal_and_cost_proof_sha256=sha(base/'preparation/source-terminal-and-cost-proof.json'),
        unsaved_probe20_recompute_required=True,actor_recompute_required=False,large_assets_via_Mac=False,
        linked_tensor_contents_rehashed=False,linked_tensor_identity='same_device_inode_size_as_original_saved_states')
    atomic_json(target,result)
    return dict(status=result['status'],source_base=str(source),outer=outer,phase_hashes=phases,
        assets_count=len(assets),actor_recompute_required=False,unsaved_probe20_recompute_required=True)
