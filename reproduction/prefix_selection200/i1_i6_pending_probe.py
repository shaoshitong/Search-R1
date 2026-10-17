"""Saved actor / zero-write pending probe recovery under the I1–I6 contract.

This boundary is not an earlier full joint. All completed actor writes and the
reserved adaptive pool are retained. A genuine stopped-writer proof plus a
separate CPU full-state qualification is required before hardlink import. No
tensor is read by planning/import, no source file is changed, and no process is
signalled. Partial unsaved probe writes are explicitly outside this adapter.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import socket
import subprocess
import time

from .contract import WORLD
from .periodic import is_probe_update, probe_boundary
from .pool import reserve_for_outer, state_sha
from .candidate_schedule import candidate_count, consumed_candidates_through, schedule_metadata
from .storage import actual_actor_boundary
from .i1_i6_joint_resume import (SCIENCE_CONFIG, canonical_path, read, sha, _once,
    _joint, _grounding, _seal, _writer_barrier, _code_inventory, _critical_ast,
    _native_writers_present, verify_source_compatibility, scientific_binding, source_envelope)
from .grounding import expected_round

ADAPTER = 'i1_i6_pending_probe_v1'
QUALIFIED = 'I1_I6_SAVED_ACTOR_PENDING_PROBE_FULL_STATE_QUALIFIED'


def validate_boundary(config, joint, actor, probe, pool, reservation):
    """Check real write counts and the exact retained pool transition, not20*n."""
    n = actor['completed_outer']
    if type(n) is not int or not is_probe_update(n):
        raise ValueError('Pending probe requires an actual periodic actor boundary10..190')
    previous, before = joint['actor'], probe_boundary(n - 1)
    if (joint['run_id'] != config['run_id'] or joint['outer'] != n - 1
            or not actual_actor_boundary(previous, n - 1) or not actual_actor_boundary(actor, n)
            or actor['run_id'] != config['run_id'] or actor['scheduler_completed_outer'] != n
            or previous['completed_outer'] != n - 1 or previous['scheduler_completed_outer'] != n - 1
            or [a-b for a,b in zip(actor['actual_adam_steps_per_rank'],actor['new_adam_steps_per_rank'])]
                != previous['actual_adam_steps_per_rank']
            or actor['sampling_actor_version'] != previous['actor_version']
            or joint['probe'] != probe or probe['run_id'] != config['run_id']
            or probe['completed_outer'] != before['last_update_outer']
            or probe['actual_adam_steps_per_rank'] != [before['actual_adam_per_rank']]*WORLD
            or actor['grounding_state'] != joint['grounding']
            or actor['grounding_probe_version'] != joint['grounding']['version']):
        raise ValueError('Saved actor, prior joint, unchanged probe or grounding boundary differs')
    old = read(joint['pool_snapshot'])
    if sha(joint['pool_snapshot']) != joint['pool_sha256']:
        raise ValueError('Prior joint pool snapshot SHA changed')
    ids, cycles, after = reserve_for_outer(old,config,n)
    if (candidate_count(config,n) != 2560 or reservation['outer'] != n
            or reservation['before_sha256'] != state_sha(old)
            or reservation['after_sha256'] != state_sha(pool)
            or reservation['indices'] != ids or reservation['cycles'] != cycles
            or reservation['after'] != pool or pool != after
            or pool['next_outer'] != n + 1
            or pool['consumed_candidates'] != consumed_candidates_through(config,n)
            or any(reservation.get(k) != v for k,v in schedule_metadata(config,n).items())):
        raise ValueError('Pending periodic pool must retain its exact adaptive history and reservation')
    return n


def _writer_absence(config,terminal):
    if (socket.gethostname()==terminal['host']
            and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==terminal['boot_id']):
        return _native_writers_present(terminal)
    # /proc on a new allocation cannot establish absence on the old node. On
    # HPC3 require fresh authoritative termination of the exact owned Slurm
    # allocation, in addition to the original all-births-gone terminal proof.
    job=str(config.get('slurm_job',''))
    if (config.get('execution_backend')!='slurm_hpc3' or re.fullmatch('[0-9]+',job) is None
            or config.get('slurm_node')!=terminal['host']
            or any(row['uid']!=os.getuid() for row in terminal['owned_births'])):
        raise RuntimeError('Cross-node writer absence requires the exact owned HPC3 allocation identity')
    out=subprocess.check_output(['sacct','-nP','-j',job,'--format=JobIDRaw,User,State,NodeList'],text=True,timeout=20)
    rows=[line.split('|') for line in out.splitlines() if line.split('|')[0].strip()==job]
    terminal_states={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}
    if (len(rows)!=1 or rows[0][1]!=pwd.getpwuid(os.getuid()).pw_name
            or rows[0][2].split()[0] not in terminal_states or rows[0][3]!=terminal['host']):
        raise RuntimeError('Original Slurm allocation is not proven terminated; do not admit another writer')
    return []


def _terminal(config,terminal,ledger,n,actor):
    if terminal.get('termination_cause') != 'FAULT_EXIT':
        raise ValueError('Pending-probe adapter is for a genuine fault, never a healthy-run handoff')
    _writer_barrier(config,terminal,ledger,completed_outer=n,
        absence_check=lambda value:_writer_absence(config,value))
    d=terminal.get('uncommitted_after_joint',{})
    if (d.get('actor_actual_writes_per_rank') != actor['new_adam_steps_per_rank']
            or d.get('probe_actual_writes_per_rank') != [0]*WORLD
            or d.get('optimizer_work_in_flight') is not False
            or d.get('all_updates_accounted') is not True):
        raise ValueError('Unaccounted or unsaved probe writes cannot be reset to an earlier checkpoint')


def inputs(base,config, *, validate_grounding=True):
    """Small-metadata semantic checks shared by CPU qualification and import."""
    base=canonical_path(base);joint=read(base/'latest-joint.json')
    actor=read(base/'actor/latest.json');probe=read(base/'probe/latest.json')
    n=actor['completed_outer'];root=base/'outers'/('outer-%06d'%n)
    if (root/'complete.json').exists() or (root/'probe-update/complete.json').exists():
        raise ValueError('Already completed joint/probe is a different recovery boundary')
    if (base/'probe/latest.pt.partial').exists():
        raise ValueError('Uncommitted probe checkpoint must be reconciled; never discard it')
    pool=read(base/'question-pool/state.json');reservation=read(root/'pool-reservation.json')
    validate_boundary(config,joint,actor,probe,pool,reservation)
    if validate_grounding:
        _joint(config,joint,n-1)
        _grounding(config,actor['grounding_state'],expected_round(n-1),adopted=True)
    if read(Path(actor['state_path'])/'complete.json') != actor:
        raise ValueError('Saved actor latest is not its immutable completed attempt')
    score=read(root/'probe-score/manifest.json');full=read(root/'full/complete.json')
    full_pool=read(root/'full-pool/complete.json');diagnostic=read(root/'diagnostic-complete.json')
    prefix=read(root/'prefix/complete.json');version=actor['sampling_actor_version']
    for row in (score,full,full_pool,prefix,diagnostic):
        if row['run_id']!=config['run_id'] or row['outer']!=n or row['actor_version']!=version:
            raise ValueError('Pending inputs must retain the actual sampling actor and outer')
    if (score['probe_before_checkpoint_sha256']!=probe['sha256']
            or score['prefix_scan_sha256']!=prefix['scan_sha256']
            or full_pool['questions']!=2560 or full_pool['trajectories']!=12800
            or full['questions']!=512 or full['trajectories']!=2560
            or actor['original_rollout_path']!=str(root/'full/complete.json')
            or actor['original_rollout_sha256']!=sha(root/'full/complete.json')
            or full['full_pool_complete_sha256']!=sha(root/'full-pool/complete.json')
            or full['selection_sha256']!=sha(root/'probe-score/actual-em-selection.json')
            or diagnostic['questions']!=512 or diagnostic['training_questions_used'] is not False):
        raise ValueError('Pending actor/probe inputs changed their original G5 generation event')
    # A directory may have been created before the first write. Its actual
    # update logs, if present, must be empty as well as the terminal proof.
    for path in (root/'probe-update').glob('*'):
        if path.name.endswith('updates.jsonl') and path.stat().st_size:
            raise ValueError('Probe wrote real updates; this zero-write adapter cannot roll them back')
        if path.name.startswith('progress-rank-'):
            raise ValueError('Probe progress exists and requires its own retained-update recovery')
    return dict(n=n,root=root,joint=joint,actor=actor,probe=probe,pool=pool,
                reservation=reservation,score=score,full_pool=full_pool,diagnostic=diagnostic)


def qualify(base,terminal_proof,terminal_sha,output):
    """Read actual actor/probe/Adam/RNG and pending input payloads once on CPU."""
    from .checkpoint_native_pending_probe import Seals, _hf
    from .i1_i6_checkpoint_cpu import actor_state, actor_ledger, probe_state
    from .checkpoint_cpu_once import validate_rng, need
    base=canonical_path(base);output=canonical_path(output,exists=False)
    config=read(base/'preparation/launch-config.json');v=inputs(base,config)
    actor,probe,n=v['actor'],v['probe'],v['n']
    terminal=read(terminal_proof);ledger=read(base/'logs/process-ledger.json')
    if sha(terminal_proof)!=terminal_sha or terminal['process_ledger_sha256']!=sha(base/'logs/process-ledger.json'):
        raise ValueError('Exact actual terminal/writer proof is required')
    _terminal(config,terminal,ledger,n,actor)
    if output.exists():
        result=read(output)
        if (result['status']!=QUALIFIED or result['pending_outer']!=n
                or result['source_execution_id']!=config['execution_id']
                or result['terminal_proof_sha256']!=terminal_sha
                or result['source_config_sha256']!=sha(base/'preparation/launch-config.json')
                or result['source_joint_sha256']!=sha(base/'latest-joint.json')):
            raise ValueError('Existing once-only qualification belongs to another boundary')
        seals={r['path']:r for r in result['verified_files']}
        for path,row in seals.items():_seal(canonical_path(path),row,seals,required=True)
        return dict(result,existing_proof_reused_without_payload_rescan=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Actual CPU qualification requires CUDA_VISIBLE_DEVICES=')
    lock=os.open(output.with_name(output.name+'.lock'),os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        intent=output.with_name(output.name+'.intent.json')
        if output.exists() or intent.exists():raise RuntimeError('Reconcile an existing qualification; never blindly replay')
        _once(intent,dict(run_id=config['run_id'],source_execution_id=config['execution_id'],pending_outer=n,
            terminal_proof_sha256=terminal_sha,started_unix=time.time()))
        history={i:read(base/'outers'/('outer-%06d'%i)/'complete.json') for i in range(1,n)}
        allowed=[Path(config['official_model_dir'])/'config.json',Path(actor['hf_model_dir']),Path(terminal_proof)]
        allowed.extend(Path(row['path']) for row in actor['rank_states']+[actor['driver_state']])
        allowed.extend(Path(actor['hf_model_dir'])/name for name in actor['hf_files'])
        for item in history.values():
            allowed.extend(Path(item['grounding'][key]) for key in ('manifest_path','commit','boundary','adoption_receipt'))
        allowed.extend(Path(actor['grounding_state'][key]) for key in ('manifest_path','commit','boundary','adoption_receipt'))
        seals=Seals(base,allowed=allowed);seals.file(terminal_proof)
        for name in ('preparation/launch-config.json','logs/process-ledger.json','latest-joint.json',
                     'actor/latest.json','probe/latest.json','probe/initialization-complete.json',
                     'question-pool/binding.json','question-pool/state.json'):
            seals.json(base/name)
        previous=[0]*WORLD
        for i,item in history.items():
            _joint(config,item,i,previous_actual=previous);previous=item['actor']['actual_adam_steps_per_rank']
            root=base/'outers'/('outer-%06d'%i)
            for path in root.rglob('*.json'):
                # Old large token/trajectory records are not recovery payloads.
                if path.name in ('complete.json','metrics.json','probe-frozen.json','grounding-refresh.json','candidate-budget.json'):
                    seals.json(path)
            for path in (item['grounding_snapshot'],item['pool_snapshot']):seals.json(path)
            state=item['grounding'];seals.json(state['manifest_path']);seals.json(state['adoption_receipt'])
            for key,name in (('commit','commit_probe.pkl'),('boundary','boundary_probe.pkl')):
                path=Path(state[key]);seals.file(path,dict(bytes=path.stat().st_size,sha256=state['head_sha256'][name]))
        hf=_hf(seals,actor,config);ranks=[]
        for rank in range(WORLD):
            saved,file=seals.mmap_torch(actor['rank_states'][rank]['path'],actor['rank_states'][rank])
            ranks.append(dict(actor_state(saved,actor,config,n,rank),file=file,ledger=actor_ledger(seals,actor,n,rank)))
            del saved;gc.collect()
        driver,file=seals.mmap_torch(actor['driver_state']['path'],actor['driver_state'])
        captured=driver['candidate_pool_state'];selected=driver['prefix_probe_receipt']
        need(driver['run_id']==config['run_id'] and driver['completed_outer']==n
            and driver['actor_version']==actor['actor_version'] and captured['value']==v['pool']
            and captured['sha256']==sha(base/'question-pool/state.json')
            and driver['grounding_state']==actor['grounding_state']
            and selected['value']==probe and selected['sha256']==v['score']['probe_before_checkpoint_sha256'],
            'Actual saved driver pool/head/grounding snapshot differs')
        driver_proof=dict(file=file,rng=validate_rng(driver['rng']));del driver;gc.collect()
        saved,file=seals.mmap_torch(base/'probe/latest.pt',probe)
        head=dict(probe_state(saved,probe,config,n-1,history),file=file);del saved;gc.collect()
        for path in v['root'].iterdir():
            if path.is_file():seals.file(path)
        # All features, original token continuations, own-EM and diagnostic
        # inputs stay remote and must pass actual full-file SHA qualification.
        for name in ('prefix','probe-score','full-pool','full','diagnostic'):
            seals.tree(v['root']/name)
        for key in ('score','full_pool','diagnostic'):
            seals.verify_producer_references(v[key])
        scan=Path(v['score']['native_hidden_scan']);test=Path(v['diagnostic']['scan_path'])
        for path,count in ((scan,12800),(test,2560)):
            manifest=seals.json(path/'scan-complete.json')
            need(manifest['count']==count and manifest['actor_version']==actor['sampling_actor_version']
                 and manifest['actor_hidden_native'] is True and manifest['independent_encoder_forward'] is False,
                 'Pending hidden input is not its actual current sampling actor')
            seals.verify_producer_references(manifest)
        need(sha(scan/'scan-complete.json')==v['score']['prefix_scan_sha256']
             and sha(test/'scan-complete.json')==v['diagnostic']['scan_sha256'], 'Pending hidden scan SHA differs')
        own=Path(v['full_pool']['own_em']['path']);seals.file(own,v['full_pool']['own_em'])
        from .probe import join_own_em,candidate_groups
        index=seals.json(v['root']/'probe-score/index.json')
        need(isinstance(index,list),'Require actual scored trajectory index')
        index={row['trajectory_id']:row for row in index}
        labels=[json.loads(line) for line in own.open()]
        joined=join_own_em(index,labels,actor['sampling_actor_version'],v['score']['prefix_scan_sha256'],v['score']['selection_sha256'])
        candidate_groups(joined,2560)
        files=seals.finish();_terminal(config,read(terminal_proof),read(base/'logs/process-ledger.json'),n,actor)
        result=dict(status=QUALIFIED,run_id=config['run_id'],source_execution_id=config['execution_id'],
            source_commit=config['source_commit'],source_config_sha256=sha(base/'preparation/launch-config.json'),
            pending_outer=n,completed_joint_outer=n-1,source_joint_sha256=sha(base/'latest-joint.json'),
            terminal_proof_sha256=terminal_sha,actor_actual_adam_steps_per_rank=actor['actual_adam_steps_per_rank'],
            probe_actual_adam_steps_per_rank=probe['actual_adam_steps_per_rank'],
            actor_rank_optimizer_rng_scheduler_verified=WORLD,actor_driver_rng_verified=True,
            probe_model_optimizer_rank_rng_verified=True,pool_exact_permutation_verified=True,
            pending_inputs_full_sha_verified=True,payload_full_sha_read_once=True,
            actual_cpu_deserialization=True,cuda_restore_verified=False,gpu_forward_performed=False,
            actor_hf=hf,actor_rank_states=ranks,actor_driver=driver_proof,probe=head,verified_files=files,
            optimizer_reset=False,automatic_rollback_performed=False,completed_unix=time.time())
        _once(output,result);return result
    finally:os.close(lock)


def plan(base,config=None):
    base=canonical_path(base);config=config or read(base/'preparation/launch-config.json')
    spec=config['resume_pending_probe']
    if spec.get('adapter')!=ADAPTER or config.get('resume_joint') or config.get('resume_import'):
        raise ValueError('Choose the explicit I1-I6 pending-probe adapter without another recovery mode')
    source=canonical_path(spec['source_base']);original=read(source/'preparation/launch-config.json')
    if (source==base or source.parent!=base.parent or config['base']!=str(base)
            or config['execution_id']==original['execution_id']
            or spec['source_execution_id']!=original['execution_id']
            or spec['source_commit']!=original['source_commit']
            or sha(source/'preparation/launch-config.json')!=spec['source_config_sha256']
            or any(config.get(k)!=original.get(k) for k in SCIENCE_CONFIG)):
        raise ValueError('Retain exact science in a distinct canonical recovery execution')
    proof=verify_source_compatibility(config)
    if (proof.get('authorized_candidate_delta') is not None
            or proof['parent_runtime_source_commit']!=original['source_commit']
            or _code_inventory(source/'repo')!=proof['parent_publication']['files_sha256']
            or _critical_ast(source/'repo')!=proof['parent_publication'].get('critical_ast_sha256',{})):
        raise ValueError('Pending recovery permits controller repairs only, not an algorithm change')
    v=inputs(source,original);n=v['n'];actor=v['actor'];joint=v['joint'];probe=v['probe']
    if (spec['pending_outer']!=n or spec['completed_joint_outer']!=n-1
            or sha(source/'latest-joint.json')!=spec['source_joint_sha256']
            or sha(source/'outers'/('outer-%06d'%(n-1))/'complete.json')!=spec['source_joint_sha256']
            or actor['binding']!=scientific_binding(dict(config,grounding_state=actor['grounding_state']),proof=proof)):
        raise ValueError('Explicit pending actor/previous joint/scientific actor binding differs')
    terminal=read(spec['terminal_proof']);ledger=read(source/'logs/process-ledger.json')
    if (sha(spec['terminal_proof'])!=spec['terminal_proof_sha256']
            or sha(source/'logs/process-ledger.json')!=terminal['process_ledger_sha256']):
        raise ValueError('Actual stopped-writer proof changed')
    _terminal(original,terminal,ledger,n,actor)
    q=read(spec['qualification_proof'])
    expected=dict(status=QUALIFIED,run_id=config['run_id'],source_execution_id=original['execution_id'],
        source_commit=original['source_commit'],pending_outer=n,completed_joint_outer=n-1,
        actor_actual_adam_steps_per_rank=actor['actual_adam_steps_per_rank'],
        probe_actual_adam_steps_per_rank=probe['actual_adam_steps_per_rank'],
        source_joint_sha256=spec['source_joint_sha256'],source_config_sha256=spec['source_config_sha256'],
        terminal_proof_sha256=spec['terminal_proof_sha256'],actor_rank_optimizer_rng_scheduler_verified=WORLD,
        actor_driver_rng_verified=True,probe_model_optimizer_rank_rng_verified=True,
        pool_exact_permutation_verified=True,payload_full_sha_read_once=True,pending_inputs_full_sha_verified=True)
    if sha(spec['qualification_proof'])!=spec['qualification_proof_sha256'] or any(q.get(k)!=v for k,v in expected.items()):
        raise ValueError('A genuine full-state CPU qualification for this exact pending boundary is required')
    seals={row['path']:row for row in q['verified_files']}
    if len(seals)!=len(q['verified_files']):raise ValueError('Duplicate qualified path')
    assets={}
    def add(path, target=None, expected=None):
        path=canonical_path(path);destination=base/(target or path.relative_to(source))
        if str(path) not in seals:raise ValueError('Required pending-state input was not CPU sealed: '+str(path))
        row=_seal(path,expected or seals[str(path)],seals,required=True)
        if path.suffix=='.json' and sha(path)!=row['sha256']:raise ValueError('Sealed metadata changed')
        item=dict(row,source=str(path),target=str(destination))
        if str(destination) in assets and assets[str(destination)]!=item:raise ValueError('Conflicting recovery aliases')
        assets[str(destination)]=item
    for name in ('actor/latest.json','probe/latest.json','probe/latest.pt','probe/initialization-complete.json',
                 'question-pool/binding.json','question-pool/state.json','latest-joint.json'):
        add(source/name)
    previous=[0]*WORLD
    for outer in range(1,n):
        root=source/'outers'/('outer-%06d'%outer);item=read(root/'complete.json')
        _joint(original,item,outer,previous_actual=previous);previous=item['actor']['actual_adam_steps_per_rank']
        for name in ('complete.json','probe-frozen.json','probe-update/metrics.json','probe-update/complete.json',
                     'validation/complete.json','grounding-refresh.json','candidate-budget.json'):
            if (root/name).is_file():add(root/name)
    # Only tiny input metadata is aliased; tensors/rollouts remain at the exact
    # sealed source paths consumed by the pending update, not copied/resampled.
    for path in sorted(v['root'].iterdir()):
        if path.is_file() and path.suffix=='.json':add(path)
    for name in ('prefix/complete.json','probe-score/manifest.json','probe-score/selection.json',
                 'probe-score/actual-em-selection.json','full/complete.json','full-pool/complete.json'):
        add(v['root']/name)
    for row in actor['rank_states']+[actor['driver_state']]:
        _seal(canonical_path(row['path']),row,seals,required=True)
    for name,row in actor['hf_files'].items():_seal(canonical_path(Path(actor['hf_model_dir'])/name),row,seals,required=True)
    # All sampled inputs remain live dependencies, including already sealed
    # native hidden tensors. Re-stat every seal before admitting CUDA.
    for path,row in seals.items():_seal(canonical_path(path),row,seals,required=True)
    return dict(status='I1_I6_QUALIFIED_PENDING_PROBE_IMPORT_PLAN',adapter=ADAPTER,
        run_id=config['run_id'],execution_id=config['execution_id'],source_base=str(source),base=str(base),
        pending_outer=n,completed_joint_outer=n-1,source_inputs_root=str(v['root']),
        actor_actual_adam_steps_per_rank=actor['actual_adam_steps_per_rank'],
        probe_actual_adam_steps_per_rank=probe['actual_adam_steps_per_rank'],
        consumed_candidates=v['pool']['consumed_candidates'],assets=list(assets.values()),
        qualification_proof_sha256=spec['qualification_proof_sha256'],
        source_joint_sha256=spec['source_joint_sha256'],source_binding=source_envelope(config,proof),
        tensor_payloads_read=0,actor_updates_to_reexecute=0,candidates_to_resample=0,
        probe_initialized_again=False,optimizer_reset=False,automatic_rollback_performed=False,
        cuda_restore_verified=False,source_modified=False,planned_unix=time.time())


def import_pending(base):
    base=canonical_path(base);config=read(base/'preparation/launch-config.json')
    fd=os.open(base/'preparation/pending-probe-import.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=base/'preparation/resume-pending-probe-receipt.json'
        if receipt.exists():validate_imported_pending(base,config);return read(receipt)
        intent=base/'preparation/resume-pending-probe-intent.json'
        if intent.exists():raise RuntimeError('Interrupted import requires reconciliation, never blind replay')
        result=plan(base,config);_once(intent,result)
        for row in result['assets']:
            source=canonical_path(row['source']);target=canonical_path(row['target'],exists=False)
            if not target.is_relative_to(base) or target.exists():raise ValueError('Recovery alias occupied or outside target')
            target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target,follow_symlinks=False)
        result=dict(result,status='I1_I6_SAVED_ACTOR_PENDING_PROBE_IMPORTED')
        _once(receipt,result);validate_imported_pending(base,config);return result
    finally:os.close(fd)


def validate_imported_pending(base,config):
    base=canonical_path(base);actual=plan(base,config)
    receipt=read(base/'preparation/resume-pending-probe-receipt.json')
    if (receipt['status']!='I1_I6_SAVED_ACTOR_PENDING_PROBE_IMPORTED'
            or receipt['execution_id']!=config['execution_id'] or receipt['adapter']!=ADAPTER
            or any(receipt[k]!=actual[k] for k in ('assets','pending_outer','qualification_proof_sha256','source_joint_sha256','source_binding'))
            or (base/'outers'/('outer-%06d'%actual['pending_outer'])/'probe-update').exists()):
        raise ValueError('Imported pending recovery changed or has already begun its probe phase')
    for row in receipt['assets']:
        source=canonical_path(row['source']);target=canonical_path(row['target'])
        if (source.stat().st_dev,source.stat().st_ino)!=(target.stat().st_dev,target.stat().st_ino):
            raise ValueError('Retained recovery alias changed')
    return actual['completed_joint_outer']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('plan','import','qualify'));p.add_argument('--base',required=True)
    p.add_argument('--terminal-proof');p.add_argument('--terminal-sha256');p.add_argument('--output')
    a=p.parse_args()
    if a.phase=='qualify':
        if not all((a.terminal_proof,a.terminal_sha256,a.output)):p.error('qualify requires terminal proof/SHA/output')
        result=qualify(a.base,a.terminal_proof,a.terminal_sha256,a.output)
    else:result=plan(a.base) if a.phase=='plan' else import_pending(a.base)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
