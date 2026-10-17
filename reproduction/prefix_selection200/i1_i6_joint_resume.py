"""Stopped-writer, full-joint recovery for the I1–I6 scientific contract.

Planning reads JSON/code and stat metadata only. Import requires a separately
recorded genuine once-only full-state qualification; it never deserializes a
checkpoint, rehashes tensor/weight payloads, signals a process or admits CUDA.
Historical records and state blobs retain their original bytes/source identity.
"""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import time

from .actor import canonical, _state_binding
from .contract import CANDIDATE_QUESTIONS, WORLD
from .grounding import expected_round, static_binding
from .periodic import probe_boundary
from .pool import state_sha
from .storage import actual_actor_boundary

ADAPTER = 'i1_i6_joint_v1'
QUALIFIED = 'I1_I6_ACTUAL_FULL_STATE_QUALIFIED_ONCE'
HPC3_BASE = Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017')
CONTROL_FILES = frozenset({
    'reproduction/prefix_selection200/launch.py',
    'reproduction/prefix_selection200/i1_i6_joint_resume.py',
    'reproduction/prefix_selection200/i1_i6_actor_resume.py',
    'reproduction/prefix_selection200/i1_i6_checkpoint_cpu.py',
    'reproduction/prefix_selection200/i1_i6_pending_probe.py',
    'reproduction/prefix_selection200/test_i1_i6_pending_probe.py',
    'reproduction/hpc3_prefix5/check_recovery_cpu.py',
    'reproduction/hpc3_prefix5/observe.py',
    'reproduction/hpc3_prefix5/test_observe.py',
    'reproduction/hpc3_prefix5/test_diagnostic_scope.py',
    'reproduction/hpc3_prefix5/terminal.py',
    'reproduction/hpc3_prefix5/test_terminal.py',
    'reproduction/hpc3_prefix5/recover_diagnostic.py',
    'reproduction/operations/a800_prefix5_variable_pool_control.py',
    'reproduction/operations/a800_i1_i6_boundary_handoff.py',
    'reproduction/prefix_selection200/test_i1_i6_joint_resume.py',
    'reproduction/prefix_selection200/test_candidate_schedule.py',
})
CANDIDATE_DELTA_FILES = frozenset('reproduction/prefix_selection200/' + name for name in (
    'candidate_schedule.py','contract.py','periodic.py','pool.py','rollout.py','probe.py',
    'periodic_data.py','storage.py','native_features.py','guard_retry.py'))
PROBE_CRITICAL = ('update','load_online','initialize','initial_online_state','atomic_checkpoint','setup_ddp',
                  'batch_subset','concatenate','inference_logits','mixed_probabilities','stable_top512','join_own_em',
                  'PrefixEncoder','SelectedFeatures')


def _critical_ast(repo):
    path=Path(repo)/'reproduction/prefix_selection200/probe.py'
    if not path.exists():return {}
    tree=ast.parse(path.read_text());nodes={node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    return {name:hashlib.sha256(ast.dump(nodes[name],include_attributes=False).encode()).hexdigest()
            for name in PROBE_CRITICAL if name in nodes}


def _candidate_delta(value):
    if value is None:return None
    binding=value['candidate_schedule']
    expected=dict(effective_from_outer=binding['effective_from_outer'],legacy_candidates=2560,ordinary_candidates=5120,
        update_candidates=2560,probe_update_period=10,probe_update_at_final=False,selected_questions=512)
    if (binding!=expected or type(binding['effective_from_outer']) is not int or not 2<=binding['effective_from_outer']<=200 or
            re.fullmatch('[0-9a-f]{64}',value.get('human_intent_sha256','')) is None or
            value.get('only_authorized_algorithm_delta')!='ordinary_candidate_pool_5120_periodic2560'):
        raise ValueError('Bind the exact latest human-authorized candidate schedule and intent SHA')
    return value

SCIENCE_CONFIG = ('run_id','outers','world_size','model_revision','model_files_sha256','official_model_dir',
    'canonical_train','canonical_train_sha256','source_question_pool','source_question_pool_sha256',
    'common_manifest_sha256','composite_manifest_sha256','candidate_questions','selected_questions',
    'responses_per_question','prefix_policy_tokens','exploration_questions','actor_global_mini_batch',
    'actor_ppo_epochs','actor_loss_reduction','actor_loss_virtual_world','policy_safeguards','seed',
    'nonmixed_advantage_mode','actual_em_selection_fallback','probe_learning_rate','probe_update_period',
    'probe_updates_per_period','probe_update_at_final','probe_epochs_per_update','probe_global_batch_average',
    'probe_source_sha256','probe_source_epoch','probe_source_model_kind','fixed_test_manifest_sha256',
    'fixed_test_identity_sha256','interaction_protocol','hidden_capture_contract','independent_encoder_forward',
    'grounding_refresh_enabled','grounding_refit_rounds','grounding_calibration_question_ids',
    'grounding_calibration_sha256','grounding_calibration_data_sha256','validation_questions_sha256',
    'final_questions_sha256','adaptive_candidates')


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()


def canonical_path(path, *, exists=True):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or path.resolve(strict=exists) != path:
        raise ValueError('Require an actual canonical absolute recovery path')
    cursor = path
    while cursor != cursor.parent:
        if cursor.is_symlink(): raise ValueError('Recovery paths may not follow symlinks')
        cursor = cursor.parent
    return path


def _raw(path):
    path=canonical_path(path); descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        # Shared-storage metadata can settle after the creator closes/fsyncs a
        # new receipt. Retry a rejected read, never return bytes from an unstable
        # attempt. The accepted attempt still requires identical before/after
        # device/inode/size/mtime/ctime and callers retain exact SHA checks.
        for _ in range(3):
            os.lseek(descriptor,0,os.SEEK_SET)
            before=os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode): raise ValueError('Recovery metadata must be a regular file')
            with os.fdopen(os.dup(descriptor),'rb') as stream: result=stream.read()
            if _signature(before)==_signature(os.fstat(descriptor)): return result
        raise ValueError('Recovery metadata changed while read after three bounded attempts')
    finally: os.close(descriptor)


def read(path): return json.loads(_raw(path))


def sha(path): return hashlib.sha256(_raw(path)).hexdigest()


def _signature(info): return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)


def _once(path,value):
    path=canonical_path(path,exists=False);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(descriptor,'wb') as stream:
        stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())


def source_tree(repo):
    """Admit only our published HPC3 alias; payload paths stay symlink-free."""
    repo=Path(repo)
    if repo==HPC3_BASE/'repo' and repo.is_symlink():
        target=repo.resolve(strict=True)
        if (target.parent!=HPC3_BASE/'source' or re.fullmatch('[0-9a-f]{40}',target.name) is None
                or target.stat().st_uid!=os.getuid() or repo.lstat().st_uid!=os.getuid()):
            raise ValueError('HPC3 source alias must name this user-owned immutable publication')
        return canonical_path(target)
    return canonical_path(repo)


def _code_inventory(repo):
    repo=source_tree(repo);result={}
    for path in sorted(repo.rglob('*')):
        relative=path.relative_to(repo)
        if any(part in ('.git','__pycache__','.pytest_cache','.venv') for part in relative.parts): continue
        if path.suffix in ('.py','.sh'):
            result[relative.as_posix()]=sha(canonical_path(path))
    return result


def source_manifest(repo, source_commit, *, science_source_commit=None):
    """Small code publication manifest; the publisher pins its actual Git commit."""
    for value in (source_commit,science_source_commit or source_commit):
        if re.fullmatch('[0-9a-f]{40}',value or '') is None: raise ValueError('Published source commit must be explicit')
    repo=source_tree(repo)
    return dict(schema_version=1,source_commit=source_commit,science_source_commit=science_source_commit or source_commit,
                files_sha256=_code_inventory(repo),critical_ast_sha256=_critical_ast(repo),payload_files_read=0)


def compatibility_proof(parent_manifest,target_manifest, *, authorized_candidate_delta=None):
    parent=parent_manifest['files_sha256'];target=target_manifest['files_sha256']
    delta=_candidate_delta(authorized_candidate_delta)
    allowed=CONTROL_FILES | (CANDIDATE_DELTA_FILES if delta else frozenset())
    names=set(parent)|set(target);changed={name for name in names if parent.get(name)!=target.get(name)}
    if not changed <= allowed or not (names-allowed):
        raise ValueError('Only explicitly enumerated recovery/controller files may differ; actor/math has no exception')
    science={name:parent[name] for name in sorted(names-allowed) if name in parent}
    if any(target.get(name)!=value for name,value in science.items()): raise ValueError('Scientific source SHA differs')
    if parent_manifest.get('critical_ast_sha256',{})!=target_manifest.get('critical_ast_sha256',{}):
        raise ValueError('Head optimizer/training/restore critical AST changed outside the candidate metadata delta')
    return dict(schema_version=1,adapter=ADAPTER,authorized_candidate_delta=delta,parent_publication=parent_manifest,target_publication=target_manifest,
        parent_runtime_source_commit=parent_manifest['source_commit'],
        science_source_commit=parent_manifest['science_source_commit'],
        runtime_source_commit=target_manifest['source_commit'],scientific_files_sha256=science,
        controller_changes={name:dict(parent_sha256=parent.get(name),target_sha256=target.get(name)) for name in sorted(changed)},
        actor_file_sha_unchanged=True,head_training_critical_ast_unchanged=True,
        scientific_payload_or_algorithm_changed=bool(delta),only_scientific_delta='candidate_schedule' if delta else None)


def verify_source_compatibility(config):
    path=config['source_compatibility'];proof=read(path)
    if sha(path)!=config['source_compatibility_sha256'] or proof!=compatibility_proof(proof['parent_publication'],proof['target_publication'],
            authorized_candidate_delta=proof.get('authorized_candidate_delta')):
        raise ValueError('Explicit source compatibility proof differs')
    delta=proof.get('authorized_candidate_delta')
    if delta and (config.get('candidate_schedule')!=delta['candidate_schedule'] or
            config.get('candidate_schedule_authorization_sha256')!=delta['human_intent_sha256'] or
            sha(config['candidate_schedule_authorization'])!=delta['human_intent_sha256']):
        raise ValueError('Candidate schedule changed without its exact pinned human authorization')
    if (config['source_commit']!=proof['runtime_source_commit'] or
            config.get('science_source_commit')!=proof['science_source_commit'] or
            proof['target_publication']['science_source_commit']!=proof['science_source_commit'] or
            _code_inventory(Path(config['base'])/'repo')!=proof['target_publication']['files_sha256'] or
            _critical_ast(Path(config['base'])/'repo')!=proof['target_publication'].get('critical_ast_sha256',{})):
        raise ValueError('Runtime source/scientific source or actual target code differs from its publication')
    return proof


def source_envelope(config, proof):
    return dict(runtime_source_commit=config['source_commit'],science_source_commit=proof['science_source_commit'],
        source_compatibility=config['source_compatibility'],source_compatibility_sha256=config['source_compatibility_sha256'],
        legacy_binding_source_commit_role='immutable_actor_and_unchanged_contract_source_commit',
        authorized_candidate_delta=proof.get('authorized_candidate_delta'))


def scientific_binding(config, *, original_binding=_state_binding, proof=None):
    proof=proof or verify_source_compatibility(config)
    if (config['source_commit']!=proof['runtime_source_commit'] or config.get('science_source_commit')!=proof['science_source_commit']):
        raise ValueError('Cannot change source_commit to bypass actor restore')
    # This is the sole compatibility mapping. The caller's runtime config and
    # top-level runtime provenance retain the new controller commit unchanged.
    result=original_binding(dict(config,source_commit=proof['science_source_commit']))
    if result['source_commit']!=proof['science_source_commit']: raise ValueError('Unexpected scientific binding schema')
    return result


def _native_writers_present(terminal):
    """Future execution-host check; permission/host uncertainty is never absence."""
    if socket.gethostname()!=terminal['host'] or Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=terminal['boot_id']:
        raise RuntimeError('Confirm owned writers on their original execution host/boot')
    from reproduction.a800.owned_processes import identity
    present=[]
    for birth in terminal['owned_births']:
        try: current=identity(birth['pid'])
        except (FileNotFoundError,ProcessLookupError): continue
        if all(current[key]==birth[key] for key in ('pid','start_ticks','sid','uid')) and current['state']!='Z': present.append(current)
    return present


def _writer_barrier(config,terminal,ledger, *, authorized_handoff_sha=None, completed_outer=None, absence_check=None):
    if (terminal.get('run_id')!=config['run_id'] or terminal.get('execution_id')!=config['execution_id'] or
            terminal.get('source_commit')!=config['source_commit'] or terminal.get('base')!=config['base'] or
            terminal.get('termination_cause') not in ('FAULT_EXIT','AUTHORIZED_BOUNDARY_HANDOFF') or
            terminal.get('pipeline_alive') is not False or
            terminal.get('owned_writers_gone') is not True or terminal.get('actual_births_present')!=[] or
            not terminal.get('owned_births') or ledger.get('run_id')!=config['run_id'] or
            any(row.get('state')!='STOPPED' for row in ledger.get('jobs',{}).values())):
        raise ValueError('Only a genuine fault with stopped owned writers may be imported; never stop a healthy source')
    if terminal['termination_cause']=='AUTHORIZED_BOUNDARY_HANDOFF' and (
            authorized_handoff_sha is None or terminal.get('human_intent_sha256')!=authorized_handoff_sha or
            terminal.get('stopped_after_complete_outer')!=completed_outer):
        raise ValueError('Planned boundary handoff requires the exact latest explicit human authorization')
    recorded={(row['pid'],row['start_ticks'],row['sid'],row['uid']) for row in terminal['owned_births']}
    known={(row['pid'],row['start_ticks'],row['sid'],row['uid']) for job in ledger.get('jobs',{}).values()
           for row in job.get('ledger',{}).values()}
    if absence_check is None and config.get('execution_backend') == 'slurm_hpc3':
        from reproduction.hpc3_prefix5.terminal import writers_present
        absence_check = lambda value: writers_present(value, config)
    if not known or not known<=recorded or (absence_check or _native_writers_present)(terminal):
        raise RuntimeError('Actual source owned writers remain present or their birth coverage is incomplete')


def _seal(path, expected, seals, *, required):
    """Consume a once-verified digest and immutable inode; do not read payload."""
    path=canonical_path(path);info=path.stat();row=seals.get(str(path))
    if not stat.S_ISREG(info.st_mode) or info.st_size!=expected['bytes']:
        raise ValueError('Producer payload path/size differs')
    if not re.fullmatch('[0-9a-f]{64}',expected['sha256']): raise ValueError('Actual producer SHA is missing')
    if row is None:
        if required: raise ValueError('Genuine full-state qualification omitted a required payload')
        return dict(path=str(path),bytes=info.st_size,sha256=expected['sha256'],dev=info.st_dev,inode=info.st_ino,
                    mtime_ns=info.st_mtime_ns,qualification='UNKNOWN_NOT_REHASHED')
    if (row['sha256']!=expected['sha256'] or row['bytes']!=info.st_size or row['dev']!=info.st_dev or
            row['inode']!=info.st_ino or row['mtime_ns']!=info.st_mtime_ns):
        raise ValueError('Payload changed since its genuine once-only qualification')
    return dict(row,qualification='EXISTING_ACTUAL_ONCE_ONLY_SEAL_NOT_REPEATED')


def _grounding(config,state,round, *, adopted):
    # JSON identity only. Head/model tensor bytes belong to the once-only seal.
    manifest=read(state['manifest_path'])
    if (state['run_id']!=config['run_id'] or state['round']!=round or state['lifecycle']!=static_binding(config) or
            sha(state['manifest_path'])!=state['manifest_sha256'] or
            any(state[key]!=manifest[key] for key in ('round','version','encoder','encoder_sha256','head_sha256','boundary','commit'))):
        raise ValueError('Saved grounding state/immutable manifest/scientific lifecycle differs')
    if round and manifest['run_id']!=config['run_id']: raise ValueError('Scheduled grounding fit belongs to another run')
    if adopted:
        receipt=read(state['adoption_receipt'])
        if (sha(state['adoption_receipt'])!=state['adoption_receipt_sha256'] or receipt['run_id']!=config['run_id'] or
                any(receipt[key]!=state[key] for key in ('round','version','manifest_sha256')) or
                sorted(row['rank'] for row in receipt['replicas'])!=list(range(WORLD)) or
                any(row.get('ok') is not True or any(row[key]!=state[key] for key in ('round','version','encoder')) for row in receipt['replicas'])):
            raise ValueError('Retain the actual four-replica adopted grounding receipt')
    return manifest


def _joint(config,joint,outer, *, previous_actual=None):
    actor=joint['actor'];probe=joint['probe'];b=probe_boundary(outer)
    if config.get('adaptive_candidates') is not None:
        from .candidate_schedule import schedule_metadata
        if joint.get('candidate_schedule')!=schedule_metadata(config,outer):
            raise ValueError('Joint checkpoint lost its exact adaptive candidate budget/feedback')
    if (joint['run_id']!=config['run_id'] or joint['outer']!=outer or not actual_actor_boundary(actor,outer) or
            actor['run_id']!=config['run_id'] or actor['completed_outer']!=outer or actor['scheduler_completed_outer']!=outer or
            probe['run_id']!=config['run_id'] or probe['completed_outer']!=b['last_update_outer'] or
            probe['actual_adam_steps_per_rank']!=[b['actual_adam_per_rank']]*WORLD or
            actor['grounding_state']!=joint['grounding'] or actor['grounding_probe_version']!=joint['grounding']['version'] or
            read(joint['grounding_snapshot'])!=joint['grounding'] or sha(joint['grounding_snapshot'])!=joint['grounding_snapshot_sha256']):
        raise ValueError('Actual complete joint actor/probe/grounding boundary differs')
    if previous_actual is not None and [a-b for a,b in zip(actor['actual_adam_steps_per_rank'],actor['new_adam_steps_per_rank'])]!=previous_actual:
        raise ValueError('History does not retain each actual partial Adam contribution')
    _grounding(config,joint['grounding'],expected_round(outer-1),adopted=True)
    return actor,probe


def pending_candidate_questions(config, original, outer):
    """Planning precedes import: adaptive provenance still lives at the source."""
    from .candidate_schedule import candidate_count
    if config.get('adaptive_candidates') is not None:
        if (config['adaptive_candidates'] != original.get('adaptive_candidates')
                or config.get('candidate_schedule') != original.get('candidate_schedule')):
            raise ValueError('Adaptive recovery must retain its original schedule and feedback policy')
        return candidate_count(original, outer)
    return candidate_count(config, outer)


def plan(base,config=None):
    """Read-only recovery plan; UNKNOWN qualification cannot authorize import."""
    base=canonical_path(base);config=config or read(base/'preparation/launch-config.json');spec=config['resume_joint']
    if (spec.get('adapter')!=ADAPTER or config.get('resume_pending_probe') or config.get('resume_import') or
            config['base']!=str(base) or config.get('outers')!=200 or config.get('world_size')!=WORLD):
        raise ValueError('Use an explicit new I1–I6 full-joint recovery namespace and total budget200')
    source=canonical_path(spec['source_base']);original=read(source/'preparation/launch-config.json')
    if (source==base or source.parent!=base.parent or config['run_id']!=original['run_id'] or
            config['execution_id']==original['execution_id'] or sha(source/'preparation/launch-config.json')!=spec['source_config_sha256'] or
            spec['source_commit']!=original['source_commit'] or spec['source_execution_id']!=original['execution_id'] or
            any(config.get(key)!=original.get(key) for key in SCIENCE_CONFIG if key!='candidate_questions')):
        raise ValueError('Full-joint recovery must retain science in a distinct execution')
    proof=verify_source_compatibility(config)
    delta=proof.get('authorized_candidate_delta')
    if (config.get('candidate_questions')!=original.get('candidate_questions') and not delta):
        raise ValueError('Candidate budget differs without its explicit authorized schedule')
    if delta and delta['candidate_schedule']['effective_from_outer']!=spec['completed_outer']+1:
        raise ValueError('Candidate schedule must start after the imported complete logical boundary')
    if proof['parent_runtime_source_commit']!=original['source_commit'] or (_code_inventory(source/'repo')!=proof['parent_publication']['files_sha256'] or
            _critical_ast(source/'repo')!=proof['parent_publication'].get('critical_ast_sha256',{})):
        raise ValueError('Parent scientific source publication differs from the actual source checkout')
    terminal=read(spec['terminal_proof']);ledger=read(source/'logs/process-ledger.json')
    if sha(spec['terminal_proof'])!=spec['terminal_proof_sha256'] or sha(source/'logs/process-ledger.json')!=terminal['process_ledger_sha256']:
        raise ValueError('Actual fault/owned-writer proof changed')
    n=spec['completed_outer']
    _writer_barrier(original,terminal,ledger,authorized_handoff_sha=config.get('candidate_schedule_authorization_sha256'),completed_outer=n)
    if type(n) is not int or not 1<=n<=200: raise ValueError('A complete joint1..200 is required; fresh0 is forbidden')
    joint=read(source/'latest-joint.json')
    if sha(source/'latest-joint.json')!=spec['source_joint_sha256'] or sha(source/'outers'/('outer-%06d'%n)/'complete.json')!=spec['source_joint_sha256']:
        raise ValueError('Source joint and its immutable history differ')
    actor,probe=_joint(original,joint,n)
    if read(source/'actor/latest.json')!=actor or read(source/'probe/latest.json')!=probe:
        raise ValueError('A committed actor/pending probe is a different boundary; never discard its actual writes')
    binding_config=dict(config,grounding_state=joint['grounding'])
    if actor['binding']!=scientific_binding(binding_config,proof=proof): raise ValueError('Scientific actor source/model/data/math binding differs')
    disposition=terminal.get('uncommitted_after_joint',{})
    if (disposition.get('actor_actual_writes_per_rank')!=[0]*WORLD or disposition.get('probe_actual_writes_per_rank')!=[0]*WORLD or
            disposition.get('optimizer_work_in_flight') is not False or disposition.get('all_updates_accounted') is not True):
        raise ValueError('Uncommitted actual/unknown updates require their own full-state recovery; no silent rollback')
    qpath=spec.get('qualification_proof');q=read(qpath) if qpath else {}
    qualified=bool(qpath and sha(qpath)==spec['qualification_proof_sha256'] and q.get('status')==QUALIFIED and
        q.get('run_id')==config['run_id'] and q.get('source_execution_id')==original['execution_id'] and
        q.get('source_commit')==original['source_commit'] and q.get('source_joint_sha256')==spec['source_joint_sha256'] and
        q.get('completed_outer')==n and q.get('actual_adam_steps_per_rank')==actor['actual_adam_steps_per_rank'] and
        q.get('actor_rank_optimizer_rng_scheduler_verified')==WORLD and q.get('actor_driver_rng_verified') is True and
        q.get('probe_model_optimizer_rank_rng_verified') is True and q.get('pool_exact_permutation_verified') is True and
        q.get('payload_full_sha_read_once') is True)
    seals={row['path']:row for row in q.get('verified_files',[])}
    if len(seals)!=len(q.get('verified_files',[])): raise ValueError('Full-state qualification repeats a payload identity')
    assets=[];metadata={};history=[]
    def add(path,expected=None, *, target=None):
        path=canonical_path(path)
        relative=(path.relative_to(source) if path.is_relative_to(source) else Path('recovery-assets')/hashlib.sha256(str(path).encode()).hexdigest()/path.name)
        destination=base/(target or relative)
        if path.suffix=='.json':
            value=read(path);checksum=sha(path);metadata[str(path)]=checksum
            if expected and checksum!=expected['sha256']: raise ValueError('Producer JSON metadata SHA differs')
            expected=dict(bytes=path.stat().st_size,sha256=checksum)
        elif expected is None:
            if str(path) not in seals: raise ValueError('Payload has no existing qualification/producer receipt')
            expected=seals[str(path)]
        row=_seal(path,expected,seals,required=qualified and path.suffix!='.json')
        item=dict(row,source=str(path),target=str(destination));assets.append(item);return item
    for name in ('actor/latest.json','probe/latest.json','probe/latest.pt','probe/initialization-complete.json','question-pool/binding.json','latest-joint.json'):
        expected=probe if name=='probe/latest.pt' else None
        add(source/name,expected)
    from .candidate_schedule import candidate_count,consumed_candidates_through
    pool=read(joint['pool_snapshot']);consumed=consumed_candidates_through(original,n)
    if (pool['next_outer']!=n+1 or pool['consumed_candidates']!=consumed or
            pool['cycle']*len(pool['order'])+pool['cursor']!=consumed or
            sorted(pool['order'])!=list(range(len(pool['order']))) or sha(joint['pool_snapshot'])!=joint['pool_sha256']):
        raise ValueError('Exact saved candidate permutation/cycle/cursor differs')
    add(joint['pool_snapshot'],target='question-pool/state.json');add(joint['pool_snapshot'])
    add(joint['probe_snapshot'],probe)
    if (source/'probe/latest.pt').stat().st_ino!=Path(joint['probe_snapshot']).stat().st_ino:
        raise ValueError('Joint probe snapshot must be the actual online state inode')
    for item in actor['rank_states']:add(item['path'],item)
    if len(actor['rank_states'])!=WORLD or {Path(row['path']).name for row in actor['rank_states']}!={'rank-%d.pt'%rank for rank in range(WORLD)}:
        raise ValueError('All four actor optimizer/scheduler/RNG blobs are required')
    add(actor['driver_state']['path'],actor['driver_state'])
    for name,item in actor['hf_files'].items():add(Path(actor['hf_model_dir'])/name,item)
    previous=[0]*WORLD
    for outer in range(1,n+1):
        root=source/'outers'/('outer-%06d'%outer);saved=read(root/'complete.json')
        historical_actor,_=_joint(original,saved,outer,previous_actual=previous);previous=historical_actor['actual_adam_steps_per_rank']
        if re.fullmatch('[0-9a-f]{40}',saved['source_commit']) is None:raise ValueError('Historical source commit must remain explicit')
        history.append(dict(outer=outer,source_commit=saved['source_commit'],sha256=sha(root/'complete.json')))
        add(root/'complete.json');add(saved['grounding_snapshot'])
        for name in ('probe-frozen.json','probe-update/complete.json','probe-update/metrics.json','validation/complete.json','grounding-refresh.json','candidate-budget.json'):
            path=root/name
            if path.is_file():add(path)
    state=joint['grounding'];action='RESTART_SAME_ADOPTED_VERSION';pending_fit=None
    if n in (50,100,150):
        fitted=source/'grounding/rounds'/('round-%06d'%n)/'fit-complete.json'
        if fitted.is_file():
            state=read(fitted);manifest=_grounding(original,state,n,adopted=False)
            if state.get('source_actor_version')!=actor['actor_version'] or state['encoder']!=actor['hf_model_dir'] or manifest['encoder_sha256']!={key:row['sha256'] for key,row in actor['hf_files'].items()}:
                raise ValueError('Existing due fit does not bind the completed current actor')
            pending_fit=str(fitted);add(fitted);action='ADOPT_EXISTING_FIT_NO_REFIT'
        elif (fitted.parent/'fit-intent.json').exists():action='RECONCILE_INTERRUPTED_REFIT_NO_BLIND_REPLAY'
        else:action='FIT_SAVED_CURRENT_ACTOR_ONCE_BEFORE_NEXT_OUTER'
    active=source/'grounding/latest.json'
    if active.is_file():
        live=read(active)
        if live['round']==expected_round(n):
            _grounding(original,live,expected_round(n),adopted=True)
            if n in (50,100,150) and (pending_fit is None or live['version']!=state['version']):
                raise ValueError('Adopted due evaluator lacks its real matching completed fit')
            state=live;action='RESTART_ADOPTED_EXISTING_VERSION_NO_REFIT';add(active)
    # Preserve both the evaluator actually used by jointN and the due evaluator.
    for value in (joint['grounding'],state):
        add(value['manifest_path'])
        for key,name in (('commit','commit_probe.pkl'),('boundary','boundary_probe.pkl')):
            path=Path(value[key]);add(path,dict(bytes=path.stat().st_size,sha256=value['head_sha256'][name]))
        if 'adoption_receipt' in value:add(value['adoption_receipt'])
    pending=n+1;pending_root=source/'outers'/('outer-%06d'%pending);phases={}
    if n<200 and (pending_root/'pool-reservation.json').exists():
        reservation=read(pending_root/'pool-reservation.json');current=read(source/'question-pool/state.json')
        if (reservation['outer']!=pending or reservation['before_sha256']!=state_sha(pool) or
                reservation['after']!=current or reservation['after_sha256']!=state_sha(current) or
                current['next_outer']!=pending+1 or current['consumed_candidates']!=consumed_candidates_through(original,pending) or
                len(reservation['indices'])!=pending_candidate_questions(config,original,pending)):
            raise ValueError('Pending reservation must be the exact next saved-pool transition')
        for name in ('pool-reservation.json','selection.json','questions.json','actor-binding.json'):add(pending_root/name)
        if original.get('adaptive_candidates') is not None:add(pending_root/'candidate-budget.json')
        for phase,marker in (('prefix','complete.json'),('probe-score','manifest.json'),('full','complete.json')):
            path=pending_root/phase/marker
            if not path.exists():continue
            if not qualified or q.get('phase_hashes',{}).get(phase)!=sha(path):continue
            value=read(path)
            if value['run_id']!=config['run_id'] or value['outer']!=pending or value['actor_version']!=actor['actor_version']:
                raise ValueError('Qualified pending sampling event uses another actual actor')
            phases[phase]=dict(marker=phase+'/'+marker,sha256=sha(path))
            for path,row in seals.items():
                if Path(path).is_relative_to(pending_root/phase):add(path,row)
    elif read(source/'question-pool/state.json')!=pool:raise ValueError('Source pool advanced without a sealed next reservation')
    # Deduplicate identical aliases while preserving distinct hardlinks needed by consumers.
    by_target={}
    for item in assets:
        if item['target'] in by_target and by_target[item['target']]['source']!=item['source']:raise ValueError('Conflicting recovery aliases')
        by_target[item['target']]=item
    return dict(schema_version=1,adapter=ADAPTER,status='READY_QUALIFIED_FULL_JOINT_IMPORT' if qualified else 'NEEDS_GENUINE_FULL_STATE_QUALIFICATION',
        run_id=config['run_id'],execution_id=config['execution_id'],source_base=str(source),base=str(base),
        completed_outer=n,next_outer=n+1,total_logical_budget=200,remaining_logical_outers=200-n,
        actual_adam_steps_per_rank=actor['actual_adam_steps_per_rank'],probe_adam_per_rank=probe['actual_adam_steps_per_rank'],
        scheduler_completed_outer=n,consumed_candidates_at_boundary=consumed,
        effective_from_outer=(delta['candidate_schedule']['effective_from_outer'] if delta else None),grounding_state=state,joint_grounding_state=joint['grounding'],grounding_action=action,
        phase_reuse=phases,history_provenance=history,assets=list(by_target.values()),metadata_sha256=metadata,
        source_joint_sha256=spec['source_joint_sha256'],source_compatibility_sha256=config['source_compatibility_sha256'],
        terminal_proof_sha256=spec['terminal_proof_sha256'],qualification_proof_sha256=spec.get('qualification_proof_sha256'),
        genuine_full_state_qualification=qualified,tensor_payloads_read=0,payload_full_sha_repeated=False,
        cuda_restore_verified=False,optimizer_reset=False,automatic_rollback_performed=False,
        uncommitted_actual_updates_discarded=False,source_namespace_still_required=True,
        source_modified=False,plan_created_unix=time.time())


def import_complete(base):
    """Mutating entry point: genuine fault/qualification and actual gone checks twice."""
    base=canonical_path(base);config=read(base/'preparation/launch-config.json')
    lock_path=base/'preparation/i1-i6-joint-import.lock'
    descriptor=os.open(lock_path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=plan(base,config)
        if not result['genuine_full_state_qualification'] or result['grounding_action']=='RECONCILE_INTERRUPTED_REFIT_NO_BLIND_REPLAY':
            raise RuntimeError('No import: genuine full-state qualification or interrupted-refit reconciliation is still required')
        receipt_path=base/'preparation/resume-joint-receipt.json';intent=base/'preparation/resume-joint-intent.json'
        if receipt_path.exists():
            validate_imported_boundary(base,config);return read(receipt_path)
        if intent.exists():raise RuntimeError('Existing interrupted import requires reconciliation, never blind replay')
        _once(intent,result)
        source=Path(result['source_base']);original=read(source/'preparation/launch-config.json');spec=config['resume_joint']
        _writer_barrier(original,read(spec['terminal_proof']),read(source/'logs/process-ledger.json'),
            authorized_handoff_sha=config.get('candidate_schedule_authorization_sha256'),completed_outer=result['completed_outer'])
        for row in result['assets']:
            src=canonical_path(row['source']);dst=canonical_path(row['target'],exists=False)
            if not dst.is_relative_to(base) or dst.exists():raise ValueError('Import target is occupied or outside the new execution')
            info=src.stat()
            if (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)!=(row['dev'],row['inode'],row['bytes'],row['mtime_ns']):
                raise ValueError('Qualified immutable payload changed before linking')
            dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst,follow_symlinks=False)
        result=dict(result,status='I1_I6_FULL_JOINT_IMPORTED',source_binding=source_envelope(config,verify_source_compatibility(config)))
        _once(receipt_path,result)
        validate_imported_boundary(base,config)
        return result
    finally:os.close(descriptor)


def validate_imported_boundary(base,config):
    base=canonical_path(base);receipt=read(base/'preparation/resume-joint-receipt.json')
    current=plan(base,config)
    if (receipt['status']!='I1_I6_FULL_JOINT_IMPORTED' or receipt['adapter']!=ADAPTER or
            receipt['execution_id']!=config['execution_id'] or receipt['source_joint_sha256']!=current['source_joint_sha256'] or
            receipt['source_compatibility_sha256']!=config['source_compatibility_sha256'] or
            read(base/'latest-joint.json')!=read(Path(receipt['source_base'])/'latest-joint.json')):
        raise ValueError('Imported raw joint/provenance changed')
    for item in receipt['assets']:
        path=canonical_path(item['target']);source=canonical_path(item['source'])
        if (path.stat().st_dev,path.stat().st_ino)!=(source.stat().st_dev,source.stat().st_ino):raise ValueError('Imported full-state alias changed')
        if source.suffix=='.json' and sha(path)!=item['sha256']:raise ValueError('Imported immutable metadata changed')
    return receipt['completed_outer']


def inherited_phase(base,root,phase,config):
    receipt=read(Path(base)/'preparation/resume-joint-receipt.json');item=receipt['phase_reuse'].get(phase)
    if root!=Path(base)/'outers'/('outer-%06d'%receipt['next_outer']) or item is None:return False
    path=root/item['marker']
    if sha(path)!=item['sha256']:raise ValueError('Inherited sampling event changed')
    return True


def inherited_actor_binding(base,root,config,actual):
    receipt=read(Path(base)/'preparation/resume-joint-receipt.json')
    return (root==Path(base)/'outers'/('outer-%06d'%receipt['next_outer']) and
        (root/'actor-binding.json').exists() and read(root/'actor-binding.json')==actual)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('plan','import'));parser.add_argument('--base',required=True)
    args=parser.parse_args();print(json.dumps(plan(args.base) if args.phase=='plan' else import_complete(args.base),allow_nan=False))


if __name__=='__main__':main()
