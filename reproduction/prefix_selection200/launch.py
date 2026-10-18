"""One native allocation for the approved prefix-selection actor200 experiment."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import urllib.request
import uuid

from reproduction.prefix20_online.launch import read, sha, once, publish
from .contract import (CANDIDATE_QUESTIONS, PROBE_LR, PROBE_PERIOD, PROBE_UPDATES,
                       WORLD, PREFIX_TOKENS, HIDDEN_CAPTURE_CONTRACT)
from .a800_runtime import query, native_allocation, UUIDLocks
from . import a800_runtime

PREFIX = a800_runtime.PREFIX


def validate_config(c, base):
    from .candidate_schedule import validate_schedule
    validate_schedule(c, required=c.get('resume_joint', {}).get('adapter') == 'i1_i6_joint_v1')
    from .h100_replica_hardware import enabled as h100_layout, validate as validate_h100, PREFIXES as H100_PREFIXES
    if c['base'] != str(base) or not str(base).startswith(H100_PREFIXES if h100_layout(c) else PREFIX):
        raise ValueError('Use the isolated remote prefix-selection namespace')
    required = dict(outers=200, candidate_questions=CANDIDATE_QUESTIONS, selected_questions=512,
                    responses_per_question=5, prefix_policy_tokens=PREFIX_TOKENS, exploration_questions=0,
                    probe_learning_rate=PROBE_LR, probe_epochs_per_update=1, probe_update_period=PROBE_PERIOD,
                    probe_updates_per_period=PROBE_UPDATES, actor_hidden_native=True, probe_global_batch_average=256,
                    actor_global_mini_batch=128, actor_ppo_epochs=1, world_size=WORLD,
                    probe_update_at_final=False, nonmixed_advantage_mode='zero_anchor_all_correct_only')
    if any(c.get(k) != v for k, v in required.items()):
        raise ValueError('The approved prefix selection or training budget differs')
    if c.get('actual_em_selection_fallback') not in ('probe_score_fill', 'require_512_mixed'):
        raise ValueError('Bind an explicit fallback for periodic pools with fewer than512 actual mixed groups')
    if c.get('selection_probe_experiment'):
        from .ab_policy import validate as validate_ab
        validate_ab(c)
    import re
    if (not re.fullmatch('[0-9a-f]{64}', c['probe_source_sha256'])
            or type(c.get('probe_source_epoch')) is not int or c['probe_source_epoch'] < 1
            or c.get('probe_source_model_kind') != 'ema'
            or c.get('hidden_capture_contract') != HIDDEN_CAPTURE_CONTRACT
            or c.get('independent_encoder_forward') is not False):
        raise ValueError('Bind an explicitly retained complete EMA source and native question5 capture')
    from .interaction_protocol import CONTRACT as INTERACTION_CONTRACT
    from .grounding import validate_config as validate_grounding
    if c.get('interaction_protocol') != INTERACTION_CONTRACT:
        raise ValueError('Freeze the shared new BF16 train/QA interaction contract')
    validate_grounding(c)
    from .actor import OBJECTIVE
    from .actor_safety import POLICY_SAFEGUARDS
    if c.get('actor_loss_reduction') != OBJECTIVE or c.get('policy_safeguards') != POLICY_SAFEGUARDS:
        raise ValueError('Freeze the original64 equal micro2 loss and authorized policy safeguards')
    if c['model_revision'] != 'aa8e72537993ba99e69dfaafa59ed015b17504d1':
        raise ValueError('Official actor/ref/encoder revision differs')
    if len(set(c['physical_gpu_uuids'])) != WORLD:
        raise ValueError('Four actual physical training GPUs are required')
    if h100_layout(c):
        validate_h100(c)
        if c.get('probe_source_run') != '8af926a49630428097b94027864da68b':
            raise ValueError('Use the exact reference warmup source run')
    else:
        if (len(c.get('physical_gpu_indices', [])) != WORLD
                or c.get('auxiliary_gpu_indices') != [5, 6]
                or len(c.get('auxiliary_gpu_uuids') or []) != 2
                or set(c['physical_gpu_indices']) & {5, 6}
                or len(set(c['physical_gpu_uuids'] + c['auxiliary_gpu_uuids'])) != 6
                or c.get('probe_source_run') != '8af926a49630428097b94027864da68b'):
            raise ValueError('Bind actual four training/two auxiliary UUIDs and the approved warmup source')
    for key in ('environment_root', 'model_dir', 'official_model_dir', 'encoder_model_dir',
                'canonical_train', 'source_question_pool', 'probe_source_checkpoint',
                'fixed_test_features', 'retrieval_index', 'retrieval_corpus', 'grounding_manifest'):
        if not str(c[key]).startswith(('/mnt/', '/dataset/', '/data/')):
            raise ValueError('All large inputs and outputs must stay remote: ' + key)
    return c


def config(base):
    base = Path(base).resolve()
    return validate_config(read(base / 'preparation/launch-config.json'), base)


def submit(base):
    return a800_runtime.submit(base, validate_config)


def _json_request(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def reused_phase(base, root, phase, c):
    """Only explicitly imported complete source phases may bypass execution."""
    if c.get('resume_joint', {}).get('adapter') == 'i1_i6_joint_v1':
        from .i1_i6_joint_resume import inherited_phase
        return inherited_phase(base, root, phase, c)
    if c.get('resume_joint', {}).get('adapter') == 'native_a800_joint_v1':
        from .native_joint_resume import inherited_phase
        return inherited_phase(base, root, phase, c)
    path = root / phase / 'complete.json'
    if not path.exists():
        return False
    imported = read(base / 'preparation/resume-import-receipt.json')
    if (c.get('resume_import', {}).get('source_base') != imported['source_base']
            or imported['run_id'] != c['run_id'] or imported['outer'] != 1
            or imported['phase_hashes'][phase] != sha(path)):
        raise ValueError('Existing phase is not the explicitly inherited complete source')
    value = read(path)
    if value['run_id'] != c['run_id'] or value['outer'] != 1:
        raise ValueError('Inherited phase belongs to another logical run/outer')
    return True


def commit_outer(base, c, outer, root, actor_selection_path, timings, recovery=None, grounding_state=None):
    """Publish only an actual actor/probe/pool boundary, including repaired10."""
    if c.get('selection_probe_experiment'):
        from .ab_lifecycle import commit_outer as commit_ab
        return commit_ab(base,c,outer,root,actor_selection_path,timings,grounding_state)
    from .periodic import is_probe_update, probe_boundary
    from .storage import actual_actor_boundary
    from . import grounding
    periodic_update = is_probe_update(outer)
    if periodic_update:
        probe_metrics = read(root/'probe-update/metrics.json')
        updated = read(root/'probe-update/complete.json')
        if (updated['status'] != 'UNSELECTED2560G5_PERIODIC_PROBE_50_ADAM_COMPLETE'
                or updated['outer'] != outer or updated['run_id'] != c['run_id']
                or updated['new_adam_steps_per_rank'] != PROBE_UPDATES
                or updated['metrics_sha256'] != sha(root/'probe-update/metrics.json')):
            raise RuntimeError('Periodic probe must actually finish its complete pass before joint commit')
    else:
        probe_metrics = read(root/'probe-frozen.json')
    actor, probe = read(base/'actor/latest.json'), read(base/'probe/latest.json')
    boundary = probe_boundary(outer)
    if (actor['completed_outer'] != outer or actor['scheduler_completed_outer'] != outer
            or not actual_actor_boundary(actor, outer)
            or probe['completed_outer'] != boundary['last_update_outer']
            or probe['actual_adam_steps_per_rank'] != [boundary['actual_adam_per_rank']]*WORLD
            or (periodic_update and updated['checkpoint'] != probe)):
        raise RuntimeError('Actual actor/probe Adam boundaries differ from the completed outer')
    pool = read(base/'question-pool/state.json')
    from .candidate_schedule import consumed_candidates_through, schedule_metadata
    consumed = consumed_candidates_through(c, outer)
    if (pool['next_outer'] != outer+1 or pool['consumed_candidates'] != consumed
            or pool['cycle']*len(pool['order'])+pool['cursor'] != consumed):
        raise RuntimeError('Joint commit cannot advance beyond its actual reserved candidate pool')
    snapshot = root/'pool-state-at-completion.json'
    once(snapshot, pool)
    probe_snapshot = root/'probe-state-at-completion.pt'
    os.link(base/'probe/latest.pt', probe_snapshot)
    grounding_state = grounding.checked_state(c, grounding_state, require_adopted=True)
    if actor.get('grounding_state') != grounding_state:
        raise RuntimeError('Joint grounding state differs from actual actor-used state')
    grounding_snapshot = root/'grounding-state-at-completion.json'
    once(grounding_snapshot, grounding_state)
    complete = dict(run_id=c['run_id'], execution_id=c.get('execution_id'), outer=outer, actor=actor, probe=probe,
        grounding=grounding_state, grounding_snapshot=str(grounding_snapshot),
        grounding_snapshot_sha256=sha(grounding_snapshot),
        grounding_refresh_due=outer if outer in grounding.REFIT_ROUNDS else None,
        probe_snapshot=str(probe_snapshot), pool_snapshot=str(snapshot), pool_sha256=sha(snapshot),
        selection_sha256=sha(actor_selection_path),
        actor_selection_path=str(actor_selection_path), actor_selection_sha256=sha(actor_selection_path),
        predicted_selection_sha256=sha(root/'probe-score/selection.json'),
        actor_selection_outcome_labels_used=periodic_update, rollout=read(root/'full/complete.json'),
        probe_metrics=probe_metrics, periodic_probe_update=periodic_update,
        probe_boundary=boundary, source_commit=c['source_commit'], observed_unix=time.time())
    complete['candidate_schedule'] = schedule_metadata(c, outer)
    if c.get('resume_joint', {}).get('adapter') == 'i1_i6_joint_v1':
        from .i1_i6_joint_resume import verify_source_compatibility, source_envelope
        complete.update(source_envelope(c, verify_source_compatibility(c)))
        parent = c['resume_joint']
        complete['source_lineage'] = dict(parent_source_commit=parent['source_commit'],
            parent_execution_id=parent['source_execution_id'], parent_base=parent['source_base'],
            inherited_complete_through=parent['completed_outer'],
            qualification_proof_sha256=parent['qualification_proof_sha256'],
            imported_history_receipt=str(base/'preparation/resume-joint-receipt.json'),
            effective_candidate_schedule_from_outer=c['candidate_schedule']['effective_from_outer'])
    complete['phase_timings'] = {key:value for key,value in timings.items()
                                if key.startswith('outer-%06d-'%outer)}
    if c.get('resume_pending_probe'):
        parent = c['resume_pending_probe']
        complete['source_lineage'] = dict(parent_source_commit=parent['source_commit'],
            parent_execution_id=parent['source_execution_id'], parent_base=parent['source_base'],
            inherited_complete_through=parent['completed_joint_outer'],
            repaired_pending_outer=parent['pending_outer'],
            qualification_proof_sha256=parent['qualification_proof_sha256'])
    elif c.get('resume_joint', {}).get('adapter') == 'native_a800_joint_v1':
        parent = c['resume_joint']
        complete['source_lineage'] = dict(parent_source_commit=parent['source_commit'],
            parent_execution_id=parent['source_execution_id'], parent_base=parent['source_base'],
            inherited_complete_through=parent['completed_outer'],
            inherited_prefix_outer=parent['inherited_prefix_outer'],
            qualification_proof_sha256=parent['checkpoint_cpu_proof_sha256'],
            parent_source_lineage=read(Path(parent['source_base'])/'latest-joint.json').get('source_lineage'),
            imported_history_receipt=str(base/'preparation/resume-joint-receipt.json'))
    if recovery is not None:
        complete['pending_probe_recovery'] = dict(recovery,
            imported_actor_updates_reexecuted=0, native_rollouts_reexecuted=0,
            joint_was_incomplete_before_this_probe=True, probe_update_reexecuted_before_writes_only=True)
    once(root/'complete.json', complete)
    publish(base/'latest-joint.json', complete)
    return complete


def run(base):
    from reproduction.prefix_only_correctness_ab.launch import torchrun_owned_class, DeferredInterrupt, birth_states
    from .pool import prepare
    base = Path(base).resolve(); c = config(base)
    if sha(base / 'preparation/launch-config.json') != read(base / 'preparation/submit-intent.json')['config_sha256']:
        raise ValueError('Submitted configuration changed')
    if (base / 'logs/process-ledger.json').exists():
        raise RuntimeError('Existing execution needs explicit recovery; do not replay its launch')
    compatible_resume = c.get('resume_joint', {}).get('adapter') == 'i1_i6_joint_v1'
    if c.get('resume_pending_probe') or c.get('resume_import') or (c.get('resume_joint') and not compatible_resume):
        raise ValueError('I1-I6 fresh source requires an explicitly compatible recovery adapter; old hardcoded-count adapters are forbidden')
    pending_recovery = c.get('resume_pending_probe')
    if pending_recovery and c.get('resume_joint'):
        raise ValueError('Complete-joint and committed-actor/pending-probe recovery are distinct boundaries')
    outer = int(c.get('resume_joint', {}).get('completed_outer', 0))
    # All CPU seals are checked before acquiring leases or admitting CUDA.  A
    # failed import cannot leave a new allocation or a started child behind.
    if pending_recovery:
        from .pending_probe_resume import validate_imported_pending
        outer = validate_imported_pending(base, c)
    elif outer:
        if compatible_resume:
            from .i1_i6_joint_resume import validate_imported_boundary
        elif c['resume_joint'].get('adapter') == 'native_a800_joint_v1':
            from .native_joint_resume import validate_imported_boundary
        else:
            from .resume import validate_imported_boundary
        validate_imported_boundary(base, c)
    lease = UUIDLocks(c['physical_gpu_uuids'] + c['auxiliary_gpu_uuids'], root=c.get('gpu_uuid_lock_root'))
    try:
        native_allocation(base, c)
    except BaseException:
        lease.close()
        raise
    owned = torchrun_owned_class()(base / 'logs/process-ledger.json', c['run_id'])
    interrupt = DeferredInterrupt(); handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    for sig in handlers:
        signal.signal(sig, interrupt.handler)
    env = dict(os.environ, PYTHONPATH=str(base / 'repo'), CUDA_VISIBLE_DEVICES=','.join(map(str, c['physical_gpu_indices'])),
        SEARCH_G1_BASE=str(base), OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', NCCL_NVLS_ENABLE='0',
        HF_HOME=str(base / 'cache/hf'), XDG_CACHE_HOME=str(base / 'cache'),
        PYTHONDONTWRITEBYTECODE='1',
        TRITON_CACHE_DIR=str(base / 'cache/triton'), TORCHINDUCTOR_CACHE_DIR=str(base / 'cache/inductor'),
        TMPDIR=str(base / 'tmp'), PYTHONUNBUFFERED='1', NO_PROXY='127.0.0.1,localhost')
    for key in ('PYTHONHOME', 'RAY_ADDRESS', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    for name in ('tmp', 'cache/hf', 'cache/triton', 'cache/inductor'):
        (base / name).mkdir(parents=True, exist_ok=True)
    python = str(Path(c['environment_root']) / 'bin/python')
    conf = str(base / 'preparation/launch-config.json')
    services = {}; success = False; timings = {}; actor_micro = c['actor_global_micro_batch_size']
    grounding_state = None
    retriever_name = None
    retriever_generation = 0
    if c.get('python_header_cpath'):
        env['CPATH'] = c['python_header_cpath']

    def stage(name, **extra):
        publish(base / 'logs/pipeline-status.json', dict(run_id=c['run_id'], stage=name,
            outer=outer, checked_unix=time.time(), **extra))

    def start(name, command, child_env=None):
        interrupt.check()
        p = owned.start(name, command, child_env or env, str(base / 'logs' / (name + '.log')), str(base / 'repo'), pass_fds=tuple(lease.fds))
        owned.observe(name)
        return p

    def phase(name, command, child_env=None, allow_failed=False):
        began = time.monotonic()
        stage(name); p = start(name, command, child_env)
        while p.poll() is None:
            owned.observe(name); interrupt.check()
            for service_name, service in services.items():
                owned.observe(service_name)
                if service.poll() is not None:
                    raise RuntimeError('Own required service exited: ' + service_name)
            time.sleep(2)
        owned.observe(name); owned.jobs[name]['exit_code'] = p.returncode; owned.save(); owned.stop(name)
        interrupt.check()
        timings[name] = dict(elapsed_seconds=time.monotonic()-began, exit_code=p.returncode)
        publish(base / 'logs/phase-timings.json', timings)
        if p.returncode and not allow_failed:
            raise RuntimeError('Actual phase failed: ' + name + '; preserve its original log and artifacts')
        return p.returncode

    def distributed(module, args):
        return [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=%d' % WORLD, '-m', module] + args

    def await_service(name, url, expected):
        deadline = time.monotonic() + 1800
        while True:
            interrupt.check(); owned.observe(name)
            if services[name].poll() is not None:
                raise RuntimeError('Own service failed while loading: ' + name)
            try:
                value = _json_request(url)
                if value.get('ok') and all(value.get(k) == v for k, v in expected.items()):
                    return value
            except (OSError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Own service did not become ready: ' + name)
            time.sleep(3)

    def ensure_retriever():
        nonlocal retriever_name, retriever_generation
        if retriever_name is not None:
            if services[retriever_name].poll() is not None:
                raise RuntimeError('Own retriever exited unexpectedly')
            return
        retriever_generation += 1
        retriever_name = 'retriever' if retriever_generation == 1 else 'retriever-load-%06d' % retriever_generation
        began = time.monotonic()
        services[retriever_name] = start(retriever_name, [python, '-m', 'reproduction.prefix20_online.retriever',
            '--index', c['retrieval_index'], '--corpus', c['retrieval_corpus'], '--encoder', c['retrieval_encoder'],
            '--cache', str(base / 'retrieval-cache'), '--run-id', c['run_id'], '--port', str(c['retrieval_port']),
            '--data-cache', c.get('retrieval_data_cache', str(base / 'retrieval-cache')),
            '--batch-queries', str(c.get('retrieval_batch_queries', 1024)), '--gpu-shards', str(c.get('retrieval_gpu_shards', 2))],
            dict(env, CUDA_VISIBLE_DEVICES=','.join(map(str,c['auxiliary_gpu_indices'])), PYTHONPATH=c['retriever_gpu_lib'] + ':' + str(base / 'repo'), LD_LIBRARY_PATH=':'.join(map(str, libs))))
        await_service(retriever_name, c['search_url'].replace('/retrieve', '/health'),
                      dict(run_id=c['run_id'], gpu_shards=c.get('retrieval_gpu_shards', 2)))
        timings['outer-%06d-retriever-load-%06d' % (outer,retriever_generation)] = dict(
            elapsed_seconds=time.monotonic()-began, exit_code=0)

    def release_retriever_for_actor():
        nonlocal retriever_name
        if c.get('retriever_phase_residency') != 'rollout_only' or retriever_name is None:
            return
        name = retriever_name
        owned.stop(name)
        services.pop(name)
        retriever_name = None

    def start_grounding(state):
        from . import grounding
        live = grounding.publish_live(base, c, state)
        replicas = []
        for rank in range(WORLD):
            name = 'grounding-%06d-rank-%d' % (state['round'], rank)
            services[name] = start(name, [python, 'probe_server.py', '--manifest', str(live),
                '--port', str(c['grounding_probe_base_port'] + rank)],
                dict(env, CUDA_VISIBLE_DEVICES=str(c.get('grounding_physical_gpu_by_rank',
                    [c['grounding_physical_gpu']] * WORLD)[rank]),
                     SEARCH_G1_BASE=str(base / 'grounding' / ('rank-%d' % rank)), OMP_NUM_THREADS='2'))
        for rank in range(WORLD):
            name = 'grounding-%06d-rank-%d' % (state['round'], rank)
            actual = await_service(name, 'http://127.0.0.1:%d/health' % (c['grounding_probe_base_port'] + rank),
                dict(version=state['version'], round=state['round'], encoder=state['encoder']))
            replicas.append(dict(actual, rank=rank, owned_service=name, pid=services[name].pid))
        return grounding.adopt(base, c, state, replicas)

    def refresh_grounding(completed):
        nonlocal grounding_state
        from . import grounding
        if completed not in grounding.REFIT_ROUNDS:
            return
        if grounding_state['round'] != completed:
            stage('GROUNDING_REFIT_REQUIRED', complete_outer=completed)
            # Stop only our four evaluator replicas; retriever and other tasks
            # remain bound to their existing owned allocation.
            for name in [key for key in services if key.startswith('grounding-')]:
                owned.stop(name); services.pop(name)
            phase('outer-%06d-grounding-refit' % completed,
                [python, '-m', 'reproduction.prefix_selection200.grounding', '--base', str(base),
                 '--outer', str(completed)],
                dict(env, CUDA_VISIBLE_DEVICES=str(c['grounding_refit_visible_device'])))
            fitted = read(base / 'grounding/rounds' / ('round-%06d' % completed) / 'fit-complete.json')
            grounding_state = start_grounding(fitted)
        path = base / 'outers' / ('outer-%06d' % completed) / 'grounding-refresh.json'
        if not path.exists():
            once(path, dict(run_id=c['run_id'], outer=completed, grounding=grounding_state,
                            required_before_outer=completed + 1))

    previous_qa = Path((pending_recovery or c.get('resume_joint', {})).get('source_base', str(base))) / 'logs/qa-pending.json'
    pending_qa = read(previous_qa) if previous_qa.is_file() else {}

    def validation(ordinal, model, evaluation_config, output, name):
        ensure_retriever()
        code = phase(name, [python, '-m', 'reproduction.prefix_selection200.eval',
            '--config', str(evaluation_config), '--model', model,
            '--questions', c['validation_questions'], '--expected-count', '512', '--output', str(output)],
            allow_failed=True)
        if code:
            pending_qa[str(ordinal)] = dict(outer=ordinal, model=model, evaluation_config=str(evaluation_config),
                actual_exit_code=code, original_output=str(output), original_log=str(base/'logs'/(name+'.log')),
                status='QA_PENDING_ACTUAL_WORKER_FAILURE_TRAINING_CONTINUES')
        else:
            pending_qa.pop(str(ordinal), None)
        publish(base / 'logs/qa-pending.json', pending_qa)
        return code

    try:
        if pending_recovery:
            # Actor10 has already returned all20 Adam writes and committed its
            # complete optimizer/scheduler/four-rank RNG.  The failed probe
            # wrote zero updates; reuse its exact sampling actor hidden/EM and
            # diagnostic.  Do not reserve, generate, score, or run actor10 again.
            outer = pending_recovery['pending_outer']
            root = base/'outers'/('outer-%06d'%outer)
            source_root = Path(pending_recovery['source_base'])/'outers'/root.name
            diagnostic = read(source_root/'diagnostic-complete.json')
            phase('outer-%06d-probe-update'%outer,
                distributed('reproduction.prefix_selection200.probe', [
                    'update', '--state-dir', str(base/'probe'),
                    '--scored-features', str(source_root/'probe-score'),
                    '--own-em', str(source_root/'full-pool/own-em.jsonl'),
                    '--output', str(root/'probe-update'),
                    '--own-em-sha256', read(source_root/'full-pool/complete.json')['own_em']['sha256'],
                    '--outer', str(outer), '--model-dir', c['official_model_dir'],
                    '--test-cache', diagnostic['scan_path'],
                    '--test-manifest-sha', diagnostic['scan_sha256']]))
            recovery = read(base/'preparation/resume-pending-probe-receipt.json')
            commit_outer(base, c, outer, root, root/'probe-score/actual-em-selection.json',
                         timings, recovery=recovery)
            stage('RECOVERED_ACTOR_PROBE_POOL_JOINT_COMPLETE', complete_outer=outer,
                  actor_adam_per_rank=20*outer,
                  probe_adam_per_rank=read(base/'probe/latest.json')['actual_adam_steps_per_rank'][0],
                  inherited_actor_updates_not_reexecuted=20)
        elif not outer:
            phase('probe-initialize-model-only', [python, '-m', 'reproduction.prefix_selection200.probe', 'initialize',
                '--state-dir', str(base / 'probe'), '--source-checkpoint', c['probe_source_checkpoint'],
                '--source-sha256', c['probe_source_sha256'], '--source-epoch', str(c['probe_source_epoch']),
                '--source-model-kind', c['probe_source_model_kind'], '--run-id', c['run_id'],
                '--test-cache', c['fixed_test_features'], '--test-manifest-sha', c['fixed_test_manifest_sha256']],
                dict(env, CUDA_VISIBLE_DEVICES=''))
        stage('STARTING_AUX_TWO_SHARD_RETRIEVER_AND_FOUR_GROUNDING_REPLICAS')
        env_root = Path(c['environment_root'])
        libs = list((env_root / 'lib/python3.10/site-packages/nvidia').glob('*/lib'))
        libs += [env_root / 'lib/python3.10/site-packages/torch/lib']
        if not c.get('startup_sampling_resume'):
            ensure_retriever()
        from . import grounding
        if outer:
            # A future compatible recovery must retain the exact adopted state.
            recovered = grounding.recovery_state(Path(c['resume_joint']['source_base']), c,
                read(base/'latest-joint.json'))
            grounding_state = start_grounding(recovered)
        else:
            grounding_state = start_grounding(grounding.initialize(base, c))
        refresh_grounding(outer)
        # Retry the interrupted milestone QA with its saved actor, then continue
        # from the joint actor/probe/pool boundary without redoing any updates.
        if outer in (5, 50, 100, 150, 200):
            actor = read(base / 'actor/latest.json')
            recovery_config = base / 'preparation/recovered-evaluation-config.json'
            once(recovery_config, dict(c, actor_version=actor['actor_version']))
            validation(outer, actor['hf_model_dir'], recovery_config, base/'recovered-validation',
                       'recovered-outer-%06d-validation' % outer)
        for outer in range(outer + 1, 201):
            from .b_progressive import enabled as b_progressive,prefix as b_prefix,ENV as b_env
            if b_progressive(c):
                length=b_prefix(outer);env[b_env]=str(length);os.environ[b_env]=str(length)
                if outer==51 or length!=c['prefix_policy_tokens']:
                    phase('outer-%06d-prefix-dimension'%outer,[python,'-m','reproduction.prefix_selection200.b_progressive_control','migrate','--base',str(base),'--prefix',str(length)],dict(env,CUDA_VISIBLE_DEVICES=''))
                c=dict(c,prefix_policy_tokens=length,hidden_capture_contract='sample%d_keep%d_postblock27_native_residual_sum_v1'%(length+1,length))
                conf=str(base/'preparation'/('outer-%06d-launch-config.json'%outer));once(conf,c)
            outer_wall_started=time.monotonic()
            root = base / 'outers' / ('outer-%06d' % outer)
            stage('CANDIDATE_POOL_RESERVATION')
            prepare(c, outer)
            actor = read(base / 'actor/latest.json') if outer > 1 else None
            actor_binding = dict(run_id=c['run_id'], outer=outer,
                actor_model_dir=actor['hf_model_dir'] if actor else c['official_model_dir'],
                actor_version=actor['actor_version'] if actor else c['run_id'] + '-official0',
                actor_model_files_sha256={name:item['sha256'] for name,item in actor['hf_files'].items()}
                    if actor else c['model_files_sha256'])
            actor_binding_path = root / 'actor-binding.json'
            if actor_binding_path.exists():
                inherited_binding = False
                if compatible_resume:
                    from .i1_i6_joint_resume import inherited_actor_binding
                    inherited_binding = inherited_actor_binding(base, root, c, actor_binding)
                elif c.get('resume_joint', {}).get('adapter') == 'native_a800_joint_v1':
                    from .native_joint_resume import inherited_actor_binding
                    inherited_binding = inherited_actor_binding(base, root, c, actor_binding)
                if ((not c.get('resume_import') and not inherited_binding)
                        or read(actor_binding_path) != actor_binding):
                    raise ValueError('Inherited actual actor binding differs')
            else:
                once(actor_binding_path, actor_binding)
            from .periodic import is_probe_update, probe_boundary
            periodic_update = is_probe_update(outer)

            def sample_outer(sampling_attempt=None):
                ensure_retriever()
                sampling_conf = conf
                if sampling_attempt:
                    sampling_conf = str(root/('sampling-attempt-%s-config.json' % sampling_attempt))
                    once(sampling_conf, dict(c, sampling_attempt_id=sampling_attempt))
                def job(phase_name):
                    return 'outer-%06d-%s' % (outer, phase_name) + (
                        '-sample-' + sampling_attempt if sampling_attempt else '')
                if not reused_phase(base, root, 'prefix', c):
                    phase(job('prefix'), [python, '-m', 'reproduction.prefix_selection200.rollout',
                        '--config', sampling_conf, '--outer', str(outer), '--phase', 'prefix'])
                if not reused_phase(base, root, 'probe-score', c):
                    phase(job('score'), distributed('reproduction.prefix_selection200.probe',
                        ['score', '--state-dir', str(base / 'probe'), '--prefix-inputs', str(root / 'prefix/scan'),
                         '--output', str(root / 'probe-score'), '--outer', str(outer), '--model-dir', c['official_model_dir'],
                         '--encoder-micro-batch-size', str(c.get('encoder_micro_batch', 256))]))
                actor_selection_path=root/'probe-score/selection.json'
                if periodic_update:
                    # The label-free predicted selection remains immutable. Complete
                    # every candidate once, then separately bind actual-mixed actor
                    # selection to those exact native outcomes and project its G5s.
                    phase(job('complete-unselected-pool'),
                        [python,'-m','reproduction.prefix_selection200.periodic_data','complete-pool',
                         '--config',sampling_conf,'--outer',str(outer)])
                    actor_selection_path=root/'probe-score/actual-em-selection.json'
                    actual_selection=read(actor_selection_path)
                    if (actual_selection.get('outcome_labels_used') is not True
                            or actual_selection.get('fallback_mode') != c['actual_em_selection_fallback']):
                        raise ValueError('Periodic actual-EM actor selection is missing its bound fallback policy')
                    diagnostic_path = root/'diagnostic-complete.json'
                    if diagnostic_path.is_file():
                        diagnostic = read(diagnostic_path)
                        if (not sampling_attempt or diagnostic['run_id'] != c['run_id'] or
                                diagnostic['outer'] != outer or diagnostic['actor_version'] != actor_binding['actor_version'] or
                                diagnostic['fixed_test_identity_sha256'] != c['fixed_test_identity_sha256'] or
                                diagnostic['questions'] != 512 or diagnostic['trajectories'] != 2560):
                            raise ValueError('Only the same zero-write actor diagnostic may be reused')
                    else:
                        phase(job('current-actor-diagnostic'),
                            [python,'-m','reproduction.prefix_selection200.periodic_data','diagnostic',
                             '--config',sampling_conf,'--outer',str(outer)])
                elif __import__('reproduction.prefix_selection200.b_progressive',fromlist=['ordinary']).ordinary(c,outer):
                    phase(job('expanded-continuation-rerank'),[python,'-m','reproduction.prefix_selection200.b_progressive_control','complete','--config',sampling_conf,'--outer',str(outer)])
                    actor_selection_path=root/'probe-score/actual-em-selection.json'
                else:
                    phase(job('continue'), [python, '-m', 'reproduction.prefix_selection200.rollout',
                        '--config', sampling_conf, '--outer', str(outer), '--phase', 'continue', '--selection', str(actor_selection_path)])
                return actor_selection_path

            if outer == 1 and c.get('startup_sampling_resume'):
                from .startup_sampling_resume import validate_reused_sampling
                actor_selection_path = validate_reused_sampling(base,c)
            else:
                actor_selection_path = sample_outer()
            if compatible_resume:
                from .i1_i6_actor_resume import collect_oom_retry, archive_zero_write_attempt
            else:
                from .actor import collect_oom_retry
                from .guard_retry import archive_zero_write_attempt
            actor_module = ('reproduction.prefix_selection200.i1_i6_actor_resume' if compatible_resume
                            else 'reproduction.prefix_selection200.actor')
            attempt_id = uuid.uuid4().hex
            while True:
                release_retriever_for_actor()
                attempt_config = dict(grounding.bind_attempt(c, grounding_state, actor, outer), actor_attempt_id=attempt_id,
                    actor_global_micro_batch_size=actor_micro,
                    pool_state_path=str(base/'question-pool/state.json'),
                    prefix_probe_receipt=str(base/'probe/latest.json'))
                attempt_path = root / ('actor-attempt-%s-config.json' % attempt_id)
                once(attempt_path, attempt_config)
                code = phase('outer-%06d-actor-%s' % (outer, attempt_id), distributed(actor_module,
                    ['--config', str(attempt_path), '--outer', str(outer), '--rollout', str(root / 'full/complete.json'),
                     '--output', str(base / 'actor')]), allow_failed=True)
                if code == 0:
                    guarded = base/'actor/outers'/('outer-%04d-attempt-%s' % (outer, attempt_id))/'guard-result.json'
                    if guarded.is_file():
                        rejected = archive_zero_write_attempt(base, attempt_config, outer, root, attempt_id)
                        stage('OUTER_GUARD_ZERO_WRITE_RESAMPLE', logical_outer_not_completed=outer,
                              zero_write_attempt=attempt_id, scheduler_advanced=False)
                        actor_selection_path = sample_outer(rejected['sampling_attempt_id'])
                        attempt_id = uuid.uuid4().hex
                        continue
                    if read(base/'actor/latest.json').get('completed_outer') != outer:
                        raise RuntimeError('Actor exit0 without a complete boundary or typed zero-write guard')
                    break
                retry = collect_oom_retry(base / 'actor', attempt_path, outer, root / 'full/complete.json',
                    attempt_id, phase_exit_code=code)
                once(root / ('actor-attempt-%s-recovery.json' % attempt_id), retry)
                if retry['status'] == 'ACTOR_ALREADY_COMMITTED':
                    break
                if not retry.get('retry'):
                    raise RuntimeError('Actual actor failure requires repair: '+json.dumps(retry))
                actor_micro = retry['actor_global_micro_batch_size']
                attempt_id = retry['new_attempt_id']
            from .ab_policy import mode as ab_mode, BALANCED, FROZEN
            do_probe_update = ab_mode(c)==BALANCED or (not ab_mode(c) and periodic_update)
            if do_probe_update:
                scope = 'full-pool' if periodic_update else 'full'
                update_args = ['update','--state-dir',str(base/'probe'),'--scored-features',str(root/'probe-score'),
                    '--own-em',str(root/scope/'own-em.jsonl'),'--output',str(root/'probe-update'),
                    '--own-em-sha256',read(root/scope/'complete.json')['own_em']['sha256'],
                    '--outer',str(outer),'--model-dir',c['official_model_dir']]
                if periodic_update:
                    diagnostic_receipt=read(root/'diagnostic-complete.json')
                    update_args += ['--test-cache',diagnostic_receipt['scan_path'],
                                    '--test-manifest-sha',diagnostic_receipt['scan_sha256']]
                phase('outer-%06d-probe-update' % outer,
                      distributed('reproduction.prefix_selection200.probe',update_args))
                probe_metrics=read(root/'probe-update/metrics.json')
            elif ab_mode(c)==FROZEN:
                scope='full-pool' if periodic_update else 'full'
                report_args=['frozen-report','--state-dir',str(base/'probe'),
                    '--scored-features',str(root/'probe-score'),'--own-em',str(root/scope/'own-em.jsonl'),
                    '--outer',str(outer),'--output',str(root/'probe-frozen.json')]
                if periodic_update:
                    diagnostic_receipt=read(root/'diagnostic-complete.json')
                    target=root/'frozen-diagnostic-score'
                    phase('outer-%06d-frozen-diagnostic-score'%outer,distributed('reproduction.prefix_selection200.probe',
                        ['score','--state-dir',str(base/'probe'),'--prefix-inputs',diagnostic_receipt['scan_path'],
                         '--output',str(target),'--outer',str(outer),'--model-dir',c['official_model_dir']]))
                    report_args += ['--frozen-test-score',str(target),'--test-own-em',
                        str(Path(diagnostic_receipt['full_complete']).parent/'own-em.jsonl')]
                phase('outer-%06d-frozen-metrics'%outer,
                      [python,'-m','reproduction.prefix_selection200.probe']+report_args,
                      dict(env,CUDA_VISIBLE_DEVICES=''))
            else:
                current_probe=read(base/'probe/latest.json')
                probe_metrics=dict(status='PROBE_FROZEN_NO_OPTIMIZER_UPDATE',outer=outer,
                    new_adam_steps_per_rank=0,actual_adam_steps_per_rank=current_probe['actual_adam_steps_per_rank'],
                    last_update_outer=current_probe['completed_outer'])
                once(root/'probe-frozen.json',probe_metrics)
            if c.get('selection_probe_experiment'):
                timings['outer-%06d-training-wall'%outer]=time.monotonic()-outer_wall_started
            complete = commit_outer(base, c, outer, root, actor_selection_path, timings, grounding_state=grounding_state)
            actor, probe = complete['actor'], complete['probe']
            stage('OUTER_JOINT_COMPLETE', complete_outer=outer, actor_adam_per_rank=actor['actual_adam_steps_per_rank'][0],
                  processed_opportunities=actor['opportunities_processed'],
                  new_actor_adam_per_rank=actor['new_adam_steps_per_rank'][0],
                  skipped_update_opportunities=actor['skipped_update_opportunities'][0],
                  probe_adam_per_rank=probe['actual_adam_steps_per_rank'][0])
            # Training and QA share the new interaction protocol and BF16.
            if outer in (5, 50, 100, 150, 200):
                evaluation_config = root / 'evaluation-config.json'
                once(evaluation_config, dict(c, actor_version=actor['actor_version']))
                validation(outer, actor['hf_model_dir'], evaluation_config, root/'validation',
                           'outer-%06d-validation' % outer)
            refresh_grounding(outer)
            # Only completed run-owned derived features are removable; retain tokens/labels.
            from .storage import prune_completed
            try:
                prune_completed(base, outer)
            except Exception as error:
                # The scientific joint checkpoint is already committed. A
                # housekeeping/ownership mismatch preserves files and reports
                # a warning; it must not cancel valid actor/probe training.
                import traceback
                once(root/'storage-warning.json',dict(run_id=c['run_id'],outer=outer,
                    error=repr(error),traceback=traceback.format_exc(),
                    scientific_joint_complete=True,training_continues=True,
                    source_assets_preserved=True))
            if c.get('selection_probe_experiment'):
                once(root/'iteration-wall-time.json',dict(outer=outer,
                    training_wall_seconds=timings['outer-%06d-training-wall'%outer],
                    total_wall_seconds_including_qa_grounding_storage=time.monotonic()-outer_wall_started))
        for ordinal, pending in list(pending_qa.items()):
            validation(int(ordinal), pending['model'], pending['evaluation_config'],
                base/'validation-retries'/('outer-%06d-%s'%(int(ordinal),uuid.uuid4().hex)),
                'retry-outer-%06d-validation'%int(ordinal))
        if pending_qa:
            raise RuntimeError('Actor/probe200 complete; actual QA failures remain pending: '+json.dumps(pending_qa))
        final_config = base / 'preparation/final-evaluation-config.json'
        once(final_config, dict(c, actor_version=read(base / 'actor/latest.json')['actor_version']))
        ensure_retriever()
        phase('final51713', [python, '-m', 'reproduction.prefix_selection200.eval', '--config', str(final_config),
            '--model', read(base / 'actor/latest.json')['hf_model_dir'], '--questions', c['final_questions'],
            '--expected-count', '51713', '--output', str(base / 'final-evaluation')])
        once(base / 'logs/pipeline-complete.json', dict(run_id=c['run_id'], complete_outers=200,
            latest_joint=read(base / 'latest-joint.json'), final_evaluation=read(base / 'final-evaluation/complete.json')))
        success = True
    finally:
        try:
            owned.stop_all(); owned.reap()
            births = [row for job in owned.jobs.values() for row in job['ledger'].values()]
            present = birth_states(births)
            once(base / 'logs/cleanup-status.json', dict(run_id=c['run_id'], cleaned=not any(row['state'] != 'Z' for row in present),
                owned_births_all_gone=not present, remaining_owned_births=[row['pid'] for row in present], checked_unix=time.time()))
        finally:
            lease.close()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return dict(status='PIPELINE_COMPLETE' if success else 'INCOMPLETE', base=str(base))


def observe(base):
    from reproduction.prefix20_online.observe import small
    base = Path(base)
    value = dict(base=str(base), observed_unix=time.time(), stage=small(base / 'logs/pipeline-status.json'),
        latest_joint=small(base / 'latest-joint.json'), actor=small(base / 'actor/latest.json'),
        probe=small(base / 'probe/latest.json'), cleanup=small(base / 'logs/cleanup-status.json'),
        pipeline_complete=small(base / 'logs/pipeline-complete.json'))
    receipt = small(base / 'preparation/submit-receipt.json')
    if receipt:
        value['native'] = query(base)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    value['gpu_actual'] = dict(returncode=gpu.returncode, stdout=gpu.stdout, stderr=gpu.stderr)
    current = (value['stage'] or {}).get('outer', 0)
    if current:
        root = base / 'outers' / ('outer-%06d' % current)
        value['current_prefix'] = [small(root / 'prefix' / ('rank-%d-progress.json' % rank)) for rank in range(WORLD)]
        value['current_continuation'] = [small(root / 'full' / ('rank-%d-progress.json' % rank)) for rank in range(WORLD)]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('run', 'submit', 'query', 'observe'))
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    result = dict(run=run, submit=submit, query=query, observe=observe)[args.phase](args.base)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
