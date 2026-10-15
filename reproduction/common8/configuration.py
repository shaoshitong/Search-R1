"""One Hydra recipe for actual common8 training and actual model admission."""
import hashlib
import json
import os
import socket
import time
from pathlib import Path

from reproduction.common8.batch_contract import check_global
from reproduction.common8.formal_contract import assert_formal_run, verify_formal_config, formal_mathematics
from reproduction.common8.topology import world_size, actor_global_micro, monitor_trajectories_per_rank, execution_job_id
from reproduction.common8.precision_contract import overrides as precision_overrides, verify as verify_precision_config

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / 'reproduction/common8/recipe.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arguments(base):
    base = Path(base).resolve()
    run = json.loads((base / 'logs/run-manifest.json').read_text())
    budget = json.loads((REPO / 'reproduction/common8/budget.json').read_text())
    budget.update(formal_mathematics(run))
    assert run['profile'] == 'common8' and run['budget'] == budget
    assert_formal_run(run)
    assert run['model_revision'] == budget['model_revision']
    assert run['initial_model'] == run['reference_model']
    assert run['completed_updates'] == 0, 'Admission is for a fresh run only'
    for name in ('reproduction/common8/recipe.json', 'verl/trainer/config/ppo_trainer.yaml'):
        assert sha(REPO / name) == run['source_sha256'][name], 'Recipe changed after run binding'
    values = json.loads(RECIPE.read_text())
    precision = precision_overrides(run)
    overrides = []
    for value in values:
        key, raw = value.split('=', 1)
        raw = precision.get(key, raw)
        if key=='actor_rollout_ref.actor.ppo_micro_batch_size':
            raw=str(actor_global_micro(run))
        elif key=='trainer.n_gpus_per_node':
            raw=str(world_size(run))
        elif key=='actor_rollout_ref.rollout.gpu_memory_utilization':
            raw=str(run.get('vllm_gpu_memory_utilization',0.4))
        elif key=='+trainer.val_before_train' and run.get('pretraining_baseline_policy')=='reuse_historical':
            raw='false'
        elif key=='+data.mixed_first_selection' and run.get('question_probe') is not None:
            raw='false'
        assert key not in [x.split('=', 1)[0] for x in overrides], 'Duplicate recipe override'
        if '@BASE@' in raw or '@MODEL@' in raw:
            raw = raw.replace('@BASE@', str(base)).replace('@MODEL@', run['initial_model'])
            # Hydra receives these as individual argv strings, never shell code.
            raw = json.dumps(raw)
        overrides.append(key + '=' + raw)
    overrides.append('actor_rollout_ref.actor.policy_safety.monitor_trajectories_per_rank='+
                     str(monitor_trajectories_per_rank(run)))
    diagnostic = run.get('finite_thresholds_diagnostic_only',False)
    if type(diagnostic) is not bool:
        raise ValueError('Finite-threshold monitoring manifest flag must be bool')
    if diagnostic and (run.get('question_probe') or {}).get('probe_architecture')!='qwen2_decoder3_binary_mix_v1':
        raise ValueError('Diagnostic-only finite thresholds require explicitly bound binary run')
    overrides.append('actor_rollout_ref.actor.policy_safety.finite_thresholds_diagnostic_only='+str(diagnostic).lower())
    extra = run.get('extra_policy_monitor_forwards',True)
    if type(extra) is not bool or (not extra and not diagnostic):
        raise ValueError('No-extra-monitor mode requires explicit binary diagnostic-only binding')
    overrides.append('actor_rollout_ref.actor.policy_safety.extra_policy_monitor_forwards='+str(extra).lower())
    from reproduction.common8.qprobe_configuration import arguments as qprobe_arguments
    overrides.extend(qprobe_arguments(run, base))
    from search_g1.rollout_phase_contract import hydra_arguments
    overrides.extend(hydra_arguments(run))
    if run.get('full_state_restore'):
        restore=run['full_state_restore']
        overrides += ['+actor_rollout_ref.model.actor_checkpoint_path='+json.dumps(restore['source_checkpoint']),
                      '+trainer.full_state_restore_receipt='+json.dumps(restore['source_receipt'])]
    return run, overrides


def resolved(base):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    run, overrides = arguments(base)
    with initialize_config_dir(config_dir=str(REPO / 'verl/trainer/config'), version_base=None):
        config = compose(config_name='ppo_trainer', overrides=overrides)
    OmegaConf.resolve(config)
    check_global(config.actor_rollout_ref, config.trainer.n_gpus_per_node,
                 config.actor_rollout_ref.actor.ulysses_sequence_parallel_size, run=run)
    assert config.data.train_batch_size == 512
    assert config.actor_rollout_ref.actor.grad_clip == 1.0
    verify_precision_config(OmegaConf.to_container(config, resolve=True), run)
    assert config.trainer.total_training_steps == 201
    assert config.actor_rollout_ref.model.path == run['initial_model']
    assert config.trainer.default_local_dir == run['checkpoint_dir']
    assert config.data.train_files == str(Path(base).resolve() / 'data/train.parquet')
    assert config.data.val_files == str(Path(base).resolve() / 'data/stage.parquet')
    verify_formal_config(OmegaConf.to_container(config, resolve=True), run, base)
    assert config.actor_rollout_ref.actor.policy_safety.finite_thresholds_diagnostic_only is run.get('finite_thresholds_diagnostic_only',False)
    assert config.actor_rollout_ref.actor.policy_safety.extra_policy_monitor_forwards is run.get('extra_policy_monitor_forwards',True)
    from reproduction.common8.qprobe_configuration import validate as validate_qprobe
    validate_qprobe(config, run, base)
    return run, config, overrides


def launch(base):
    import sys
    from omegaconf import OmegaConf
    base = Path(base).resolve()
    run, config, overrides = resolved(base)
    claim = json.loads((base / 'logs/training-launch-claim.json').read_text())
    assert claim['run_id'] == run['run_id']
    assert claim['slurm_job_id'] == execution_job_id(run)
    if run.get('execution_backend')=='ssh_local':
        assert claim['execution_backend']=='ssh_local' and claim['execution_id']==run['execution_id']
        assert claim['hostname']==run['execution_host']==socket.gethostname()
    assert claim['shell_pid'] == os.getppid(), 'Only the claimed shell may launch'
    stat = Path(f'/proc/{claim["shell_pid"]}/stat').read_text().rsplit(')', 1)[1].split()
    assert int(stat[19]) == claim['shell_start_ticks']
    document = OmegaConf.to_container(config, resolve=True)
    canonical = json.dumps(document, sort_keys=True, separators=(',', ':')).encode()
    proof = dict(run_id=run['run_id'], slurm_job_id=claim['slurm_job_id'],
                 checked_unix=time.time(), recipe_sha256=sha(RECIPE),
                 resolved_config_sha256=hashlib.sha256(canonical).hexdigest(),
                 resolved_config=document, actual_argv_overrides=overrides,
                 formal_contract=verify_formal_config(document, run, base),
                 semantics='resolved launch configuration; not actual optimizer or model evidence')
    if run.get('execution_backend')=='ssh_local':
        proof.update(execution_backend='ssh_local',execution_id=run['execution_id'],
                     hostname=socket.gethostname())
    with (base / 'logs/resolved-training-config.json').open('x') as f:
        json.dump(proof, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.execv(sys.executable, [sys.executable, '-m', 'verl.trainer.main_ppo_eco', *overrides])


if __name__ == '__main__':
    launch(os.environ['SEARCH_G1_BASE'])
