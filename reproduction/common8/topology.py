"""Manifest-bound actual training topology; legacy common8 remains eight ranks."""
import json
import os
import re
import sys
from pathlib import Path


def _bound_run():
    base = os.environ.get('SEARCH_G1_BASE')
    if base is None:
        return None
    path = Path(base) / 'logs/run-manifest.json'
    return json.loads(path.read_text()) if path.is_file() else None


def world_size(run=None):
    """Use a bound manifest, or the launcher's explicit pre-manifest topology."""
    run = _bound_run() if run is None else run
    explicit = os.environ.get('SEARCH_G1_TRAINING_WORLD_SIZE')
    if explicit is not None:
        assert explicit in ('4', '8'), 'Unsupported explicit Search-G1 training topology'
    size = (run.get('training_world_size', 8) if run is not None else
            int(explicit) if explicit is not None else 8)
    assert type(size) is int and size in (4, 8), 'Only actual four/eight-rank training is supported'
    if run is not None:
        assert run['profile'] == 'common8'
        if explicit is not None:
            assert size == int(explicit), 'Launch topology differs from bound run'
        if 'budget' in run:
            assert type(run['budget']['world_size']) is int and run['budget']['world_size'] == size, \
                'Bound budget differs from actual training topology'
    distributed = sys.modules.get('torch.distributed')
    if distributed is not None and distributed.is_initialized():
        assert distributed.get_world_size() == size, 'Actual process group differs from bound training topology'
    return size


def ranks(run=None):
    return range(world_size(run))


def actor_local_micro(run=None):
    run = _bound_run() if run is None else run
    value = run.get("actor_micro_batch_per_rank", 2) if run is not None else 2
    grouped = (run is not None and world_size(run) == 4 and
               (run.get('actor_logical_micro_batch_per_rank') == 4 or
                (run.get('dynrpo7b_selection') is True and
                 run.get('actor_logical_micro_batch_per_rank') == 2)))
    allowed = (2, 4, 8, 16, 32) if grouped else (2, 4)
    assert type(value) is int and value in allowed
    assert actor_local_mini(run) % value == 0
    return value


def actor_logical_micro(run=None):
    run = _bound_run() if run is None else run
    physical = actor_local_micro(run)
    value = run.get('actor_logical_micro_batch_per_rank', physical) if run is not None else physical
    assert type(value) is int and value in (2, 4)
    assert physical % value == 0, 'Physical micros must contain whole logical micros'
    return value


def actor_global_micro(run=None):
    return actor_local_micro(run) * world_size(run)


def actor_local_mini(run=None):
    return 128 // world_size(run)


def local_trajectories(run=None):
    return 2560 // world_size(run)


def logprob_local_micro(run=None):
    return 32 // world_size(run)


def monitor_trajectories_per_rank(run=None):
    return 32 // world_size(run)


def formal_monitor_geometry(run):
    """Bound masked-mean monitor mass, with no change to worker normalization."""
    assert isinstance(run,dict), 'Formal monitor geometry requires a bound run'
    size=world_size(run)
    if size==4:
        assert 'actor_micro_batch_per_rank' in run, 'Native4 requires an explicit actor micro binding'
    micro=actor_local_micro(run);logical=actor_logical_micro(run);count=monitor_trajectories_per_rank(run)
    assert count%logical==0, 'Monitor rows must form complete logical micros'
    return dict(training_world_size=size,actor_micro_batch_per_rank=micro,
                actor_logical_micro_batch_per_rank=logical,
                monitor_trajectories_per_rank=count,
                expected_complete_effective_weight_mass=float(size*(count//logical)))


def monitor_scope(run=None):
    count = monitor_trajectories_per_rank(run)
    word = 'four' if count == 4 else 'eight'
    return 'fixed ' + word + ' trajectories per rank across length strata; not full rollout'


def execution_job_id(run=None):
    """Return a real native job ID, or null for an explicitly bound local run."""
    run = _bound_run() if run is None else run
    assert run is not None, 'Execution identity requires a bound run manifest'
    backend = run.get('execution_backend', 'slurm')
    if backend == 'ssh_local':
        assert run['slurm_job_id'] is None
        assert isinstance(run['execution_id'], str) and re.fullmatch('[0-9a-f]{32}', run['execution_id'])
        assert os.environ.get('SLURM_JOB_ID') is None, 'Local execution cannot inherit or spoof a Slurm job'
        return None
    assert backend == 'slurm', 'Unknown execution backend'
    job = os.environ['SLURM_JOB_ID']
    assert re.fullmatch('[1-9][0-9]*', job), 'Missing actual native Slurm job ID'
    if 'slurm_job_id' in run:
        assert run['slurm_job_id'] == job, 'Actual native job differs from bound run'
    return job


def validate_base(base, run):
    """Keep legacy native namespace strict; admit only the bound fresh A800 base."""
    base = Path(base)
    assert base.is_absolute() and base.resolve() == base
    if run.get('execution_backend', 'slurm') == 'ssh_local':
        execution_job_id(run)
        assert re.fullmatch(r'/data/shaoshitong/Search-G1-a800-fresh200-[A-Za-z0-9._-]+', str(base)), \
            'Local training requires its fresh A800 run namespace'
    elif run.get('execution_site') == 'hpc3':
        if run.get('dynrpo7b_selection') is True:
            from reproduction.dynrpo7b.infra_profile import own_base, native4
            own_base(base)
            assert native4(run), 'DynRPO requires its explicit four-rank recipe'
        elif run.get('easy2560_selection') is True:
            assert re.fullmatch(r'/data/user/jxu957/experiments/easy2560-corollout-[A-Za-z0-9._-]+',str(base)), \
                'Easy2560 requires its dedicated run namespace'
            assert world_size(run)==4, 'Easy2560 requires four native ranks'
        else:
            assert re.fullmatch(r'/data/user/jxu957/experiments/qprobe-teacher-warmup200-[A-Za-z0-9._-]+',str(base)), \
                'HPC3 requires the dedicated question-probe run namespace'
        assert run.get('execution_backend','slurm')=='slurm' and world_size(run) in (4,8)
    else:
        assert str(base).startswith('/mnt/data/shaoshitong/'), 'Native training namespace changed'
    return base
