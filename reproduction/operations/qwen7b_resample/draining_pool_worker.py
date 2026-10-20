"""New-stage wrapper adding a genuine whole-batch drain to the frozen loop."""
import importlib.util
import json
import os
from pathlib import Path
import time

_spec = importlib.util.spec_from_file_location('q7b_drain_original_pool', Path(__file__).with_name('pool_worker.py'))
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)
globals().update({key: value for key, value in vars(_core).items() if not key.startswith('_') and key != 'sample_loop'})


class _Drained(Exception):
    pass


def sample_loop(args):
    jobdir = Path(args.job_dir)
    binding = read(jobdir / 'allocation.json')
    original_handoff, original_sampler = _core.handoff, _core.single_gpu_sampler

    def before_claim(base, marker):
        result = original_handoff(base, marker)
        own = jobdir / 'drain.json'
        owner = os.environ.get('Q7B_SLOT_OWNER_DRAIN')
        for path, kind in ((own, 'own'), (Path(owner) if owner else None, 'owner')):
            if path is None or not path.exists():
                continue
            value = read(path)
            require(value['run_id'] == RUN_ID, 'Drain run identity differs')
            if kind == 'own':
                require(value['status'] == 'DRAIN_AT_WHOLE_BATCH_BOUNDARY' and value['job_id'] == binding['job_id']
                    and value['allocation_sha256'] == sha(jobdir / 'allocation.json'), 'Own drain allocation binding differs')
            else:
                require(value['status'] == 'DRAIN_RETRIEVAL_OWNER'
                    and value['owner_pid'] == binding['source_owner']['pid']
                    and value['owner_allocation_sha256'] == binding['source_owner_allocation_sha256'],
                    'Owner drain allocation binding differs')
            raise _Drained()
        return result

    _core.handoff = before_claim
    _core.single_gpu_sampler = single_gpu_sampler
    try:
        _core.sample_loop(args)
    except _Drained:
        progress = read(jobdir / 'progress.json') if (jobdir / 'progress.json').exists() else {}
        require(not progress or (progress['run_id'] == RUN_ID and progress['job_id'] == binding['job_id']), 'Drain progress identity differs')
        counts = {key: progress.get(key, 0) for key in ('completed_own_batches', 'questions', 'trajectories')}
        require(all(type(value) is int and value >= 0 for value in counts.values())
            and counts['trajectories'] == counts['questions'] * 5, 'Drain whole-batch counters differ')
        result = dict(status='POOL_WHOLE_BATCH_DRAINED', run_id=RUN_ID, job_id=binding['job_id'],
            **counts, drained_before_next_claim=True, global_goal_complete=False, epoch=time.time())
        publish(jobdir / 'sampler-exit.json', result)
        print(json.dumps(result), flush=True)
    finally:
        _core.handoff, _core.single_gpu_sampler = original_handoff, original_sampler
