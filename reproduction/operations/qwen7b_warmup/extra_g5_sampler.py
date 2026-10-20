"""Isolated NEW-process adapter around the unchanged qualified G5 collector.

Only this process's imported initial_state binding is adapted to indices5..9.
No source file is patched, and no other process/module address space is touched.
The original collector retains every sampling/retrieval/strict-EM contract.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from .extra_g5_prepare import (ROLE, REVISION, atomic, extra_initial_state,
                               read, remote, sha, validate_config)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--shard', required=True, type=int, choices=range(4))
    a = p.parse_args()
    c = validate_config(read(a.config))
    base = remote(c['base'])
    if Path(a.config).resolve() != base/'config.json':
        raise ValueError('Sampler configuration must be its isolated phase config')
    for key in ('original_actor_complete', 'original_bank', 'original_config', 'original_bank_complete', 'final_plan'):
        proof = c[key]
        if sha(proof['path']) != proof['sha256']:
            raise ValueError('Pinned immutable source changed: '+key)
    root = Path(c['code_root'])
    for relative, digest in c['code_files_sha256'].items():
        if sha(root/relative) != digest:
            raise ValueError('Sampler source differs from frozen published config: '+relative)
    actor = read(c['original_actor_complete']['path'])
    if actor['revision'] != REVISION:
        raise ValueError('Wrong official initial actor revision')
    # Read all snapshot files, not only a model-path/revision string.
    for name, proof in actor['files'].items():
        path = Path(c['model_dir'])/name
        if Path(name).name != name or path.stat().st_size != proof['bytes'] or sha(path) != proof['sha256']:
            raise ValueError('Official initial actor file SHA/size changed: '+name)
    controller = base/'sampling'/('shard-%02d' % a.shard)
    controller.mkdir(parents=True, exist_ok=True)
    lock = (controller/'extra-g5-adapter.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic(controller/'extra-g5-adapter-controller.json', dict(role=ROLE, pid=os.getpid(),
        run_id=c['run_id'], sampling_attempt_id=c['sampling_attempt_id'], source_commit=c['source_commit'],
        job_id=os.environ.get('SLURM_JOB_ID'), original_actor_fresh_all_file_SHA=True,
        response_indices=c['extra_response_indices'], source_files_mutated=False, epoch=time.time()))
    from . import sampling, shard
    from .native_capture import PREFIX, QMAX
    if sampling.PREFIX_TOKENS != 20 or PREFIX != 20 or QMAX != 273:
        raise ValueError('Runtime prefix/feature environment changed from frozen20-token protocol')
    if sampling._CACHED:
        raise ValueError('Extra-G5 adapter requires a fresh process with no prior actor engine')
    original_factory = sampling.initial_state
    def state_factory(config, outer, group, response_index, tokenizer):
        return extra_initial_state(config, outer, group, response_index, tokenizer, factory=original_factory)
    sampling.initial_state = state_factory
    try:
        # shard.main parses the same --config/--shard argv. Its outputs live only
        # under the distinct new base and retain independent writer/lock state.
        shard.main()
    finally:
        sampling.initial_state = original_factory
    from .extra_g5_merge import verify_extra_shard
    receipt = verify_extra_shard(base, a.shard)
    atomic(controller/'extra-g5-complete.json', receipt)
    print(json.dumps(dict(status=receipt['status'], shard=a.shard,
                          questions=receipt['questions'], trajectories=receipt['trajectories'],
                          actor_updates=0, genuine_new_G5_verified=True)), flush=True)


if __name__ == '__main__':
    main()
