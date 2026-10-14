"""CPU-only check of the actual authorized alpha1 recovery lineage and bindings.

Never load rank optimizer tensors on the login node or submit GPU work here.
"""
import ast
import hashlib
import json
import math
import pathlib
import textwrap
import time

ROOT = pathlib.Path('/data/user/jxu957/experiments/topd-20261008/runtime')
RUN = ROOT/'logs/topd-hpc3-713717/evidence'
TOOLS = ROOT/'tools/topd-alpha1-final51713-v1-20261014'


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    code = (TOOLS/'verify_topd_alpha1_final51713.py').read_text()
    ast.parse(code)
    start = code.index(" outer=json.loads((RUN/'actual-outer-updates.json').read_text())")
    end = code.index(" identity=json.loads(", start)
    namespace = dict(ROOT=ROOT, RUN=RUN, sha=sha, json=json, math=math)
    exec(textwrap.dedent(code[start:end]), namespace)
    assert namespace['math_steps'] == list(range(1, 201))
    assert len(namespace['rank_audits']) == 4
    for rank in range(4):
        proof = json.loads((RUN/('resume-loaded-rank%d.json'%rank)).read_text())
        assert proof['status'] == 'ACTUAL_GPU_WORKER_STEP5_FULL_RESTORE_EQUALITY_PASS'
        assert proof['outer_step'] == 5 and proof['optimizer_steps'] == 100
        assert all(proof[key] for key in ['model_local_equal', 'adam_equal',
                                         'scheduler_equal', 'rng_equal',
                                         'actual_update_history_equal'])
    driver = json.loads((RUN/'resume-loaded-driver.json').read_text())
    assert driver['rng_equal'] and driver['next_outer_step'] == 6 and driver['next_batch'] == 5
    original5 = ROOT/'logs/topd-hpc3-712204/evidence/checkpoints/actor/global_step_5'
    for outer in [5, 50, 100, 150, 200]:
        directory = original5 if outer == 5 else RUN/('checkpoints/actor/global_step_%d'%outer)
        if outer == 5 and not (directory/'resume-complete.json').is_file():
            continue
        receipt = json.loads((directory/'resume-complete.json').read_text())
        assert receipt['status'] == 'ALL_FOUR_RANKS_AND_DRIVER_RESUME_STATE_VERIFIED'
        assert receipt['outer_step'] == outer and len(receipt['ranks']) == 4
        for item in receipt['ranks'] + [receipt['driver']]:
            path = pathlib.Path(item['path'])
            assert path.parent == directory and path.stat().st_size == item['bytes']
    manifest = ROOT/'assets/common-instruct-0b002b/manifest.json'
    data = ROOT/'assets/common-instruct-0b002b/final.parquet'
    assert sha(manifest) == '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
    assert sha(data) == '55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1'
    result = dict(status='ALPHA1_ACTUAL_200_OUTERS_AND_AUTHORIZED_STEP5_LINEAGE_CPU_GUARDS_PASS',
                  epoch=time.time(), training_job=713717, alpha=1,
                  rank_audits=namespace['rank_audits'], math_outers=200,
                  frozen_teacher_rounds=200, checkpoint_paths_and_sizes_pass=True,
                  full_checkpoint_SHA_and_CPU_tensor_read_deferred_to_allocated_job=True,
                  historical_step5_present=(original5/'resume-complete.json').is_file())
    path = ROOT.parent/'outputs/training-preparation/alpha1-final51713-lineage-preflight.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
