"""One-off CPU-only checks of published d915 against immutable job730226 logs."""
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import unittest

COMMIT = 'd91575dbca221f66c467dccafa8508d9fb703688'
OLD = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018')
NEW = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-recovery5-after730226-20261018')
SOURCE = NEW / 'source' / COMMIT
AUDIT = NEW / 'preparation' / 'd915-independent-cpu-reader-check-20261018'
RUN = 'dfe9c94084bb48b2a9bb7b60f9d882da'
TRAINING = '8a5bac94b8230a99b81977f5290589e3d3708dfe'
MODULES = ('search_g1.test_training_state', 'search_g1.test_full_state_restore',
           'search_g1.test_easy2560_restore', 'search_g1.test_easy2560_restore_runtime',
           'reproduction.common8.test_formal_contract', 'reproduction.common8.test_configuration')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(name, value):
    value.update(checked_unix=time.time(), host=socket.gethostname(), uid=os.getuid(),
                 operator_commit=COMMIT, source_root=str(SOURCE), cpu_only=True,
                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 actual_cuda_restore_accepted=False)
    with (AUDIT / name).open('x') as out:
        json.dump(value, out, sort_keys=True, indent=2); out.write('\n')
        out.flush(); os.fsync(out.fileno())
    print(json.dumps(dict(receipt=str(AUDIT/name), status=value['status'], checked_unix=value['checked_unix'])), flush=True)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.getuid() == 14090
    sys.path[:0] = [str(SOURCE)]
    AUDIT.mkdir(parents=True, exist_ok=False)
    publish('started.json', dict(status='CPU_CHECKS_STARTED', pid=os.getpid(), python=sys.executable))
    os.environ.update(SEARCH_G1_BASE=str(OLD), SEARCH_G1_TRAINING_WORLD_SIZE='4')
    identities = {}
    audit_ok = False
    try:
        names = ('reproduction.common8.behavior_runtime', 'reproduction.common8.runtime',
                 'reproduction.common8.formal_inline_safety', 'reproduction.common8.formal_attempt_audit',
                 'reproduction.common8.safety_calibration_evidence')
        modules = [importlib.import_module(name) for name in names]
        for name, module in zip(names, modules):
            path = Path(module.__file__).resolve()
            assert path.is_relative_to(SOURCE), (name, str(path))
            identities[name] = dict(path=str(path), sha256=sha(path))
        behavior, runtime, inline = modules[:3]
        run = json.loads((OLD/'logs/run-manifest.json').read_text())
        assert (run['run_id'], run['source_commit'], run['training_world_size']) == (RUN, TRAINING, 4)
        records = [json.loads(line) for line in (OLD/'logs/actual-updates.jsonl').read_text().splitlines()]
        complete = [row for row in records if row['step'] <= 5]
        assert [row['step'] for row in complete] == list(range(1, 6))
        gates = {str(row['step']): behavior.read_gate_receipt(OLD, run, row['step'], update_record=row) for row in complete}
        print('CPU_PROGRESS=READ_FIVE_REAL_BEHAVIOR_GATES', flush=True)
        schedulers = runtime.verify_scheduler_receipts(OLD, run, 1, 5, gates)
        plan = json.loads((OLD/inline.PLAN).read_text())
        _, plan_sha = inline.verify_plan(OLD, plan, run)
        print('CPU_PROGRESS=READ_FOUR_REAL_SCHEDULER_PREFIXES_AND_PLAN', flush=True)
        proof = inline.verify_first_five(OLD, run, gates, schedulers)
        assert proof['structural_accepted'] is True and proof['actual_complete_outers'] == 5
        assert proof['actual_optimizer_writes_per_rank'] == 100
        keep = ('role', 'run_id', 'structural_accepted', 'actual_complete_outers', 'actual_full_budget_outers',
                'actual_optimizer_writes_per_rank', 'unique_global_monitor_count', 'actual_monitor_forward_count',
                'reused_monitor_forward_count', 'guard_triggered_monitor_count',
                'completed_outer_optimizer_opportunities_per_rank', 'selected_evidence_sha256')
        result = {key: proof[key] for key in keep if key in proof}
        publish('actual-first-five-reader.json', dict(status='ACTUAL_OLD_FIRST_FIVE_D915_READER_PASS',
            source_base=str(OLD), training_commit=TRAINING, source_job='730226', run_id=RUN,
            loaded_readers=identities, plan_sha256=plan_sha, behavior_gate_steps=sorted(gates),
            scheduler_ranks=sorted(schedulers), proof=result, model_tensors_reloaded=False,
            old_acceptance_or_log_mutated=False))
        audit_ok = True
    except BaseException:
        publish('actual-first-five-reader.json', dict(status='ACTUAL_OLD_FIRST_FIVE_D915_READER_FAILED',
            loaded_readers=identities, traceback=traceback.format_exc(), old_acceptance_or_log_mutated=False))
    # Remove production environment from isolated fixture tests. They use only temp files.
    for key in ('SEARCH_G1_BASE', 'SEARCH_G1_TRAINING_WORLD_SIZE', 'SLURM_JOB_ID', 'SLURM_JOBID'):
        os.environ.pop(key, None)
    stream = io.StringIO()
    try:
        import torch
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()
        suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        assert not torch.cuda.is_initialized()
        with (AUDIT/'targeted-tests.txt').open('x') as output:
            output.write(stream.getvalue()); output.flush(); os.fsync(output.fileno())
        publish('targeted-tests.json', dict(status='TARGETED_CPU_TESTS_PASS' if result.wasSuccessful() else 'TARGETED_CPU_TESTS_FAILED',
            modules=list(MODULES), tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
            skipped=len(result.skipped), log_sha256=sha(AUDIT/'targeted-tests.txt'), log_path=str(AUDIT/'targeted-tests.txt'),
            failure_details=[dict(test=str(test), traceback=error) for test,error in result.failures+result.errors],
            torch_version=torch.__version__, cuda_initialized=False))
        tests_ok = result.wasSuccessful()
    except BaseException:
        publish('targeted-tests.json', dict(status='TARGETED_CPU_TESTS_FAILED', traceback=traceback.format_exc()))
        tests_ok = False
    publish('complete.json', dict(status='INDEPENDENT_CPU_CHECKS_PASS' if audit_ok and tests_ok else 'INDEPENDENT_CPU_CHECKS_FAILED',
        actual_first_five_pass=audit_ok, targeted_tests_pass=tests_ok, training_started_or_modified=False))
    return 0 if audit_ok and tests_ok else 1


if __name__ == '__main__':
    sys.exit(main())
