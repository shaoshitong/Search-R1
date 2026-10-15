"""Reuse measured round-0 reward heads only after an owned pristine failed run.

This relocates historical calibration, never current-node service admission or
the separate native question probe's weights/Adam state.
"""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import subprocess
import time
import uuid

from reproduction.common8.verify_data import digest
from .common import DEFAULT_BASE, MODEL
from .retry_before_training import SOURCE_LOCKS, pristine_student

PRODUCERS = ('reproduction/common8/refit.py', 'reproduction/a800/probe_admission.py',
    'reproduction/a800/recovery_binding.py', 'ts_probe_dump.py', 'ts_probe_train.py',
    'search_g1/signals.py', 'search_g1/calibration_evidence.py',
    'search_r1/qa_protocol.py', 'verl/utils/reward_score/qa_em.py')


def read(path):
    return json.loads(path.read_text())


def owned(path, *, directory=False):
    assert path.is_absolute() and path.resolve() == path and not path.is_symlink(), str(path)
    assert path.stat().st_uid == os.getuid(), 'Historical artifact has a different owner: '+str(path)
    assert path.is_dir() if directory else path.is_file(), str(path)


def matching_bindings(old, run):
    for key in ('initial_model', 'model_revision', 'initial_model_sha256', 'input_sha256',
                'calibration_question_ids'):
        assert old[key] == run[key], 'Historical initial calibration binding differs: '+key
    assert old.get('completed_updates', 0) == run.get('completed_updates', 0) == 0
    assert not old.get('full_state_restore') and not run.get('full_state_restore')
    return {name: old['source_sha256'][name] for name in PRODUCERS
            if old['source_sha256'][name] == run['source_sha256'][name]}


def reuse_initial_calibration(base, run):
    certificate = base/'preparation/zero-update-retry.json'
    if not certificate.exists():
        return False
    retry = read(certificate)
    source = Path(retry['source_base'])
    pattern = re.escape(str(DEFAULT_BASE))+r'(?:-r[1-9][0-9]*)?'
    assert source != base and re.fullmatch(pattern, str(source))
    owned(source, directory=True); owned(base, directory=True)
    live_path = source/'artifacts/live/manifest.json'
    if not live_path.exists():
        return False
    assert retry['student_updates_before_failure'] == 0 and not retry.get('deferred_cleanup')
    source_job = retry['source_job']
    assert source_job.isdecimal()
    submission = read(source/'preparation/formal-submission.json')
    assert submission['job'] == source_job
    username = pwd.getpwuid(os.getuid()).pw_name
    rows = subprocess.check_output(['sacct', '-nP', '-X', '-j', source_job,
        '--format=JobID,State,ExitCode,User'], text=True, timeout=30).strip().splitlines()
    assert len(rows) == 1
    row = rows[0].split('|')
    assert row[0] == source_job and row[1].split()[0] in ('FAILED', 'OUT_OF_MEMORY', 'TIMEOUT')
    assert row[3] == username and rows[0] == retry['actual_source_terminal']
    assert source_job not in subprocess.check_output(['squeue', '-h', '-u', str(os.getuid()),
        '-o', '%i'], text=True, timeout=30).split()
    with contextlib.ExitStack() as stack:
        for name in SOURCE_LOCKS:
            lock = stack.enter_context((source/name).open('r+'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pristine_student(source)
        old = read(source/'logs/run-manifest.json')
        cleanup = read(source/'logs/cleanup-status.json')
        assert cleanup['cleaned'] is True and cleanup['run_id'] == old['run_id'] == retry['source_run_id']
        assert run['initial_model'] == str(MODEL)
        assert run['model_revision'] == 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
        producers = matching_bindings(old, run)
        assert set(producers) == set(PRODUCERS), 'Historical calibration producer changed'
        for name, sha in producers.items():
            assert digest(source/'repo'/name) == digest(base/'repo'/name) == sha
        for name, sha in run['input_sha256'].items():
            assert digest(source/'data'/name) == digest(base/'data'/name) == sha
        overlays = [root/'env/overlay-installed.json' for root in (source, base)]
        if any(path.exists() for path in overlays):
            from .environment_overlay import inventory
            assert all(path.exists() for path in overlays), 'Calibration dependency proof missing'
            previous, current = map(read, overlays)
            for key in ('pins', 'core_before', 'files_sha256'):
                assert previous[key] == current[key], 'Calibration dependency differs: '+key
            for root, proof in ((source, previous), (base, current)):
                assert proof['base'] == str(root)
                assert inventory(root/'env/overlay') == proof['files_sha256']
        owned(live_path)
        live = read(live_path)
        from reproduction.a800.probe_admission import calibrated_version
        version = calibrated_version(old, live)
        proof, report = live['label_proof'], live['calibration_report']
        assert live['round'] == 0 and set(proof['question_ids']) == set(run['calibration_question_ids'])
        assert report['require_all_cv_folds'] is True and report['configured_cv_folds'] == 2
        assert proof['calibration_protocol']['rollout_max_new'] == 500
        assert proof['calibration_protocol']['correction_feedback'] is True
        assert proof['evidence_deletion'] == 'typed_retrieval_ranges_only_preserve_model_and_correction'
        origin = Path(live['commit']).parent.parent
        owned(origin, directory=True)
        assert origin.parent == source/'artifacts' and origin.name.startswith(('round_0-attempt-', 'round_0-reused-'))
        assert Path(live['commit']) == origin/'heads/commit_probe.pkl'
        assert Path(live['boundary']) == origin/'heads/boundary_probe.pkl'
        hashes = {}
        for item in sorted(origin.rglob('*')):
            owned(item, directory=item.is_dir())
            if item.is_file():
                hashes[item.relative_to(origin).as_posix()] = digest(item)
        assert read(origin/'version-manifest.json') == live
        assert read(origin/'heads/calibration-report.json') == report
        assert read(origin/'dumps/calibration-label-proof.json') == proof
        assert hashes['heads/calibration-report.json'] == live['calibration_report_sha256']
        assert {p.name: hashes[p.relative_to(origin).as_posix()] for p in (origin/'dumps').iterdir()
                if p.is_file()} == live['dump_sha256']
        original_manifest_sha = digest(live_path)
        target = base/'artifacts'/('round_0-attempt-'+uuid.uuid4().hex)
        live_target = base/'artifacts/live'
        assert not target.exists() and not live_target.exists(), 'Never overwrite calibration artifacts'
        shutil.copytree(origin, target)
        for name, sha in hashes.items():
            assert digest(target/name) == sha
        historical = target/('historical-version-manifest-'+original_manifest_sha+'.json')
        assert not historical.exists(), 'Never overwrite an older historical calibration manifest'
        (target/'version-manifest.json').rename(historical)
        moved = dict(live, run_id=run['run_id'], version=run['run_id']+':0:'+uuid.uuid4().hex,
            commit=str(target/'heads/commit_probe.pkl'), boundary=str(target/'heads/boundary_probe.pkl'),
            published_unix=time.time(), historical_origin=dict(source_base=str(source),
                source_job=source_job, source_run_id=old['run_id'], source_version=version,
                source_manifest=str(live_path), source_manifest_sha256=original_manifest_sha,
                source_producer_sha256=producers, source_artifact_sha256=hashes,
                measured_calibration_reused=True, new_rollouts_generated=False,
                current_node_service_admission=False))
        # Validate relocated hashes/run identity, without inventing HTTP evidence.
        calibrated_version(run, moved)
        for path in (target/'version-manifest.json', live_target/'manifest.json'):
            path.parent.mkdir(exist_ok=True)
            with path.open('x') as output:
                json.dump(moved, output, indent=2, allow_nan=False)
                output.flush(); os.fsync(output.fileno())
            parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        return moved
