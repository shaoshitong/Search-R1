"""Seal the terminal own outer41 CUDA OOM; leave all original assets in place."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from reproduction.easy2560_joint4.control import publish, sha

BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-recovery5-after730961-20261018')
JOB = '731063'
COMMIT = 'a61b48139386b346e4fef12511ece1ac0d09a8e3'


def preserve():
    target = BASE/('failed-job-'+JOB)
    receipt = target/'failure-evidence.json'
    if receipt.is_file():
        result = json.loads(receipt.read_bytes())
        if result['source_job'] != JOB or result['source_commit'] != COMMIT:
            raise ValueError('731063 preserved identity differs')
        for row in result['files']:
            if sha(target/row['relative_path']) != row['sha256']:
                raise ValueError('731063 preserved file SHA differs')
        return result
    if target.exists():
        raise ValueError('Partial731063 preservation exists; inspect without overwriting')
    accounting = subprocess.run(['sacct', '-j', JOB, '--noheader', '--parsable2',
        '--format=JobIDRaw,State,ExitCode,NodeList,Start,End'],
        capture_output=True, text=True, timeout=30)
    states = {r[0]:r[1:] for r in (line.split('|') for line in accounting.stdout.splitlines()) if len(r)>=6}
    expected = {JOB:('FAILED','1:0'), JOB+'.batch':('FAILED','1:0'), JOB+'.extern':('COMPLETED','0:0')}
    if accounting.returncode or any(tuple(states.get(job,[])[:2]) != state or
            states[job][2] != 'ACD1-13' for job, state in expected.items()):
        raise ValueError('731063 is not the observed terminal own failure')
    run = json.loads((BASE/'logs/run-manifest.json').read_bytes())
    updates = [json.loads(line) for line in (BASE/'logs/actual-updates.jsonl').read_text().splitlines()]
    if ((BASE/'job.id').read_text().strip() != JOB or run['source_commit'] != COMMIT or
            run['actor_micro_batch_per_rank'] != 16 or
            [row['step'] for row in updates] != list(range(6,41))):
        raise ValueError('731063 completed/source boundary differs')
    log = (BASE/'logs/search-g1-common8-training.log').read_text()
    if 'torch.OutOfMemoryError: CUDA out of memory' not in log or 'response_logits = logits_rmpad[response_positions].float()' not in log:
        raise ValueError('731063 actual CUDA OOM differs')
    names = ['job.id', 'preparation/training-submit-claim.json',
        'preparation/training-submission.json', 'preparation/run-source.json']
    names += [str(path.relative_to(BASE)) for path in sorted((BASE/'logs').iterdir())
              if path.is_file() and not path.is_symlink() and path.suffix in ('.json','.jsonl','.log')]
    # Selected/native traces and failed candidate raw remain at their original
    # remote paths. This seal duplicates only the small failure/log evidence.
    target.mkdir(mode=0o770)
    publish(target/'terminal-observation.json', dict(checked_unix=time.time(),
        returncode=accounting.returncode, stdout=accounting.stdout, stderr=accounting.stderr))
    records = []
    for name in names:
        original = BASE/name
        if original.is_symlink() or not original.is_file() or original.stat().st_size > 256*1024**2:
            raise ValueError('731063 only small regular evidence files may be copied')
        before = sha(original)
        copied = target/name
        copied.parent.mkdir(parents=True, exist_ok=True)
        with original.open('rb') as src, copied.open('xb') as dst:
            shutil.copyfileobj(src, dst, 1024*1024)
            dst.flush(); os.fsync(dst.fileno())
        if sha(original) != before or sha(copied) != before:
            raise ValueError('731063 failure copy SHA differs')
        records.append(dict(relative_path=name, bytes=copied.stat().st_size, sha256=before))
    result = dict(status='FAILED731063_OUTER41_CUDA_OOM_PRESERVED', source_job=JOB,
        source_commit=COMMIT, source_base=str(BASE), completed_outer=40, failed_outer=41,
        selected_trace_and_candidate_raw_retained_at_original_paths=True, files=records,
        latest_durable_checkpoint_outer=5, resumed_training_must_recompute_from_outer=6,
        tensors_copied=False, original_evidence_modified=False, jobs_submitted=0,
        operator_sha256=sha(__file__), checked_unix=time.time())
    publish(receipt, result)
    return result


if __name__ == '__main__':
    result = preserve()
    print(json.dumps(dict(status=result['status'], files=len(result['files']),
        bytes=sum(row['bytes'] for row in result['files']),
        receipt=str(BASE/('failed-job-'+JOB)/'failure-evidence.json'),
        receipt_sha256=sha(BASE/('failed-job-'+JOB)/'failure-evidence.json'),
        completed_outer=40, failed_outer=41, latest_durable_checkpoint_outer=5,
        jobs_submitted=0)))
