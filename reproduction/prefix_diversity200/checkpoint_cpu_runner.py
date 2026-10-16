"""Durable CPU reader entry: writes its own identity, exit and cost receipt."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def once(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True); handle.flush(); os.fsync(handle.fileno())


def main():
    root = Path(__file__).resolve().parents[2]
    deploy = json.loads((root/'deploy-receipt.json').read_text()); c = deploy['config']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and c['outer'] == 5
    once(root/'reader-start.json', dict(pid=os.getpid(), proc_stat=Path('/proc/self/stat').read_text(),
         started_unix=time.time(), helper_commit=deploy['helper_commit'], CUDA_VISIBLE_DEVICES=''))
    started = time.monotonic()
    with (root/'reader.log').open('xb') as log:
        process = subprocess.run([sys.executable, '-u', '-m', 'reproduction.prefix_diversity200.checkpoint_cpu_once',
            '--base', c['base'], '--outer', str(c['outer'])], cwd=root, stdout=log, stderr=subprocess.STDOUT)
        log.flush(); os.fsync(log.fileno())
    proof_path = Path(c['base'])/'preparation/joint-milestone-0005-independent-full-sha-cpu-once-v1.json'
    proof = json.loads(proof_path.read_text()) if proof_path.exists() else {}
    passed = process.returncode == 0 and proof.get('status') == 'PREFIX_SELECTION200_JOINT_MILESTONE_FULL_SHA_CPU_PASS'
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    once(root/'reader-exit.json', dict(status='CPU_ACCEPTANCE_PASS' if passed else 'CPU_ACCEPTANCE_FAILED_NO_REPLAY',
        subprocess_exit=process.returncode, exit_code=0 if passed else 1, finished_unix=time.time(),
        elapsed_seconds=time.monotonic()-started, cpu_user_seconds=usage.ru_utime,
        cpu_system_seconds=usage.ru_stime, max_rss_kib=usage.ru_maxrss,
        proof_path=str(proof_path), proof_status=proof.get('status'), error=proof.get('error'),
        proof_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest() if proof_path.exists() else None,
        helper_commit=deploy['helper_commit'], training_modified=False, gpu_forward_performed=False))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__': main()
