"""Recover the real111 boundary after h100-5 lacked a linker libcuda.so.

Only task-local driver linking and explicit terminal-owner recovery change.
The deployed d15f311 training source, frozen config and native state stay intact.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import fcntl


def write(path, value):
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(value, sort_keys=True) + '\n')
    os.replace(temp, path)


def main():
    assert socket.gethostname().split('.')[0] == 'h100-5' and os.getuid() == 0
    base = Path('/dataset/shaoshitong/recursive-ablation150-20261014')
    source = base / 'source-h1005-shared-d15f311'
    reference = Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013')
    runtime = Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009')
    control = base / 'shared-h1005-driver-recovery2-20261018'
    control.mkdir(exist_ok=True)
    lock = (control / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (control / 'started.json').exists():
        raise RuntimeError('One-shot recovery already started; inspect actual state')
    prior = json.loads((base / 'shared-h1005-cuda-recovery-20261018/actual-exit.json').read_text())
    assert prior['rc'] == 1 and prior['stage'] == 'native-study-continuation'
    old = json.loads((base / 'budget-5/receipts/current-entry.json').read_text())
    assert (old['pid'], old['uid'], old['tick']) == (3527991, 0, 1534179663)
    terminal = json.loads((Path(old['output']) / 'entry-exit.json').read_text())
    assert terminal['rc'] == 1 and terminal['pid'] == old['pid']
    assert not Path('/proc', str(old['pid'])).exists()
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    assert not query.stdout.strip(), 'Do not overlap any current GPU process'
    receipts = base / 'budget-5/receipts'
    request_path = receipts / 'explicit-recovery.json'
    (control / 'previous-explicit-recovery.json').write_bytes(request_path.read_bytes())
    request = json.loads(request_path.read_text())
    assert request['kind'] == 'COMPLETE_NATIVE_BOUNDARY'
    assert request['sha256'] == '79bdfda99542d38244b10b3381ac47d286327212a1e5627e581a44c5e29fa9cc'
    request.update(previous_pid=old['pid'], previous_uid=old['uid'], previous_tick=old['tick'])
    write(request_path, request)
    linkdir = runtime / 'lib/driver'
    driver = Path('/usr/lib/x86_64-linux-gnu/libcuda.so.1').resolve(strict=True)
    old_links = {}
    for name in ('libcuda.so', 'libcuda.so.1'):
        path = linkdir / name
        assert path.is_symlink() and not path.exists(), 'Only replace observed stale task links'
        old_links[name] = str(path.readlink())
        path.rename(control / ('preserved-' + name))
        path.symlink_to(driver)
    write(control / 'task-driver-links.json', dict(previous=old_links, actual_driver=str(driver), scope='Task runtime links only; system libraries untouched'))
    check = subprocess.run(['/usr/bin/cc', '-shared', '-fPIC', '-x', 'c', '-', '-L' + str(linkdir), '-lcuda', '-o', str(control / 'link-check.so')], input='int task_link_probe(void) { return 0; }\n', capture_output=True, text=True)
    write(control / 'link-check.json', dict(rc=check.returncode, stderr=check.stderr, driver=str(driver)))
    check.check_returncode()
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('SLURM_') or key in ('LD_LIBRARY_PATH', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'RAY_ADDRESS'):
            env.pop(key)
    env.update(RBA_H100_SHARED_EXECUTION_ID='shared-h1005-ablation150-20261018',
        RBA_ABLATION_BASE=str(base), RBA_ABLATION_SOURCE_COMMIT='233b0dc95de3f7b797d5b424d3b7dcaafca5328f',
        RBA_ABLATION_ENTRY_COMMIT='d15f3119643b4a95364f3180f7730db102651106',
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1', NVIDIA_TF32_OVERRIDE='0', TRITON_LIBCUDA_PATH=str(linkdir),
        LIBRARY_PATH=str(linkdir), PYTHONPATH=str(source), PATH=str(runtime / 'env/bin') + ':/usr/local/bin:/usr/bin:/bin')
    write(control / 'started.json', dict(time=time.time(), pid=os.getpid(), uid=os.getuid(),
        tick=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        backend='h100_shared', execution_id=env['RBA_H100_SHARED_EXECUTION_ID'],
        resume_completed_outer=111, resume_Adam_each_rank=2220, driver=str(driver), library_path=str(linkdir)))
    stages = [('reuse-live-admitted-retriever', [str(runtime / 'env/bin/python'), '-B', str(source / 'release/recursive_bran_a/h100_migration/reuse_retriever.py'), str(reference / 'retriever')]),
              ('native-study-continuation', ['bash', str(source / 'release/recursive_bran_a/ablation150_v1/study.sh')])]
    for stage, command in stages:
        with (control / (stage + '.log')).open('xb') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            write(control / 'stage.json', dict(time=time.time(), stage=stage, pid=child.pid, status='RUNNING'))
            rc = child.wait()
        write(control / (stage + '-wait.json'), dict(time=time.time(), pid=child.pid, rc=rc))
        if rc:
            write(control / 'actual-exit.json', dict(time=time.time(), stage=stage, rc=rc))
            return rc
    write(control / 'actual-exit.json', dict(time=time.time(), stage='study-returned', rc=0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

