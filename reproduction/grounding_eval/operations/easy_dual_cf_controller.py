"""Easy's one verified normal source -> separate frozen-v4 and strict-v5 CF.

This is a task controller, outside both immutable scientific runtimes. It never
trains, downloads, regenerates normal, or shares CF results between protocols.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    assert job['method'] == 'Easy Selection'
    assert job['run_id'] == '15bb15365346401da1d919e9cde8f813'
    assert job['normal_recompute_authorized'] is False
    assert job['protocols']['v4']['source_commit'].startswith('21b9ccf4')
    assert job['protocols']['v5']['source_commit'] == '16c36aa777f684f245b1d38cb2939ef175131831'
    assert job['protocols']['v4']['output'] != job['protocols']['v5']['output']
    gate_path = Path(job['g1_acceptance_gate'])
    assert sha(gate_path) == job['g1_acceptance_gate_sha256']
    gate = json.loads(gate_path.read_text())
    assert gate['accepted'] and gate['source_commit'] == job['protocols']['v5']['source_commit']
    assert 'Easy Selection strict-v5 after normal complete' in gate['authorized_next']
    assert gate['normal_recompute_authorized'] is False
    normal = Path(job['normal_root'])
    normal_receipt = json.loads((normal/'normal-independent-verification.json').read_text())
    assert normal_receipt['state'] == 'VERIFIED' and normal_receipt['count'] == 3610
    assert json.loads((normal/'control/exit.json').read_text())['exit_code'] == 0
    assert json.loads((normal/'complete.json').read_text())['normal_jsonl_sha256'] == sha(normal/'normal.jsonl')
    root = Path(job['control_output']); root.mkdir(exist_ok=True)
    lock = (root/'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def publish(name, value):
        path = root/name; temporary = path.with_name(path.name+'.tmp')
        with temporary.open('w') as stream:
            json.dump(value, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)

    if (root/'job-binding.json').exists():
        assert json.loads((root/'job-binding.json').read_text()) == job
    else:
        publish('job-binding.json', job)
    if (root/'complete.json').exists():
        assert json.loads((root/'complete.json').read_text())['state'] == 'EASY_DUAL_CF_COMPLETE'
        return
    publish('worker.json', dict(pid=os.getpid(), argv=sys.argv, started_at=time.time()))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8')

    def step(name, command, runtime, *, cpu=False):
        started = time.time()
        publish('progress.json', dict(phase=name, command=command, started_at=started))
        child_env = dict(env, PYTHONPATH=str(runtime))
        if cpu:
            child_env['CUDA_VISIBLE_DEVICES'] = ''
        with (root/(name+'.log')).open('ab') as log:
            child = subprocess.Popen(command, cwd=runtime, env=child_env,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            code = child.wait()
        publish(name+'-exit.json', dict(child_pid=child.pid, exit_code=code,
                started_at=started, completed_at=time.time()))
        if code:
            raise RuntimeError(name+' failed with exit '+str(code))

    try:
        for version in ('v4', 'v5'):
            config = job['protocols'][version]; runtime = Path(config['runtime'])
            deployment = json.loads((runtime/'deployment-receipt.json').read_text())
            assert deployment['source_commit'] == config['source_commit']
            for name, value in deployment['files'].items():
                assert sha(runtime/name) == value['sha256']
            module = [job['python'], '-m', config['module']]
            out = Path(config['output']); manifest = out/'manifest.json'
            if not manifest.exists():
                step(version+'-prepare', module+['prepare','--source',str(normal/'normal.jsonl'),
                    '--checkpoint',job['checkpoint'],'--output',str(out),'--method',job['method'],
                    '--run-id',job['run_id'],'--normal-config',str(normal/'normal-binding.json')], runtime, cpu=True)
            bound = json.loads(manifest.read_text())
            assert bound['method'] == job['method'] and bound['run_id'] == job['run_id']
            assert bound['source']['path'] == str((normal/'normal.jsonl').resolve())
            assert bound['checkpoint']['path'] == str(Path(job['checkpoint']).resolve())
            assert bound['normal_config_path'] == str((normal/'normal-binding.json').resolve())
            if version == 'v5':
                assert bound['protocol_version'] == 'strict-v5-conservative-typed'
            step(version+'-run', module+['run','--manifest',str(manifest),'--max-batch','256',
                '--max-memory-utilization','0.90','--gpu-uuid',job['gpu_uuid']], runtime)
            step(version+'-verify', module+['verify','--manifest',str(manifest)], runtime, cpu=True)
            summary = json.loads((out/'verified-summary.json').read_text())
            assert summary['status'] == 'complete' and summary['counts']['normal_questions'] == 3610
            acceptance = out/'independent-acceptance-20261019.json'
            if not acceptance.exists():
                step(version+'-independent-acceptance', [job['python'],'-B',job['content_identity_verifier'],
                     '--legacy-verifier',config['verifier'],'--manifest',str(manifest),
                     '--runtime',str(runtime),'--output',str(acceptance)], runtime, cpu=True)
            assert json.loads(acceptance.read_text())['state'] == 'INDEPENDENT_CF_FULL_READ_VERIFIED'
            replay = out/'official-changed-rule-replay.json'
            if not replay.exists():
                step(version+'-official-changed-rule', [job['python'],'-B',job['rule_replay'],
                    '--method',job['method'],'--cf-root',str(out),'--output',str(replay)], runtime, cpu=True)
            publish(version+'-accepted.json', dict(protocol=version, acceptance_sha256=sha(acceptance),
                    normal_content_order_sha256=json.loads(acceptance.read_text())['normal_content_order_sha256'],
                    verified_summary_sha256=sha(out/'verified-summary.json'), at=time.time()))
        publish('complete.json', dict(state='EASY_DUAL_CF_COMPLETE', completed_at=time.time(),
            normal_recomputed=False, protocols_kept_separate=True, job=job))
        publish('progress.json', dict(phase='COMPLETE', at=time.time()))
    except BaseException as error:
        publish('failed.json', dict(error_type=type(error).__name__, error=str(error), at=time.time()))
        raise


if __name__ == '__main__':
    main()
