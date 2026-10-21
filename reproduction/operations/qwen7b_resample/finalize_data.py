"""Plan/stage/status or CPU-seal the full data bank on physical NAS10 only.

This seals data, not GPU cleanup or completion of the surrounding goal. The
exact standalone auditor is staged beside this helper; sampling is untouched.
"""
import argparse
import base64
import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import time
from types import SimpleNamespace

AUDIT_SHA = '3def055adebfdb38e9326b9db385338c44800160c1b0a64e1ee78d8c3b68b906'
RUN_ID, SEED = '70188b8e50fd4b45bc420ce58014f46e', 20261021
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
PHYSICAL = '/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020'


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode() + b'\n'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def control_module():
    if not __package__:
        import sys
        root = str(Path(__file__).resolve().parents[3])
        if root not in sys.path:
            sys.path.insert(0, root)
    return importlib.import_module('reproduction.operations.qwen7b_resample.pool_control')


def write(path, value, immutable=False):
    data = encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        require(path.read_bytes() == data, 'Existing immutable artifact differs: ' + str(path))
        return
    target = path.with_name(path.name + '.tmp-%d-%d' % (os.getpid(), time.time_ns()))
    try:
        with target.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            try:
                os.link(target, path)
            except FileExistsError:
                require(path.read_bytes() == data, 'Existing immutable artifact differs: ' + str(path))
        else:
            os.replace(target, path)
    finally:
        target.unlink(missing_ok=True)


def qualified(row):
    expected = dict(status='COMPLETE', run_id=RUN_ID, seed=SEED, model_revision=REVISION,
                    auditor_sha256=AUDIT_SHA, failed_batches=0, verified_batches=1322,
                    discovered_input_batches=1322, canonical_unique_input_qids=169028,
                    verified_unique_qids=169028, verified_unique_trajectories=845140,
                    questions=169028, trajectories=845140, labels=845140,
                    unsealed_batches=0, sealed_pending_batches=0)
    return all(row.get(key) == value for key, value in expected.items())


def remote(action, base, code):
    require(os.geteuid() == 0 and socket.gethostname().split('.')[0] == 'h100-10', 'Physical NAS10 only')
    base = Path(base)
    root = base / 'preparation/data-bank-finalization-v1'
    receipt = root / 'complete.json'
    if action == 'status':
        audit = read(base / 'audit/latest.json') if (base / 'audit/latest.json').exists() else {}
        return dict(status='DATA_FINALIZATION_STATUS', finalization=read(receipt) if receipt.exists() else None,
                    global_data_seal=read(base / 'complete.json') if (base / 'complete.json').exists() else None,
                    audit={key: audit.get(key) for key in ('status', 'verified_batches', 'verified_unique_qids',
                           'verified_unique_trajectories', 'failed_batches')}, entire_goal_complete=False)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'seal.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            old = read(receipt)
            require(old['status'] == 'FULL_DATA_CPU_AUDIT_SEALED' and old['run_id'] == RUN_ID
                    and old['seed'] == SEED and old['questions'] == 169028 and old['trajectories'] == 845140,
                    'Existing finalization receipt belongs to different data')
            return dict(status='EXISTING_FINALIZATION_RECEIPT_NO_REPLAY', receipt=old)
        latest_path = base / 'audit/latest.json'
        if not latest_path.exists():
            return dict(status='PENDING', reason='Full exact-run COMPLETE CPU audit is required')
        latest_bytes = latest_path.read_bytes()
        if not qualified(json.loads(latest_bytes)):
            return dict(status='PENDING', reason='Full exact-run COMPLETE CPU audit is required')
        audit = load('finalization_audit', Path(__file__).with_name('audit_data.py'))
        with contextlib.redirect_stdout(io.StringIO()):
            checked = audit.Audit(SimpleNamespace(base=str(base), audit_dir=str(base / 'audit'),
                                                 path_map=[], max_new_batches=1)).run()
        write(root / 'recheck-latest.json', checked)
        if not qualified(checked) or checked['cache_reused_batches'] + checked['independently_read_batches'] != 1322:
            return dict(status='FAILED' if checked['status'] == 'FAILED' else 'PENDING',
                        audit_status=checked['status'], reason='Current file bindings do not cover the full bank',
                        recheck=str(root / 'recheck-latest.json'))
        membership, totals = [], {0: [0, 0, 0], 1: [0, 0, 0]}
        for inp in sorted((base / 'inputs').glob('shard-*/batch*.json')):
            seal = base / 'sampling' / inp.parent.name / 'batches' / inp.stem / 'complete.json'
            data, row = seal.read_bytes(), read(seal)
            shard = int(inp.parent.name.split('-')[1])
            require(shard in totals and row['run_id'] == RUN_ID and row['input_sha256'] == digest(inp.read_bytes()), 'Batch/input binding differs')
            membership.append(dict(input=str(inp.relative_to(base)), input_sha256=row['input_sha256'],
                                   seal=str(seal.relative_to(base)), seal_sha256=digest(data)))
            totals[shard] = [a + b for a, b in zip(totals[shard], [1, row['questions'], row['trajectories']])]
        require(len(membership) == 1322 and all(value == [661, 84514, 422570] for value in totals.values()), 'Shard totals differ')
        manifest_sha, member_sha = digest((base / 'inputs/manifest.json').read_bytes()), digest(encoded(membership))
        member_path = root / ('batch-membership-' + member_sha + '.json')
        write(member_path, membership, immutable=True)
        proof = dict(original_full_audit=json.loads(latest_bytes), original_full_audit_sha256=digest(latest_bytes),
                     current_full_binding_audit=checked, input_manifest_sha256=manifest_sha,
                     batch_membership_sha256=member_sha, code_sha256=code,
                     strict_em_recomputed=False, states_shards_independently_audited=False)
        proof_sha = digest(encoded(proof))
        proof_path = root / ('full-cpu-proof-' + proof_sha + '.json')
        write(proof_path, proof, immutable=True)
        intent_path = root / 'intent.json'
        if intent_path.exists():
            intent = read(intent_path)
            require(intent['input_manifest_sha256'] == manifest_sha and intent['batch_membership_sha256'] == member_sha
                    and intent['code_sha256'] == code, 'Previous finalization intent binds different data/code')
            require(digest(Path(intent['full_cpu_proof']).read_bytes()) == intent['full_cpu_proof_sha256'], 'Intent CPU proof differs')
        else:
            intent = dict(status='FULL_DATA_CPU_AUDIT_SEALED', run_id=RUN_ID, seed=SEED,
                          model_revision=REVISION, questions=169028, trajectories=845140, batches=1322,
                          latest_full_audit_sha256=digest(latest_bytes), input_manifest_sha256=manifest_sha,
                          batch_membership=str(member_path), batch_membership_sha256=member_sha,
                          full_cpu_proof=str(proof_path), full_cpu_proof_sha256=proof_sha, code_sha256=code,
                          strict_em_recomputed=False, states_shards_independently_audited=False,
                          data_complete=True, owned_resource_cleanup_complete=False, entire_goal_complete=False, epoch=time.time())
            write(intent_path, intent, immutable=True)
        common = dict(run_id=RUN_ID, seed=SEED, model_revision=REVISION, hidden_cached=False,
                      actor_updates=0, probe_updates=0, actor_optimizer_updates=0,
                      full_cpu_proof_sha256=intent['full_cpu_proof_sha256'], full_cpu_proof=intent['full_cpu_proof'],
                      strict_em_recomputed=False, states_shards_independently_audited=False,
                      data_complete=True, owned_resource_cleanup_complete=False, entire_goal_complete=False, epoch=intent['epoch'])
        shards = [dict(common, status='ALL_NEW_SEED_G5_BATCHES_COMPLETE', shard=i,
                       questions=84514, trajectories=422570, batches=661) for i in range(2)]
        records = [(base / 'sampling' / ('shard-%02d' % i) / 'complete.json', row) for i, row in enumerate(shards)]
        records.append((base / 'complete.json', dict(common, status='ALL_169028_QUESTIONS_NEW_SEED_G5_COMPLETE',
                                                   questions=169028, trajectories=845140, batches=1322, shards=shards)))
        for path, row in records:
            if path.exists():
                old = read(path)
                require(all(old.get(key, row[key]) == row[key] for key in ('status', 'run_id', 'questions', 'trajectories', 'batches')), 'Existing data seal differs')
            write(path, row)
        write(receipt, intent, immutable=True)
        return dict(status='DATA_BANK_SEALED', receipt=str(receipt), full_cpu_proof_sha256=intent['full_cpu_proof_sha256'],
                    data_complete=True, owned_resource_cleanup_complete=False, entire_goal_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'stage', 'seal', 'status'))
    args = parser.parse_args()
    files = {name: Path(__file__).with_name(name).read_bytes() for name in ('finalize_data.py', 'audit_data.py')}
    require(digest(files['audit_data.py']) == AUDIT_SHA, 'Exact audited source SHA is required')
    code = {name: digest(data) for name, data in files.items()}
    identity = digest(encoded(code))
    stage = PHYSICAL + '/preparation/data-bank-finalization-v1/code/' + identity
    if args.action == 'plan':
        print(json.dumps(dict(status='LOCAL_ONLY_PLAN', stage=stage, code_sha256=code, gpu_actions=False), indent=2))
        return
    values = dict(ACTION=args.action, PHYSICAL=PHYSICAL, STAGE=stage, CODE=code,
                  FILES={name: base64.b64encode(data).decode() for name, data in files.items()})
    script = '\n'.join(key + '=' + repr(value) for key, value in values.items()) + '\n' + r'''
import base64,hashlib,importlib.util,json,os,socket
from pathlib import Path
assert os.geteuid()==0 and socket.gethostname().split('.')[0]=='h100-10'
stage=Path(STAGE)
if ACTION=='stage':
    stage.mkdir(parents=True,exist_ok=True)
    for name,text in FILES.items():
        path=stage/name;data=base64.b64decode(text)
        if not path.exists():
            with path.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==CODE[name]
        os.chmod(path,0o444)
    result=dict(status='EXACT_FINALIZER_AND_AUDITOR_STAGE_VERIFIED',stage=STAGE,code_sha256=CODE)
else:
    for name,sha in CODE.items():assert hashlib.sha256((stage/name).read_bytes()).hexdigest()==sha
    spec=importlib.util.spec_from_file_location('data_finalizer',stage/'finalize_data.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.remote(ACTION,PHYSICAL,CODE)
print(json.dumps(result,sort_keys=True))
raise SystemExit(1 if result['status']=='FAILED' else 2 if result['status']=='PENDING' else 0)
'''
    control = control_module()
    result = control.transport(script)
    envelope = dict(action=args.action, returncode=result.returncode, stdout=result.stdout,
                    stderr=result.stderr, code_sha256=code, epoch=time.time())
    record = control.ROOT / 'work' / ('qwen7b-data-finalization-%s-%d.json' % (args.action, time.time_ns()))
    write(record, envelope)
    print(json.dumps(dict(envelope, local_receipt=str(record)), indent=2))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
