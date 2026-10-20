"""CPU evidence for an r6 pipeline failure on the withdrawn h100-11.

This is an alternative recovery proof, never a claim that the inaccessible
native unit, cgroup, or GPUs were inspected or cleaned. It changes no source
runtime, Slurm state, actor, or partial data. Run `observe`, then `seal` at least
60 seconds later. Both independently read and hash original shared files.
Local `plan` is inert; stage/observe/seal require exact Git publication checks.
"""

import argparse
import base64
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


ROOT = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
REPO = Path(__file__).resolve().parents[3]
RELATIVE_SOURCE = 'reproduction/operations/qwen7b_resample/source_terminal_handoff.py'
BASE = '/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020'
PHYSICAL = '/mnt/data/shaoshitong/' + BASE.rsplit('/', 1)[1]
RUN_ID = '70188b8e50fd4b45bc420ce58014f46e'
SEED = 20261021
DIRECTORY = 'preparation/r6-shared-terminal-recovery-v1'
FAILURE = 'Sampling worker terminal returncodes: [-9, -9, -9, -9, -9, -9, -9, -9]'
WORKER_PIDS = [2950545, 2950546, 2950547, 2950548, 2950549, 2950550, 2950556, 2950565]
UNAVAILABLE = 'UNAVAILABLE_SOURCE_HOST_WITHDRAWN'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value) + b'\n'
    if path.exists():
        require(path.read_bytes() == data, 'Refusing to overwrite immutable proof: ' + str(path))
        return
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o444)
    try:
        os.link(temporary, path)
    except FileExistsError:
        require(path.read_bytes() == data, 'Concurrent different immutable proof: ' + str(path))
    finally:
        temporary.unlink()


def identity(path):
    def stat():
        value = path.stat()
        return dict(bytes=value.st_size, inode=value.st_ino, device=value.st_dev,
                    mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)
    require(path.is_file() and not path.is_symlink(), 'Expected original regular shared file: ' + str(path))
    before = stat()
    result = dict(before, sha256=digest(path))
    require(before == stat(), 'Shared source changed while hashing: ' + str(path))
    return result


def node_observation():
    commands = [['scontrol', 'show', 'node', 'h100-11', '--oneliner'],
                ['squeue', '--noheader', '--nodelist=h100-11', '--format=%A|%i|%T|%N']]
    reports = []
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        require(result.returncode == 0, 'Slurm source node observation failed: ' + result.stderr)
        reports.append(dict(argv=command, returncode=result.returncode,
                            stdout=result.stdout, stderr=result.stderr, epoch=time.time()))
    text = reports[0]['stdout']
    require(re.search(r'\bNodeName=h100-11\b', text), 'Slurm returned another source node')
    state = re.search(r'\bState=(\S+)', text)
    require(state and 'DOWN' in state.group(1).split('+'), 'Source node is no longer DOWN')
    require(not reports[1]['stdout'].strip(), 'Slurm still lists jobs on the withdrawn source node')
    return dict(source_node_down=True, source_node_state=state.group(1),
                source_node_jobs=[], source_node_commands=reports)


def input_plan(base, config):
    manifest = read(base / 'inputs/manifest.json')
    require(manifest['questions'] == 169028 and manifest['trajectories'] == 845140
            and manifest['seed'] == SEED and len(manifest['batches']) == 1322,
            'The original global input bank differs')
    require(config['train_questions'] == 169028 and config['train_trajectories'] == 845140
            and config['sampling_shards'] == 2, 'Root global plan differs')
    rows, files, seen, source_qids = 0, [], set(), {}
    planned = set()
    for item in manifest['batches']:
        recorded = item['path']
        require(recorded.startswith(BASE + '/inputs/'), 'Unexpected original input path')
        relative = recorded[len(BASE) + 1:]
        require(relative not in planned, 'Duplicate planned input')
        planned.add(relative)
        path = base / relative
        fingerprint = identity(path)
        require(fingerprint['sha256'] == item['sha256'], 'Original input SHA differs: ' + relative)
        batch = read(path)
        require(batch['split'] == 'train' and len(batch['rows']) == item['questions']
                and batch['model_revision'] == config['model_revision'], 'Original batch plan differs')
        for row in batch['rows']:
            require(row['split'] == 'train' and row['qid'] not in seen, 'Global bank qid duplicates or split differs')
            seen.add(row['qid'])
            require(row['source_ordinal'] not in source_qids, 'Global bank ordinal duplicates')
            source_qids[row['source_ordinal']] = row['qid']
        rows += len(batch['rows'])
        files.append(dict(path=recorded, **fingerprint))
    require(rows == len(seen) == 169028, 'Actual original input bank does not contain 169028 unique questions')
    ordered = [source_qids[x] for x in sorted(source_qids)]
    require(hashlib.sha256(json.dumps(ordered).encode()).hexdigest() == manifest['ordered_qid_sha256'],
            'Global bank ordered qid binding differs')
    require({str(x.relative_to(base)) for x in (base / 'inputs').glob('shard-*/batch-*.json')} == planned,
            'Actual input files differ from original manifest')
    return dict(verified_input_plan_questions=rows, verified_input_plan_trajectories=rows * 5,
                verified_input_plan_batches=len(files), input_plan_fingerprint_sha256=hashlib.sha256(canonical(files)).hexdigest())


def observe(base):
    started = time.time()
    first_node = node_observation()
    config = read(base / 'config.json')
    require(config['run_id'] == RUN_ID and config['seed'] == SEED and config['base'] == BASE,
            'Root run/seed/base differs')
    require(config['warmup_only'] is True and config['actor_hidden_native'] is False
            and config['actor_optimizer_updates'] == 0, 'Root actor science differs')
    failure, exits = read(base / 'failure.json'), read(base / 'worker-exits.json')
    controller, children = read(base / 'controller.json'), read(base / 'children.json')
    require(failure['error_type'] == 'RuntimeError' and failure['error'] == FAILURE
            and exits['returncodes'] == [-9] * 8, 'Original eight-worker terminal failure differs')
    require(controller['run_id'] == RUN_ID and controller['seed'] == SEED
            and controller['pid'] == 2950135 and str(controller['start_ticks']) == '728996799'
            and len(controller['cards']) == 8 and controller['actor_updates'] == controller['probe_updates'] == 0,
            'Original r6 controller binding differs')
    require(children['attempt'] == 'r6' and children['worker_pids'] == WORKER_PIDS
            and children['retriever_pid'] == 2950141, 'Original r6 eight-child binding differs')
    require(controller['epoch'] < children['epoch'] < exits['epoch'] <= failure['epoch'],
            'Original terminal chronology differs')
    names = ['config.json', 'inputs/manifest.json', 'failure.json', 'worker-exits.json',
             'controller.json', 'children.json', 'progress.json']
    names += [str(x.relative_to(base)) for x in sorted(base.glob('worker-failure-*.json'))]
    logs = sorted((base / 'logs/r6').glob('*.log'))
    expected_logs = {'retriever.log'} | {'sampling-%02d%s.log' % (i, suffix)
                                             for i in range(8) for suffix in ('', '-signals')}
    require({x.name for x in logs} == expected_logs, 'Original r6 log inventory differs')
    names += [str(x.relative_to(base)) for x in logs]
    terminal = {BASE + '/' + name: identity(base / name) for name in names}
    plan = input_plan(base, config)
    closed, questions, trajectories = [], 0, 0
    for path in sorted((base / 'sampling').glob('shard-*/batches/batch-*/complete.json')):
        value = read(path)
        require(value['run_id'] == RUN_ID and value['trajectories'] == value['questions'] * 5,
                'Closed batch run/G5 differs')
        questions += value['questions']
        trajectories += value['trajectories']
        closed.append(dict(path=BASE + '/' + str(path.relative_to(base)), **identity(path)))
    require(0 < questions < 169028, 'Expected a partial original terminal dataset')
    second_node = node_observation()
    # A second stat after all input/batch hashing catches writers during this scan.
    require(terminal == {name: identity(base / name[len(BASE) + 1:]) for name in terminal},
            'Original terminal files changed during observation')
    return dict(observation_started_epoch=started, observed_epoch=time.time(),
        source_node_down=True, source_node_state=second_node['source_node_state'], source_node_jobs=[],
        source_node_commands=first_node['source_node_commands'] + second_node['source_node_commands'],
        terminal_files=terminal, actual_worker_returncodes=exits['returncodes'],
        original_pipeline_failure=failure, controller_binding=dict(pid=controller['pid'],
            start_ticks=str(controller['start_ticks']), run_id=RUN_ID, seed=SEED),
        actual_worker_pids=children['worker_pids'], closed_batch_count=len(closed),
        closed_questions=questions, closed_trajectories=trajectories,
        closed_batch_fingerprint_sha256=hashlib.sha256(canonical(closed)).hexdigest(), **plan)


def marker_for(receipt):
    return dict(status='R6_NATIVE_PIPELINE_FAILED_NODE_DOWN', run_id=RUN_ID, seed=SEED,
                r6_terminal_receipt=BASE + '/' + DIRECTORY + '/complete.json',
                r6_terminal_receipt_sha256=digest(receipt))


def summary(value):
    return {key: value[key] for key in ('status', 'run_id', 'seed', 'observations_elapsed_seconds',
            'source_native_unit_status', 'source_gpu_cleanup_status', 'global_goal_complete') if key in value}


def remote(action, base):
    require(sys.platform == 'linux' and os.geteuid() == 0
            and socket.gethostname().split('.')[0] == 'h100-10', 'Run this CPU helper only on NAS10')
    out = base / DIRECTORY
    if action == 'status':
        receipt = out / 'complete.json'
        result = dict(status='READ_ONLY_RECOVERY_STATUS', proof=summary(read(receipt)) if receipt.exists() else None,
                      observations=len(list((out / 'observations').glob('*.json'))),
                      marker=read(base / 'preparation/pool-handoff.json') if (base / 'preparation/pool-handoff.json').exists() else None)
        print(json.dumps(result));return
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = out / 'complete.json'
        if receipt.exists():
            result = read(receipt)
            require(result['status'] == 'R6_NATIVE_WORKERS_TERMINAL_NODE_WITHDRAWN'
                    and result['run_id'] == RUN_ID and result['seed'] == SEED, 'Different immutable proof exists')
            immutable(base / 'preparation/pool-handoff.json', marker_for(receipt))
            print(json.dumps(dict(summary(result), existing_immutable_proof=True, marker=marker_for(receipt))));return
        observation = observe(base)
        path = out / 'observations' / ('%d.json' % time.time_ns())
        immutable(path, observation)
        if action == 'observe':
            print(json.dumps(dict(status='FIRST_OR_ADDITIONAL_ACTUAL_OBSERVATION_RECORDED',
                path=BASE + '/' + str(path.relative_to(base)), observed_epoch=observation['observed_epoch'],
                closed_batch_count=observation['closed_batch_count'],
                closed_batch_fingerprint_sha256=observation['closed_batch_fingerprint_sha256'])));return
        earlier = [read(p) for p in sorted((out / 'observations').glob('*.json')) if p != path]
        earlier = [x for x in earlier if observation['observation_started_epoch'] - x['observed_epoch'] >= 60]
        require(earlier, 'A previous actual completed observation at least 60 seconds earlier is required')
        previous = earlier[-1]
        for key in ('terminal_files', 'closed_batch_count', 'closed_batch_fingerprint_sha256',
                    'verified_input_plan_questions', 'verified_input_plan_trajectories', 'verified_input_plan_batches',
                    'input_plan_fingerprint_sha256', 'actual_worker_returncodes', 'controller_binding',
                    'actual_worker_pids', 'original_pipeline_failure'):
            require(previous[key] == observation[key], 'Original shared evidence changed between observations: ' + key)
        result = dict(status='R6_NATIVE_WORKERS_TERMINAL_NODE_WITHDRAWN',run_id=RUN_ID,seed=SEED,
            source_node='h100-11',source_attempt='r6',actual_worker_returncodes=observation['actual_worker_returncodes'],
            original_pipeline_failure=observation['original_pipeline_failure'],controller_binding=observation['controller_binding'],
            actual_worker_pids=observation['actual_worker_pids'], observations=[previous, observation],
            observations_elapsed_seconds=observation['observation_started_epoch']-previous['observed_epoch'],
            stable_terminal_files=True,stable_closed_batch_fingerprint=True,
            original_pipeline_failure_observed=True,original_sampling_worker_wrappers_terminal_observed=True,
            original_shared_batch_writer_quiescence_observed=True,
            source_native_unit_status=UNAVAILABLE,source_gpu_cleanup_status=UNAVAILABLE,r6_cgroup_status=UNAVAILABLE,
            unsealed_partials_retained=True,shared_data_deleted=False,actor_updates=0,probe_updates=0,
            global_goal_complete=False,source_sha256=digest(Path(__file__)),epoch=time.time())
        immutable(receipt,result)
        immutable(base / 'preparation/pool-handoff.json',marker_for(receipt))
        print(json.dumps(dict(summary(result),marker=marker_for(receipt),
            closed_batch_count=observation['closed_batch_count'],closed_questions=observation['closed_questions'])))


def publication(commit, reference, source):
    require(re.fullmatch(r'[0-9a-f]{40}',commit or '') and re.fullmatch(r'refs/heads/[A-Za-z0-9._/-]+',reference or ''),
            'Require published full SHA and refs/heads/BRANCH before remote mutation')
    require(subprocess.check_output(['git','show',commit+':'+RELATIVE_SOURCE],cwd=REPO)==source,
            'Published commit does not contain this exact helper')
    require(subprocess.check_output(['git','ls-remote','--exit-code','origin',reference],cwd=REPO,text=True).strip()
            ==commit+'\t'+reference,'Origin publication SHA differs')


def dispatch(args):
    source=Path(__file__).read_bytes();source_sha=hashlib.sha256(source).hexdigest()
    staged=PHYSICAL+'/'+DIRECTORY+'/code/'+source_sha+'/source_terminal_handoff.py'
    if args.action=='plan':
        print(json.dumps(dict(status='LOCAL_ONLY_PLAN',source_sha256=source_sha,physical_source=staged,
            receipt=BASE+'/'+DIRECTORY+'/complete.json',minimum_observation_separation_seconds=60,
            unavailable_native_checks=UNAVAILABLE),indent=2));return
    if args.action!='status':publication(args.published_commit,args.published_ref,source)
    if args.action=='stage':
        require(re.fullmatch(r'[A-Za-z0-9._-]{1,120}',args.intent_id or ''),'stage requires explicit --intent-id')
        script='DATA='+repr(base64.b64encode(source).decode())+'\nSHA='+repr(source_sha)+'\nPATH='+repr(staged)+'\nINTENT='+repr(args.intent_id)+'\n'+r'''
import base64,hashlib,json,os,socket,time
from pathlib import Path
assert os.geteuid()==0 and socket.gethostname().split('.')[0]=='h100-10'
p=Path(PATH);p.parent.mkdir(parents=True,exist_ok=True);data=base64.b64decode(DATA)
assert hashlib.sha256(data).hexdigest()==SHA
if not p.exists():
 with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
assert p.read_bytes()==data;os.chmod(p,0o555)
receipt=p.parent/'stage-receipt.json'
if not receipt.exists():
 with receipt.open('x')as f:json.dump(dict(status='EXACT_CPU_RECOVERY_HELPER_STAGED',source_sha256=SHA,intent_id=INTENT,epoch=time.time()),f)
print(receipt.read_text())
'''
    else:
        script='import hashlib,subprocess\nfrom pathlib import Path\np=Path('+repr(staged)+')\nassert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(source_sha)+'\nr=subprocess.run(["/usr/bin/python3",str(p),"--remote-action",'+repr(args.action)+',"--base",'+repr(PHYSICAL)+'],text=True,capture_output=True,timeout=95)\nprint(r.stdout,end="");print(r.stderr,end="",file=__import__("sys").stderr);raise SystemExit(r.returncode)\n'
    modules={}
    for name,path in [('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py'),
                      ('route','/Users/shaoshitong/Documents/Search-G1/h100-prefix5-reference-replica-20261018/reproduction/operations/h100_reference_replica/readonly_management_route.py')]:
        spec=importlib.util.spec_from_file_location('recovery_'+name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules[name]=module
    os.environ['PREFIX5_AB_MGMT_EXISTING_PROXY']='1'
    record=ROOT/'work'/('qwen7b-r6-recovery-%s-%d.json'%(args.action,time.time_ns()))
    result=subprocess.run(modules['route'].management_argv(modules['ctl'].route('h100-10','/usr/bin/python3 -')),
                          input=script,text=True,capture_output=True,timeout=110)
    value=dict(action=args.action,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
               source_sha256=source_sha,local_receipt=str(record),epoch=time.time())
    record.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
    if result.returncode:raise SystemExit(result.returncode)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',choices=('plan','stage','observe','seal','status'),default='plan')
    parser.add_argument('--remote-action',choices=('observe','seal','status'),help=argparse.SUPPRESS)
    parser.add_argument('--base',default=PHYSICAL,help=argparse.SUPPRESS)
    parser.add_argument('--intent-id')
    parser.add_argument('--published-commit')
    parser.add_argument('--published-ref')
    args=parser.parse_args()
    if args.remote_action:remote(args.remote_action,Path(args.base))
    else:dispatch(args)


if __name__=='__main__':main()
