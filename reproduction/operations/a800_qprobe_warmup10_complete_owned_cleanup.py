"""Complete failed native cleanup for fixed old f847/c200 only, exactly once.

Default/--execute-native-cleanup-once repairs only the recovered old owned ledger
and its private corpus cache. --inspect-only seals identity and reports the known
cleanup error without mutation. No launch, runtime edit, global Ray stop, model
read, tensor work, test, or evaluation is performed. Original failure survives.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time

sys.dont_write_bytecode = True
ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
ORIGINAL_OPS = ROOT / 'repo'
ORIGINAL_COMMIT = '331e35825d696e28dbd486759f1c2a21e4cce50a'
FAILED_OPERATOR_COMMIT = '0abab1dadf449e38108793cbf6c523f3025edc53'
PREDECESSOR = 'reproduction/operations/a800_qprobe_warmup10_switch_stop.py'
spec = importlib.util.spec_from_file_location('published_warmup10_switch', ORIGINAL_OPS / PREDECESSOR)
switch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(switch)


NATIVE = r'''
import csv,fcntl,hashlib,importlib.util,io,json,os,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
p=json.load(sys.stdin);base=Path(p['old_base']);repo=base/'repo';library=Path(p['library_repo'])
lock=(base/'gpu-job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
run=json.loads((base/'logs/run-manifest.json').read_text())
assert (run['run_id'],run['execution_id'],run['source_commit'])==(p['old_run'],p['old_execution'],p['old_source'])
sys.path.insert(0,str(repo))
from reproduction.a800.owned_processes import OwnedProcesses,publish
from reproduction.a800_common.cache import cleanup
for relative in ('reproduction/a800/owned_processes.py','reproduction/a800_common/cache.py'):
 assert hashlib.sha256((repo/relative).read_bytes()).hexdigest()==p['old_intent']['source_sha256'][relative]
owned=OwnedProcesses(base/'logs/process-ledger.json',run['run_id']);assert owned.recovery_only
phase='native_owned_stop_all'
owned.stop_all()
assert all(job['state']=='STOPPED' for job in owned.jobs.values())
phase='native_private_corpus_cache_cleanup'
cleanup(base)
# Load the published birth-aware audit helper under a separate module name;
# importing old runtime cleanup must not shadow the C4 verifier's helpers.
phase='native_birth_session_cuda_readback'
spec=importlib.util.spec_from_file_location('published_c4_owned_audit',library/'reproduction/a800/owned_processes.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
records=[run['pipeline_identity']]+[row for job in owned.jobs.values() for row in [job['leader']]+list(job['ledger'].values())]
keys=('pid','start_ticks','sid','uid');leaders=[job['leader'] for job in owned.jobs.values()]
if run['pipeline_identity']['pid']==run['pipeline_identity']['sid']:leaders.append(run['pipeline_identity'])
for record in records:
 try:actual=audit.identity(record['pid'])
 except (FileNotFoundError,ProcessLookupError):continue
 assert actual['state'] in ('Z','X') or any(actual[k]!=record[k] for k in keys), 'An owned birth remains live'
survivors,reused_sessions=audit.owned_session_survivors(leaders)
assert not survivors and not reused_sessions, 'An owned or foreign reused session remains'
deadline=time.monotonic()+30;queries=0
while True:
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10)
 remaining,reused_pids=audit.remaining_cuda_contexts(records,list(csv.reader(io.StringIO(raw))));queries+=1
 if not remaining:break
 assert time.monotonic()<deadline, 'Owned CUDA contexts did not release'
 time.sleep(.5)
assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
phase='native_publish_canonical_cleanup_status'
publish(base/'logs/cleanup-status.json',dict(run_id=run['run_id'],execution_backend='ssh_local',
 execution_id=run['execution_id'],slurm_job_id=None,cleaned=True,checked_unix=time.time(),
 scope='only recorded project PID births and private local corpus cache'))
print('NATIVE_CLEANUP_JSON:'+json.dumps(dict(actual_native_cleanup=True,owned_cuda_contexts_remaining=0,
 nvml_queries=queries,reused_pids_belonging_to_other_births=reused_pids)),flush=True)
'''


REMOTE = "phase='old_and_replacement_identity_preflight'\n" + switch.REMOTE_COMMON + switch.REMOTE_STOP[
    switch.REMOTE_STOP.index("new=p['new_intent']"):
    switch.REMOTE_STOP.index('def counters(')
] + r'''
import importlib.util
phase='stop_receipt_preflight'
snapshot=document(base/p['stop_intent']);signal_receipt=document(base/p['stop_signal'])
for row in (snapshot,signal_receipt):
 assert (row['run_id'],row['execution_id'],row['source_commit'])==(run['run_id'],run['execution_id'],run['source_commit'])
 assert row['pipeline_identity']=={k:producer[k] for k in keys}
assert hashlib.sha256(sealed(base/p['stop_intent'])).hexdigest()==signal_receipt['stop_intent_sha256']==p['stop_intent_sha256']
assert hashlib.sha256(sealed(base/p['stop_signal'])).hexdigest()==p['stop_signal_sha256']
assert signal_receipt['signal_sent'] is True and signal_receipt['signal']=='SIGTERM'
assert snapshot['human_intent_sha256']==p['human_intent_sha256'] and snapshot['human_scope_sha256']==signal_receipt['human_scope_sha256']==p['human_scope_sha256']
assert snapshot['reviewed_prepared_evidence_sha256']==p['evidence_sha256']
assert (snapshot['new_run_id'],snapshot['new_execution_id'],snapshot['new_source_commit'],snapshot['new_base'])==(p['new_run'],p['new_execution'],new['source_commit'],new['base'])
paths=dict(repair_intent='preparation/warmup10-native-cleanup-repair-intent.json',
 repair_before='preparation/warmup10-native-cleanup-repair-before.json',
 repair_complete='preparation/warmup10-native-cleanup-repair-complete.json',
 resume_intent='preparation/warmup10-native-cleanup-repair-resume-prearchive-intent.json',cleanup_status='logs/cleanup-status.json')
flags={key+'_exists':(base/relative).exists() for key,relative in paths.items()}
assert not any((base/relative).is_symlink() for relative in paths.values())
prior=document(base/paths['repair_intent']) if flags['repair_intent_exists'] else None
flags.update(prior_operator_commit=prior.get('operator_commit') if prior else None,
 prior_intent_sha256=hashlib.sha256(sealed(base/paths['repair_intent'])).hexdigest() if prior else None,
 ledger_bytes=(base/'logs/process-ledger.json').stat().st_size)
status=document(base/'logs/pipeline-status.json');assert status['stage']=='OWNED_CLEANUP_FAILED'
phase='published_birth_session_preflight'
library=Path(p['library_repo']);verify_source(library,p['library_source'],p['library_source_sha256'],455,allow_hydra=True)
sys.path.insert(0,str(library))
from reproduction.a800.owned_processes import identity,owned_session_survivors
from reproduction.common8.actor_probe_owned_cleanup import birth_inventory,verify_preserved_births,verify_cleanup
current=birth_inventory(base,run);verify_preserved_births(snapshot['preserved_births'],current)
def gone(record):
 try:actual=identity(record['pid'])
 except (FileNotFoundError,ProcessLookupError):return
 assert any(actual[k]!=record[k] for k in keys), 'Exact stopped leader birth still exists'
gone(producer);gone(ledger['jobs']['trainer']['leader'])
leaders=[job['leader'] for job in ledger['jobs'].values()]+[producer]
survivors,reused=owned_session_survivors(leaders)
recorded={tuple(row[k] for k in keys) for job in current['jobs'].values() for row in job['births']}
assert not reused and all(tuple(row[k] for k in keys) in recorded for row in survivors), 'Foreign or unbound session member; refuse cleanup'
error=ledger['jobs']['trainer'].get('cleanup_error','');assert isinstance(error,str)
known=next((message for message in ('Unbound process after session leader exit','Recorded subprocess changed identity','Owned component cleanup incomplete','Owned identity changed before signal','No such process') if message in error),None)
sanitized=dict(cleanup_error_type=re.match(r'[A-Za-z_][A-Za-z0-9_.]*',error).group() if re.match(r'[A-Za-z_][A-Za-z0-9_.]*',error) else None,
 known_cleanup_error=known,cleanup_error_sha256=hashlib.sha256(error.encode()).hexdigest())
receipt=dict(role='warmup10_oldf847_native_owned_cleanup_repair',run_id=run['run_id'],execution_id=run['execution_id'],source_commit=run['source_commit'],
 operator_commit=p['operator_commit'],operator_sha256=p['operator_sha256'],stop_intent_sha256=p['stop_intent_sha256'],stop_signal_sha256=p['stop_signal_sha256'],
 actual_native_cleanup=False,no_launch=True,no_runtime_changes=True,no_pipeline_signal=True,original_failure_preserved=True,
 repair_flags=flags,ledger_sha256=hashlib.sha256(sealed(base/'logs/process-ledger.json',256*1024**2)).hexdigest(),
 pipeline_status_sha256=hashlib.sha256(sealed(base/'logs/pipeline-status.json')).hexdigest(),**sanitized)
def once(path,value,limit=8*1024**2):
 raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode();assert len(raw)<=limit
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
if p['inspect_only']:
 receipt.update(status='INSPECTED_NO_MUTATION',known_recorded_session_survivors=len(survivors),pipeline_and_trainer_births_gone=True)
else:
 phase='exclusive_repair_claim'
 assert ledger['jobs']['trainer']['state']=='CLEANUP_FAILED' and all(job['state']=='STOPPED' for name,job in ledger['jobs'].items() if name!='trainer')
 assert not flags['cleanup_status_exists'] and not flags['repair_before_exists'] and not flags['repair_complete_exists'] and not flags['resume_intent_exists']
 if p['resume_prearchive_only']:
  assert flags['repair_intent_exists'] and prior['operator_commit']==p['failed_operator_commit']
  assert prior['operator_sha256']==p['failed_operator_sha256']
  for key in ('run_id','execution_id','source_commit','stop_intent_sha256','stop_signal_sha256'):assert prior[key]==receipt[key]
  inspected=p['confirmed_inspection'];assert inspected['repair_flags']==flags
  assert inspected['ledger_sha256']==receipt['ledger_sha256'] and inspected['pipeline_status_sha256']==receipt['pipeline_status_sha256']
  once(base/paths['resume_intent'],dict(receipt,original_intent_sha256=flags['prior_intent_sha256'],
   original_operator_commit=p['failed_operator_commit'],original_local_claim_sha256=p['failed_local_claim_sha256'],
   original_local_result_sha256=p['failed_local_result_sha256'],confirmed_inspection_sha256=p['confirmed_inspection_sha256'],
   scope='original attempt stopped before before-archive; native cleanup was never reached',claimed_unix=time.time(),no_automatic_retry=True))
 else:
  assert not flags['repair_intent_exists']
  once(base/paths['repair_intent'],dict(receipt,claimed_unix=time.time(),no_automatic_retry=True))
 # Preserve complete old ownership/error/status before native recovery mutates its ledger.
 phase='archive_before_native_cleanup'
 before=dict(ledger=ledger,pipeline_status=status,cleanup_error=error,preserved_births=current,
  ledger_sha256=hashlib.sha256(sealed(base/'logs/process-ledger.json',256*1024**2)).hexdigest(),
  pipeline_status_sha256=hashlib.sha256(sealed(base/'logs/pipeline-status.json')).hexdigest(),**sanitized)
 once(base/paths['repair_before'],before,limit=256*1024**2)
 phase='actual_native_cleanup_child'
 child=subprocess.run(['/data/search-g1/env/bin/python','-c',p['native_code']],input=json.dumps(p).encode(),capture_output=True,timeout=65)
 child_diagnostics=[json.loads(line.split(':',1)[1]) for line in child.stdout.decode(errors='replace').splitlines() if line.startswith('NATIVE_CLEANUP_JSON:')]
 assert child.returncode==0, 'Original recovery-only native cleanup refused; inspect this claim without retry'
 phase='published_c4_cleanup_readback'
 actual=verify_cleanup(base,run,preserved_births=snapshot['preserved_births'])
 assert actual['accepted'] is True and actual['owned_cuda_contexts_remaining']==0 and actual['private_cache_removed'] is True
 assert hashlib.sha256(sealed(base/'logs/pipeline-status.json')).hexdigest()==before['pipeline_status_sha256']
 assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
 receipt.update(status='ACTUAL_NATIVE_CLEANUP_REPAIRED_AND_C4_VERIFIED',accepted=True,actual_native_cleanup=True,
  verification={k:actual[k] for k in ('actual_pipeline_and_owned_sessions_exited','owned_cuda_contexts_remaining','private_cache_removed','original_assets_present','reused_pids_belonging_to_other_births','reused_session_ids','recorded_births')},
  files=actual['files'],checked_unix=actual['checked_unix'],native_stdout_sha256=hashlib.sha256(child.stdout).hexdigest(),native_stderr_sha256=hashlib.sha256(child.stderr).hexdigest())
 once(base/'preparation/warmup10-native-cleanup-repair-complete.json',receipt)
print('WARMUP10_CLEANUP_JSON:'+json.dumps(receipt,sort_keys=True,allow_nan=False),flush=True)
'''


def diagnostic_entry(code, prefix):
    """Return exception class, execution phase and locations; never messages/code."""
    return ("import json,sys,traceback\nphase='entry'\ntry:\n" + textwrap.indent(code, ' ') +
        "\nexcept BaseException as failure:\n"
        " frames=[dict(file=frame.filename.rsplit('/',1)[-1],line=frame.lineno,function=frame.name) for frame in traceback.extract_tb(failure.__traceback__)[-8:]]\n"
        " result=dict(status='REMOTE_FAILED_NO_AUTOMATIC_RETRY',failure_type=type(failure).__name__,failure_phase=phase,frames=frames)\n"
        " if 'flags' in globals():result['repair_flags']=flags\n"
        " if 'sanitized' in globals():result.update(sanitized)\n"
        " if 'child_diagnostics' in globals():result['native_diagnostics']=child_diagnostics\n"
        " print(" + repr(prefix + ':') + "+json.dumps(result,sort_keys=True,allow_nan=False),flush=True)\n"
        " sys.exit(1)\n")


NATIVE = diagnostic_entry(NATIVE, 'NATIVE_CLEANUP_JSON')
REMOTE = diagnostic_entry(REMOTE, 'WARMUP10_CLEANUP_JSON')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--inspect-only', action='store_true')
    actions.add_argument('--execute-native-cleanup-once', action='store_true')
    actions.add_argument('--resume-prearchive-only', action='store_true')
    parser.add_argument('--confirmed-inspection', type=Path)
    parser.add_argument('--new-handle', type=Path, default=switch.NEW_HANDLE)
    parser.add_argument('--new-review', type=Path)
    parser.add_argument('--ready-receipt', type=Path, default=switch.READY)
    args = parser.parse_args()
    assert subprocess.check_output(['git','-C',str(ORIGINAL_OPS),'rev-parse','HEAD'],text=True).strip()==ORIGINAL_COMMIT
    assert subprocess.check_output(['git','-C',str(ORIGINAL_OPS),'show',ORIGINAL_COMMIT+':'+PREDECESSOR])==switch.sealed(ORIGINAL_OPS/PREDECESSOR)
    ops = Path(__file__).resolve().parents[2]; relative = str(Path(__file__).resolve().relative_to(ops))
    commit = subprocess.check_output(['git','-C',str(ops),'rev-parse','HEAD'],text=True).strip()
    assert subprocess.check_output(['git','-C',str(ops),'show',commit+':'+relative])==switch.sealed(Path(__file__)), 'Publish this exact isolated repair wrapper first'
    if args.new_review is None:
        args.new_review = Path(switch.document(args.ready_receipt)['files']['new_review']['path'])
    human, scope, recorded = switch.authority_and_human()
    new, prepared, bound, hashes = switch.ready_warmup10(args, ORIGINAL_COMMIT)
    old = switch.document(switch.OLD_HANDLE)
    original = switch.document(ROOT/'preparation/qprobe-warmup10-oldf847-stop-once.result.json')
    assert original['exit_code']==0 and original['status']=='STOP_REQUEST_RETURNED' and len(original['results'])==1
    signal = original['results'][0]
    assert signal['run_id']==switch.OLD_RUN and signal['execution_id']==switch.OLD_EXECUTION and signal['signal_sent'] is True
    mapping = switch.verify_source(switch.LIBRARY,switch.LIBRARY_SOURCE,
        switch.document(ROOT/'preparation/actor-probes-ten-production-after-3ecc-preclaim-exit-deployment.json')['source_sha256'],455)
    payload = dict(old_intent=old,old_source=switch.OLD_SOURCE,old_run=switch.OLD_RUN,old_execution=switch.OLD_EXECUTION,
        old_base=switch.OLD_BASE,old_pid=switch.OLD_PID,old_ticks=switch.OLD_TICKS,
        old_trainer_pid=switch.OLD_TRAINER_PID,old_trainer_ticks=switch.OLD_TRAINER_TICKS,
        new_intent=new,new_run=prepared['run_id'],new_execution=prepared['execution_id'],prepared_observation=prepared,bound_observation=bound,
        sigma3_loss=switch.SIGMA3_LOSS,human_intent_sha256=human,human_scope_sha256=scope,human_recorded_unix=recorded,
        evidence_sha256=hashes,stop_intent=switch.STOP_INTENT,stop_signal=switch.STOP_SIGNAL,
        stop_intent_sha256=signal['stop_intent_sha256'],stop_signal_sha256=switch.canonical_sha(signal),
        library_repo=switch.REMOTE_LIBRARY,library_source=switch.LIBRARY_SOURCE,library_source_sha256=mapping,
        inspect_only=args.inspect_only,resume_prearchive_only=args.resume_prearchive_only,
        native_code=NATIVE,operator_commit=commit,operator_sha256=switch.sha(Path(__file__)))
    # The remote signal file uses the original once() encoding, including its newline.
    payload['stop_signal_sha256'] = hashlib.sha256((json.dumps(signal,sort_keys=True,allow_nan=False)+'\n').encode()).hexdigest()
    if args.resume_prearchive_only:
        assert args.confirmed_inspection is not None, 'Resume requires the new successful read-only inspection receipt'
        args.confirmed_inspection.resolve().relative_to(ROOT/'preparation')
        inspected=switch.document(args.confirmed_inspection)
        assert inspected['exit_code']==0 and inspected['operator_commit']==commit and len(inspected['observations'])==1
        observed=inspected['observations'][0];assert observed['status']=='INSPECTED_NO_MUTATION'
        assert observed['operator_commit']==commit and observed['operator_sha256']==payload['operator_sha256']
        assert (observed['run_id'],observed['execution_id'],observed['source_commit'])==(switch.OLD_RUN,switch.OLD_EXECUTION,switch.OLD_SOURCE)
        flags=observed['repair_flags'];assert flags['repair_intent_exists'] and flags['prior_operator_commit']==FAILED_OPERATOR_COMMIT
        assert all(flags[key] is False for key in ('repair_before_exists','repair_complete_exists','resume_intent_exists','cleanup_status_exists'))
        failed_claim=ROOT/'preparation/qprobe-warmup10-oldf847-native-cleanup-once.json'
        failed_result=failed_claim.with_name(failed_claim.stem+'.result.json')
        request,result=switch.document(failed_claim),switch.document(failed_result)
        assert request['operator_commit']==result['operator_commit']==FAILED_OPERATOR_COMMIT and result['exit_code']==1 and result['observations']==[]
        original_bytes=subprocess.check_output(['git','-C',str(ops),'show',FAILED_OPERATOR_COMMIT+':'+relative])
        assert hashlib.sha256(original_bytes).hexdigest()==request['operator_sha256']
        payload.update(confirmed_inspection=observed,confirmed_inspection_sha256=switch.sha(args.confirmed_inspection),
            failed_operator_commit=FAILED_OPERATOR_COMMIT,failed_operator_sha256=request['operator_sha256'],
            failed_local_claim_sha256=switch.sha(failed_claim),failed_local_result_sha256=switch.sha(failed_result))
    else:
        assert args.confirmed_inspection is None
    stem = ('qprobe-warmup10-oldf847-native-cleanup-inspect-'+str(time.time_ns()) if args.inspect_only else
            'qprobe-warmup10-oldf847-native-cleanup-resume-prearchive-once' if args.resume_prearchive_only else
            'qprobe-warmup10-oldf847-native-cleanup-once')
    claim = ROOT/'preparation'/(stem+'.json'); output = claim.with_name(claim.stem+'.result.json')
    switch.once(claim,dict(role='warmup10_oldf847_native_cleanup_request',run_id=switch.OLD_RUN,execution_id=switch.OLD_EXECUTION,
        operator_commit=commit,operator_sha256=payload['operator_sha256'],remote_code_sha256=hashlib.sha256(REMOTE.encode()).hexdigest(),
        native_code_sha256=hashlib.sha256(NATIVE.encode()).hexdigest(),inspect_only=args.inspect_only,no_automatic_retry=True,claimed_unix=time.time()))
    assert switch.authority_and_human()==(human,scope,recorded) and switch.ready_warmup10(args,ORIGINAL_COMMIT)[3]==hashes
    spec = importlib.util.spec_from_file_location('old_published_cleanup_transport',switch.OLD_RUNTIME/'reproduction/a800_common/deploy.py')
    deploy = importlib.util.module_from_spec(spec); spec.loader.exec_module(deploy)
    route = deploy.ssh_command(REMOTE); route[-1] = shlex.quote('/data/search-g1/env/bin/python')+' -c '+shlex.quote(REMOTE)
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            completed = subprocess.run(route,input=json.dumps(payload,sort_keys=True,allow_nan=False).encode(),stdout=stdout,stderr=stderr,timeout=120)
            result = dict(exit_code=completed.returncode,status='RETURNED')
        except (OSError,subprocess.TimeoutExpired) as failure:
            result = dict(exit_code=None,status='TRANSPORT_UNKNOWN_NO_RETRY',failure_type=type(failure).__name__)
        stdout.seek(0); raw = stdout.read(256*1024+1); stderr.seek(0); error = stderr.read(64*1024)
    rows = [json.loads(line.split(':',1)[1]) for line in raw.decode(errors='replace').splitlines() if line.startswith('WARMUP10_CLEANUP_JSON:')]
    result.update(run_id=switch.OLD_RUN,execution_id=switch.OLD_EXECUTION,operator_commit=commit,
        observations=rows,stdout_bytes=len(raw),bounded_stderr_sha256=hashlib.sha256(error).hexdigest(),no_automatic_retry=True,checked_unix=time.time())
    switch.once(output,result)
    print(json.dumps(dict(receipt=str(output),**result),sort_keys=True,allow_nan=False))
    return 0 if result['exit_code']==0 and len(rows)==1 and (args.inspect_only or rows[0].get('accepted') is True) else 1


if __name__ == '__main__':
    raise SystemExit(main())
