"""REMOTE CPU-only takeover preparation after a proven terminal evaluator.

Never submit, stop, restart, load a model, or reuse a prior execution namespace.
A failed/unknown preparation is inspected, never blindly replayed.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
import uuid

from reproduction.compatible_async_evaluation.contract import digest, verify_loaded_workers, verify_source_tree
from reproduction.compatible_async_evaluation.journal import load_prefix
from reproduction.compatible_async_evaluation.prepare import prepare, once

TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
# Execution controls may change; every scientific inference/scoring byte remains pinned.
SCIENCE_PREFIXES=('search_r1/','search_g1/','verl/')
SCIENCE_EVAL=('async_controller.py','worker_runtime.py','runtime.py','runner.py','journal.py','contract.py','protocol.json')


def read(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('RESUME_REGULAR_ARTIFACT_REQUIRED:'+str(path))
    return json.loads(path.read_text())


def command(args):
    return subprocess.check_output(args,text=True,timeout=30)


def terminal_snapshot(prior,manifest,retained=None):
    """Query the actual same native job and local original process/cgroup births."""
    from reproduction.a800.owned_processes import same,identity
    proof=read(prior/'logs/compatible-native-allocation.json')
    if socket.gethostname().split('.')[0]!=proof.get('node'):raise ValueError('RESUME_LOCAL_RELEASE_CHECK_MUST_RUN_ON_ORIGINAL_NODE')
    if any(proof.get(k)!=manifest[k] for k in ('run_id','execution_id','checkpoint_step','split','recipe_sha256')) or proof.get('accepted') is not True or proof.get('node')!='h100-8':
        raise ValueError('RESUME_NATIVE_EXECUTION_BINDING_MISMATCH')
    job=str(proof['slurm_job_id'])
    if not job.isdecimal():raise ValueError('RESUME_NATIVE_JOB_ID_INVALID')
    queue=command(['squeue','-h','-o','%i|%T'])
    if any(line.split('|',1)[0].strip()==job for line in queue.splitlines()):raise ValueError('RESUME_PRIOR_JOB_STILL_QUEUED_OR_RUNNING')
    state=exitcode=None
    accounting_errors=[]
    native=raw=None
    used_retained=False
    try:
        native=command(['scontrol','show','job',job,'-o'])
        fields=dict(piece.split('=',1) for piece in native.split() if '=' in piece)
        if fields:
            if fields.get('JobId')!=job:raise ValueError('RESUME_ACCOUNTING_JOB_ID_CHANGED')
            state=fields.get('JobState');exitcode=fields.get('ExitCode')
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
        accounting_errors.append(type(error).__name__)
    if state is None:
        try:
            raw=command(['sacct','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'])
            records=[r.split('|') for r in raw.splitlines() if r.split('|')[0]==job]
            if len(records)==1 and len(records[0])>=3:
                state=records[0][1].split()[0].rstrip('+');exitcode=records[0][2]
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
            accounting_errors.append(type(error).__name__)
    if state is None:
        if retained is None:raise ValueError('RESUME_TERMINAL_ACCOUNTING_REQUIRED_NOT_SSH_FAILURE:'+','.join(accounting_errors))
        used_retained=True
        state=retained['state'];exitcode=retained['exit_code']
        if retained['slurm_job_id']!=job or retained['native_proof_sha256']!=digest(prior/'logs/compatible-native-allocation.json') or retained['process_ledger_sha256']!=digest(prior/'logs/process-ledger.json') or any(retained.get(k) is not True for k in ('original_births_released','bound_native_cgroups_unpopulated','owned_gpu_contexts_released')):
            raise ValueError('RESUME_RETAINED_TERMINAL_PROOF_DIFFERS')
    if state not in TERMINAL or not isinstance(exitcode,str) or not re.fullmatch(r'\d+:\d+',exitcode):raise ValueError('RESUME_TRUE_TERMINAL_STATE_REQUIRED')
    ledger=read(prior/'logs/process-ledger.json')
    if ledger['run_id']!=manifest['run_id'] or not ledger['jobs']:raise ValueError('RESUME_OWNERSHIP_LEDGER_REQUIRED')
    births=[r for v in ledger['jobs'].values() for r in v['ledger'].values()]
    leaders=[v['leader'] for v in ledger['jobs'].values()]
    if any(same(r) for r in births):raise ValueError('RESUME_ORIGINAL_PROCESS_BIRTH_STILL_LIVE')
    sessions={(r['sid'],r['uid']) for r in leaders}
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():continue
        try:r=identity(int(path.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if r['state']!='Z' and (r['sid'],r['uid']) in sessions:raise ValueError('RESUME_OWNED_SESSION_STILL_LIVE')
    cgroup=proof['self_cgroup'];paths=[]
    for line in cgroup.splitlines():
        fields=line.split(':',2)
        if len(fields)!=3:raise ValueError('RESUME_CGROUP_BINDING_INVALID')
        parts=Path(fields[2]).parts
        matches=[i for i,p in enumerate(parts) if p in ('job_'+job,'job'+job)]
        if not matches:continue
        mount=Path('/sys/fs/cgroup')
        if fields[1]:mount=mount/fields[1]
        paths.append(mount.joinpath(*parts[1:matches[-1]+1]))
    if not paths:raise ValueError('RESUME_BOUND_NATIVE_JOB_CGROUP_REQUIRED')
    for root in paths:
        if root.exists() and any(p.read_text().strip() for p in root.rglob('cgroup.procs')):
            raise ValueError('RESUME_NATIVE_JOB_CGROUP_STILL_POPULATED')
    gpu=command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
    for pid in {str(r['pid']) for r in births} & {x.strip() for x in gpu.splitlines()}:
        try:current=identity(int(pid))
        except (FileNotFoundError,ProcessLookupError):raise ValueError('RESUME_ORIGINAL_GPU_CONTEXT_STILL_PRESENT')
        originals=[r for r in births if str(r['pid'])==pid]
        if any(all(current[k]==r[k] for k in ('start_ticks','sid','uid')) for r in originals):raise ValueError('RESUME_ORIGINAL_GPU_CONTEXT_STILL_PRESENT')
        # A new foreign process that reused a PID is not our original GPU birth.
    return dict(slurm_job_id=job,state=state,exit_code=exitcode,actual_scontrol_stdout=native,actual_sacct_stdout=raw,accounting_query_errors=accounting_errors,retained_accounting_used=used_retained,native_proof_sha256=digest(prior/'logs/compatible-native-allocation.json'),
        process_ledger_sha256=digest(prior/'logs/process-ledger.json'),original_births_released=True,
        bound_native_cgroups_unpopulated=True,owned_gpu_contexts_released=True,
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())


def retained_terminal(path,sha,prior,manifest):
    if not sha or not re.fullmatch('[0-9a-f]{64}',sha) or digest(path)!=sha:
        raise ValueError('RESUME_INDEPENDENTLY_PINNED_TERMINAL_SHA_REQUIRED')
    value=read(path)
    if value.get('role')!='compatible_evaluation_actual_terminal_release' or value.get('prior_base')!=str(prior) or any(value.get(k)!=manifest[k] for k in ('run_id','execution_id')):
        raise ValueError('RESUME_RETAINED_TERMINAL_EXECUTION_MISMATCH')
    rows=value.get('observations',[])
    if len(rows)!=2 or any(rows[0].get(k)!=rows[1].get(k) for k in ('slurm_job_id','state','exit_code','native_proof_sha256','process_ledger_sha256')):
        raise ValueError('RESUME_TWO_ACTUAL_TERMINAL_OBSERVATIONS_REQUIRED')
    return rows[-1]


def validate_science(old,new):
    names={n for n in old if n.startswith(SCIENCE_PREFIXES)}
    names.update('reproduction/compatible_async_evaluation/'+n for n in SCIENCE_EVAL)
    # Shared dataset identity/scoring/controller dependencies cannot drift either.
    names.update(n for n in old if n.startswith('reproduction/common8/'))
    if not names or any(old.get(n)!=new.get(n) for n in names):raise ValueError('RESUME_INFERENCE_SCORING_OR_DATA_CODE_CHANGED')


def validated_prefix(prior,binding,ids,sources,job):
    journal=prior/'journal'
    if journal.resolve()!=journal or journal.is_symlink():raise ValueError('RESUME_PRIVATE_PRIOR_JOURNAL_REQUIRED')
    if not (journal/'binding.json').is_file() or (journal/'binding.json').is_symlink():raise ValueError('RESUME_DURABLE_JOURNAL_REQUIRED')
    if read(journal/'binding.json')!=binding['journal_binding']:raise ValueError('RESUME_ORIGINAL_JOURNAL_BINDING_CHANGED')
    rows=load_prefix(journal,binding['journal_binding'],ids,sources)
    if not 0<len(rows)<len(ids) or len(rows)%64:raise ValueError('RESUME_PARTIAL_COMPLETE_BATCH_PREFIX_REQUIRED')
    proof_path=journal/('loaded-workers-'+str(job)+'.json');workers=read(proof_path)
    verify_loaded_workers(workers)
    if any(p.get('checkpoint')!=binding['checkpoint'] or p.get('checkpoint_manifest_sha256')!=binding['checkpoint_manifest_sha256'] or p.get('generation_config_sha256')!=binding['generation_config_sha256'] or p.get('actual_vllm_version')!='0.6.3' or p.get('actual_vllm_config_dtype')!='torch.float32' or p.get('actual_engine_max_model_len')!=4596 for p in workers):
        raise ValueError('RESUME_PRIOR_ACTUAL_WORKER_BINDING_MISMATCH')
    names=['binding.json']+[p.name for p in sorted(journal.glob('batch-????????-????????.json'))]
    hashes={name:digest(journal/name) for name in names}
    return rows,hashes,digest(proof_path)


def verify_resume_receipt(base,manifest):
    intent_path=base/'preparation/resume-intent.json';receipt_path=base/'preparation/resume-receipt.json'
    intent=read(intent_path);receipt=read(receipt_path)
    if digest(intent_path)!=manifest['resume_intent_sha256'] or receipt.get('intent_sha256')!=digest(intent_path) or receipt.get('complete') is not True or receipt.get('execution_id')!=manifest['execution_id'] or receipt.get('run_id')!=manifest['run_id']:
        raise ValueError('RESUME_PREPARATION_INCOMPLETE_INSPECT_DO_NOT_REPLAY')
    binding=read(manifest['binding'])
    if binding['journal_binding']!=intent['journal_binding'] or receipt['copied_files_sha256']!=intent['journal_files_sha256']:
        raise ValueError('RESUME_JOURNAL_BINDING_CHANGED')
    for name,sha in receipt['copied_files_sha256'].items():
        if Path(name).name!=name or digest(base/'journal'/name)!=sha:raise ValueError('RESUME_COPIED_PREFIX_CHANGED')
    return receipt


def validate_new_execution(prior_execution,new_execution):
    if not re.fullmatch('[0-9a-f]{32}',new_execution) or new_execution[:12]==prior_execution[:12]:
        raise ValueError('RESUME_DISTINCT_EXECUTION_AND_RAY_PREFIX_REQUIRED')
    if Path('/tmp/g1ca-'+new_execution[:12]).exists():
        raise ValueError('RESUME_FRESH_RAY_NAMESPACE_REQUIRED')


def resume(args):
    if not sys.platform.startswith('linux'):raise ValueError('RESUME_PREPARATION_REMOTE_LINUX_ONLY')
    prior=Path(args.prior_base).resolve();base=Path(args.base).resolve()
    if prior==base or prior in base.parents or base in prior.parents:raise ValueError('RESUME_DISTINCT_NON_NESTED_NAMESPACE_REQUIRED')
    if base.exists() and any(base.iterdir()):raise ValueError('RESUME_OCCUPIED_OUTCOME_INSPECT_DO_NOT_REPLAY')
    manifest=read(prior/'preparation/compatible-evaluation-manifest.json');binding=read(manifest['binding'])
    validate_new_execution(manifest['execution_id'],args.execution_id)
    if Path(manifest['base']).resolve()!=prior or Path(manifest['journal']).resolve()!=prior/'journal' or binding['run_id']!=manifest['run_id'] or binding['recipe_sha256']!=manifest['recipe_sha256']:
        raise ValueError('RESUME_PRIOR_NAMESPACE_BINDING_INVALID')
    verify_source_tree(binding['source_root'],binding['source_sha256'])
    publication=read(args.control_publication_proof);validate_science(binding['source_sha256'],publication['files'])
    from reproduction.common8.validation_identity import expected
    record,frame,ids=expected(Path(manifest['source_base']),binding['split']);sources=frame['data_source'].tolist()
    if record['sha256']!=binding['validation_sha256']:raise ValueError('RESUME_SOURCE_DATA_CHANGED')
    retained=None
    if args.terminal_proof:
        retained=retained_terminal(args.terminal_proof,args.terminal_proof_sha256,prior,manifest)
    elif args.terminal_proof_sha256:raise ValueError('RESUME_TERMINAL_PROOF_PATH_REQUIRED')
    first=terminal_snapshot(prior,manifest,retained)
    rows,hashes,loaded_sha=validated_prefix(prior,binding,ids,sources,first['slurm_job_id'])
    second=terminal_snapshot(prior,manifest,retained)
    if any(first[k]!=second[k] for k in ('slurm_job_id','state','exit_code','native_proof_sha256','process_ledger_sha256')):raise ValueError('RESUME_TERMINAL_EVIDENCE_CHANGED')
    plan=dict(role='compatible_evaluation_terminal_resume',run_id=binding['run_id'],execution_id=args.execution_id,
        prior_base=str(prior),prior_execution_id=manifest['execution_id'],prior_control_commit=binding['control_source_commit'],new_control_commit=args.control_commit,prior_manifest_sha256=digest(prior/'preparation/compatible-evaluation-manifest.json'),
        prior_binding_sha256=digest(manifest['binding']),terminal_observations=[first,second],completed_occurrences=len(rows),
        prior_loaded_workers_sha256=loaded_sha,journal_binding=binding['journal_binding'],journal_files_sha256=hashes,
        scientific_recipe_unchanged=True,actual_gpu_resume_performed=False)
    new=SimpleNamespace(base=str(base),control_publication_proof=args.control_publication_proof,control_commit=args.control_commit,
        checkpoint=binding['checkpoint'],checkpoint_receipt=binding['checkpoint_manifest'],step=binding['checkpoint_step'],split=binding['split'],run_id=binding['run_id'],
        source_base=manifest['source_base'],official_model=binding['official_model'],reference_config=manifest['config'],sharing_bindings=args.sharing_bindings,
        environment_root=str((prior/'env').resolve()),gpu_uuid=manifest['physical_gpu_uuids'],execution_id=args.execution_id,
        gpu_lock_directory=manifest['gpu_lock_directory'],retrieval_url=args.retrieval_url or manifest['retrieval_url'],
        retrieval_index=manifest['retrieval_index_path'],retrieval_corpus=manifest['retrieval_corpus_path'],retrieval_encoder=manifest['retrieval_encoder'],retriever_gpu_lib=manifest['retriever_gpu_lib'])
    result=prepare(new,resume_plan=plan)
    current=read(base/'preparation/binding.json')
    # Control publication and localhost endpoint may differ; checkpoint science must not.
    for key in ('run_id','checkpoint_step','split','checkpoint','checkpoint_manifest_sha256','checkpoint_files_sha256','generation_config_sha256','official_model','resolved_config_sha256','validation_sha256','journal_binding','recipe_id','recipe_sha256'):
        if current[key]!=binding[key]:raise ValueError('RESUME_NEW_BINDING_CHANGED:'+key)
    for name,sha in hashes.items():
        source=prior/'journal'/name;target=base/'journal'/name
        if source.is_symlink() or digest(source)!=sha:raise ValueError('RESUME_PRIOR_PREFIX_CHANGED')
        with target.open('xb') as out:
            out.write(source.read_bytes());out.flush();os.fsync(out.fileno())
        if digest(target)!=sha:raise ValueError('RESUME_TARGET_PREFIX_SHA_MISMATCH')
    restored=load_prefix(base/'journal',current['journal_binding'],ids,sources)
    if restored!=rows:raise ValueError('RESUME_PREFIX_READBACK_DIFFERS')
    once(base/'preparation/resume-receipt.json',dict(complete=True,intent_sha256=digest(base/'preparation/resume-intent.json'),run_id=binding['run_id'],execution_id=args.execution_id,
        completed_occurrences=len(rows),copied_files_sha256=hashes,original_artifacts_retained=True,submitted=False,actual_gpu_resume_performed=False))
    verify_resume_receipt(base,read(base/'preparation/compatible-evaluation-manifest.json'))
    return dict(result,completed_occurrences=len(rows),original_artifacts_retained=True,actual_gpu_resume_performed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True);p.add_argument('--inspect',action='store_true')
    for name in ('prior-base','control-publication-proof','control-commit','sharing-bindings'):p.add_argument('--'+name)
    p.add_argument('--execution-id',default=uuid.uuid4().hex);p.add_argument('--retrieval-url')
    p.add_argument('--terminal-proof');p.add_argument('--terminal-proof-sha256')
    p.add_argument('--capture-terminal',help='Write two real terminal snapshots for later accounting purge; no preparation')
    args=p.parse_args()
    if args.capture_terminal:
        if not sys.platform.startswith('linux') or not args.prior_base:p.error('capture requires remote Linux and prior-base')
        prior=Path(args.prior_base).resolve();manifest=read(prior/'preparation/compatible-evaluation-manifest.json')
        observations=[terminal_snapshot(prior,manifest),terminal_snapshot(prior,manifest)]
        value=dict(role='compatible_evaluation_actual_terminal_release',prior_base=str(prior),run_id=manifest['run_id'],execution_id=manifest['execution_id'],observations=observations)
        once(Path(args.capture_terminal),value)
        print(json.dumps(dict(path=args.capture_terminal,sha256=digest(args.capture_terminal),actual_gpu_resume_performed=False)));return
    if args.inspect:
        base=Path(args.base).resolve()
        print(json.dumps({name:read(base/'preparation'/name) for name in ('resume-intent.json','resume-receipt.json','prepare-receipt.json') if (base/'preparation'/name).exists()}));return
    if any(getattr(args,n) is None for n in ('prior_base','control_publication_proof','control_commit','sharing_bindings')):p.error('resume requires prior-base, control-publication-proof, control-commit and sharing-bindings')
    print(json.dumps(resume(args)))

if __name__=='__main__':main()
