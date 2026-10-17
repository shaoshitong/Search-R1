"""CUDA-free supervision of fresh TP1 initialization attempts; never retries QA."""
import os
import fcntl
from pathlib import Path
import re
import subprocess
import sys
import time
from .actor import read, sha, atomic_json, utc


def initialization_error(error, actual_cuda_oom):
    message=str(error)
    changed=(isinstance(error,AssertionError) and re.fullmatch(
        r'Error in memory profiling\. Initial free memory [0-9]+, current free memory [0-9]+\. '
        r'This happens when the GPU memory was not properly cleaned up before initializing the vLLM instance\.',message) is not None)
    return dict(retryable_initialization=bool(actual_cuda_oom or changed),
        actual_cuda_oom=bool(actual_cuda_oom),invalid_memory_profile=changed,
        error_type=type(error).__name__,message=message[:4000])


def initialization_status(config_path, model_dir, questions_path, expected_count, rank, *, failure=None):
    c=read(config_path);control=c.get('_qa_initialization')
    if control is None:return
    value=dict(kind='INITIALIZATION_FAILED' if failure else 'INITIALIZATION_READY',
        run_id=c['run_id'],actor_version=c['actor_version'],rank=rank,pid=os.getpid(),
        attempt=control['attempt'],config_sha256=sha(config_path),
        actor_model_dir=str(model_dir),question_source_sha256=sha(questions_path),
        expected_count=expected_count,created_utc=utc())
    if failure:value.update(failure)
    atomic_json(control['status_path'],value)


def next_settings(current, failure):
    # The same model/seed/questions/decoding budget is retained in a fresh process.
    result=dict(current)
    if failure['actual_cuda_oom']:
        result['evaluation_initialization_utilization_scale']=float(current.get('evaluation_initialization_utilization_scale',1))*.85
        result['evaluation_max_num_seqs']=max(32,int(current.get('evaluation_max_num_seqs',1024))//2)
        result['evaluation_max_batched_tokens']=max(4096,int(current.get('evaluation_max_batched_tokens',32768))//2)
    return result


def supervise(config_path,model_dir,output,questions_path,expected_count,*,popen=None):
    """Serialize owners; preserve total attempts and refuse an ambiguous prior owner."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'.initialization-owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _supervise(config_path,model_dir,output,questions_path,expected_count,popen=popen)


def _supervise(config_path,model_dir,output,questions_path,expected_count,*,popen=None):
    """Each failed constructor dies and is reaped before its replacement starts."""
    popen=popen or subprocess.Popen
    config_path,model_dir,output,questions_path=map(Path,(config_path,model_dir,output,questions_path))
    config=read(config_path);question_sha=sha(questions_path);active={};codes={};reused=[]
    directory=output/'initialization-attempts';terminal=directory/'supervisor-terminal.json'
    original_sha=sha(config_path)
    binding=dict(original_config_sha256=original_sha,run_id=config['run_id'],actor_version=config['actor_version'],
        actor_model_dir=str(model_dir),question_source_sha256=question_sha,expected_count=expected_count)
    if all((output/('rank-%d-complete.json'%rank)).exists() for rank in range(8)):
        return dict(actual_worker_exit_codes={},reused_rank_receipts=list(range(8)),
            initialization_attempt_history=read(directory/'history.json') if (directory/'history.json').exists() else [],
            initialization_supervisor_cuda_initialized=False)
    if directory.exists():
        previous=read(terminal) if terminal.exists() else {}
        if previous.get('binding')!=binding or not previous.get('all_owned_workers_reaped'):
            raise RuntimeError('Previous QA owner has no matching complete reap proof; reconcile it before resuming')
    else:directory.mkdir()
    history=read(directory/'history.json') if (directory/'history.json').exists() else []
    def bound_failure(status,rank,attempt,pid,config_sha):
        return (status and status.get('kind')=='INITIALIZATION_FAILED' and status.get('pid')==pid
            and status.get('rank')==rank and status.get('attempt')==attempt
            and status.get('config_sha256')==config_sha
            and status.get('run_id')==config['run_id'] and status.get('actor_version')==config['actor_version']
            and status.get('actor_model_dir')==str(model_dir) and status.get('question_source_sha256')==question_sha
            and status.get('expected_count')==expected_count and status.get('retryable_initialization') is True)
    def resume_settings(rank):
        files=sorted(directory.glob('rank-%d-attempt-*-config.json'%rank))
        rows=[row for row in history if row['rank']==rank]
        if not files:
            if rows:raise ValueError('QA history has no corresponding attempt config')
            return dict(config),1
        # A launch intent without a reaped result is ambiguous, never a retry signal.
        if len(rows)!=len(files) or [r['attempt'] for r in rows]!=list(range(1,len(files)+1)):
            raise RuntimeError('QA attempt journal is incomplete; preserve evidence and reconcile before retry')
        row=rows[-1];last=directory/('rank-%d-attempt-%02d-config.json'%(rank,row['attempt']))
        if (not row.get('reaped') or row['exit_code']==0 or row['attempt']>=4
                or not bound_failure(row.get('status'),rank,row['attempt'],row['pid'],sha(last))):
            raise RuntimeError('Prior QA rank did not end at a retryable uncommitted constructor')
        settings=read(last);settings.pop('_qa_initialization',None)
        return next_settings(settings,row['status']),row['attempt']+1
    def start(rank,settings,attempt):
        if (output/('rank-%d-full.jsonl.gz'%rank)).exists():
            raise FileExistsError('Preserve partial QA shard; initialization retry cannot resample generated answers')
        stem=directory/('rank-%d-attempt-%02d'%(rank,attempt))
        attempt_config=stem.with_name(stem.name+'-config.json');status=stem.with_name(stem.name+'-status.json')
        c=dict(settings,_qa_initialization=dict(attempt=attempt,status_path=str(status),original_config_sha256=original_sha))
        atomic_json(attempt_config,c)
        stream=stem.with_suffix('.log').open('xb')
        try:
            process=popen([sys.executable,'-m','reproduction.prefix_diversity200.eval',
                '--config',str(attempt_config),'--model',str(model_dir),'--output',str(output),
                '--questions',str(questions_path),'--expected-count',str(expected_count),'--rank',str(rank)],
                stdout=stream,stderr=subprocess.STDOUT)
        except BaseException:stream.close();raise
        active[rank]=dict(process=process,stream=stream,attempt=attempt,config=attempt_config,
            status=status,settings=settings,started=time.time())
    def record(rank,item,code,*,cleanup=False):
        status=read(item['status']) if item['status'].exists() else None
        row=dict(rank=rank,pid=item['process'].pid,attempt=item['attempt'],exit_code=code,reaped=True,
            config_sha256=sha(item['config']),status_path=str(item['status']),
            started_unix=item['started'],ended_unix=time.time(),status=status,cleanup=cleanup)
        history.append(row);atomic_json(directory/'history.json',history,replace=True)
        return row
    # Invalidate the old terminal proof before a new child can exist.
    atomic_json(terminal,dict(binding=binding,all_owned_workers_reaped=False,started_utc=utc()),replace=True)
    cleanup_errors=[];unreaped=[]
    try:
        planned=[]
        for rank in range(8):
            if (output/('rank-%d-complete.json'%rank)).exists():reused.append(rank)
            else:
                if (output/('rank-%d-full.jsonl.gz'%rank)).exists():
                    raise FileExistsError('Preserve partial QA shard; constructor retries cannot resample answers')
                settings,attempt=resume_settings(rank);planned.append((rank,settings,attempt))
        for rank,settings,attempt in planned:start(rank,settings,attempt)
        while active:
            for rank in list(active):
                item=active[rank];p=item['process'];code=p.poll()
                if code is None:continue
                p.wait()  # Process/context is gone before any replacement is launched.
                item['stream'].close();del active[rank]
                row=record(rank,item,code);status=row['status']
                if code==0:
                    if not (output/('rank-%d-complete.json'%rank)).exists():
                        raise RuntimeError('Exited worker did not publish its complete QA receipt')
                    codes[str(rank)]=0;continue
                if (not bound_failure(status,rank,item['attempt'],p.pid,row['config_sha256'])
                        or item['attempt']>=4 or (output/('rank-%d-full.jsonl.gz'%rank)).exists()):
                    raise RuntimeError('Owned QA worker failed outside a retryable uncommitted constructor: '+str(row))
                start(rank,next_settings(item['settings'],status),item['attempt']+1)
            if active:time.sleep(.2)
    finally:
        primary=sys.exc_info()[1]
        # One vanished/racing worker must not prevent cleanup of the other seven.
        for rank,item in active.items():
            try:
                if item['process'].poll() is None:item['process'].terminate()
            except Exception as error:cleanup_errors.append(dict(rank=rank,operation='terminate',error=repr(error)))
        deadline=time.monotonic()+60
        for rank,item in active.items():
            p=item['process'];code=None;reaped=False
            try:
                try:code=p.wait(timeout=max(.01,deadline-time.monotonic()));reaped=True
                except subprocess.TimeoutExpired:
                    try:p.kill()
                    except ProcessLookupError:pass
                    code=p.wait(timeout=10);reaped=True
            except Exception as error:
                cleanup_errors.append(dict(rank=rank,operation='reap',error=repr(error)))
                # Still try a final bounded kill/wait before moving to the next handle.
                try:
                    try:p.kill()
                    except ProcessLookupError:pass
                    code=p.wait(timeout=10);reaped=True
                except Exception as last:cleanup_errors.append(dict(rank=rank,operation='final-reap',error=repr(last)))
            finally:
                try:item['stream'].close()
                except Exception as error:cleanup_errors.append(dict(rank=rank,operation='close',error=repr(error)))
            if reaped:
                try:record(rank,item,code,cleanup=True)
                except Exception as error:cleanup_errors.append(dict(rank=rank,operation='journal',error=repr(error)));unreaped.append(rank)
            else:unreaped.append(rank)
        atomic_json(terminal,dict(binding=binding,all_owned_workers_reaped=not unreaped,
            unreaped_ranks=unreaped,cleanup_errors=cleanup_errors,primary_error=repr(primary) if primary else None,
            completed_utc=utc()),replace=True)
        if unreaped and primary is None:raise RuntimeError('Owned QA cleanup incomplete: '+str(unreaped))
    return dict(actual_worker_exit_codes=codes,reused_rank_receipts=reused,
        initialization_attempt_history=history,initialization_supervisor_cuda_initialized=False,
        initialization_cleanup_errors=cleanup_errors)
