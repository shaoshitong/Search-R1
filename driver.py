import pathlib, subprocess, os, json, time, re, signal, sys

ROOT=pathlib.Path(os.environ['TASK_ROOT'])
SOURCE=ROOT/'source'
RESULTS=[]

def save():
    q=ROOT/'results.json.tmp'
    q.write_text(json.dumps(dict(updated_at=time.time(), cases=RESULTS),indent=2))
    q.replace(ROOT/'results.json')

def run_case(name, memory, max_len, lengths, limit=700, kv='auto'):
    log=ROOT/(name+'.log');events=ROOT/(name+'.jsonl')
    env=dict(os.environ, CASE_EVENTS=str(events))
    cmd=[sys.executable,str(SOURCE/'case.py'),'--memory',str(memory),
         '--max-len',str(max_len),'--lengths',','.join(map(str,lengths)), '--kv',kv]
    row=dict(name=name,memory_utilization=memory,max_model_len=max_len,
             input_lengths=lengths,kv_cache_dtype=kv,command=cmd,start_time=time.time())
    RESULTS.append(row);save()
    print('CASE_STARTED '+name,flush=True)
    with log.open('w') as f:
        p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        row['pid']=p.pid;save()
        try: row['exit_code']=p.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            row['timeout']=True
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait()
            row['exit_code']=p.returncode
        finally:
            # Only this fresh child process group. Never signal other jobs.
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    row['end_time']=time.time()
    text=log.read_text(errors='replace')
    hits=re.findall(r'GPU KV cache size:\s*([\d,]+)\s*tokens',text)
    row['capacity_tokens']=int(hits[-1].replace(',','')) if hits else None
    row['events']=[json.loads(x) for x in events.read_text().splitlines()] if events.exists() else []
    row['pass']=row['exit_code']==0 and any(x['event']=='case_pass' for x in row['events'])
    row['diagnostic_tail']=text[-5000:] if not row['pass'] else ''
    save();print('CASE_FINISHED '+json.dumps({k:row[k] for k in ['name','exit_code','pass','capacity_tokens']}),flush=True)
    time.sleep(4)
    return row

def main():
    # Seed profile retains real Qwen3-8B dimensions; only the position cap was
    # extended. No weights are read or saved, and requests use token IDs.
    base=run_case('bf16-u090-profile',.90,40960,[32768],limit=300)
    if not base['pass'] or not base['capacity_tokens']:
        raise RuntimeError('Profile did not pass; do not report memory estimates as measured context')
    cap=base['capacity_tokens']
    target=(cap-1024)//16*16
    checks=[x for x in [131072,262144] if x<target-1]+[target-1]
    large=run_case('bf16-u090-long',.90,target,checks,limit=650)
    # A deliberately larger request-length configuration tests the memory gate;
    # this is an expected failure, not a successful long-context measurement.
    run_case('bf16-u090-over-capacity',.90,(cap+8192)//16*16,[],limit=250)
    profile95=run_case('bf16-u095-profile',.95,40960,[],limit=250)
    if profile95['pass'] and profile95['capacity_tokens']:
        target95=(profile95['capacity_tokens']-1024)//16*16
        run_case('bf16-u095-long',.95,target95,[target95-1],limit=650)
    success=[c for c in RESULTS if c['pass']]
    req=[dict(case=c['name'],**e) for c in success for e in c['events'] if e['event']=='request_pass']
    summary=dict(completed_at=time.time(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                 random_weights=True,original_model_max_position_embeddings=40960,
                 debug_max_position_embeddings=524288,weight_dtype='bfloat16',kv_dtype='bfloat16',
                 tensor_parallel_size=1,max_num_seqs=1,enforce_eager=True,chunked_prefill_tokens=2048,
                 cpu_offload_gb=0,largest_validated_total_tokens=max([r['total_tokens'] for r in req],default=0),
                 requests=req,cases=[{k:c[k] for k in ['name','pass','exit_code','capacity_tokens','max_model_len']} for c in RESULTS],
                 caveat='Memory/functional test only. Synthetic token IDs and dummy weights do not establish long-context accuracy.')
    (ROOT/'complete.json').write_text(json.dumps(summary,indent=2))
    print('BENCHMARK_COMPLETE '+json.dumps(summary),flush=True)

if __name__=='__main__':main()
