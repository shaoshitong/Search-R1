"""Stop only this benchmark's unservable exact-log-capacity child group.

vLLM 0.17.1 BlockPool consumes one GPU cache block as null_block. The
advertised aggregate capacity therefore is not all available to a request.
The original driver will retain this failure and run its near-capacity case.
"""
import pathlib, json, os, signal, time

ROOT=pathlib.Path('/data/user/mou176/experiments/vllm-qwen3-8b-context-debug-20261103/high-utilization-r2')
assert os.environ.get('SLURM_JOB_ID')=='795528'
rows=json.loads((ROOT/'results.json').read_text())['cases']
c=rows[-1]
assert c['name'].endswith('-exact-capacity') and 'exit_code' not in c
events=[json.loads(s) for s in (ROOT/(c['name']+'.jsonl')).read_text().splitlines()]
ready=next(s for s in events if s['event']=='engine_ready')
request=next(s for s in events if s['event']=='request_started')
assert not any(s['event']=='request_pass' for s in events)
usable=(ready['cache_num_gpu_blocks']-1)*ready['cache_block_size']
assert request['input_tokens']>usable, 'Not the reserved-null-block boundary'
pid=int(c['pid']);proc=pathlib.Path('/proc')/str(pid)
birth=(proc/'stat').read_text().split()[21]
cmdline=(proc/'cmdline').read_bytes().split(b'\0')
assert str(ROOT/'source/case.py').encode() in cmdline
assert proc.stat().st_uid==os.getuid()
assert os.getpgid(pid)==pid and os.getsid(pid)==pid
time.sleep(.2)
assert (proc/'stat').read_text().split()[21]==birth
note=dict(at=time.time(),case=c['name'],pid=pid,start_ticks=birth,
          job_id=os.environ['SLURM_JOB_ID'],reported_capacity=ready['cache_num_gpu_blocks']*ready['cache_block_size'],
          usable_after_null_block=usable,input_tokens=request['input_tokens'],
          request_age_seconds=time.time()-request['time'],
          reason='Unservable exact advertised capacity: reserved null block. Stop only owned child group; driver automatically tests with 1024-token margin.')
with (ROOT/'null-block-boundary-interventions.jsonl').open('a') as f:
    f.write(json.dumps(note)+'\n');f.flush();os.fsync(f.fileno())
os.killpg(pid,signal.SIGTERM)
print(json.dumps(note),flush=True)
