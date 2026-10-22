import pathlib,json,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-M3to10-P20-20261022-v1');v=json.loads((B/'prepare-process.json').read_bytes());proc=pathlib.Path('/proc/%d'%v['pid']);out=dict(time=time.time(),pid=v['pid'],identity_live=proc.exists()and b'online_packed_variance_scaling'in(proc/'cmdline').read_bytes(),progress=json.loads((B/'fixture/prepare-progress.json').read_bytes())if(B/'fixture/prepare-progress.json').exists()else None,compact_shard_bytes=sum(p.stat().st_size for p in(B/'fixture/inputs').glob('tokens-rank-*.jsonl.gz')),complete=(B/'fixture/fixture-complete.json').exists())
if proc.exists():out['memory']=[x for x in(proc/'status').read_text().splitlines()if x.startswith(('VmRSS:','VmHWM:'))]
if(B/'fixture/failure-rank-cpu.json').exists():out['failure']=json.loads((B/'fixture/failure-rank-cpu.json').read_bytes())
print(json.dumps(out))
