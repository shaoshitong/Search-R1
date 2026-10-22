import pathlib,json,subprocess,time
a=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');suite=a/'online-mixed-layer-prefix-20261022-2gpu-v1';x=dict(time=time.time(),variants=[])
for variant in('layer27-prefix80','layer21-prefix80'):
 b=suite/variant;r=dict(variant=variant,complete_exists=(b/'complete.json').exists())
 if(b/'complete.json').exists():r['complete']=json.loads((b/'complete.json').read_bytes())
 if(b/'root-submission.json').exists():
  job=json.loads((b/'root-submission.json').read_bytes())['job_id'];r['job_id']=job;r['sacct']=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],universal_newlines=True,timeout=10).strip()
 x['variants'].append(r)
print(json.dumps(x))
