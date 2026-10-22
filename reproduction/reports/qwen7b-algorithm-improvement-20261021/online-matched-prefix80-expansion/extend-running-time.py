"""Extend only the two owned pilots to cover measured P80 throughput."""
import pathlib,json,subprocess,time
S=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1');out={'time':time.time(),'jobs':{}}
for layer in (15,18):
 b=S/('layer%d-prefix80'%layer);r=json.loads((b/'root-submission.json').read_bytes());job=r['job_id'];before=subprocess.run(['scontrol','show','job',job,'-o'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);assert before.returncode==0;assert'JobName='+r['name']in before.stdout and'UserId=mou176('in before.stdout
 p=subprocess.run(['scontrol','update','JobId='+job,'TimeLimit=01:30:00'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 after=subprocess.run(['scontrol','show','job',job,'-o'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 out['jobs'][str(layer)]={'job_id':job,'before':before.stdout.strip(),'update_returncode':p.returncode,'update_stdout':p.stdout.strip(),'update_stderr':p.stderr.strip(),'after':after.stdout.strip()}
(S/'prefix80-time-extension.json').write_text(json.dumps(out)+'\n');print(json.dumps(out))
