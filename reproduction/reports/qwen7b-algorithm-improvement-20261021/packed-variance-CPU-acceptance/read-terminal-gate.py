"""Metadata-only terminal gate, never opens a training checkpoint."""
import json,pathlib,subprocess,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-M3to10-P20-variance-8192-20261022-v1')
r=json.loads((B/'root-submission.json').read_bytes());job=str(r['job_id']);assert job=='749515';out=dict(time=time.time(),job_id=job,checkpoint_opened=False)
p=B/'complete.json';out['complete']=json.loads(p.read_bytes())if p.exists()else None
s=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],universal_newlines=True,timeout=15);out['sacct']=s.strip();out['terminal_ready']=bool(out['complete'])and any(row.split('|')[:3]==[job,'COMPLETED','0:0']for row in s.splitlines());print(json.dumps(out))
