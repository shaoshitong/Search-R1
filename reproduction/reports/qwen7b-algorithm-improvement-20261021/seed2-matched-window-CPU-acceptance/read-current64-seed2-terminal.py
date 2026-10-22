from pathlib import Path
import json,time,subprocess
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');b=A/'online-packed-current64-seed19137-plan2026102302-L15P80-8192-20261023-v1/current64';o=dict(time=time.time())
for n in('root-submission.json','progress-rank-0.json','progress-rank-1.json','complete.json'):
 if(b/n).exists():o[n]=json.loads((b/n).read_bytes())
job=str(o['root-submission.json']['job_id']);o['accounting']=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],universal_newlines=True,timeout=15);o['Slurm_mutated']=False;o['checkpoints_read']=False;print(json.dumps(o))
