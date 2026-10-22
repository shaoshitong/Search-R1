from pathlib import Path
import json,time,subprocess,hashlib
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');P=A/'online-packed-window2018-seed19137-plan2026102302-L15P80-8192-20261023-v1';b=P/'window2018';out=dict(time=time.time(),status='SUPERSEDED_BEFORE_ANY_UPDATE',reason='CLI choices rejected2018 before training startup; not a checkpoint/restore failure',checkpoints_opened=False,Slurm_mutated=False)
for n in('root-submission.json','controller.json','progress-rank-0.json','progress-rank-1.json','complete.json'):
 if(b/n).exists():out[n]=json.loads((b/n).read_bytes())
job=str(out['root-submission.json']['job_id']);assert job=='750023';out['accounting']=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed'],universal_newlines=True,timeout=15)
out['checkpoint_paths']=[str(p)for p in b.glob('checkpoint-epoch-*.pt')];assert not out['checkpoint_paths']and not(b/'latest.json').exists();assert not(b/'controller.json').exists()and not(b/'progress-rank-0.json').exists()and not(b/'progress-rank-1.json').exists()
pub=P/'source-publication.json';raw=pub.read_bytes();out['source_publication']=dict(path=str(pub),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),content=json.loads(raw));logs=list(b.glob('*750023*.log'));out['logs']={str(p):p.read_text(errors='replace')[-5000:]for p in logs};assert any("invalid choice: 2018"in text for text in out['logs'].values());print(json.dumps(out))
