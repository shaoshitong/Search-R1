import json,sys,subprocess
from pathlib import Path
p=subprocess.run([sys.executable,'-B','/tmp/easy7b_diag_remote.py',str(Path(__file__).with_name('observe_candidate_retrieval_pair_r3_once.py')),'candidate-retrieval-pair-r3-live'],capture_output=True,text=True)
paths=[line for line in p.stdout.splitlines() if line.startswith('/Users/') and line.endswith('.json')]
if not paths:
 print(p.stdout[-2000:]);print(p.stderr[-1000:]);raise SystemExit(p.returncode or 1)
f=Path(paths[0]);raw=json.loads(f.read_text())
if raw['returncode']:
 print(json.dumps(dict(evidence=str(f),returncode=raw['returncode'],error=raw['stderr'][-2000:])));raise SystemExit(raw['returncode'])
x=json.loads(raw['stdout']);s=x['service'];pair=x['pair']
def docs(section):return {k:v.get('summary') for k,v in section.get('documents',{}).items() if k in ('status.json','logs/service-ready.json','progress.json','failure.json','result.json','completed.json')}
r=dict(evidence=str(f),epoch=x['epoch'],slurm=[call.get('own_rows') for call in x['slurm']['calls']],service=docs(s),service_errors=[row.get('errors') for row in s['logs']],runner=docs(pair.get('runner',{})),variants=[dict(root=v['root'],documents=docs(v),candidate=docs(v.get('candidate',{})),chunks=len(v['chunks']),raw_metadata=v['raw_chunk_metadata'],logs=v['logs']) for v in pair['variants']],pair_errors=[row.get('errors') for row in pair['logs']])
print(json.dumps(r,ensure_ascii=False))
