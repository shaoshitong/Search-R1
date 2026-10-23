import json,sys,subprocess
from pathlib import Path
observer=Path(__file__).with_name('observe_search_action_stop_legacy_gpu_once.py')
p=subprocess.run([sys.executable,'-B','/tmp/easy7b_diag_remote.py',str(observer),'search-action-stop-legacy-gpu-live'],capture_output=True,text=True)
paths=[line for line in p.stdout.splitlines() if line.startswith('/Users/') and line.endswith('.json')]
if not paths:
 print(p.stdout[-1500:]);print(p.stderr[-1000:]);raise SystemExit(p.returncode or 1)
f=Path(paths[0]);raw=json.loads(f.read_text())
if raw['returncode']:
 print(json.dumps(dict(evidence=str(f),returncode=raw['returncode'],error=raw['stderr'][-1500:])));raise SystemExit(raw['returncode'])
x=json.loads(raw['stdout']);s=x['service'];pair=x['pair']
def docs(section):return {k:v.get('summary') for k,v in section.get('documents',{}).items() if k in ('progress.json','failure.json','result.json','completed.json')}
def brief(section):
 result=docs(section)
 if 'result.json' in result and isinstance(result['result.json'],dict):
  v=result['result.json'];result['result.json']={k:v.get(k) for k in ('status','completed_chunks','completed_questions','generation_phase_wall_seconds','pipeline_seconds_including_input_io_score_and_output_save','actual_raw_generated_tokens','actual_retrieval_calls','actual_fa3_prefill_and_decode','selection','full') if k in v}
 return result
r=dict(evidence=str(f),epoch=x['epoch'],slurm=[call.get('own_rows') for call in x['slurm']['calls']],service={k:v.get('summary') for k,v in s.get('documents',{}).items() if k=='logs/service-ready.json'},runner=brief(pair.get('runner',{})),variants=[dict(name=Path(v['root']).name,candidate=brief(v.get('candidate',{})),chunks=len(v['chunks']),raw_bytes=sum(z['bytes'] for z in v['raw_chunk_metadata']),errors=[e for z in v['logs'] for e in z.get('errors',[])]) for v in pair['variants']],errors=[e for z in pair['logs']+s['logs'] for e in z.get('errors',[])])
print(json.dumps(r,ensure_ascii=False))
