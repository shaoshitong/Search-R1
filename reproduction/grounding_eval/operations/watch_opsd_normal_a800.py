import sys,subprocess,json,pathlib,time,shlex,argparse
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot96','full3610'],required=True);args=p.parse_args()
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
root='/data/shaoshitong/grounding-eval-20261019/'+('opsd3304-normal-pilot96-v4' if args.phase=='pilot96' else 'opsd3304-normal-v4')
code=r'''
import pathlib,json,time,subprocess,hashlib
root=pathlib.Path(ROOT);out={'at':time.time(),'root':str(root)}
def read(p):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return None
out['control']={n:read(root/'control'/n) for n in ['worker.json','exit.json','launch.json']};out['normal']={n:read(root/n) for n in ['gpu-admission.json','progress.json','normal-error.json','complete.json','pilot-verification.json']}
out['raw']={'files':0,'bytes':0,'records':0,'token_totals':{},'actual_batch_sizes':[],'batch_completion_history':[],'max_turn_input_tokens':0,'max_turn_raw_output_tokens':0}
for p in sorted((root/'raw').glob('*.json')):
 d=read(p);s=out['raw'];s['files']+=1;s['bytes']+=p.stat().st_size;s['records']+=len(d['rows']);s['actual_batch_sizes'].append(len(d['rows']));s['batch_completion_history'].append({'file':p.name,'completed_at':d['completed_at'],'records':len(d['rows']),'bytes':p.stat().st_size})
 for r in d['rows']:
  for k in ['model_generated_tokens','raw_generated_tokens','observation_tokens','environment_correction_tokens','initial_prompt_tokens','trajectory_total_tokens','cumulative_model_input_tokens']:
   s['token_totals'][k]=s['token_totals'].get(k,0)+r[k]
  for t in r['turns']:
   s['max_turn_input_tokens']=max(s['max_turn_input_tokens'],len(t['generation_context_token_ids']));s['max_turn_raw_output_tokens']=max(s['max_turn_raw_output_tokens'],len(t['raw_response_token_ids']))
out['logs']={}
for p in [root/'control/normal.log',root/'control/controller.log']:
 if p.exists():
  with p.open('rb')as f:f.seek(max(0,p.stat().st_size-15000));lines=f.read().decode(errors='replace').splitlines()
  out['logs'][p.name]={'bytes':p.stat().st_size,'mtime':p.stat().st_mtime,'tail':lines[-10:],'errors':[l[:1500]for l in lines if any(x in l for x in ['Traceback','Error','ERROR','out of memory','Killed'])][-12:]}
r=subprocess.run(['nvidia-smi','-i','0','--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10);out['gpu0']={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr[-300:]}
print(json.dumps(out))
'''.replace('ROOT',repr(root))
r=subprocess.run(control.route('baai-a800','/usr/bin/python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=35);d={'transport_rc':r.returncode,'stderr':r.stderr[-500:]}
try:d['audit']=json.loads(r.stdout.strip().splitlines()[-1])
except:d['stdout_tail']=r.stdout[-1000:]
q=pathlib.Path('outputs/grounding-eval-20261019')/('opsd-a800-'+args.phase+'-observe-'+time.strftime('%H%M%S')+'.json');q.write_text(json.dumps(d,indent=2));print(q)
a=d.get('audit',{});print(json.dumps({'raw':a.get('raw'),'progress':a.get('normal',{}).get('progress.json'),'normal_error':a.get('normal',{}).get('normal-error.json'),'exit':a.get('control',{}).get('exit.json'),'gpu':a.get('gpu0'),'logs':a.get('logs')}))
