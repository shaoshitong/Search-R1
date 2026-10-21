"""Read only the former supplementary producer's bounded terminal metadata."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
WORKER = r'''import pathlib,json,hashlib,time,subprocess,os
B=pathlib.Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
unit='sst-q7b-g5-relaygen-queued-supp-v1-slot-gpu4-tp1-r1-70188b8e50fd.service'
roots=list((B/'preparation').glob('*supp*'))+[B/'retrieval-relay-generation/launch/native-3731224']
safe_keys={'status','stage','phase','epoch','pid','start_ticks','birth','returncode','return_code','exit_code','completed','questions','trajectories','completed_batches','batches','assigned_batches','input_batches','attempt','run_id','seed','worker_id','unit','native_unit','actor_updates','probe_updates','completed_inputs','pending_inputs','failed_inputs','reused_inputs','successful_inputs','expected_batches','expected_questions','returncodes','all_assigned_inputs_complete','all_inputs_complete','job_id','job','attempt_dir','result','worker','reason','error_type'}
def safe(v,depth=0):
 if depth>4:return None
 if isinstance(v,dict):return{k:safe(x,depth+1)for k,x in v.items()if k in safe_keys}
 if isinstance(v,list):return[safe(x,depth+1)for x in v[:20]if isinstance(x,(dict,int,float,bool,type(None)))]
 if isinstance(v,(str,int,float,bool,type(None))):return v
 return None
rows=[];paths=set()
for root in roots:
 if not root.exists():continue
 for path in root.rglob('*.json'):
  if path in paths or path.stat().st_size>256*1024 or any(s in path.parts for s in ['.git','__pycache__']):continue
  if any(s in path.name.lower()for s in ['secret','token','credential']):continue
  paths.add(path);raw=path.read_bytes()
  try:v=json.loads(raw)
  except ValueError:continue
  if not isinstance(v,dict):continue
  projection=safe(v)
  # Receipt descriptors and assignment filenames are metadata only, never inputs/rollouts.
  locator={k:x for k,x in v.items()if isinstance(x,str)and str(B)in x and any(w in k for w in ['receipt','progress','job','attempt','controller'])and not any(w in k for w in ['secret','token','credential'])}
  rows.append(dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),mtime=path.stat().st_mtime,root_keys=sorted(v),safe_fields=projection,metadata_locators=locator))
p=subprocess.run(['journalctl','-u',unit,'-n','12','-o','json','--no-pager'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
events=[]
for line in p.stdout.splitlines():
 try:x=json.loads(line)
 except ValueError:continue
 m=x.get('MESSAGE','');m=m if isinstance(m,str)else''
 if x.get('_PID')=='1':
  events.append(dict(epoch_microseconds=x.get('__REALTIME_TIMESTAMP'),deactivated_successfully='Deactivated successfully' in m,
   failed_with_result='Failed with result' in m,main_process_exited='Main process exited' in m,
   consuming_resources='Consumed' in m,priority=x.get('PRIORITY')))
print(json.dumps(dict(status='READONLY_SUPPLEMENTARY_PRODUCER_TERMINAL_METADATA',epoch=time.time(),roots=[str(p)for p in roots],receipts=rows,journal_systemd_terminal_events=events,source_modified=False,model_or_rollout_read=False,GPU_operation=False),sort_keys=True))
'''

def main():
    spec = importlib.util.spec_from_file_location('ctl', '/Users/shaoshitong/Documents/remote_codex_control/control.py')
    ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)
    result = subprocess.run(ctl.route('h100-10', 'python3 -c '+shlex.quote(WORKER)),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-700:])
    value = json.loads(result.stdout.decode().splitlines()[-1])
    (HERE/'actual-supp-terminal-receipts.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value, sort_keys=True))

if __name__ == '__main__':
    main()
