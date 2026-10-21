"""Read-only owner admission/lifetime source metadata and existing queued jobs."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
WORKER = '''import pathlib,json,hashlib,subprocess,time
B=pathlib.Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
D=B/'preparation/relay-generation-queued-owner-v2-downstream-r1';roots=[D/'owner-ready.json']
rows=[]
for p in roots:
 raw=p.read_bytes();v=json.loads(raw);fields={k:v[k]for k in ['status','mode','generation','run_id','deadline_epoch','owner_seconds','instance','ready_epoch']if k in v}
 if isinstance(v.get('owner'),dict):fields['owner']={k:v['owner'][k]for k in ['pid','birth','start_ticks','unit','deadline_epoch','node','mode']if k in v['owner']}
 rows.append(dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),safe_fields=fields,root_keys=sorted(v)))
code=D/'072cbe91ab5ad1a3d222821f30bd79219aba837ad6479a08ef97cde5e587a834';texts=[]
for name in ['queued_relay_owner.py','retrieval_relay_owner.py']:
 p=code/name;raw=p.read_bytes();lines=raw.decode().splitlines();hits=[]
 terms=['Popen','subprocess.run','systemctl','while ','deadline','consumer','auto','slot','drain','main(','serve','Thread','end_epoch','launch']
 for i,line in enumerate(lines,1):
  if any(t in line for t in terms)and not any(t in line.lower()for t in ['secret','token','authorization']):hits.append(dict(line=i,code=line[:240]))
 texts.append(dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),selected_code_lines=hits))
p=subprocess.run(['systemctl','list-jobs','--no-legend','--no-pager'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
queued=[line for line in p.stdout.splitlines()if'70188b8e50fd'in line]
live=[]
for path in sorted((B/'pool/relay-live-slots').glob('*.json')):
 if path.stat().st_size>64*1024:continue
 raw=path.read_bytes();v=json.loads(raw)
 if not isinstance(v,dict):continue
 f={k:v[k]for k in ['status','slot','pid','start_ticks','deadline_epoch','epoch','job_id','unit','state']if k in v and isinstance(v[k],(str,int,float,bool,type(None)))}
 live.append(dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),mtime=path.stat().st_mtime,safe_fields=f,root_keys=sorted(v)))
print(json.dumps(dict(status='READONLY_OWNER_MECHANISM_STATIC_AND_QUEUE_METADATA',epoch=time.time(),owner_ready=rows,source_code=texts,queued_same_run_systemd_jobs=queued,relay_live_slot_metadata=live,source_modified=False,GPU_operation=False,launch_invoked=False),sort_keys=True))
'''

def main():
    spec = importlib.util.spec_from_file_location('ctl', '/Users/shaoshitong/Documents/remote_codex_control/control.py')
    ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)
    result = subprocess.run(ctl.route('h100-10', 'python3 -c '+shlex.quote(WORKER)),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=50)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-700:])
    value = json.loads(result.stdout.decode().splitlines()[-1])
    (HERE/'actual-owner-mechanism-metadata.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value, sort_keys=True))

if __name__ == '__main__':
    main()
