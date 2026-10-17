"""Read-only exact standalone unit and small QA progress; no runtime control."""
import json
from pathlib import Path
import time
from reproduction.h1008_qprobe.management_transport import remote


def observe(root):
    root=Path(root);output=root/'outputs/search-g1-i1i6-fresh200-20261017'
    deployment=json.loads((output/'QA5_STANDALONE_DEPLOYMENT.json').read_text())
    connection=json.loads((root/'work/search-g1-prefix1024-diversity512-20261016/MANAGEMENT_DIRECT_MASTER_CURRENT.json').read_text())
    script=r'''
import pathlib,json,subprocess,time
b=pathlib.Path(BASE);unit=UNIT
assert b.name.startswith('search-g1-qa-i1i6-4a1cc2d5-outer0005-') and unit.startswith('g1-qa-4a1cc2d5-o5-')
def read(p):return json.loads(p.read_text()) if p.exists() else None
r=subprocess.run(['systemctl','show',unit,'-p','ActiveState,SubState,Result,ExecMainStatus,MainPID,ControlGroup,CPUUsageNSec,MemoryPeak,MemoryCurrent'],capture_output=True,text=True,timeout=10)
props=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
group=props.get('ControlGroup');pids=None
if group:
 assert group=='/system.slice/'+unit
 path=pathlib.Path('/sys/fs/cgroup')/group.lstrip('/')/'cgroup.procs'
 pids=list(map(int,path.read_text().split())) if path.exists() else []
value=dict(observed_unix=time.time(),base=str(b),unit=unit,systemd=props,cgroup_pids=pids,
 resources=read(b/'logs/resource-latest.json'),retriever_ready=read(b/'logs/retriever-ready.json'),
 terminal=read(b/'logs/terminal.json'),qa=read(b/'validation/complete.json'),
 rank_progress=[read(b/'validation'/('rank-%d-progress.json'%rank)) for rank in range(8)],
 rank_complete=[read(b/'validation'/('rank-%d-complete.json'%rank)) is not None for rank in range(8)],
 attempts=read(b/'validation/initialization-attempts/history.json'),logs={})
for name in ('controller','retriever','evaluation'):
 p=b/'logs'/(name+'.log')
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2200));tail=f.read().decode(errors='replace')
  value['logs'][name]=dict(bytes=p.stat().st_size,tail=tail)
print(json.dumps(value))
'''.replace('BASE',repr(deployment['base'])).replace('UNIT',repr(deployment['unit']))
    receipt=output/('qa5-observe-'+str(time.time_ns())+'-transport.json')
    value=json.loads(remote(script,receipt,control_path=connection['control_path'],target='h100-5'))
    evidence=receipt.with_name(receipt.name.replace('-transport','-actual'))
    evidence.write_text(json.dumps(value,indent=2)+'\n')
    (output/'QA5_STANDALONE_LATEST.json').write_text(json.dumps(value,indent=2)+'\n')
    return value


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    r=observe(a.root)
    print(json.dumps(dict(observed_unix=r['observed_unix'],systemd=r['systemd'],
        completed_questions=sum((v or {}).get('completed_questions',0) for v in r['rank_progress']),
        completed_ranks=r['rank_complete'],qa=r['qa'],logs=r['logs'])))
