"""Read-only exact standalone unit and small QA progress; no runtime control."""
import json
from pathlib import Path
import time
from reproduction.h1008_qprobe.management_transport import remote, connection_options, local_connection_path


def observe(root,deployment_path=None,connection=None):
    root=Path(root);output=root/'outputs/search-g1-i1i6-fresh200-20261017'
    deployment=json.loads(Path(deployment_path or output/'QA5_STANDALONE_DEPLOYMENT.json').read_text())
    from .qa_standalone_identity import binding
    target=binding(deployment['actual_config'])
    if target['base']!=deployment['base'] or target['unit']!=deployment['unit']:
        raise ValueError('Deployment receipt and exact QA config disagree')
    connection=json.loads(Path(connection or local_connection_path(root)).read_text())
    script=r'''
import pathlib,json,subprocess,time
b=pathlib.Path(BASE);unit=UNIT
assert b.name.startswith('search-g1-qa-i1i6-4a1cc2d5-outer') and unit.startswith('g1-qa-4a1cc2d5-o')
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
 gpu_compute=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10).stdout,
 attempts=read(b/'validation/initialization-attempts/history.json'),
 initialization_statuses=[read(p) for p in sorted((b/'validation/initialization-attempts').glob('*-status.json'))],logs={})
for name in ('controller','retriever','evaluation'):
 p=b/'logs'/(name+'.log')
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2200));tail=f.read().decode(errors='replace')
  value['logs'][name]=dict(bytes=p.stat().st_size,tail=tail)
print(json.dumps(value))
'''.replace('BASE',repr(deployment['base'])).replace('UNIT',repr(deployment['unit']))
    label='QA%d_STANDALONE'%target['outer'] if target['expected_count']==512 else 'QA_FINAL_STANDALONE'
    receipt=output/(label.lower()+'-observe-'+str(time.time_ns())+'-transport.json')
    value=json.loads(remote(script,receipt,target=deployment['actual_config']['node'],**connection_options(connection)))
    evidence=receipt.with_name(receipt.name.replace('-transport','-actual'))
    evidence.write_text(json.dumps(value,indent=2)+'\n')
    (output/(label+'_LATEST.json')).write_text(json.dumps(value,indent=2)+'\n')
    return value


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--deployment');p.add_argument('--connection');a=p.parse_args()
    r=observe(a.root,a.deployment,a.connection)
    print(json.dumps(dict(observed_unix=r['observed_unix'],systemd=r['systemd'],
        completed_questions=sum((v or {}).get('completed_questions',0) for v in r['rank_progress']),
        completed_ranks=r['rank_complete'],qa=r['qa'],logs=r['logs'])))
