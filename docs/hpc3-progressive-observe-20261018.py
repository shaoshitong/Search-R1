"""One bounded read-only status query for the isolated progressive CP50 run."""
import json
from pathlib import Path
import shlex
import subprocess
from reproduction.hpc3_prefix5.direct_control import run
from reproduction.hpc3_prefix5.control import PYTHON,OUTPUT
from reproduction.hpc3_prefix5.observe import query_for

BASE='/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r3'
EXTRA=r'''
# Restrict the result to small scalar progress, never model/rollout payloads.
base=b
stage=status.get('stage','') if status else ''
result['probe_refreshes']={}
for n in (50,100,150):
 root=b/'probe-refreshes'/('round-%06d'%n)/'outers'/('outer-%06d'%n)
 if not root.exists():continue
 result['probe_refreshes'][str(n)]={phase:[read(root/phase/('rank-%d-progress.json'%rank)) for rank in range(4)] for phase in ('prefix','full-pool')}
 diag=root/'diagnostic'/'outers'/('outer-%06d'%n)
 result['probe_refreshes'][str(n)]['diagnostic']={phase:[read(diag/phase/('rank-%d-progress.json'%rank)) for rank in range(4)] for phase in ('prefix','full')}
 result['probe_refreshes'][str(n)]['reservation']=({key:(v:=read(root/'probe-only-reservation.json')).get(key) for key in ('student_checkpoint_outer','prefix_tokens','actor_pool_advanced')} if (root/'probe-only-reservation.json').exists() else None)
 result['probe_refreshes'][str(n)]['migration']=read(b/'probe'/('prefix-migration-%03d.json'%n))
# Epoch metrics contain only scalar aggregate diagnostics; raw predictions stay remote.
result['probe_fits']={}
for n in range(50,200,10):
 root=b/'outers'/('outer-%06d'%n)/'probe-update'
 if not root.exists():continue
 result['probe_fits'][str(n)]=dict(progress=[read(root/('progress-rank-%d.json'%rank)) for rank in range(4)],
   epoch_metrics=[read(root/('epoch-%03d-metrics.json'%epoch)) for epoch in (1,2,3)],latest_epoch=read(root/'latest-epoch.json'),
   complete=bool((root/'complete.json').exists()))
result['latest_selected']={key:joint['rollout'].get(key) for key in ('actual_selected_mixed_questions','actual_selected_mixed_fraction','selected_k_histogram')} if joint else None
result['running_phases']=[dict(name=name,pid=item.get('pid'),state=item.get('state')) for name,item in (ledger or {}).get('jobs',{}).items() if item.get('state')!='STOPPED']
result['recent_logs']={}
for name in ['slurm-'+job]+[row['name'] for row in result['running_phases']]:
 path=b/'logs'/(name+'.log')
 if path.is_file():
  with path.open('rb') as handle:handle.seek(max(0,path.stat().st_size-5000));tail=handle.read().decode(errors='replace')
  result['recent_logs'][name]=dict(bytes=path.stat().st_size,tail=tail)
result['shared_free_bytes']=__import__('os').statvfs(b).f_bavail*__import__('os').statvfs(b).f_frsize
'''

def main():
    code=query_for(BASE).replace('print(json.dumps(result,allow_nan=False))',EXTRA+'\nprint(json.dumps(result,allow_nan=False))')
    address=subprocess.check_output(['ipconfig','getifaddr','en0'],text=True).strip()
    result=run(PYTHON+' -c '+shlex.quote(code),account='mou176',source_address=address,timeout=25)
    if result.returncode:raise RuntimeError(result.stdout.decode()[-4000:]+result.stderr.decode()[-1000:])
    value=json.loads(result.stdout)
    (OUTPUT/'progressive-cp50-current.json').write_text(json.dumps(value,indent=2)+'\n')
    with (OUTPUT/'progressive-cp50-observations.jsonl').open('a') as handle:handle.write(json.dumps(value)+'\n')
    # Full bounded scalar audit is saved; keep the tool output compact.
    for name in value.get('recent_logs',{}):
        value['recent_logs'][name]['tail']=value['recent_logs'][name]['tail'][-1800:]
    print(json.dumps(value))

if __name__=='__main__':main()
