"""Read all three replacement units in one SSH connection; no persistent watch."""
import base64
import contextlib
import io
import json
import subprocess
import time
from . import source_tp1_replacement_control as control

PROOF_SHA = '67b096a2d6852b3791184e05c8c44cf9be45cd826f367124cec27bec90a8024c'


def main():
    scripts = {}
    for group in control.GROUPS:
        identity, _, manifest = control.bundle(group, terminal_sha256=PROOF_SHA)
        values = dict(BASE=control.BASE,RUN_ID=control.RUN_ID,SOURCE_UNIT=control.SOURCE_UNIT,
            PARENT_PID=control.PARENT_PID,PARENT_BIRTH=control.PARENT_BIRTH,IDENTITY=identity,
            MANIFEST=manifest,ACTION='status',FILES={},INTENT=None,PUBLICATION=None)
        scripts[group] = '\n'.join(k+'='+repr(v) for k,v in values.items())+'\n'+control.REMOTE
    inner = 'SCRIPTS='+repr(scripts)+'\n'+r'''
import contextlib,io,json,time
rows={}
for group,script in SCRIPTS.items():
 out=io.StringIO()
 with contextlib.redirect_stdout(out):exec(script,{})
 rows[group]=json.loads(out.getvalue())
print(json.dumps(dict(groups=rows,epoch=time.time(),read_only=True)))
'''
    wrapper = ('import subprocess,sys\nr=subprocess.run(' + repr(['nsenter','--target','1490214','--mount','--net','--','/usr/bin/python3','-'])
        + ',input='+repr(inner)+',text=True,capture_output=True,timeout=85)\nprint(r.stdout,end="");print(r.stderr,end="",file=sys.stderr);raise SystemExit(r.returncode)\n')
    result = control.old.transport(wrapper,'h100-5')
    path = control.ROOT/'work'/('qwen7b-three-replacement-one-connection-actual-%d.json'%time.time_ns())
    value = dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,epoch=time.time())
    path.write_text(json.dumps(value,indent=2)+'\n')
    if result.returncode:
        print(json.dumps(dict(status='OBSERVATION_CHANNEL_ERROR_NOT_NATIVE_EXIT',raw=str(path),stderr=result.stderr)))
        raise SystemExit(result.returncode)
    raw = json.loads(result.stdout)
    summary = {}
    for group,row in raw['groups'].items():
        sampler=row.get('sampler')or{};progress=sampler.get('progress')or{};last=progress.get('latest_input','').rsplit('/',1)[-1].removesuffix('.json')
        phases=sampler.get('active_phases')or[]
        summary[group]=dict(unit=row['unit'],observed=row['observed'],own_complete_batches=progress.get('completed_own_batches',0),
            next_input_started=any(x['batch']!=last and x['metadata'].get('completed_trajectories',0)>0 for x in phases),
            failure=sampler.get('failure'),model_initialized=sampler.get('model_initialized'),progress=progress)
    print(json.dumps(dict(epoch=raw['epoch'],groups=summary,actual=str(path),read_only=True)),flush=True)


if __name__=='__main__':main()
