"""Observe the two already launched d45 SHA workers; never launch or delete."""
import ast
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import shlex
import time

WS = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6')
OUT = Path(__file__).parent
PACKAGE = json.loads((OUT/'old-cosearch200-deploy-package.json').read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BRIDGE = load('modern_bridge', WS/'work/qprobe-sigma3-loss-20261015/reproduction/hpc3_qprobe/local_account_bridge.py')
HK = load('hkctl', WS/'work/start_cosearch_flash_hf.py')


def observe(site):
    settings = PACKAGE['sites'][site]
    phase = settings['phase']
    launch_name = 'old-cosearch200-source-repair219-sha-launch-result-1792258035.json' if site == 'hpc' else 'old-cosearch200-hk-sha-launch-result-1792257482.json'
    launch = json.loads((OUT/launch_name).read_text())
    expected = json.loads(launch['stdout'])['child']
    remote = '''import json,os,socket,time
from pathlib import Path
root=Path(ROOT_VALUE);expected=EXPECTED_VALUE
output={'checked_unix':time.time(),'host':socket.gethostname(),'uid':os.getuid(),'phase':PHASE_VALUE,'expected_birth':expected}
proc=Path('/proc')/str(expected['pid']);output['process_exists']=proc.exists()
if proc.exists():
 fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
 birth={'host':socket.gethostname(),'pid':expected['pid'],'start_ticks':int(fields[19]),'uid':proc.stat().st_uid,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'pid_namespace':os.readlink(proc/'ns/pid')}
 output['actual_birth']=birth;output['birth_matches']=birth==expected;output['process_state']=fields[0]
for name in [PHASE_VALUE+'-launch-claim.json',PHASE_VALUE+'-process.json',PHASE_VALUE+'-worker.json','progress.json',PHASE_VALUE+'-validation.json',PHASE_VALUE+'-exit.json','failure.json','result.json']:
 p=root/name
 output[name]=json.loads(p.read_text()) if p.exists() else None
p=root/(PHASE_VALUE+'.log');output['log_bytes']=p.stat().st_size if p.exists() else None
if p.exists():
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-4000));output['log_tail']=f.read().decode('utf-8',errors='replace')
output['audit_files']=sorted(p.name for p in root.iterdir())
print(json.dumps(output,sort_keys=True))
'''.replace('ROOT_VALUE', repr(settings['audit_root'])).replace('EXPECTED_VALUE', repr(expected)).replace('PHASE_VALUE', repr(phase))
    ast.parse(remote)
    command = shlex.join([settings['python'], '-B', '-c', remote])
    if site == 'hpc':
        result = BRIDGE.run('mou176', command, '', str(WS/'work'), login_host='10.120.48.31',
            expected_host_identity=dict(hostname='ACD-Login-1', boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a', pid_namespace='pid:[4026531836]'), observation_timeout=55)
    else:
        result = HK.campus('hkust3', command, input='', observation_timeout=55)
    record = dict(checked_unix=time.time(), command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    path = OUT/('old-cosearch200-repair219-'+phase+'-sha-observe-'+str(int(time.time()))+'.json')
    path.write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
    summary = dict(site=site, phase=phase, path=str(path), returncode=result.returncode)
    if result.returncode:
        summary.update(stderr=result.stderr, stdout=result.stdout)
    else:
        state = json.loads(result.stdout); progress = state['progress.json'] or {}
        summary.update(birth_matches=state.get('birth_matches'), process_exists=state['process_exists'],
            actual_birth=state.get('actual_birth'), process_state=state.get('process_state'),
            progress=progress, exit=state[phase+'-exit.json'],
            validation_status=(state[phase+'-validation.json'] or {}).get('status'),
            log_bytes=state['log_bytes'], log_tail=state.get('log_tail'),
            failure=state['failure.json'], deletion_result=state['result.json'])
    return summary


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        for value in pool.map(observe, ['hpc', 'hk']):
            print(json.dumps(value, sort_keys=True))
