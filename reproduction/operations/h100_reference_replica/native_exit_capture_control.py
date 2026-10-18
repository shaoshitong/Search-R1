"""Once-start/status for an exit-only passive collector; not a progress watcher."""
import argparse
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path

from .b_progressive_remote import remote
from .follow_h10011_ab_normal_boundary import PATH_LAYOUTS,ROOT

PARENT='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/search-g1-native-exit-evidence-20261018'
BOOT='24bbc2ee-c187-492f-bce3-a3afaa2b08e6'
CAPTURE_ROOT=PARENT+'/h100-11-'+BOOT+'/ab-leaders-128824-757735-once-v1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('start','status'));args=parser.parse_args()
    sources={}
    for name in ('kernel_exit_events.py','native_exit_capture.py'):
        raw=Path(__file__).with_name(name).read_bytes()
        sources[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(raw).decode())
    state=json.loads((ROOT/'work/search-g1-prefix5-ab200-state.json').read_text())
    compact=json.loads((ROOT/'work/prefix5-ab-latest-compact.json').read_text())
    targets={}
    for arm,value in state['slots'].items():
        actual=compact['arms'][arm];receipt=actual['native']
        if actual['base']!=value['base']or receipt['run_id']!=value['run_id']or receipt['config_sha256']!=value['config_sha256']:
            raise ValueError('Primary state and current original submitted run differ')
        if receipt['leader_same_birth']is not True:raise ValueError('Current leader is not the submitted birth')
        targets[arm]=dict(base=actual['base'],run_id=receipt['run_id'],config_sha256=receipt['config_sha256'],
            leader={key:receipt['leader'][key]for key in ('pid','start_ticks','sid','uid')})
    if {arm:value['leader']['pid']for arm,value in targets.items()}!={'A':128824,'B':757735}:
        raise ValueError('This once collector is bound only to the actual current A/B leaders')
    manifest=dict(targets=targets,qualification_result=PARENT+'/h100-11-'+BOOT+'/qualification-once-v1/RESULT.json')
    data=dict(action=args.action,root=CAPTURE_ROOT,sources=sources,manifest=manifest)
    inner='PARAMS='+repr(data)+'\n'+r'''
import base64,hashlib,json,os,subprocess
from pathlib import Path
p=PARAMS;root=Path(p['root'])
if p['action']=='start':
 root.mkdir(parents=True,exist_ok=True)
 for name,item in p['sources'].items():
  path=root/name;raw=base64.b64decode(item['data'])
  if not path.exists():
   with path.open('xb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
  if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Collector source differs; do not overwrite')
 path=root/'manifest.json'
 if not path.exists():
  with path.open('x')as stream:json.dump(p['manifest'],stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
 if json.loads(path.read_text())!=p['manifest']:raise ValueError('Once collector target binding differs')
action='submit'if p['action']=='start'else'status'
run=subprocess.run(['/usr/bin/python3',str(root/'native_exit_capture.py'),action,'--root',str(root)],
 text=True,capture_output=True,timeout=20)
if run.returncode:raise RuntimeError(run.stderr)
print(run.stdout)
'''
    outer='import subprocess,os\np=subprocess.run('+repr(['unshare','--mount','--propagation','private','bash',PATH_LAYOUTS['retained-cache-v2'],'/usr/bin/python3','-'])+',input='+repr(inner)+',env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1"),text=True,capture_output=True,timeout=30)\nassert p.returncode==0,p.stderr\nprint(p.stdout)'
    with contextlib.redirect_stdout(io.StringIO())as output:
        value=remote(outer,'passive-native-exit-capture-'+args.action,namespace=False,timeout=50)
    raw=json.loads(output.getvalue())
    print(json.dumps(dict(evidence=raw['evidence'],root=CAPTURE_ROOT,collector_same_birth=value.get('collector_same_birth'),
        collector_state=value.get('collector_state'),armed='ARMED.json'in value,
        actual_exit_proofs=[arm for arm in ('A','B')if arm+'-actual-exit.json'in value],
        result=value.get('RESULT.json',{}).get('value'),training_modified=False)))


if __name__=='__main__':main()
