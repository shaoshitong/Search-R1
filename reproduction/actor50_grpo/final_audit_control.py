"""Existing SSH route for a dedicated, read-only remote CPU final acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid
from reproduction.operations.a800_prefix5_variable_pool_control import route, ROOT

REMOTE = r'''
import hashlib,json,os,pathlib,subprocess,sys,time
p=json.load(sys.stdin);h=p['deployment'];b=pathlib.Path(h['base'])
assert str(b)=='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'+h['run_id']+'-resume-'+h['execution_id']
def read(path,limit=131072):
 path=pathlib.Path(path);assert path.resolve()==path and not path.is_symlink() and path.stat().st_size<=limit
 return json.loads(path.read_text())
def who(pid):
 try:
  path=pathlib.Path('/proc')/str(pid);line=(path/'stat').read_text();v=line[line.rfind(')')+2:].split()
  return dict(pid=pid,start_ticks=int(v[19]),sid=int(v[3]),uid=path.stat().st_uid,state=v[0])
 except (FileNotFoundError,ProcessLookupError):return None
assert read(b/'preparation/deployment.json')==h
if p['action']=='launch':
 assert read(b/'logs/exit.json')['success'] and read(b/'logs/cleanup-status.json')['owned_births_all_gone']
 old=read(b/'preparation/pipeline-launch.json')['leader'];actual=who(old['pid'])
 assert actual is None or any(actual[k]!=old[k] for k in ('start_ticks','sid','uid')) or actual['state']=='Z'
 nonce=p['nonce'];assert len(nonce)==32 and all(c in '0123456789abcdef' for c in nonce)
 out=b/('final-independent-audit-'+nonce);code=b/'preparation'/('final-audit-source-'+nonce+'.py')
 assert not out.exists() and not code.exists()
 raw=p['source'].encode();assert hashlib.sha256(raw).hexdigest()==p['source_sha256'];compile(raw,str(code),'exec')
 with code.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 config=read(b/'preparation/launch-config.json')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(b/'repo'),PYTHONDONTWRITEBYTECODE='1',
  SEARCH_G1_GRPO_FROM50='1',SEARCH_G1_GRPO_PREFIX='40',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
 env.pop('PYTHONHOME',None);log=b/'logs'/('final-independent-audit-'+nonce+'.log')
 with log.open('xb') as f:
  child=subprocess.Popen([str(pathlib.Path(config['environment_root'])/'bin/python'),str(code),'--base',str(b),'--output',str(out)],
   cwd=b/'repo',env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 identity=who(child.pid);assert identity and identity['sid']==child.pid
 value=dict(run_id=h['run_id'],execution_id=h['execution_id'],base=str(b),output=str(out),source=str(code),
  source_sha256=p['source_sha256'],source_commit=p['source_commit'],leader=identity,log=str(log),checked_unix=time.time())
 with (b/'preparation'/('final-audit-launch-'+nonce+'.json')).open('x') as f:json.dump(value,f,sort_keys=True)
else:
 audit=p['audit'];assert audit['base']==str(b) and audit['run_id']==h['run_id'] and audit['execution_id']==h['execution_id']
 out=pathlib.Path(audit['output']);assert out.parent==b and out.name.startswith('final-independent-audit-')
 actual=who(audit['leader']['pid'])
 same=actual if actual and all(actual[k]==audit['leader'][k] for k in ('start_ticks','sid','uid')) else None
 value=dict(checked_unix=time.time(),leader=same,progress=None,error=None,summary=None)
 for name in ('progress','error','summary'):
  path=out/(name+'.json')
  if path.is_file():value[name]=read(path)
 if value['summary']:
  path=out/'complete.json';assert path.stat().st_size<=4*1024**2
  value['proof_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
  assert value['proof_sha256']==value['summary']['complete_sha256']
 log=pathlib.Path(audit['log'])
 if log.exists():
  with log.open('rb') as f:
   size=os.fstat(f.fileno()).st_size;f.seek(max(0,size-4096));value['log_tail']=f.read(4096).decode(errors='replace')
raw=json.dumps(value,allow_nan=False);assert len(raw.encode())<131072;print(raw)
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('launch','observe'));p.add_argument('--deployment',type=Path,required=True)
    p.add_argument('--audit-handle',type=Path,required=True);args=p.parse_args()
    h=json.loads(args.deployment.read_text());payload=dict(action=args.action,deployment=h)
    if args.action=='launch':
        state=json.loads((ROOT/'preparation/qprobe-teacher-warmup-a800-progress-handoff.json').read_text())['current_authoritative']
        current=state[state['active_experiment_key']];assert not current.get('active_launch_owner')
        assert current['execution_id']==h['execution_id'] and current['complete_outer']==200
        assert not args.audit_handle.exists()
        source=Path(__file__).with_name('final_audit.py').read_text()
        publication=json.loads((ROOT/'preparation/actor50-grpo-independent-final-audit-source-publication-20261021.json').read_text())
        assert hashlib.sha256(source.encode()).hexdigest()==publication['sha256']
        payload.update(nonce=uuid.uuid4().hex,source=source,source_sha256=publication['sha256'],source_commit=publication['source_backup_commit'])
    else:payload['audit']=json.loads(args.audit_handle.read_text())
    result=subprocess.run(route(REMOTE),input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=65)
    receipt=ROOT/'preparation'/('actor50-independent-final-audit-'+args.action+'-'+str(time.time_ns())+'.json')
    receipt.write_text(json.dumps(dict(checked_unix=time.time(),exit_code=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode()),indent=2)+'\n')
    if result.returncode:raise RuntimeError('Audit transport failed; inspect '+str(receipt))
    value=json.loads(result.stdout)
    if args.action=='launch':
        with args.audit_handle.open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(dict(receipt=str(receipt),result=value)))


if __name__=='__main__':main()
