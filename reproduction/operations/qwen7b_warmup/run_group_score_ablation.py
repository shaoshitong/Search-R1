"""Publish and start one isolated CPU-only scorer diagnosis; never submit GPUs."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[3]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    assert subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]==commit
    names=['reproduction/operations/qwen7b_warmup/'+n for n in ['group_score_ablation.py','test_group_score_ablation.py','run_group_score_ablation.py','runtime.sh']]
    payload={};hashes={}
    for rel in names:
        data=(repo/rel).read_bytes();assert data==subprocess.check_output(['git','show',commit+':'+rel],cwd=repo),'Unpublished source '+rel
        payload[rel]=base64.b64encode(data).decode();hashes[rel]=hashlib.sha256(data).hexdigest()
    spec=importlib.util.spec_from_file_location('control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    code='commit='+repr(commit)+'\npayload='+repr(payload)+'\nhashes='+repr(hashes)+'\n'+REMOTE
    assert len(code.encode())<64000
    r=control.run('python3 -c '+shlex.quote(code),account='mou176',source_address=address,timeout=45)
    data=[json.loads(line) for line in r.stdout.decode().splitlines() if line.startswith('{')]
    if r.returncode or len(data)!=1:raise RuntimeError('Diagnosis launch not confirmed; inspect existing controller before retry: '+r.stderr.decode()[-1500:]+' '+r.stdout.decode()[-1500:])
    out=a.control_root/'outputs/qwen7b-warmup-20261019/group-score-ablation-controller.json';out.write_text(json.dumps(data[0],indent=2)+'\n');print(json.dumps(data[0]))


REMOTE=r'''
import pathlib,base64,hashlib,json,os,subprocess,time,fcntl
parent=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
base=parent/'balanced-rank1-from-epoch1-prefix20-20261020';root=parent/'control-tools'/('group-score-'+commit[:12]);root.mkdir(parents=True,exist_ok=True)
with (root/'launch.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for rel,encoded in payload.items():
  raw=base64.b64decode(encoded);assert hashlib.sha256(raw).hexdigest()==hashes[rel];dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert dest.read_bytes()==raw,'Unknown source conflict'
  else:dest.write_bytes(raw)
 output=base/'analysis'/('epoch-0002-group-score-'+commit[:12]+'.json');output.parent.mkdir(exist_ok=True)
 cp=root/'controller.json'
 if cp.exists():
  old=json.loads(cp.read_text())
  assert old['source_commit']==commit and old['files']==hashes
  pid=old['pid'];path=pathlib.Path('/proc')/str(pid)/'cmdline'
  if path.exists():
   cmd=path.read_bytes();assert b'group_score_ablation.py' in cmd and str(output).encode() in cmd
   old['observed_live']=True;old['observed_at']=time.time();print(json.dumps(old))
  elif output.exists():old['completed_output_exists']=True;print(json.dumps(old))
  else:raise RuntimeError('Prior CPU diagnosis exited without result; inspect log, do not relaunch blindly')
 else:
  env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
  log=root/'analysis.log';command=['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),str(root/'reproduction/operations/qwen7b_warmup/group_score_ablation.py'),'--base',str(base),'--epoch','2','--output',str(output)]
  with log.open('xb') as h:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=h,stderr=h,start_new_session=True)
  record=dict(status='ISOLATED_CPU_GROUP_SCORE_ABLATION_STARTED',pid=proc.pid,source_commit=commit,code_root=str(root),files=hashes,output=str(output),progress=str(output.with_name(output.stem+'-progress.json')),log=str(log),base=str(base),epoch=2,time=time.time(),gpu_training_changed=False)
  cp.write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''

if __name__=='__main__':main()
