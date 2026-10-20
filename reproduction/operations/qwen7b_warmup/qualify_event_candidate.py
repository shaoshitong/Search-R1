"""Run loss regression in isolated CPU-only HPC3 source, without touching writer."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[3]
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip(),'Publish all source before qualification'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    assert subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]==commit
    names=['reproduction/operations/qwen7b_warmup/'+n for n in ['balanced_loss.py','event_candidate_loss.py','test_event_candidate_loss.py','runtime.sh','qualify_event_candidate.py']]+['reproduction/prefix20_group_joint100/loss.py']
    payload={n:base64.b64encode((repo/n).read_bytes()).decode() for n in names}
    hashes={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in names}
    spec=importlib.util.spec_from_file_location('control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    code='commit='+repr(commit)+'\npayload='+repr(payload)+'\nhashes='+repr(hashes)+'\n'+REMOTE
    assert len(code.encode())<64000
    r=control.run('python3 -c '+shlex.quote(code),account='mou176',source_address=address,timeout=45)
    lines=[json.loads(s) for s in r.stdout.decode().splitlines() if s.startswith('{')]
    if r.returncode or len(lines)!=1:raise RuntimeError('Remote CPU qualification failed: '+r.stderr.decode()[-2500:]+' '+r.stdout.decode()[-2500:])
    result=lines[0];assert result['status']=='ACTUAL_HPC3_CPU_EVENT_CANDIDATE_TESTS_PASSED'
    out=a.control_root/'outputs/qwen7b-warmup-20261019/event-candidate-actual-cpu-qualification.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


REMOTE=r'''
import pathlib,base64,hashlib,json,os,subprocess,time
parent=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
root=parent/'control-tools'/('event-candidate-cpu-'+commit[:12]);root.mkdir(parents=True,exist_ok=True)
for rel,encoded in payload.items():
    raw=base64.b64decode(encoded);assert hashlib.sha256(raw).hexdigest()==hashes[rel]
    dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert dest.read_bytes()==raw,'Unknown partial source conflict'
    else:dest.write_bytes(raw)
    assert hashlib.sha256(dest.read_bytes()).hexdigest()==hashes[rel]
for rel in ['reproduction','reproduction/operations','reproduction/operations/qwen7b_warmup','reproduction/prefix20_group_joint100']:
    (root/rel/'__init__.py').touch(exist_ok=True)
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
r=subprocess.run(['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-m','unittest','-v','reproduction.operations.qwen7b_warmup.test_event_candidate_loss'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
assert r.returncode==0,r.stdout+r.stderr
result=dict(status='ACTUAL_HPC3_CPU_EVENT_CANDIDATE_TESTS_PASSED',commit=commit,remote_sha=commit,code_root=str(root),files=hashes,test_output=r.stdout+r.stderr,cpu_only=True,training_config_changed=False,gpu_training_changed=False,time=time.time())
(root/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''

if __name__=='__main__':main()
