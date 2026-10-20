"""Publish-isolated code-only CP2 continuation deployment; no automatic submit."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[3]
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=root).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    branch='codex/qwen7b-probe-warmup-20261019'
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=root,text=True).split()[0];assert remote==commit
    spec=importlib.util.spec_from_file_location('private_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    def run(code,timeout=60):
        result=m.run('python3 -c '+shlex.quote(code),account='mou176',source_address=address,timeout=timeout)
        if result.returncode:raise RuntimeError(result.stdout.decode()[-2000:]+'\n'+result.stderr.decode()[-500:])
        return result.stdout.decode().strip()
    b='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
    phase=b+'/mixed-only-ft-cp2-20261020';code_root=phase+'/repo-'+commit[:12]
    print(run('import pathlib,shutil; p=pathlib.Path('+repr(code_root)+'); assert not p.exists(),"Inspect existing deployment before reuse"; p.parent.mkdir(exist_ok=True); shutil.copytree('+repr(b+'/repo-postprocess-96df5a23f23f')+',str(p)); print("ISOLATED_SOURCE_CREATED")'),flush=True)
    names=['mixed_loss.py','mixed_finetune.py','mixed_prepare.py','mixed_fit.sbatch','heldout_selection.py']
    manifests={}
    for name in names:
        rel='reproduction/operations/qwen7b_warmup/'+name;data=(root/rel).read_bytes();digest=hashlib.sha256(data).hexdigest();manifests[rel]=digest
        script='import pathlib,base64,hashlib; p=pathlib.Path('+repr(code_root+'/'+rel)+'); d=base64.b64decode('+repr(base64.b64encode(data).decode())+'); assert hashlib.sha256(d).hexdigest()=='+repr(digest)+'; p.write_bytes(d); assert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(digest)+'; print('+repr(name+' SHA_MATCH')+')'
        print(run(script),flush=True)
    inherited=['train.py','model.py','packed_features.py','native_capture.py','runtime.sh']
    for name in inherited:
        rel='reproduction/operations/qwen7b_warmup/'+name
        manifests[rel]=hashlib.sha256(subprocess.check_output(['git','show','96df5a23f23f6da0f24e79e67dcf0e3039f22de0:'+rel],cwd=root)).hexdigest()
    rel='reproduction/prefix20_group_joint100/loss.py';manifests[rel]=hashlib.sha256((root/rel).read_bytes()).hexdigest()
    publication=dict(commit=commit,remote_sha=remote,branch=branch,code_root=code_root,base=phase,files=manifests,time=time.time(),inherited_code_source='96df5a23f23f6da0f24e79e67dcf0e3039f22de0')
    verify='''import json,pathlib,hashlib,subprocess,os
p=PUBLICATION
for name,digest in p['files'].items():assert hashlib.sha256((pathlib.Path(p['code_root'])/name).read_bytes()).hexdigest()==digest,name
b=pathlib.Path(p['base']);(b/'preparation').mkdir(exist_ok=True)
(b/'preparation/source-publication.json').write_text(json.dumps(p,indent=2))
env=dict(os.environ,QWEN7B_CODE_ROOT=p['code_root'])
subprocess.run(['bash',p['code_root']+'/reproduction/operations/qwen7b_warmup/runtime.sh','-u','-m','reproduction.operations.qwen7b_warmup.mixed_prepare','--base',p['base'],'--source-commit',p['commit']],env=env,check=True)
print('DEPLOYED_PREPARED_SOURCE_SHA_VERIFIED')
'''.replace('PUBLICATION',repr(publication))
    print(run(verify,timeout=300),flush=True)
    output=a.control_root/'outputs/qwen7b-warmup-20261019/mixed-ft-source-publication.json';output.write_text(json.dumps(publication,indent=2))

if __name__=='__main__':main()
