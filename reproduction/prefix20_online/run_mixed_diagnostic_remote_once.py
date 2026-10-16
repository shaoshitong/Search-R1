"""Deploy a committed CPU posthoc analyzer once; return only small summaries.

Does not touch the frozen runtime, submit jobs, stop training, or transfer models,
trajectory payloads or per-question predictions to Mac. Retry an unknown outcome
with the identical --tag so an existing receipt is read rather than replayed.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from reproduction.h1008_qprobe.management_transport import remote


REMOTE = r'''
import hashlib,json,os,subprocess
from pathlib import Path
p=PAYLOAD
base=Path(p['base']).resolve()
root=base/'preparation'/('mixed-g5-v1-'+p['tag'])
root.mkdir(parents=True,exist_ok=True)
intent=root/'INTENT.json';receipt=root/'RECEIPT.json'
identity={k:p[k] for k in ('base','run_id','tag','through_epoch','source_commit','module_sha256')}
if intent.exists():
    if json.loads(intent.read_text())!=identity:raise ValueError('Existing analysis intent differs')
    if not receipt.exists():raise RuntimeError('Existing intent without receipt: inspect this namespace, do not replay')
    result=json.loads(receipt.read_text())
else:
    with intent.open('x') as f:json.dump(identity,f,sort_keys=True)
    code=root/'mixed_diagnostic.py'
    with code.open('x') as f:f.write(p['module_source'])
    if hashlib.sha256(code.read_bytes()).hexdigest()!=p['module_sha256']:raise ValueError('Analyzer code SHA differs')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('LD_LIBRARY_PATH',None)
    out=root/'results'
    cmd=['/dataset/shaoshitong/Search-G1-common8/env/bin/python','-u',str(code),'--base',str(base),'--out',str(out),'--through-epoch',str(p['through_epoch'])]
    proc=subprocess.run(cmd,env=env,cwd=root,capture_output=True,text=True,timeout=180)
    (root/'STDOUT.txt').write_text(proc.stdout);(root/'STDERR.txt').write_text(proc.stderr)
    result=dict(role='CPU_MIXED_G5_POSTHOC_ONCE_RECEIPT',**identity,returncode=proc.returncode,
                namespace=str(root),output=str(out),gpu_used=False,training_modified=False,
                full_predictions_transferred_to_Mac=False,stdout=proc.stdout,stderr=proc.stderr)
    with receipt.open('x') as f:json.dump(result,f,sort_keys=True,indent=2)
answer=dict(receipt=result,receipt_path=str(receipt),receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
if result['returncode']==0:
    out=Path(result['output'])
    for name,key in [('summary.json','summary'),('metrics.csv','csv')]:
        data=(out/name).read_bytes()
        if len(data)>2*1024*1024:raise ValueError('Small summary transfer limit exceeded')
        answer[key+'_text']=data.decode();answer[key+'_sha256']=hashlib.sha256(data).hexdigest()
print(json.dumps(answer,allow_nan=False))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',required=True)
    parser.add_argument('--through-epoch',required=True,type=int)
    parser.add_argument('--tag',default=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9-]{1,64}',args.tag):raise ValueError('Invalid namespace tag')
    work=Path(args.work).resolve();state=json.loads((work/'prefix20-online1000-state.json').read_text())
    repo=Path(__file__).resolve().parents[2]
    source=Path(__file__).with_name('mixed_diagnostic.py').read_bytes()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if subprocess.check_output(['git','show',commit+':reproduction/prefix20_online/mixed_diagnostic.py'],cwd=repo)!=source:
        raise ValueError('Analyzer must match committed source')
    relative='reproduction/prefix20_online/run_mixed_diagnostic_remote_once.py'
    if subprocess.check_output(['git','show',commit+':'+relative],cwd=repo)!=Path(__file__).read_bytes():
        raise ValueError('Control adapter must match committed source')
    payload=dict(base=state['base'],run_id=state['run_id'],tag=args.tag,through_epoch=args.through_epoch,
                 source_commit=commit,module_sha256=hashlib.sha256(source).hexdigest(),module_source=source.decode())
    script=REMOTE.replace('PAYLOAD',repr(payload),1)
    observed=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    transport=work/('prefix20-mixed-g5-'+args.tag+'-transport-'+observed+'.json')
    answer=json.loads(remote(script,transport,control_path=None,target='h100-8'))
    small=work/('prefix20-mixed-g5-'+args.tag+'-receipt-'+observed+'.json')
    small.write_text(json.dumps({k:v for k,v in answer.items() if not k.endswith('_text')},indent=2)+'\n')
    if answer['receipt']['returncode']:
        raise RuntimeError('CPU analysis failed; remote receipt retained: '+answer['receipt_path'])
    for key,suffix in [('summary','.json'),('csv','.csv')]:
        data=answer[key+'_text'].encode()
        if hashlib.sha256(data).hexdigest()!=answer[key+'_sha256']:raise ValueError('Summary readback SHA differs')
        path=work/('prefix20-mixed-g5-through%04d-'%args.through_epoch+args.tag+suffix)
        if path.exists() and path.read_bytes()!=data:raise ValueError('Existing local summary differs')
        if not path.exists():path.write_bytes(data)
        answer[key+'_local']=str(path)
    print(json.dumps({k:v for k,v in answer.items() if not k.endswith('_text')},ensure_ascii=True))


if __name__=='__main__':main()
