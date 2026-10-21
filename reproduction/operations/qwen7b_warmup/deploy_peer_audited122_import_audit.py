"""Stage/run the published terminal subset identity audit in an isolated root."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from . import deploy_peer_audited122_transfer as t


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--action',choices=('stage','run'),required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[3];t.publication(repo,a.commit,'codex/peer-audited122-delta-transfer-20261021')
    names={'audit.py':t.P+'audit_peer_audited122_import.py','selection.json':t.PLAN};files={name:t.blob(repo,a.commit,source) for name,source in names.items()}
    refs={name:dict(bytes=len(data),sha256=t.sha(data)) for name,data in files.items()};root=str(t.HPCDEP/('independent-audit-'+a.commit[:12]))
    def call(script,label):return t.hpc_call(a.control_root,script,label)
    common='v='+repr(dict(root=root,refs=refs,commit=a.commit,ingest=str(t.INGEST)))+'\n'+r'''
import base64,gzip,hashlib,json,pathlib,os,subprocess,sys,time
b=pathlib.Path(v['root']);b.mkdir(parents=True,exist_ok=True)
def fp(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def atomic(p,val):
 q=p.with_name(p.name+'.tmp-'+str(os.getpid()));q.write_text(json.dumps(val,sort_keys=True));q.replace(p)
'''
    if a.action=='stage':
        raw=gzip.compress(json.dumps({name:base64.b64encode(data).decode() for name,data in files.items()},sort_keys=True).encode(),mtime=0)
        for i in range(0,len(raw),6000):
            part=raw[i:i+6000];script=common+'part='+repr(dict(offset=i,data=base64.b64encode(part).decode(),sha=t.sha(part)))+'\n'+r'''
p=b/'stage.gz.upload';data=base64.b64decode(part['data']);assert hashlib.sha256(data).hexdigest()==part['sha'];size=p.stat().st_size if p.exists() else 0
if size==part['offset']:
 with p.open('ab') as f:f.write(data);f.flush();os.fsync(f.fileno())
else:
 assert size>=part['offset']+len(data)
 with p.open('rb') as f:f.seek(part['offset']);assert f.read(len(data))==data
print(json.dumps(dict(status='PUBLISHED_AUDIT_METADATA_CHUNK_ACCEPTED',offset=part['offset'])))
''';call(script,'final-audit-stage-chunk')
        result=call(common+'bundle_sha='+repr(t.sha(raw))+'\n'+r'''
raw=(b/'stage.gz.upload').read_bytes();assert hashlib.sha256(raw).hexdigest()==bundle_sha;values=json.loads(gzip.decompress(raw));assert set(values)==set(v['refs'])
for name,encoded in values.items():
 data=base64.b64decode(encoded);item=v['refs'][name];assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'];p=b/name
 if p.exists():assert p.read_bytes()==data
 else:p.write_bytes(data);p.chmod(0o444)
pub=dict(status='PUBLISHED_DELTA_IMPORT_INDEPENDENT_AUDIT_STAGED',commit=v['commit'],root=str(b),files={name:fp(b/name) for name in v['refs']});atomic(b/'source-publication.json',pub);print(json.dumps(pub))
''','final-audit-stage')
    else:
        result=call(common+r'''
for name,item in v['refs'].items():assert fp(b/name)==dict(item,path=str(b/name))
assert (pathlib.Path(v['ingest'])/'complete.json').exists(),'No actual strictEM ingest completion yet'
start=time.time();args=[sys.executable,str(b/'audit.py'),'--selection',str(b/'selection.json'),'--ingest',v['ingest']]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,universal_newlines=True,timeout=40)
(b/'audit.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-1800:];val=json.loads([x for x in r.stdout.splitlines() if x.startswith('{')][-1]);receipt=fp(pathlib.Path(v['ingest'])/'independent-final-audit.json')
answer=dict(status='ACTUAL_INDEPENDENT_DELTA_IMPORT_AUDIT_EXITED_ZERO',exit_code=r.returncode,elapsed_seconds=time.time()-start,audit=receipt,summary=val,source_publication=fp(b/'source-publication.json'));atomic(b/'terminal.json',answer);print(json.dumps(answer))
''','final-audit-run')
    out=a.control_root/'outputs/qwen7b-warmup-20261019/peer-audited122-transfer';out.mkdir(parents=True,exist_ok=True);(out/('final-audit-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
