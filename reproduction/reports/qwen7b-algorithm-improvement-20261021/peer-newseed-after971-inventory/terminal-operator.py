"""Stage an existing stdlib terminal auditor and exact source-plan metadata."""
import argparse,base64,hashlib,json,pathlib,subprocess,sys,zlib
R=pathlib.Path(__file__).resolve().parents[4];HERE=pathlib.Path(__file__).parent
sys.path.insert(0,str(R))
from reproduction.operations.qwen7b_warmup import deploy_peer_frozen_delta_transfer as t
MOD='reproduction/operations/qwen7b_warmup/audit_peer_frozen_delta_import.py'
TARGET='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-after971-terminal-audit-v1'
INGEST='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed-ingest-audited138-after971-20261021-51057f37'

def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','run'),required=True);p.add_argument('--commit',required=True);p.add_argument('--control-root',type=pathlib.Path,required=True);a=p.parse_args()
    t.publication(R,a.commit,'codex/peer-donor-after971-20261021');t.configure(json.loads((HERE/'migration-config.json').read_text()),str((HERE/'migration-config.json').relative_to(R)))
    if a.action=='stage':
        code=t.blob(R,a.commit,MOD);plan=t.canonical(json.loads((HERE/'frozen-audited-delta-plan.json').read_text()))+b'\n'
        # Actual source uses UTF-8 canonical JSON, not ASCII escaping.
        plan=json.dumps(json.loads(plan),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\n'
        if len(plan)!=274282 or t.sha(plan)!='ecec9cbb76934cf300b532f985562338aaa74bfe4cd895d83a5d46abcd476d00':raise ValueError('Source frozen plan bytes differ')
        packet=dict(commit=a.commit,code=code.decode(),selection=plan.decode());wire=zlib.compress(json.dumps(packet).encode())
        for offset in range(0,len(wire),6000):
            part=wire[offset:offset+6000];v=dict(base=TARGET,offset=offset,part=base64.b64encode(part).decode(),sha=t.sha(part))
            script='v='+repr(v)+'\n'+r'''import pathlib,base64,hashlib,json,os
b=pathlib.Path(v['base']);b.mkdir(parents=True,exist_ok=True);p=b/'immutable-packet.upload';raw=base64.b64decode(v['part']);assert hashlib.sha256(raw).hexdigest()==v['sha'];n=p.stat().st_size if p.exists()else 0
if n==v['offset']:
 with p.open('ab')as f:f.write(raw);f.flush();os.fsync(f.fileno())
else:
 assert n>=v['offset']+len(raw)
 with p.open('rb')as f:f.seek(v['offset']);assert f.read(len(raw))==raw
print(json.dumps(dict(status='METADATA_CODE_CHUNK_ACCEPTED',offset=v['offset'])))'''
            t.hpc_call(a.control_root,script,'after971-audit-chunk')
        v=dict(base=TARGET,wire_bytes=len(wire),wire_sha=t.sha(wire),commit=a.commit)
        script='v='+repr(v)+'\n'+r'''import pathlib,json,hashlib,zlib,os
b=pathlib.Path(v['base']);wire=(b/'immutable-packet.upload').read_bytes();assert len(wire)==v['wire_bytes']and hashlib.sha256(wire).hexdigest()==v['wire_sha'];packet=json.loads(zlib.decompress(wire));assert packet['commit']==v['commit'];compile(packet['code'],'auditor.py','exec');refs={}
for name,key in(('auditor.py','code'),('frozen-selection.json','selection')):
 p=b/name;raw=packet[key].encode()
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
 refs[name]=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
out=dict(status='EXISTING_STDLIB_TERMINAL_AUDITOR_AND_SOURCE_PLAN_STAGED',source_commit=v['commit'],files=refs);p=b/'source-publication.json'
if p.exists():assert json.loads(p.read_bytes())==out
else:p.write_text(json.dumps(out,sort_keys=True)+'\n')
print(json.dumps(out))'''
    else:
        script='TARGET='+repr(TARGET)+'\nINGEST='+repr(INGEST)+'\n'+r'''import pathlib,json,hashlib,importlib.util,os,resource
b=pathlib.Path(TARGET);pub=json.loads((b/'source-publication.json').read_bytes())
for item in pub['files'].values():
 p=pathlib.Path(item['path']);raw=p.read_bytes();assert len(raw)==item['bytes']and hashlib.sha256(raw).hexdigest()==item['sha256']
if not(pathlib.Path(INGEST)/'complete.json').exists():print(json.dumps(dict(status='WAITING_FOR_ACTUAL_INGEST_COMPLETE',ingest=INGEST)));raise SystemExit(0)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(45,45))
s=importlib.util.spec_from_file_location('qualified_existing_terminal_auditor',b/'auditor.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
print(json.dumps(m.run(INGEST,b/'frozen-selection.json','ecec9cbb76934cf300b532f985562338aaa74bfe4cd895d83a5d46abcd476d00','7e0553541d532699ca5cf209ee80b7cde6cbbddbb6b747fd854c2c879a1ee964')))'''
    value=t.hpc_call(a.control_root,script,'after971-terminal-'+a.action);(HERE/('actual-terminal-'+a.action+'.json')).write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))

if __name__=='__main__':main()
