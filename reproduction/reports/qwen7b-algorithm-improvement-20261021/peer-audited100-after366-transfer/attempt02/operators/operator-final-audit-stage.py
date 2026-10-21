v={'root': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-delta-import-runtime-audited100-after366-20261021-636015f5-a02/independent-audit-e21e61e8fbfa', 'refs': {'audit.py': {'bytes': 8820, 'sha256': 'f5b43eb257050642d1f15548d25f5ea14a29a7fb6d1cfedc3b93bd9a06b0a1e2'}, 'selection.json': {'bytes': 170253, 'sha256': '00e5021284cb8f6614670452b749c7add10c028e50578822432600762074662a'}}, 'commit': 'e21e61e8fbfac1674b6e7c04bbb5218107577bd5', 'ingest': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed-ingest-audited100-after366-20261021-636015f5-a02', 'selection_sha256': '00e5021284cb8f6614670452b749c7add10c028e50578822432600762074662a', 'record_sha256': '636015f5b3cf84ad49688a1cee6b8d15959ccbc59451e91f6dae5d625df103ce'}

import base64,gzip,hashlib,json,pathlib,os,subprocess,sys,time
b=pathlib.Path(v['root']);b.mkdir(parents=True,exist_ok=True)
def fp(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def atomic(p,val):
 q=p.with_name(p.name+'.tmp-'+str(os.getpid()));q.write_text(json.dumps(val,sort_keys=True));q.replace(p)
bundle_sha='d2d9f130fc6fd11bd10680ad0463899c6751efb1e0ccdb581c1b0f5ce3c64b63'

raw=(b/'stage.gz.upload').read_bytes();assert hashlib.sha256(raw).hexdigest()==bundle_sha;values=json.loads(gzip.decompress(raw));assert set(values)==set(v['refs'])
for name,encoded in values.items():
 data=base64.b64decode(encoded);item=v['refs'][name];assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'];p=b/name
 if p.exists():assert p.read_bytes()==data
 else:p.write_bytes(data);p.chmod(0o444)
pub=dict(status='PUBLISHED_DELTA_IMPORT_INDEPENDENT_AUDIT_STAGED',commit=v['commit'],root=str(b),files={name:fp(b/name) for name in v['refs']});atomic(b/'source-publication.json',pub);print(json.dumps(pub))
