"""Read actual intermediate archive receipts; no payload read or new writer."""
import json,pathlib,shlex,subprocess
HERE=pathlib.Path(__file__).resolve().parent
SNAPSHOT='audited138-after971-20261021-51057f37'
source=json.loads((HERE/'actual-transfer-source-status.json').read_text())
expected={v['group']:dict(bytes=v['archive_bytes'],sha256=v['archive_sha256'])for v in source['archives']}
worker=r'''import pathlib,json,hashlib,socket,sys,time,os
p=json.loads(sys.stdin.buffer.read());b=pathlib.Path(p['base']);expected=p['expected'];receipts=[];total=0
for group,want in sorted(expected.items()):
 path=b/p['folder']/(group+'.json');raw=path.read_bytes();v=json.loads(raw)
 assert v['status']==p['status']and v['bytes']==want['bytes']and v['sha256']==want['sha256'];archive=b/'archives'/('%s-%s.tar'%(b.name,group));assert archive.is_file()and not archive.is_symlink()and archive.stat().st_size==want['bytes'];total+=want['bytes']
 receipts.append(dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),archive_bytes=want['bytes'],archive_sha256=want['sha256']))
complete=b/'complete.json';raw=complete.read_bytes();c=json.loads(raw);assert c['groups']==9 and c['archive_bytes']==984002560 and c['fresh_all_file_SHA256']is True
print(json.dumps(dict(status='ACTUAL_REMOTE_HOP_ALL_NINE_SOURCE_SHA_BOUND_ARCHIVE_RECEIPTS_PASSED',host=socket.gethostname(),base=str(b),resolved=str(b.resolve()),device=b.stat().st_dev,groups=len(receipts),archive_bytes=total,receipts=receipts,complete=dict(path=str(complete),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),payload_SHA_recomputed_in_this_metadata_read=False,fresh_payload_SHA_inherited_from_actual_receiver=True,writer_started=False,Mac_payload_transit=False,epoch=time.time())))'''
for host,base,folder,status in [('hkust2','/data/shaoshitong/qwen7b-peer-newseed-to-hpc3-20261021','download-accepted','SOURCE_HF_HKUST2_ARCHIVE_SHA_VERIFIED'),('hkust3','/backup/shaoshitong/qwen7b-peer-newseed-to-hpc3-20261021','accepted','HKUST3_ARCHIVE_FRESH_SHA_VERIFIED')]:
    value=dict(base=base+'/'+SNAPSHOT,folder=folder,status=status,expected=expected)
    r=subprocess.run(['ssh',host,'python3 -c '+shlex.quote(worker)],input=json.dumps(value).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=40)
    if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-1200:])
    result=json.loads(r.stdout.decode().splitlines()[-1]);(HERE/('actual-'+host+'-all-receipts.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='receipts'}))
