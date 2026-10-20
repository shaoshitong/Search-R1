"""Independent read-only full remote archive/member SHA audit; aggregate output only."""
import importlib.util,json,pathlib,shlex,subprocess
CONTROL=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
CODE=r'''
import hashlib,json,pathlib,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/peer-newseed-h100-20261021/newseed-20261021-185747-445fc691')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while True:
   data=f.read(1048576)
   if not data:break
   h.update(data)
 return h.hexdigest()
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def verify(p,size,digest):
 assert p.is_file() and not p.is_symlink() and p.stat().st_size==size
 assert not any(q.is_symlink() for q in p.parents)
 assert sha(p)==digest
started=time.time();cp=b/'complete.json';complete=json.loads(cp.read_bytes());catalog=json.loads((b/'catalog.json').read_bytes());binding=json.loads((b/'catalog-binding.json').read_bytes())
assert complete['status']=='PEER_NEWSEED_RAW_ALL_ARCHIVES_AND_FILES_FRESH_SHA256_VERIFIED'
assert binding['catalog_commit']==complete['catalog_commit']=='55923224d54e209aedc40505ceb175c8a7ff1276'
assert hashlib.sha256(binding['catalog_bytes'].encode()).hexdigest()==binding['catalog_sha256']==complete['catalog_sha256']
assert json.loads(binding['catalog_bytes'])==catalog
seen=set();archive_bytes=payload_bytes=0
for record in catalog['groups']:
 name=pathlib.PurePosixPath(record['archive']).name
 verify(b/'archives'/name,record['archive_bytes'],record['archive_sha256']);archive_bytes+=record['archive_bytes']
 accepted=json.loads((b/'accepted'/(record['group']+'.json')).read_bytes())
 assert accepted['status']=='TARGET_ARCHIVE_AND_ALL_MEMBER_FILES_FRESH_SHA_VERIFIED' and accepted['catalog_sha256']==binding['catalog_sha256']
 for item in record['files']:
  part=pathlib.PurePosixPath(item['path']);assert not part.is_absolute() and '..' not in part.parts and item['path'] not in seen
  seen.add(item['path']);verify(b/'raw'/item['path'],item['bytes'],item['sha256']);payload_bytes+=item['bytes']
assert archive_bytes==complete['archive_bytes']==1366896640
assert len(seen)==complete['files'] and len(catalog['groups'])==complete['groups']==16
assert complete['questions']*5==complete['trajectories']==catalog['source_snapshot']['trajectories']
assert complete['actor_updates']==complete['probe_updates']==0 and complete['feature_bank_created'] is False and complete['Mac_data_transit'] is False
print(json.dumps(dict(status='INDEPENDENT_FRESH_ALL_TARGET_ARCHIVES_AND_MEMBER_SHA256_PASSED',complete=ref(cp),catalog=ref(b/'catalog.json'),catalog_binding=ref(b/'catalog-binding.json'),catalog_commit=binding['catalog_commit'],original_HF_catalog_sha256=binding['catalog_sha256'],raw=str(b/'raw'),groups=len(catalog['groups']),files=len(seen),archive_bytes=archive_bytes,payload_bytes=payload_bytes,questions=complete['questions'],trajectories=complete['trajectories'],actor_updates=0,probe_updates=0,feature_bank_created=False,Mac_payload_transit=False,started_epoch=started,finished_epoch=time.time(),elapsed_seconds=time.time()-started)))
'''
p=CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py'
spec=importlib.util.spec_from_file_location('peer_receive_final_audit_control',str(p));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
result=m.run('python3 -c '+shlex.quote(CODE),account='mou176',source_address=ip,timeout=300)
if result.returncode:raise RuntimeError('Independent target readback exited unsuccessfully')
value=json.loads(result.stdout);path=pathlib.Path(__file__).with_name('peer-newseed-independent-final-fresh-SHA.json');path.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
