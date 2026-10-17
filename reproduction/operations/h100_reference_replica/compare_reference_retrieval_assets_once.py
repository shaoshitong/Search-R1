from pathlib import Path
import json,importlib.util,subprocess
ROOT=Path(__file__).resolve().parent
expected=json.loads((ROOT/'actorhidden-fixed-recovery-assets-independent-complete-small-summary.json').read_text())['file_sha_records']
rows=[]
for r in expected:
 p=r['original_path']
 if '/e5-base-v2/'in p:source='/data/search-r1/assets/e5-base-v2/'+p.split('/e5-base-v2/')[1]
 elif p.endswith('/e5_Flat.index'):source='/data/search-r1/assets/wiki/e5_Flat.index'
 elif p.endswith('/wiki-18.jsonl'):source='/data/search-r1/assets/wiki/wiki-18.jsonl'
 else:continue
 rows.append({'source':source,'h100_path':p,'bytes':r['bytes'],'expected_h100_full_sha256':r['source_full_sha256']})
spec=importlib.util.spec_from_file_location('route','/Users/shaoshitong/Documents/Search-G1/actorhidden-prefix5-prune-aware-resume-20261017/reproduction/operations/a800_actorhidden_prefix5_control.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
code='ROWS='+repr(rows)+'\n'+r'''
import pathlib,hashlib,json,time
p=pathlib.Path('/data/shaoshitong/reference-prefix5-assets-transfer-model-only-20261018/retrieval-h100-comparison-once.json')
if p.exists():print(p.read_text());raise SystemExit
out=[]
for r in ROWS:
 f=pathlib.Path(r['source']);before=f.stat();h=hashlib.sha256()
 with f.open('rb')as stream:
  for chunk in iter(lambda:stream.read(32*1024**2),b''):h.update(chunk)
 after=f.stat();assert(before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 out.append(dict(r,source_full_sha256=h.hexdigest(),source_bytes=after.st_size,match=h.hexdigest()==r['expected_h100_full_sha256']and after.st_size==r['bytes']))
v={'files':out,'all_same':all(r['match']for r in out),'reference_current_source_full_reads':True,'h100_existing_independent_full_sha_proof_reused':True,'gpu_used':False,'finished_unix':time.time()}
with p.open('x')as f:json.dump(v,f,indent=2)
print(json.dumps(v))
'''
r=subprocess.run(m.route(code),capture_output=True,timeout=240);p=ROOT/'prefix5-reference-retrieval-assets-comparison.json';v={'returncode':r.returncode,'stdout':r.stdout.decode(),'stderr':r.stderr.decode()};p.write_text(json.dumps(v,indent=2)+'\n');assert r.returncode==0,v['stderr'];v=json.loads(v['stdout']);print(json.dumps({'all_same':v['all_same'],'files':len(v['files']),'bytes':sum(x['source_bytes']for x in v['files'])}))
