from pathlib import Path
import json,subprocess,hashlib,sys,base64,datetime
R=Path('/Users/shaoshitong/Documents/Search-G1/h100-prefix5-reference-replica-20261018');previous=json.loads(Path('work/prefix5-reference-code-stage.json').read_text());commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
names=subprocess.check_output(['git','ls-tree','-rz',commit],cwd=R).split(b'\0');manifest={};changed={}
for row in names:
 if not row:continue
 meta,n=row.split(b'\t',1);name=n.decode();data=subprocess.check_output(['git','cat-file','blob',meta.split()[2].decode()],cwd=R);manifest[name]=hashlib.sha256(data).hexdigest()
 old=subprocess.run(['git','show',previous['commit']+':'+name],cwd=R,capture_output=True)
 if old.returncode or old.stdout!=data:changed[name]=base64.b64encode(data).decode()
P={'commit':commit,'previous':previous,'manifest':manifest,'changed':changed}
script='P='+repr(P)+'\n'+r'''
import pathlib,json,hashlib,os,base64,time
root=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix5-reference-preparation-20261018');d=root/'code'/P['commit'];repo=d/'repo';proof=d/'source-staged.json'
if proof.exists():print(proof.read_text());raise SystemExit
repo.mkdir(parents=True,exist_ok=False);old=pathlib.Path(P['previous']['repo'])
for name,h in P['manifest'].items():
 t=repo/name;t.parent.mkdir(parents=True,exist_ok=True)
 if name in P['changed']:
  raw=base64.b64decode(P['changed'][name]);assert hashlib.sha256(raw).hexdigest()==h;t.write_bytes(raw)
 else:
  p=old/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h;os.link(p,t)
assert {str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()for p in repo.rglob('*')if p.is_file()}==P['manifest']
v={'status':'EXACT_REFERENCE_REPLICA_CODE_STAGED_FULL_SHA_PASS','repo':str(repo),'commit':P['commit'],'files':len(P['manifest']),'all_target_source_sha_match':True,'infrastructure':str(root/'infrastructure.json'),'training_submitted':False,'checked_unix':time.time(),'delta_files':len(P['changed'])};proof.write_text(json.dumps(v,indent=2));print(json.dumps(v))
'''
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016');from reproduction.h1008_qprobe.established_controller_transport import remote
tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');v=json.loads(remote(script,'work/prefix5-reference-code-delta-stage-'+tag+'.json'));Path('work/prefix5-reference-code-stage.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
