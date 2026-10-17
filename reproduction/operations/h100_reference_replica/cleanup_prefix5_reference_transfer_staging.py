"""Remove only successful-task temporary transfer payloads, retain all proof."""
from pathlib import Path
import json,subprocess,importlib.util,sys,datetime,shlex
from reference_prefix5_remote_transfer_route import controller
ROOT=Path(__file__).resolve().parent
prepared=json.loads((ROOT/'prefix5-reference-prepared-small-summary.json').read_text());assert prepared['actual_target_EMA_tensor_and_buffer_hashes_compared']and not prepared['training_submitted']
code=r'''
import pathlib,json,time,os
b=pathlib.Path('/tmp/prefix5-reference-parallel-assets-20261018');proof=b/'staging-cleanup.json'
if proof.exists():print(proof.read_text());raise SystemExit
deleted=[]
for p in [b/('part-%02d'%i)for i in range(16)]+[b/'reference-assets.tar.gz']:
 if p.exists():
  assert p.is_file()and not p.is_symlink()and p.stat().st_uid==os.getuid();deleted.append({'path':str(p),'bytes':p.stat().st_size});p.unlink()
v={'role':'only_successful_reference_transfer_temporary_payload_cleanup','target_full_SHA_and_actual_EMA_tensor_compare_verified':True,'deleted':deleted,'weights_transit_Mac':False,'time':time.time()};proof.write_text(json.dumps(v,indent=2));print(json.dumps(v))
'''
r=subprocess.run(controller().ssh('mgmt-host')+['python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr;results={'management':json.loads(r.stdout)}
spec=importlib.util.spec_from_file_location('source_route','/Users/shaoshitong/Documents/Search-G1/actorhidden-prefix5-prune-aware-resume-20261017/reproduction/operations/a800_actorhidden_prefix5_control.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
code=r'''
import pathlib,json,time,os
items=[pathlib.Path('/data/shaoshitong/reference-prefix5-assets-transfer-20261018/reference-assets.tar.gz'),pathlib.Path('/data/shaoshitong/reference-prefix5-assets-transfer-model-only-20261018/reference-assets-model-only.tar.gz'),pathlib.Path('/data/shaoshitong/reference-prefix5-assets-transfer-model-only-20261018/warmup-epoch5-EMA-model-only.pt')]
p=items[-1].parent/'successful-staging-cleanup.json'
if p.exists():print(p.read_text());raise SystemExit
deleted=[]
for f in items:
 if f.exists():
  assert f.is_file()and not f.is_symlink()and f.stat().st_uid==os.getuid();deleted.append({'path':str(f),'bytes':f.stat().st_size});f.unlink()
v={'role':'only_task_created_pack_export_cleanup','original_warmup_b5_preserved':True,'reference_experiment_untouched':True,'deleted':deleted,'time':time.time()};p.write_text(json.dumps(v,indent=2));print(json.dumps(v))
'''
r=subprocess.run(m.route(code),capture_output=True,text=True,timeout=65);assert r.returncode==0,r.stderr;results['source']=json.loads(r.stdout)
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016');from reproduction.h1008_qprobe.established_controller_transport import remote
code=r'''
import pathlib,json,time
b=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix5-reference-assets-20261018');f=b/'probe_source_checkpoint.pt.partial';rows=[]
if f.exists():rows.append({'path':str(f),'bytes':f.stat().st_size});f.unlink()
v={'role':'only_interrupted_task_copy_partial_cleanup_after_verified_EMA_copy','deleted':rows,'actual_prepared_assets_preserved':True,'time':time.time()};p=b/'superseded-partial-cleanup.json'
if not p.exists():p.write_text(json.dumps(v,indent=2))
print(p.read_text())
'''
tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');results['target_superseded_partial']=json.loads(remote(code,str(ROOT/('prefix5-reference-partial-cleanup-'+tag+'.transport.json'))));p=ROOT/'prefix5-reference-transfer-staging-cleanup.json';p.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({'actual':str(p),'source_original_warmup_preserved':True,'prepared_assets_preserved':True,'management_bytes_removed':sum(x['bytes']for x in results['management']['deleted'])}))
