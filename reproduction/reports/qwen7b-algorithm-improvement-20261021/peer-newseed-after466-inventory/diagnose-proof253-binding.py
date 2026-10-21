"""Only small metadata fields of the one failed pinned audit member."""
import importlib.util,json,pathlib,subprocess,shlex
HERE=pathlib.Path(__file__).resolve().parent;ref=json.loads((HERE/'actual-source-after466.json').read_bytes());bad=[r for r in ref['delta_batches_metadata']if r['existing_audit']['status']!='VERIFIED'][252]
worker='''import pathlib,json,sys,hashlib,time
r=json.loads(sys.stdin.buffer.read());B=pathlib.Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020');inp=pathlib.Path(r['input']);dest=B/'sampling'/inp.parent.name/'batches'/inp.stem;outer=dest/'outers/outer-000001';out=dict(input=r['input'],seal_expected_sha256=r['seal_sha256'],epoch=time.time(),files={})
for name,p in dict(seal=dest/'complete.json',config=dest/'config.json',scan=outer/'prefix/scan.json',prefix_complete=outer/'prefix/complete.json',prefix_rank=outer/'prefix/rank-0-complete.json',selection=outer/'probe-score/selection.json',full_complete=outer/'full/complete.json',full_rank=outer/'full/rank-0-complete.json').items():
 b=p.read_bytes();v=json.loads(b);out['files'][name]=dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),mtime=p.stat().st_mtime,selected={k:z for k,z in v.items()if k not in('groups','trajectory_ids','trajectories','questions','files','config','selection')})
 if name in('scan','selection'):out['files'][name]['keys']=list(v)
 if name in('prefix_complete','full_complete','full_rank'):out['files'][name]['receipt_refs']={k:z for k,z in v.items()if isinstance(z,dict)and'path'in z}
out['mismatches']=dict(prefix_scan=out['files']['full_rank']['selected'].get('prefix_scan_sha256')!=out['files']['scan']['sha256'],selection=out['files']['full_rank']['selected'].get('selection_sha256')!=out['files']['selection']['sha256']);print(json.dumps(out))'''
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl);r=subprocess.run(ctl.route('h100-10','python3 -c '+shlex.quote(worker)),input=json.dumps(bad).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=55)
if r.returncode:raise RuntimeError(r.stderr.decode()[-1200:])
x=json.loads(r.stdout.decode().splitlines()[-1]);(HERE/'actual-proof253-binding-diagnosis.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
