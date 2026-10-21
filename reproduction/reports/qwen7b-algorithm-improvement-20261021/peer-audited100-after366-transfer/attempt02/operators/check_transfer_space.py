import subprocess,json,pathlib,time
queries=[('hkust2','/data/shaoshitong/qwen7b-peer-newseed-to-hpc3-20261021'),('hkust3','/backup/shaoshitong/qwen7b-peer-newseed-to-hpc3-20261021')]
out=[]
for host,target in queries:
 script='import pathlib,shutil,json,os,subprocess; p=pathlib.Path('+repr(target)+'); p.mkdir(parents=True,exist_ok=True);q=p/(".audited100-space-probe-%d"%os.getpid());q.write_bytes(b"ok");assert q.read_bytes()==b"ok";q.unlink();print(json.dumps(dict(host='+repr(host)+',path=str(p),free_bytes=shutil.disk_usage(p).free,mount=subprocess.check_output(["findmnt","-n","-o","SOURCE,FSTYPE,TARGET","-T",str(p)],universal_newlines=True).strip(),write_probe=True)))'
 r=subprocess.run(['ssh','-T','-oBatchMode=yes','-oConnectTimeout=8',host,'python3 -'],input=script,text=True,capture_output=True,timeout=25)
 assert r.returncode==0,r.stderr[-300:];out.append(json.loads(r.stdout));assert out[-1]['free_bytes']>5*1024**3
v=dict(status='ACTUAL_NEW_DELTA_CAMPUS_STAGING_SPACE_AND_WRITE_PROBES_PASSED',epoch=time.time(),targets=out);p=pathlib.Path('outputs/qwen7b-warmup-20261019/peer-frozen-delta-transfer/audited100-after366-20261021-636015f5-a02/campus-storage-preflight.json');p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps(v))
