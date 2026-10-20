import json,pathlib,os,re,stat,time,uuid,hashlib,subprocess,collections
base=pathlib.Path('/data/user/jxu957/experiments');assert os.getuid()==21930
groups=[];deletes=[];kept=[]
def payload(root):
 rows=[]
 for d,ds,fs in os.walk(root,followlinks=False):
  ds[:]=[x for x in ds if not (pathlib.Path(d)/x).is_symlink()]
  for name in fs:
   p=pathlib.Path(d)/name
   if p.suffix not in ('.pt','.bin','.safetensors'):continue
   st=p.lstat();assert stat.S_ISREG(st.st_mode) and p.resolve()==p
   rows.append(dict(path=str(p),bytes=st.st_size,allocated_bytes=st.st_blocks*512,device=st.st_dev,inode=st.st_ino,nlink=st.st_nlink,mtime_ns=st.st_mtime_ns,uid=st.st_uid))
 return rows
def retain(name,points):
 complete=[]
 for step,p in points:
  files=payload(p)
  if not files:continue
  needed=([p/('rank-'+str(i)+'.pt') for i in range(4)]+[p/'driver.pt']) if name.startswith('DynRPO') else ([p/('resume-rank'+str(i)+'.pt') for i in range(4)]+[p/'driver-resume.pt',p/'resume-complete.json'])
  assert all(x.is_file() for x in needed) and any(x['path'].endswith('.safetensors') for x in files),str(p)
  complete.append((step,p,files))
 complete.sort(key=lambda x:x[0]);steps=sorted(set(x[0] for x in complete))[-2:]
 keep=[x for x in complete if x[0] in steps];drop=[x for x in complete if x[0] not in steps]
 if name.startswith('TOPD') and any(x[0]==200 for x in complete):assert 200 in steps
 for _,_,files in keep:kept.extend(files)
 for _,_,files in drop:deletes.extend(files)
 groups.append(dict(name=name,kept_steps=steps,kept_checkpoint_dirs=[str(x[1]) for x in keep],delete_steps=[x[0] for x in drop],delete_checkpoint_dirs=[str(x[1]) for x in drop],delete_tensor_files=sum(len(x[2]) for x in drop),delete_allocated_path_bytes=sum(y['allocated_bytes'] for x in drop for y in x[2])))
for name in ['dynrpo-prefix5-thread5-20261017-r2','dynrpo-prefix5-thread5-20261017-r3','dynrpo-prefix5-thread5-20261017-r4']:
 root=base/name/'actor/outers';points=[(int(re.match(r'outer-(\d+)',p.name).group(1)),p) for p in root.glob('outer-*') if p.is_dir() and not p.is_symlink()];retain('DynRPO '+name.rsplit('-',1)[-1],points)
for root in (base/'topd-20261008/runtime/logs').glob('topd-hpc3-*/evidence/checkpoints/actor'):
 points=[(int(p.name.split('_')[-1]),p) for p in root.glob('global_step_*') if p.is_dir() and not p.is_symlink()]
 if any(payload(p) for _,p in points):retain('TOPD '+root.parents[2].name,points)
protected={(x['device'],x['inode']) for x in kept};assert all((x['device'],x['inode']) not in protected for x in deletes);assert len({x['path'] for x in deletes})==len(deletes)
q=subprocess.run(['squeue','-h','-u','mou176','-o','%i|%T|%j'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);assert q.returncode==0
refs=[]
for line in q.stdout.splitlines():
 job=line.split('|')[0];r=subprocess.run(['scontrol','write','batch_script',job,'-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);assert r.returncode==0
 texts=[r.stdout]
 for name in re.findall(r'/data/user/[^\s\"\x27<>]+',r.stdout):
  p=pathlib.Path(name.rstrip(';,)]'))
  if p.suffix in ('.json','.yaml','.yml') and p.is_file() and p.stat().st_size<=262144:texts.append(p.read_text(errors='replace'))
 for x in deletes:
  if any(x['path'] in text or str(pathlib.Path(x['path']).parent) in text for text in texts):refs.append(dict(job=job,path=x['path']))
assert not refs,refs
links=collections.defaultdict(list)
for x in deletes:links[(x['device'],x['inode'])].append(x)
free=sum(v[0]['allocated_bytes'] for v in links.values() if len(v)==v[0]['nlink'])
root=pathlib.Path('/tmp')/('easy-authorized-latest2-'+uuid.uuid4().hex);root.mkdir(mode=0o700)
plan=dict(schema_version=1,created_unix=time.time(),plan_root=str(root),user_authorization='用户2026-10-20明确：TOPD保留200，DynRPO r2/r3/r4、CoSearch只保留最新两个，删除更早checkpoint',scope='HPC3 checkpoint tensor payloads only, latest two distinct complete steps per independent experiment',groups=groups,delete_files=deletes,preserved_payload_members=kept,expected_reclaimable_allocated_bytes=free,logical_unlink_path_allocated_bytes=sum(x['allocated_bytes'] for x in deletes),active_mou_queue=q.stdout,active_retired_checkpoint_references=refs,source_code_or_logs_deleted=False,Easy_or_q7b_modified=False,remote_archives_modified=False,execution_performed=False)
raw=(json.dumps(plan,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
with (root/'plan.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(plan_root=str(root),plan_sha256=hashlib.sha256(raw).hexdigest(),plan=plan)))
