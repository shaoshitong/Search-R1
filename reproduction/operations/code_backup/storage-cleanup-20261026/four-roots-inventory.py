import os,json,stat,time,concurrent.futures,pathlib
R='/data/user/jxu957/experiments/'
A='/data/user/jxu957/storage-cleanup-audit-20261026'
roots=[R+n for n in ['topd-20261008','cosearch-continue300-x203-20261008','easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022','dynrpo7b-formal-20261023-recoveryfix-r6']]
new=R+'dynrpo7b-formal-20261025-cp50-mathfix-r1';retr=R+'dynrpo7b-formal-20261023'
links=[]; configs={}
for base in [new,retr]:
 for sub in ['env','preparation/flash3-runtime','bootstrap-tools','data','assets']:
  for d,ds,fs in os.walk(base+'/'+sub,followlinks=False):
   for n in ds+fs:
    p=d+'/'+n
    if os.path.islink(p):
     dest=os.path.realpath(p)
     if any(dest.startswith(r+'/') for r in roots):links.append([p,dest])
   for n in fs:
    if n.endswith('.pth') or n=='pyvenv.cfg':
     p=d+'/'+n
     if os.path.getsize(p)<16384:configs[p]=pathlib.Path(p).read_text(errors='replace')
 for p in pathlib.Path(base+'/preparation').glob('*.json'):
  if p.stat().st_size<30000:configs[str(p)]=p.read_text()
entries=[];errors=[]
targets=[]
for root in roots:
 for x in pathlib.Path(root).iterdir():
  if x.is_dir() and not x.is_symlink():targets.append(str(x))
def walk(target):
 rows=[];err=[]
 for d,ds,fs in os.walk(target,followlinks=False,onerror=lambda e:err.append(str(e))):
  ds[:]=[n for n in ds if n!='.git' and not os.path.islink(d+'/'+n)]
  for n in fs:
   p=d+'/'+n
   try:
    s=os.lstat(p)
    if stat.S_ISREG(s.st_mode) and s.st_size>=1048576:
     rows.append({'path':p,'size':s.st_size,'blocks':s.st_blocks*512,'inode':[s.st_dev,s.st_ino],'nlink':s.st_nlink,'mtime':s.st_mtime})
   except OSError as e:err.append(str(e))
 return rows,err
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for rows,err in pool.map(walk,targets):entries.extend(rows);errors.extend(err)
out={'time':time.time(),'roots':roots,'entries':entries,'errors':errors,'current_dependency_links':links,'current_configs':configs}
pathlib.Path(A+'/four-roots-inventory.json').write_text(json.dumps(out))
print(json.dumps({'finished':True,'files':len(entries),'errors':len(errors)}))
