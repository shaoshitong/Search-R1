from pathlib import Path
import json,subprocess,importlib.util,datetime,hashlib
spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
a=json.loads(Path('work/prefix5-ab-assigned-arms.json').read_text())
script=r'''import json,os,time,hashlib
from pathlib import Path
ARMS=__ARMS__
ret=Path('/mnt/shared/workspace/training_data_node_10/shaoshitong');src=Path('/mnt/shared/workspace/training_data_node_8/shaoshitong')
backup=ret/'search-g1-prefix-selection200-independent-backup-20261016/a008d7ce9833-fixed-recovery-assets-once-v1'
r={'checked_unix':time.time(),'mounts':[l for l in Path('/proc/self/mountinfo').read_text().splitlines()if '/mnt/shared' in l], 'configs':{},'backup_metadata':{},'source_inventory':[]}
for s,a in ARMS.items():
 b=Path(a['base'].replace('/mnt/data/sihe-share5','/mnt/shared'));c=json.loads((b/'preparation/launch-config.json').read_text());r['configs'][s]={k:v for k,v in c.items()if isinstance(v,str)and (v.startswith('/mnt')or v.startswith('/dataset'))}
for name in ['RESULT.json','checkpoint/RESTORE_MAPPING.json']:
 p=backup/name
 if p.exists():
  raw=p.read_bytes();r['backup_metadata'][name]={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'data':json.loads(raw)}
for role,base in [('bundle',src/'search-g1-prefix5-reference-assets-20261018-model-only'),('cache',src/'search-g1-prefix20-online1000-20261016-ca1c60f/retrieval-cache')]:
 for parent,dirs,names in os.walk(base):
  for name in names:
   p=Path(parent)/name;st=p.stat();r['source_inventory'].append(dict(role=role,path=str(p),relative=str(p.relative_to(base)),bytes=st.st_size,device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns,symlink=p.is_symlink()))
st=os.statvfs(ret);r['destination_capacity']={'available':st.f_bavail*st.f_frsize,'physical_free':st.f_bfree*st.f_frsize}
print(json.dumps(r))
'''.replace('__ARMS__',repr(a))
p=subprocess.run(ctl.route('h100-11','/usr/bin/python3 -'),input=script,text=True,capture_output=True,timeout=90);assert p.returncode==0,p.stderr;d=json.loads(p.stdout);tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=Path('work')/('prefix5-h10011-retained-assets-inspect-'+tag+'.json');out.write_text(json.dumps(d,indent=2)+'\n');print('actual',str(out),'sha256',hashlib.sha256(out.read_bytes()).hexdigest());print('capacity',d['destination_capacity']);print('configs',json.dumps(d['configs']));print('backup',json.dumps(d['backup_metadata'])[:18000]);print('inventory',json.dumps(d['source_inventory']))
