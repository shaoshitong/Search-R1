import sys,json,datetime,hashlib
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016')
from reproduction.h1008_qprobe.established_controller_transport import remote
base='/mnt/data/shaoshitong/search-g1-prefix-selection200-probe20k-dataset-20261017-0bdddc20840b-r48'
script=r'''
import pathlib,json,subprocess,hashlib,os,time
b=pathlib.Path(BASE);logs=[]
for root in (b/'logs',b/'outers/outer-000053/prefix'):
 for p in root.rglob('*'):
  if not p.is_file()or p.suffix not in('.log','.out'):continue
  stat=p.stat();digest=hashlib.sha256()
  with p.open('rb')as f:
   while True:
    block=f.read(8*1024*1024)
    if not block:break
    digest.update(block)
  with p.open('rb')as f:f.seek(max(0,stat.st_size-20000));tail=f.read().decode(errors='replace')
  logs.append(dict(path=str(p),bytes=stat.st_size,inode=stat.st_ino,device=stat.st_dev,sha256=digest.hexdigest(),tail=tail))
native=subprocess.run(['scontrol','show','job','3468','-o'],capture_output=True,text=True).stdout
sys=__import__('sys');sys.path.insert(0,str(b/'repo'))
from reproduction.prefix_only_correctness_ab.launch import birth_states
ledger=json.loads((b/'logs/process-ledger.json').read_text());births=[v for j in ledger.values()if isinstance(j,dict)for v in j.get('ledger',{}).values()]
value=dict(native=native,logs=logs,cleanup=json.loads((b/'logs/cleanup-status.json').read_text()),remaining_original_births=birth_states(births),checked_unix=time.time(),gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True,capture_output=True).stdout)
p=b/'preparation/terminal3468-original-inodes-once-v1.json'
if not p.exists():p.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
'''.replace('BASE',repr(base))
tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');p=Path('work')/('probe20k3468-terminal-error-'+tag+'.json');out=json.loads(remote(script,str(p.with_name(p.stem+'.transport.json'))));p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(actual=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),native=out['native'],cleanup=out['cleanup'],remaining_original_births=out['remaining_original_births'],logs=[dict(path=v['path'],tail=v['tail'][-9000:])for v in out['logs']if'53'in v['path']])))
