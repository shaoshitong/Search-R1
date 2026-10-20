from pathlib import Path
import json,time
b=Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020');v=dict(epoch=time.time(),files={},logs={})
for name in ['progress.json','failure.json','worker-exits.json','controller.json','children.json']+['worker-failure-%d.json'%i for i in range(8)]:
 p=b/name
 if p.exists():v['files'][name]=dict(mtime=p.stat().st_mtime,value=json.loads(p.read_text()))
for p in sorted((b/'logs/r6').glob('*.log')):
 with p.open('rb')as f:
  f.seek(max(0,p.stat().st_size-2500));tail=f.read().decode(errors='replace')
 v['logs'][p.name]=dict(bytes=p.stat().st_size,inode=p.stat().st_ino,mtime=p.stat().st_mtime,tail=tail)
print(json.dumps(v))
