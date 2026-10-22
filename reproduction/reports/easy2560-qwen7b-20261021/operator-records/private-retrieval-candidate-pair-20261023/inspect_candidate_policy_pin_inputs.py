import json,time
from pathlib import Path
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-leanmetrics-r1-20261023')
p=base/'preparation/rolling-checkpoints/latest.json'
v=json.loads(p.read_text());cp=Path(v['checkpoint']);manifest=json.loads(Path(v['manifest_path']).read_text())
print(json.dumps(dict(epoch=time.time(),latest=v,manifest_keys=list(manifest),checkpoint_files=[dict(name=x.name,bytes=x.stat().st_size) for x in cp.iterdir() if x.is_file()],manifest_files_type=type(manifest.get('files')).__name__,manifest_first_record=next(iter(manifest.get('files',{}).items()),None) if isinstance(manifest.get('files'),dict) else manifest.get('files',[])[:1])))
