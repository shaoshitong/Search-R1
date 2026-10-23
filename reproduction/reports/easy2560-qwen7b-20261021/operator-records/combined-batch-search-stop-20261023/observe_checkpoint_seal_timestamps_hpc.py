"""Read saved ordinary-checkpoint manifest timestamps only; no tensor reads."""
import json,time,hashlib
from pathlib import Path
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-batch-searchstop-r1-20261023')
rows=[]
for step in (101,102,103):
 path=base/'preparation/rolling-checkpoints'/('global_step_'+str(step))/'full-manifest.json'
 raw=path.read_bytes();m=json.loads(raw)
 files=m['files'];boundary=m['boundary_files']
 def last(values):
  return max((r['mtime_ns']/1e9 for r in values.values()),default=None)
 a=last(files);b=last(boundary);latest=max(x for x in (a,b) if x is not None)
 rows.append(dict(step=step,manifest_path=str(path),manifest_sha256=hashlib.sha256(raw).hexdigest(),status=m['status'],source_commit=m['source_commit'],checkpoint_file_count=len(files),checkpoint_bytes=sum(r['bytes'] for r in files.values()),boundary_file_count=len(boundary),checkpoint_last_recorded_mtime=a,boundary_last_recorded_mtime=b,sealed_unix=m['sealed_unix'],after_latest_payload_mtime_to_seal_seconds=m['sealed_unix']-latest,payload_paths_reopened=False,timestamps_from_saved_seal_manifest=True))
print(json.dumps(dict(observed_unix=time.time(),rows=rows,tensor_files_read=False,model_loaded=False,gpu_forward_triggered=False,training_modified=False,scope='Saved seal metadata, not a fresh payload integrity audit or an independently timed SHA/fsync profiler.')))
