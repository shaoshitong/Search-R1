"""Bounded read-only observer of the existing single uploader; starts no jobs."""
import json, pathlib, subprocess, sys, time, fcntl
D=pathlib.Path('outputs/grounding-eval-20261019');lock=(D/'.easy-upload-observer.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
previous=None
for tick in range(80):
 r=subprocess.run([sys.executable,str(D/'check_easy_hf_upload.py')],capture_output=True,text=True,timeout=30)
 if r.returncode:raise RuntimeError('Read-only upload observation failed; inspect existing PID before any retry')
 outer=json.loads((D/'easy-hf-upload-status-latest.json').read_text());s=outer['status'];position=sum(x['source_read_position_bytes']for name,x in s.get('file_progress',{}).items()if name.endswith('.safetensors'))
 summary={'at':outer['at'],'phase':'EXISTING_UPLOAD_OBSERVATION','state':s['state'],'source_stream_position_bytes_not_verified':position,'HF_verified_bytes':s.get('verified_bytes',0)}
 if previous:summary['recent_source_stream_MBps_not_HF_verified']=(position-previous[1])/(outer['at']-previous[0])/1e6
 print(json.dumps(summary),flush=True);previous=(outer['at'],position)
 if s['state']=='HF_FILES_VERIFIED':break
 if s['state'].startswith('FAILED'):raise RuntimeError('Existing uploader failed; preserve source and inspect safe status')
 time.sleep(45)
else:raise TimeoutError('Observer bounded wait reached; does not stop existing uploader')
