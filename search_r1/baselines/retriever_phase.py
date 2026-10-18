"""Release only the private retrieval worker before HF allocation; CPU service stays."""
import json,os,time
from pathlib import Path

def release_retriever_gpu(config,stage):
 settings=config.get('retriever',{})
 url=settings.get('gpu_release_url')
 if not url:return False
 expected='http://10.120.17.101:18006/release-gpu'
 if os.environ.get('SEARCH_R1_EXECUTION_BACKEND') in ('slurm_h100','h100_shared'):
  expected='http://127.0.0.1:18086/release-gpu'
 if url!=expected:raise ValueError('Unbound retrieval lifecycle endpoint')
 path=Path(settings['control_token_file'])
 if path.is_symlink() or path.stat().st_uid!=os.getuid():raise ValueError('Unowned retrieval control file')
 import requests
 started=time.perf_counter()
 response=requests.post(url,headers={'x-e1-probe-token':path.read_text().strip()},timeout=(5,60))
 response.raise_for_status();proof=response.json()
 if proof.get('released') is not True or proof.get('gpu_worker_pid') is not None:
  raise RuntimeError('Private retrieval GPU worker did not release: '+str(proof))
 previous=proof.get('previous_gpu_worker_pid')
 if previous is not None and Path('/proc',str(previous)).exists():raise RuntimeError('Retrieval worker exit not observed')
 print('RETRIEVER_PHASE='+json.dumps(dict(stage=stage,wall_seconds=time.perf_counter()-started,**proof)),flush=True)
 return True
