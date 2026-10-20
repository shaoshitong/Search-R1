"""Standalone native service entry: exact code fetch, prepare, sample; no credentials."""
import base64,hashlib,json,os,subprocess,sys,time
from pathlib import Path
BASE=Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
WRAPPER='/root/.local/share/search-g1/source8-fault-recovery/b2d62068695e156dcc90fc42e112ad24c676a1e7531fc26e2c0c6a01dbbfc1c1/retained-fault-paths.sh'
def main():
 control=Path(os.environ.get('Q7B_CONTROL',str(BASE)));c=json.loads((control/'code/direct-plan.json').read_text())[0]
 reuse=Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-probe-assets-20261020/code/migration-44944c26745213d7c2761037639f59be006b8b62');delta=json.loads((control/'code/delta.json').read_text());dest=BASE/'code'/('repo-'+c['commit']);dest.mkdir(exist_ok=True)
 for name,row in c['files'].items():
  path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
  if not path.exists():
   if name in delta:path.write_bytes(base64.b64decode(delta[name]))
   else:os.link(reuse/name,path)
  data=path.read_bytes();assert len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256'],name
 (control/'code/complete.json').write_text(json.dumps(dict(status='ALL_CODE_FILES_FRESH_SHA_VERIFIED_SHARED_REUSE',commit=c['commit'],files=len(c['files']),changed_files=len(delta),epoch=time.time())))
 repo=BASE/'code'/('repo-'+c['commit']);assets='/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-probe-assets-20261020'
 runtime=control/'runtime.sh';runtime.write_text('''#!/bin/bash
set -euo pipefail
export PYTHONPATH="'''+str(repo)+''':/dataset/shaoshitong/Search-G1-common8/retriever-gpu-lib"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export SEARCH_G1_B_PREFIX_TOKENS=20 SEARCH_G1_PHYSICAL_WORLD_SIZE=4
export SEARCH_G1_BASE="'''+str(BASE)+'''"
export HF_HOME="$SEARCH_G1_BASE/cache/huggingface" TORCH_EXTENSIONS_DIR="$SEARCH_G1_BASE/cache/torch_extensions" TMPDIR="$SEARCH_G1_BASE/tmp"
mkdir -p "$HF_HOME" "$TORCH_EXTENSIONS_DIR" "$TMPDIR"
export VLLM_ATTENTION_BACKEND=XFORMERS
export CPATH=/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include:/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/python3.10:/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include/x86_64-linux-gnu/python3.10
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
cd "'''+str(repo)+'''"
exec /dataset/shaoshitong/Search-G1-common8/env/bin/python -u "$@"
''')
 # The bootstrap already owns the retained mount namespace; do not stack it twice.
 prefix=['bash',str(runtime)]
 subprocess.run(prefix+['-m','reproduction.operations.qwen7b_resample.prepare','--base',str(BASE),'--assets',assets,'--retrieval-config',str(control/'retrieval-config.json')],check=True)
 subprocess.run(prefix+['-m','reproduction.operations.qwen7b_resample.pipeline','--base',str(BASE)],check=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:(Path(os.environ.get('Q7B_CONTROL',str(BASE)))/'entry-failure.json').write_text(json.dumps(dict(error_type=type(e).__name__,detail=str(e)[:300],epoch=time.time())));raise
