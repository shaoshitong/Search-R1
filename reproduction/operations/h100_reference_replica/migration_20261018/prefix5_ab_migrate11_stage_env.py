# Archived actual operation. Reconcile existing intent/receipt before any use.
exec(open('work/prefix5_ab_migrate11_inspect.py').read().split('with concurrent.futures.ThreadPoolExecutor')[0])
script=r'''import json,os,subprocess,time
from pathlib import Path
root=Path('/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/search-g1-prefix5-h10011-runtime-assets-20261018')
root.mkdir(exist_ok=True);intent=root/'copy-runtime-intent.json';receipt=root/'copy-runtime-receipt.json'
if receipt.exists():print(receipt.read_text())
elif intent.exists():print(json.dumps({'status':'EXISTING_INTENT_QUERY_ONLY','intent':str(intent)}))
else:
 targets={'env':'/dataset/shaoshitong/Search-G1-common8/env','bootstrap-assets/instruct300/assets/qwen2.5-3b-instruct-aa8e725':'/dataset/shaoshitong/Search-G1-common8/bootstrap-assets/instruct300/assets/qwen2.5-3b-instruct-aa8e725','assets/e5-base-v2':'/dataset/shaoshitong/Search-G1-common8/assets/e5-base-v2','retriever-gpu-lib':'/dataset/shaoshitong/Search-G1-common8/retriever-gpu-lib'}
 script=root/'copy-runtime.sh'
 import shlex
 lines=['#!/bin/bash','set -euo pipefail']
 for rel,src in targets.items():
  dst=root/'common8'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  lines.append('cp -a --reflink=auto '+shlex.quote(str(Path(src).resolve()))+' '+shlex.quote(str(dst)))
 lines.append('date -u > '+shlex.quote(str(root/'COPY_COMPLETE')))
 script.write_text('\n'.join(lines)+'\n');intent.write_text(json.dumps({'created_unix':time.time(),'sources':targets,'target':str(root),'model_via_mac':False}))
 with(root/'copy.log').open('ab')as log:p=subprocess.Popen(['bash',str(script)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 v={'pid':p.pid,'target':str(root),'status':'COPY_RUNNING','same_remote_cluster':True};receipt.write_text(json.dumps(v));print(json.dumps(v))
'''
do('h100-8',script,'migrate11-env-copy')
script=r'''import json,subprocess
from pathlib import Path
p=subprocess.run(['unshare','--mount','--propagation','private','/bin/true'],capture_output=True,text=True)
print(json.dumps({'unshare_rc':p.returncode,'stderr':p.stderr,'env_available':Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/search-g1-prefix5-h10011-runtime-assets-20261018/common8').exists()}))
'''
do('h100-11',script,'migrate11-namespace-capability')
