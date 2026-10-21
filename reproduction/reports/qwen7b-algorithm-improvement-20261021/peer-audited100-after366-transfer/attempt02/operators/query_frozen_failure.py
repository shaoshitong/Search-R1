import json,sys,pathlib
r=pathlib.Path('work/qwen7b-probe-warmup-20261019').resolve();sys.path.insert(0,str(r));from reproduction.operations.qwen7b_warmup import deploy_peer_frozen_delta_transfer as op
code='/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/_code-frozen-delta-audited100-after366-20261021-636015f5-9cd8662fbab3'
script='code='+repr(code)+'\n'+'''import sys,json,pathlib,hashlib,time
sys.path.insert(0,code)
from reproduction.operations.qwen7b_warmup import peer_newseed_delta as d
from reproduction.operations.qwen7b_warmup import peer_newseed_export as e
p=pathlib.Path(code)/'frozen-selection.json';v=json.loads(p.read_text());x=v['selected_batches'][56];b=pathlib.Path(v['source_base']);f=d.cached_audit(b,x['input'],v['source_config']['sha256'],x['input_sha256'],x['questions'])
out=dict(status='READ_ONLY_FIRST_FROZEN_EXPORT_BINDING_FAILURE',index=56,input=x['input'],frozen_proof=x['audit']['proof'],frozen_ownIDs=x['expected_own_G5_IDs_sha256'],fresh=f,epoch=time.time())
print(json.dumps(out))
'''
v=op.source_call(script);p=pathlib.Path('outputs/qwen7b-warmup-20261019/peer-frozen-delta-transfer/audited100-after366-20261021-636015f5/first-source-failure-detail.json');p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n');print(json.dumps(v))
