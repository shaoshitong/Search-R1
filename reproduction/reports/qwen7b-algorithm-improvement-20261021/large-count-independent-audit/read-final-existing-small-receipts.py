import json,pathlib,hashlib,time
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):
 assert proof(r['path'])==r, 'small audit descriptor mismatch';return read(r['path'])
expected={'bytes': 45317, 'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-count-large-independent-audit-v3/audits/snapshot-ab2e83b08f781db4.json', 'sha256': '9552c4666235f182f31980278d3d3e93d135c8030ed80beede3f14efec44b893'}
snapshot=checked(expected);assert snapshot['status']=='ALL12_NODES_INDEPENDENTLY_AUDITED' and snapshot['actual_scored_nodes']==12
for node in snapshot['nodes']:checked(node['audit'])
assert not snapshot['external_payload_opened'] and not snapshot['final_payload_opened'] and not snapshot['external_scoring_permitted']
cp1path='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-count-large-smallCP1-independent-v2/independent-CP1-acceptance.json'
familypath='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozenE1-head-objective-family-v1/coarse-fork-inputs.json'
cp1=read(cp1path);family=read(familypath)
out=dict(status='ALL12_EXISTING_AUDITS_AND_SMALL_CP1_RECEIPT_FRESH_VERIFIED',epoch=time.time(),snapshot=expected,node_audits_fresh_sha_checked=12,CP1_acceptance=proof(cp1path),CP1_result=cp1,family_inputs=proof(familypath),family_status=family.get('status'),external_payload_opened=False,final_payload_opened=False,new_forward=False,new_optimizer_update=False,large_model_or_hidden_payload_opened=False)
print(json.dumps(out,sort_keys=True))
