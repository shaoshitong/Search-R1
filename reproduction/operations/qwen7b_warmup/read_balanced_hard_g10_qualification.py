import json,pathlib,hashlib
p=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/balanced-hard-sampler-qualification-v1/qualification.json')
raw=p.read_bytes();original=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
format_repaired=False
try:v=json.loads(raw.decode())
except ValueError:
 assert raw.endswith(bytes([92,110]))
 v=json.loads(raw[:-2].decode());format_repaired=True
 v['original_raw_receipt']=original;v['serialization_format_correction']='Literal backslash-n suffix from operator quoting; original preserved, no test/optimizer/native rerun'
 p=p.with_name('qualification-format-accepted.json');encoded=json.dumps(v,sort_keys=True).encode()+bytes([10])
 if p.exists():assert p.read_bytes()==encoded
 else:p.write_bytes(encoded)
 raw=p.read_bytes();json.loads(raw.decode())
assert v['first_batch_Mixed_views']==v['first_batch_nonMixed_views']==64 and v['restored_next_batch_exact']and not v['actual_training_or_update']
for ref in v['files'].values():
 content=pathlib.Path(ref['path']).read_bytes();assert len(content)==ref['bytes']and hashlib.sha256(content).hexdigest()==ref['sha256']
print(json.dumps(dict(status='INDEPENDENT_METADATA_RECEIPT_AND_SOURCE_READBACK_PASSED',receipt=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),format_repaired=format_repaired,result=v)))
