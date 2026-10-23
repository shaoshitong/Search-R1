"""Remote CPU readback of real PrefixState samples; no training mutations."""
import pathlib,json,torch,time,io
from dataclasses import asdict
from reproduction.dynrpo7b.collection_resume import snapshot_states
from reproduction.dynrpo7b.generation import PrefixState
from reproduction.dynrpo7b.deploy import prepare,publish
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo7b-formal-20261023-recoveryfix-r5')
new=pathlib.Path('/data/user/jxu957/experiments/dynrpo7b-formal-20261023-recoveryfix-r6')
v=json.loads((b/'preparation/dynrpo7b-collection/outer-0005.json').read_text())
part=torch.load(v['parts'][1]['path'],map_location='cpu',weights_only=False,mmap=True)
rows=part['states']; sample=[PrefixState(**rows[i]) for i in range(0,len(rows),max(1,len(rows)//16))][:16]
t=time.perf_counter();old=[asdict(s) for s in sample];old_time=time.perf_counter()-t
t=time.perf_counter();fast=snapshot_states(sample);new_time=time.perf_counter()-t
assert old==fast
out=io.BytesIO();torch.save(fast,out);out.seek(0);assert torch.load(out,weights_only=False)==old
result=dict(status='REAL_PREFIX_SNAPSHOT_FIELDS_AND_RELOAD_IDENTICAL',samples=len(sample),asdict_seconds=old_time,field_snapshot_seconds=new_time,serialized_bytes=out.getbuffer().nbytes,source_part_sha256=v['parts'][1]['sha256'],GPU_used=False)
prepare(new,'ee74bfa3c6d1141859abb19e882b65c93ea6f0ba',predecessor='/data/user/jxu957/experiments/dynrpo7b-formal-20261023')
publish(new/'preparation/snapshot-serialization-cpu.json',result)
publish(new/'preparation/external-retriever.json',json.loads((b/'preparation/external-retriever.json').read_text()))
print(json.dumps(result))
