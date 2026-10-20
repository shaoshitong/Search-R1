"""User-approved raw-only scope: native hidden arrays rebuilt at destination."""
import json
from pathlib import Path
import source_export as export
PARENT=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/migration-export-h100-20261020')
def raw_plan(base):
 p=json.loads((PARENT/'plan.json').read_text());groups=[]
 for group in p['groups']:
  hashes={r['path']:r['sha256']for r in json.loads((PARENT/(group['group']+'.json')).read_text())['files']}
  rows=[]
  for old in group['files']:
   if old['path'].endswith(('.bin','.pt')):continue
   row=dict(old);row['expected_sha256']=hashes[row['path']];rows.append(row)
  assert rows;groups.append(dict(group=group['group'],files=rows))
 p.update(status='SOURCE_RAW_ONLY_EXPORT_PLAN',groups=groups,payload_bytes=sum(r['bytes']for g in groups for r in g['files']),files=sum(len(g['files'])for g in groups),hidden_arrays_included=False,probe_checkpoint_previously_delivered=True,hidden_recompute_required=True,user_scope='Raw tokens/context/masks/positions/strict-EM and metadata; no hidden bin arrays')
 return p
export.plan=raw_plan
if __name__=='__main__':export.main()
