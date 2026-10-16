import hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
b=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/tree-nested-branch-handoff-20261013');oldp=b/'PUBLIC_GRPO_CONTROL_REFRESH_20261016T004236Z.json';old=json.loads(oldp.read_bytes());scopep=b/old['explicit_scope']['file'];raw=scopep.read_bytes();assert len(raw)==old['explicit_scope']['bytes'] and hashlib.sha256(raw).hexdigest()==old['explicit_scope']['sha256'];scope=json.loads(raw)
forced_stat=set(old['stat_only_paths']);prev={x['path']:x for x in old['actual_readbacks'][-1]['rows']};started=datetime.now(timezone.utc).isoformat();rows=[];changed=[];contents={}
for spec in scope['pathlist']:
 p=Path(spec['path']);before=p.stat();row={'path':str(p),'read_at_utc':datetime.now(timezone.utc).isoformat(),'bytes':before.st_size,'mtime_ns':str(before.st_mtime_ns),'content_limit_bytes':scope['content_limit_bytes']}
 if str(p) in forced_stat or before.st_size>scope['content_limit_bytes']:
  after=p.stat();row.update(read_mode='STAT_ONLY_ORIGINAL_SCOPE_POLICY',sha256=None,content_read=False,stable_observation=before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns)
 else:
  raw=p.read_bytes();after=p.stat();row.update(read_mode='SMALL_METADATA_SHA_READBACK',sha256=hashlib.sha256(raw).hexdigest(),content_read=True,stable_observation=len(raw)==before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns);contents[str(p)]=raw.decode('utf-8',errors='replace')
 q=prev[str(p)];row['reference']={k:q[k] for k in ['bytes','mtime_ns','sha256']};row['new_or_changed_since_prior_readback']=any(row[k]!=q[k] for k in ['bytes','mtime_ns','sha256'])
 if row['new_or_changed_since_prior_readback']:changed.append(str(p))
 rows.append(row)
pass1={'started_utc':started,'ended_utc':datetime.now(timezone.utc).isoformat(),'rows':rows}
tmp=Path('/tmp/public-grpo-control-refresh-pass1-cost-owner176.json');assert not tmp.exists();tmp.write_text(json.dumps(pass1,ensure_ascii=False,allow_nan=False))
print('ACTUAL_PASS1',json.dumps({'started':started,'ended':pass1['ended_utc'],'scope':len(rows),'readable_count':sum(v['content_read'] for v in rows),'stat_only_count':sum(not v['content_read'] for v in rows),'bytes_read':sum(v['bytes'] for v in rows if v['content_read']),'changed':changed,'stable':all(v['stable_observation'] for v in rows)},ensure_ascii=False))
for name in changed:
 if name in contents:print('CHANGED_SMALL_METADATA',name,contents[name][:7000])
print('CURRENT_SMALL_GRPO_INVENTORY',contents.get('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-fresh200/LATEST_GRPO_CONTROL_READONLY.json','')[:14000])
