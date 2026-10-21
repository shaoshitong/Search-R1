"""Explicit frozen sidecar proof dispatch; never writes donor audit state."""
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
from . import peer_newseed_export as e
from . import peer_newseed_delta as d
from . import peer_newseed_audit_reference as original
RESEALED_SHA='e01c2b14c7e78f10141f03fd22385d08c65017379f7b515f75b705f82fd9ceae'

def checked(ref):
 p=e.regular(Path(ref['path']).parent,Path(ref['path']))
 if p.stat().st_size!=ref['bytes'] or e.sha(p)!=ref['sha256']:raise ValueError('Frozen audit proof/attestation SHA differs')
 return p

def proof(base,item,config_sha):
 info=item['audit'];ref=info['proof'];p=checked(ref);v=e.read(p);inp=Path(item['input']);dest=Path(base)/'sampling'/inp.parent.name/'batches'/inp.stem;outer=dest/'outers/outer-000001'
 files=dict(batch_seal=dest/'complete.json',config=dest/'config.json',questions=outer/'questions.json',prefix_seal=outer/'prefix/complete.json',prefix_scan=outer/'prefix/scan.json',prefix_rank=outer/'prefix/rank-0-complete.json',full_seal=outer/'full/complete.json',full_rank=outer/'full/rank-0-complete.json',selection=outer/'probe-score/selection.json',prefix=outer/'prefix/rank-0-prefix.jsonl.gz',full=outer/'full/rank-0-full.jsonl.gz',own_em=outer/'full/rank-0-own-em.jsonl')
 auditor=info.get('auditor_sha256',e.AUDIT_SHA)
 if auditor not in(e.AUDIT_SHA,RESEALED_SHA):raise ValueError('Unknown explicit proof auditor')
 expected=dict(cache_version=1,auditor_sha256=auditor,run_id=e.RUN_ID,seed=e.SEED,root_config_sha256=config_sha,input_sha256=item['input_sha256'],files={k:d.fstat(v)for k,v in files.items()})
 counts=v.get('proof',{})
 if v.get('status')!='VERIFIED'or v.get('binding')!=expected or counts.get('questions')!=item['questions']or counts.get('trajectories')!=item['trajectories']or counts.get('labels')!=item['trajectories']:raise ValueError('Explicit sidecar proof is not current all-file-fstat/count bound')
 if auditor==RESEALED_SHA:
  att=info.get('metadata_resealed_attestation');checked(att)
  a=counts.get('metadata_reseal_attestation',{})
  if info.get('original_receipt_chain_equal_current')is not False or a.get('original_receipt_chain_equal_current')is not False or any(a.get(k)is not True for k in('same_prefix_rank_receipt_and_payload','same_all_G5_IDs','full_rows_original_chain_and_current_prefix_causal_tokens_exact'))or a.get('producer_metadata_modified')is not False:raise ValueError('Explicit metadata-resealed proof semantics differ')
  rank=e.read(files['full_rank'])
  if(a['original_continuation_scan_sha256']!=rank['prefix_scan_sha256']or a['original_continuation_selection_sha256']!=rank['selection_sha256']or a['fresh_current_scan_sha256']!=e.sha(files['prefix_scan'])or a['fresh_current_selection_sha256']!=e.sha(files['selection'])):raise ValueError('Original and current receipt roles differ')
 elif info.get('metadata_resealed_attestation')is not None or info.get('original_receipt_chain_equal_current',True)is not True:raise ValueError('Original auditor cannot attest resealed metadata')
 return dict(status='EXISTING_INDEPENDENT_AUDIT_CURRENT_FSTAT_BOUND',proof=e.fp(p),verified_epoch=v['verified_epoch'],own_trajectory_ids_sha256=counts['actual_trajectory_ids_sha256'],bulk_payload_SHA_recomputed=False,strict_EM_tokenizer_recomputed=False)

class Dispatch:
 def __init__(self,base,root,plan):
  self.base,self.root=Path(base),Path(root);self.by={x['input']:x for x in plan['selected_batches']};args=SimpleNamespace(base=str(base),audit_dir=str(root/'audit-proof-cache'),path_map=[],max_new_batches=0)
  self.original=original.Audit(args);self.original.root_config=e.read(self.base/'config.json');self.original.root_config_hash=e.sha(self.base/'config.json');self.alternate=None
  if any(x.get('audit_dispatch',{}).get('auditor_sha256')==RESEALED_SHA for x in self.by.values()):
   from . import peer_newseed_audit_resealed_metadata as alt
   if e.sha(alt.__file__)!=RESEALED_SHA:raise ValueError('Pinned alternate auditor bytes differ')
   self.alternate=alt.Audit(args);self.alternate.root_config=self.original.root_config;self.alternate.root_config_hash=self.original.root_config_hash
 @property
 def reused(self):return self.original.reused+(self.alternate.reused if self.alternate else 0)
 @property
 def fresh(self):return self.original.fresh+(self.alternate.fresh if self.alternate else 0)
 def batch(self,inp,batch,input_hash):
  item=self.by[str(Path(inp).relative_to(self.base))];info=item.get('audit_dispatch')
  if info:
   proof(self.base,dict(item,audit=info),self.original.root_config_hash)
   p=checked(info['proof']);dest=self.root/'audit-proof-cache/batches'/Path(inp).parent.name/(Path(inp).stem+'.json')
   if not dest.exists():e.stable_copy(p.parent,p,dest,info['proof'])
  selected=self.alternate if info and info.get('auditor_sha256')==RESEALED_SHA else self.original
  return selected.batch(inp,batch,input_hash)
