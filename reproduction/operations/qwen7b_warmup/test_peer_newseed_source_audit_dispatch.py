import json,tempfile,unittest
from pathlib import Path
from . import peer_newseed_source_audit_dispatch as a

class Tests(unittest.TestCase):
 def fixture(self,root):
  base=Path(root);inp='inputs/shard-01/batch-000001.json';dest=base/'sampling/shard-01/batches/batch-000001';outer=dest/'outers/outer-000001';paths=dict(batch_seal=dest/'complete.json',config=dest/'config.json',questions=outer/'questions.json',prefix_seal=outer/'prefix/complete.json',prefix_scan=outer/'prefix/scan.json',prefix_rank=outer/'prefix/rank-0-complete.json',full_seal=outer/'full/complete.json',full_rank=outer/'full/rank-0-complete.json',selection=outer/'probe-score/selection.json',prefix=outer/'prefix/rank-0-prefix.jsonl.gz',full=outer/'full/rank-0-full.jsonl.gz',own_em=outer/'full/rank-0-own-em.jsonl')
  for p in paths.values():p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
  p=base/'sidecar/proof.json';p.parent.mkdir();v=dict(status='VERIFIED',verified_epoch=1,binding=dict(cache_version=1,auditor_sha256=a.e.AUDIT_SHA,run_id=a.e.RUN_ID,seed=a.e.SEED,root_config_sha256='c'*64,input_sha256='i'*64,files={k:a.d.fstat(v)for k,v in paths.items()}),proof=dict(questions=1,trajectories=5,labels=5,actual_trajectory_ids_sha256='t'*64));p.write_text(json.dumps(v));item=dict(input=inp,input_sha256='i'*64,questions=1,trajectories=5,audit=dict(proof=a.e.fp(p),auditor_sha256=a.e.AUDIT_SHA,independent_sidecar_audit_root=str(p.parent),original_receipt_chain_equal_current=True));return base,item,paths
 def test_current_sidecar_and_payload_stat_binding(self):
  with tempfile.TemporaryDirectory()as t:
   base,item,paths=self.fixture(t);self.assertEqual(a.proof(base,item,'c'*64)['own_trajectory_ids_sha256'],'t'*64)
   paths['full'].write_text('{"changed":true}')
   with self.assertRaisesRegex(ValueError,'current all-file'):a.proof(base,item,'c'*64)
 def test_auditor_or_proof_sha_cannot_be_substituted(self):
  with tempfile.TemporaryDirectory()as t:
   base,item,paths=self.fixture(t);item['audit']['auditor_sha256']='b'*64
   with self.assertRaisesRegex(ValueError,'Unknown'):a.proof(base,item,'c'*64)
   item['audit']['auditor_sha256']=a.e.AUDIT_SHA;item['audit']['proof']['sha256']='0'*64
   with self.assertRaisesRegex(ValueError,'SHA'):a.proof(base,item,'c'*64)
if __name__=='__main__':unittest.main()
