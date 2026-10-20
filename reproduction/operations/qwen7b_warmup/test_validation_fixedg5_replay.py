"""Metadata-only fixedG5 replay tests; no model/hidden tensors/GPU/remote actions."""
import copy,gzip,json,tempfile,unittest
from pathlib import Path
from . import validation_fixedg5_replay as m
from .test_peer_newseed_ingest import trajectory


def fixed(n=2):
 groups=[dict(qid='q%d'%i,trajectory_ids=['q%d-t%d'%(i,j) for j in range(5)]) for i in range(n)]
 return dict(groups=groups,summary=dict(role='validation',labels_consulted=False,questions=n,trajectories=5*n,selected_g5_sha256=m.digest(groups)))

class FixedValidationTests(unittest.TestCase):
 def test_exact_existing_fixed_order_only(self):
  plan=fixed();rows=m.fixed_declarations(plan,2)
  self.assertEqual([x['trajectory_id'] for x in rows],[t for g in plan['groups'] for t in g['trajectory_ids']])
  self.assertTrue(all(set(x)=={'qid','trajectory_id'} for x in rows))
  plan['groups'][0]['labels']=[1,0,0,0,0];plan['summary']['selected_g5_sha256']=m.digest(plan['groups'])
  with self.assertRaisesRegex(ValueError,'identity-only'):m.fixed_declarations(plan,2)
 def test_duplicate_missing_or_oracle_view_rejected(self):
  for edit in ('duplicate','missing','oracle','role','sha'):
   v=fixed()
   if edit=='duplicate':v['groups'][1]['trajectory_ids'][0]=v['groups'][0]['trajectory_ids'][0]
   if edit=='missing':v['groups'][0]['trajectory_ids'].pop()
   if edit=='oracle':v['summary']['labels_consulted']=True
   if edit=='role':v['summary']['role']='final'
   v['summary']['selected_g5_sha256']=m.digest(v['groups']) if edit!='sha' else '0'*64
   with self.assertRaises(ValueError):m.fixed_declarations(v,2)
 def test_prefix_source_discovery_never_follows_full_em(self):
  a=dict(all_source_payload_SHA_freshly_verified=True,events={k:[dict(phase_payloads=dict(prefix={'path':k},full={'path':'NEVER_OPEN_FULL'},own_em={'path':'NEVER_OPEN_EM'}))] for k in ('event-A','event-B')})
  self.assertEqual(m.prefix_sources(a),[{'path':'event-A'},{'path':'event-B'}])
 def test_real_causal_projection_preserves_precommitted_own_G5(self):
  _,prefix,_=trajectory(dict(run_id='metadata-test',seed=1729));plan=fixed();decl=m.fixed_declarations(plan,2)
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder).resolve()/'prefix.jsonl.gz'
   with gzip.open(path,'wt') as out:
    for x in reversed(decl):
     row=copy.deepcopy(prefix);row.update(x);row['prefix_sha256']=m.digest({k:v for k,v in row.items() if k!='prefix_sha256' and not k.startswith('feature_')});out.write(json.dumps(row)+'\n')
   selection=dict(role=m.ROLE,questions=2,trajectories=decl,capture_layers=[26],label_blind_selection=True,new_G5_selected=False,prefix_files=[m.ref(path)])
   rows=m.load_selection(selection,True,2)
   self.assertEqual([x['trajectory_id'] for x in rows],[x['trajectory_id'] for x in decl])
   self.assertEqual(rows[0]['policy_position_ids'],prefix['policy_position_ids']);self.assertEqual(len(rows[0]['segments']),2)
   path.write_bytes(b'tampered')
   with self.assertRaisesRegex(ValueError,'SHA'):m.load_selection(selection,True,2)
 def test_em_cannot_enter_projected_prefix_and_missing_never_generated(self):
  _,prefix,_=trajectory(dict(run_id='metadata-test',seed=1729))
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder).resolve()/'prefix.jsonl.gz';row=copy.deepcopy(prefix);row['strict_em']=1;row['prefix_sha256']=m.digest({k:v for k,v in row.items() if k!='prefix_sha256'})
   with gzip.open(path,'wt') as out:out.write(json.dumps(row)+'\n')
   with self.assertRaisesRegex(ValueError,'without EM'):list(m.projected_rows([m.ref(path)],{row['trajectory_id']:row['qid']},True))
   with self.assertRaisesRegex(ValueError,'Incomplete'):list(m.projected_rows([m.ref(path)],{'missing':'q'},True))
 def test_unsupported_batch_fails_before_source_read(self):
  with self.assertRaisesRegex(ValueError,'batch16 or32'):m.batching_binding({'path':'NOT_OPENED'},8)

if __name__=='__main__':unittest.main()
