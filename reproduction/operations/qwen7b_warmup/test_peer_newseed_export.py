"""Tiny CPU fixtures only; the pinned production deep auditor is not replaced live."""
import copy,gzip,json,os,tarfile,tempfile,unittest
from pathlib import Path
from unittest import mock
from . import peer_newseed_export as e


def jsonfile(path,value):e.atomic(path,value)
def writegz(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(path,'wt') as f:
  for row in rows:f.write(json.dumps(row)+'\n')
def proof(path,count=None):
 result={k:v for k,v in e.fp(path).items() if k!='source_stat'}
 if count is not None:result['count']=count
 return result


def fixture(base,*,two_sealed=False,duplicate=False,conflict=False):
 base.mkdir();c=dict(run_id=e.RUN_ID,seed=e.SEED,model_revision=e.REVISION,sampling_attempt_id=e.ATTEMPT,warmup_only=True,actor_hidden_native=False,actor_optimizer_updates=0,actor_version='official7b-'+e.REVISION)
 jsonfile(base/'config.json',c);entries=[]
 for index in range(2):
  ordinal=0 if duplicate else index;qid='q%d'%ordinal
  inp=base/'inputs/shard-00'/('batch-%06d.json'%index)
  row=dict(qid=qid,question_hash='hash-'+qid,candidate_ordinal=ordinal,source_ordinal=ordinal,split='train',initial_prompt_token_ids=[1,2],reward_model=dict(ground_truth=dict(target=['a'])))
  jsonfile(inp,dict(rows=[row],split='train',model_revision=e.REVISION));entries.append(dict(proof(inp),questions=1,shard=0))
  if index and not two_sealed:continue
  dest=base/'sampling/shard-00/batches'/inp.stem;outer=dest/'outers/outer-000001'
  files=e.batch_files(base,inp)
  tids=['%s-outer000001-c%04d-g%d'%(e.RUN_ID,ordinal,j) for j in range(5)]
  labels=[dict(trajectory_id=t,qid=qid,strict_em=(j%2 if not(conflict and index) else 1-j%2)) for j,t in enumerate(tids)]
  for key in ('states','prefix','full'):writegz(files[key],[dict(trajectory_id=t,qid=qid) for t in tids])
  files['own_em'].parent.mkdir(parents=True,exist_ok=True);files['own_em'].write_text(''.join(json.dumps(v)+'\n' for v in labels))
  prefix=dict(states=proof(files['states'],5),prefix=proof(files['prefix'],5));full=dict(full=proof(files['full'],5),own_em=proof(files['own_em'],5))
  for key in ('config','questions','prefix_seal','prefix_scan','full_seal','selection'):jsonfile(files[key],dict(needed_metadata=key,source_run_id=e.RUN_ID))
  jsonfile(files['prefix_rank'],prefix);jsonfile(files['full_rank'],full)
  jsonfile(files['batch_seal'],dict(run_id=e.RUN_ID,input_sha256=e.sha(inp),batch=inp.stem,questions=1,trajectories=5,hidden_cached=False,model_revision=e.REVISION,actor_optimizer_updates=0,prefix=prefix,full=full))
 jsonfile(base/'inputs/manifest.json',dict(batches=entries,ordered_qid_sha256='fixture-not-real-data'))
 return base


class QualifiedFixtureAuditor:
 """Tests only packaging/control after a returned qualified audit proof.
 Production always loads the byte-identical real auditor, never this stub.
 """
 reused=0;fresh=0
 def batch(self,inp,batch,input_hash):
  self.fresh+=1
  return dict(questions=len(batch['rows']),trajectories=5*len(batch['rows']),labels=5*len(batch['rows']))


class PeerExportTests(unittest.TestCase):
 def test_readonly_discover_excludes_unsealed_and_export_proofs_no_source_mutation(self):
  with tempfile.TemporaryDirectory() as raw:
   root=Path(raw).resolve();base=fixture(root/'source');before={str(p):e.sha(p) for p in base.rglob('*') if p.is_file()}
   plan=e.discover(base);self.assertEqual(len(plan['selected_batches']),1);self.assertEqual(len(plan['unsealed_inputs_excluded']),1)
   self.assertEqual(before,{str(p):e.sha(p) for p in base.rglob('*') if p.is_file()})
   with mock.patch.object(e,'load_auditor',return_value=QualifiedFixtureAuditor()):done=e.export(base,root/'exports','fixture-fixed-sealed-snapshot')
   self.assertEqual(done['questions'],1);self.assertEqual(done['trajectories'],5);self.assertFalse(done['source_all_inputs_sealed_at_freeze'])
   self.assertEqual(done['unsealed_batches_excluded'],1);self.assertFalse(done['strict_em_recomputed']);self.assertFalse(done['hidden_exported'])
   self.assertEqual(before,{str(p):e.sha(p) for p in base.rglob('*') if p.is_file()})
   group=done['groups'][0];self.assertEqual(group['archive_format'],'tar');self.assertTrue(group['archive'].endswith('.tar'))
   with tarfile.open(group['archive'],'r:') as tar:
    members=tar.getmembers();self.assertTrue(all(x.isfile() and not x.issym() and not x.islnk() for x in members))
    self.assertTrue(all(x.name.startswith('fixture-fixed-sealed-snapshot/group-00000/') for x in members))
    self.assertFalse(any(x.name.endswith(e.FORBIDDEN) for x in members));self.assertTrue(any(x.name.endswith('provenance/canonical-membership.json') for x in members))
   self.assertEqual(group['archive_sha256'],e.sha(group['archive']))
   from . import peer_newseed_hf_stream as hf
   with mock.patch.object(hf,'SOURCE_PARENT',root/'exports'):
    self.assertIs(hf.validate_snapshot(done,root/'exports/fixture-fixed-sealed-snapshot'),done)
   self.assertEqual(done,e.export(base,root/'exports','fixture-fixed-sealed-snapshot'))

 def test_exact_own_five_deduplicated_but_conflicting_question_fails(self):
  with tempfile.TemporaryDirectory() as raw:
   root=Path(raw).resolve();base=fixture(root/'source',two_sealed=True,duplicate=True)
   with mock.patch.object(e,'load_auditor',return_value=QualifiedFixtureAuditor()):done=e.export(base,root/'exports','exact-duplicate')
   self.assertEqual(done['questions'],1);self.assertEqual(done['trajectories'],5);self.assertEqual(done['duplicates_removed'],1)
  with tempfile.TemporaryDirectory() as raw:
   root=Path(raw).resolve();base=fixture(root/'source',two_sealed=True,duplicate=True,conflict=True)
   with mock.patch.object(e,'load_auditor',return_value=QualifiedFixtureAuditor()),self.assertRaisesRegex(ValueError,'Conflicting duplicate'):
    e.export(base,root/'exports','conflicting-duplicate')
   self.assertFalse((root/'exports/conflicting-duplicate/complete.json').exists())

 def test_fake_complete_missing_full_never_becomes_snapshot_complete(self):
  with tempfile.TemporaryDirectory() as raw:
   root=Path(raw).resolve();base=fixture(root/'source');inp=base/'inputs/shard-00/batch-000000.json';files=e.batch_files(base,inp);files['full'].unlink()
   with mock.patch.object(e,'load_auditor',return_value=QualifiedFixtureAuditor()),self.assertRaises(FileNotFoundError):e.export(base,root/'exports','fake-complete')
   self.assertFalse((root/'exports/fake-complete/complete.json').exists())

 def test_tar_paths_symlinks_tensor_and_source_races_rejected(self):
  for bad in ('../escape','/absolute','a//b','a/./b','a\\b','a\nb'):
   with self.assertRaises(ValueError):e.safe_name(bad)
  with tempfile.TemporaryDirectory() as raw:
   root=Path(raw).resolve();source=root/'source';source.mkdir();(source/'file.json').write_text('{}');(root/'outside.json').write_text('{}');(source/'link.json').symlink_to(root/'outside.json');(source/'hidden.bin').write_bytes(b'no-real-tensor')
   for name in ('link.json','hidden.bin'):
    with self.assertRaises(ValueError):e.stable_copy(source,source/name,root/'copy.json')
   original=e.fs;calls=[]
   def race(p):
    value=original(p)
    if Path(p)==source/'file.json':
     calls.append(1)
     if len(calls)>1:value['mtime_ns']+=1
    return value
   with mock.patch.object(e,'fs',side_effect=race),self.assertRaisesRegex(ValueError,'changed during'):e.stable_copy(source,source/'file.json',root/'race-copy.json')
   self.assertFalse((root/'race-copy.json').exists())
   with self.assertRaises(ValueError):e.tar_group(source,{'link.json':dict(bytes=2,sha256=e.sha(root/'outside.json'))},root/'bad.tar','snapshot/group')

 def test_pinned_peer_auditor_is_exact_source(self):
  path=Path(e.__file__).with_name('peer_newseed_audit_reference.py');self.assertEqual(e.sha(path),e.AUDIT_SHA)

if __name__=='__main__':unittest.main()
