"""Small continuation/state/async-token fixtures, no real assets or models."""
import copy,gc,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from . import probe_ablation_train as u,mixed_group_checkpoint as ck
from . import joint_mixed_direct_fast_continuation as w
from . import direct_update_fast as f
from .test_direct_update_fast import setup


class PreparedReader:
 def __init__(self,reader):self.reader=reader;self.events=[];self.token=None
 def prepare(self,sampler):
  saved=sampler.state_dict();clone=copy.deepcopy(sampler);self.token=clone.next_batch()['token'];self.events.append('prepare')
  if sampler.state_dict()!=saved:raise AssertionError('Synthetic producer advanced real sampler')
 def accept_batch(self,batch):
  self.events.append('accept')
  if batch['token']!=self.token:raise ValueError('stale prefetched token')
 def get(self,ids,device):
  if self.events[-1]!='accept':raise AssertionError('Model/feature work before real token acceptance')
  self.events.append('get');return self.reader.get(ids,device)
 def reset_prefetch(self):self.token=None;self.events.append('reset')


class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2)
 def test_binding_only_migration_preserves_nine_clocks_and_exact_micro32_next_update(self):
  m,e,b,h,s,r,l=setup();r=PreparedReader(r);device=torch.device('cpu');seen=set();old=dict(config_sha256='old',matched_binding_sha256='old');new=dict(config_sha256='new',matched_binding_sha256='new',binding_migration={'source_cursor':2});prov=dict(initialization_source='official_initial_Qwen2.5-7B-Instruct_block27',synthetic_fixture=True)
  for _ in range(2):r.prepare(s);_,batch=f.direct_update_fast(m,e,b,h,r,l,s,device,micro_questions=32);seen.update(batch['source_qids'])
  saved=ck.snapshot(m,e,b,h,device,s,2,old,prov,seen,'training');moved=w.migrate_snapshot(saved,m,e,b,h,device,s,old,new)
  self.assertEqual(saved['accepted_hashes'],moved['accepted_hashes']);self.assertEqual(moved['cursor'],2);self.assertEqual(ck.optimizer_steps(b.state_dict()),2);self.assertEqual(ck.optimizer_steps(h.state_dict()),2)
  with self.assertRaisesRegex(ValueError,'binding'):ck.validate(moved,old)
  with tempfile.TemporaryDirectory(dir='/tmp')as tmp:
   base=Path(tmp);receipt=ck.save(base,moved,pin=True,owned_allocation_required=False);self.assertTrue(receipt['full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed'])
   r.prepare(s);_,batch1=f.direct_update_fast(m,e,b,h,r,l,s,device,micro_questions=32);expected=ck.snapshot(m,e,b,h,device,s,3,new,prov,seen|set(batch1['source_qids']),'training')['accepted_hashes']
   loaded=torch.load(u.checked(receipt),map_location='cpu',mmap=True,weights_only=False);cursor,restored=ck.restore(loaded,m,e,b,h,device,s,new);r.reset_prefetch();r.prepare(s);_,batch2=f.direct_update_fast(m,e,b,h,r,l,s,device,micro_questions=32)
   actual=ck.snapshot(m,e,b,h,device,s,3,new,prov,restored|set(batch2['source_qids']),'training')['accepted_hashes'];self.assertEqual(cursor,2);self.assertEqual(batch1,batch2);self.assertEqual(expected,actual);del loaded;gc.collect()
 def test_highest_pinned_receipt_cannot_substitute_older_cursor_or_alter_science(self):
  cfg=dict(path=str(w.BASE/'source-config.json'),bytes=9615,sha256=w.SOURCE_CONFIG_SHA);old=dict(path=str(w.source.MAIN/'checkpoints/cursor-000256.pt'),cursor=256,native_steps=256,head_steps=256,config_sha256=w.SOURCE_CONFIG_SHA,full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed=True,accepted_hashes={k:'synthetic-'+k for k in w.NINE});pin=dict(old,path=str(w.BASE/'inputs/source-checkpoint.pt'));spec=dict(source_checkpoint_original=old,source_checkpoint=pin,source_config=cfg,source_main_config=dict(cfg,path=str(w.source.MAIN/'config.json')),source_handoff_receipt={'path':'/synthetic/handoff','sha256':'synthetic'});c={'continuation':spec}
  self.assertEqual(w.validate_receipts(c),256)
  wrong=copy.deepcopy(c);wrong['continuation']['source_checkpoint']['cursor']=128
  with self.assertRaisesRegex(ValueError,'pin'):w.validate_receipts(wrong)
  handoff=dict(job_id='746083',cursor=256,terminal_exit_0_0=True,highest_accepted_complete_cursor_equal=True,no_unresolved_tail=True,training_lock_free=True,checkpoint=old,pinned_checkpoint=pin)
  with patch.object(w.u,'checked',return_value=Path('/synthetic/handoff')),patch.object(w.u,'read',return_value=handoff):self.assertEqual(w.validate_handoff(spec),handoff)
  handoff['cursor']=128
  with patch.object(w.u,'checked',return_value=Path('/synthetic/handoff')),patch.object(w.u,'read',return_value=handoff):
   with self.assertRaisesRegex(ValueError,'highest'):w.validate_handoff(spec)
  scientific={'architecture':{'micro_questions':1,'batch_questions':32},'lr':5e-6,'update_budget':21132,'mode':'main','base':'old'};fast=copy.deepcopy(scientific);fast.update(base='new',mode='fast_continuation',prefetch_batches=8);fast['architecture']['micro_questions']=32;self.assertEqual(w.scientific_config(scientific),w.scientific_config(fast));fast['lr']=1e-4;self.assertNotEqual(w.scientific_config(scientific),w.scientific_config(fast))
 def test_stale_async_token_rejected_before_feature_or_model_work(self):
  m,e,b,h,s,reader,l=setup();r=PreparedReader(reader);r.prepare(s);r.token='wrong';before=u.model_hash(m)
  with self.assertRaisesRegex(ValueError,'stale'):f.direct_update_fast(m,e,b,h,r,l,s,torch.device('cpu'),micro_questions=32)
  self.assertEqual(reader.calls,[]);self.assertEqual(u.model_hash(m),before);self.assertEqual(b.state,{});self.assertEqual(h.state,{});self.assertEqual(r.events,['prepare','accept'])


if __name__=='__main__':unittest.main()
