"""Narrow synthetic CPU full persistence/next-update qualification only."""
import copy,gc,json,tempfile,unittest
from pathlib import Path
import torch
from . import probe_ablation_train as u
from . import direct_hidden_count as h
from .direct_hidden_checkpoint import joint_blob,restore_joint,save_joint,verify_joint_file,state_hashes
from .test_direct_hidden_count import setup,update

class JointCheckpointTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def binding(self):return dict(plan=dict(path='/remote/plan.json',sha256='a'*64),source='same-E1',update_budget=938)
 def blob(self,values,cursor):
  m,e,o,head,he,ho,own,old,new=values
  return copy.deepcopy(joint_blob(m,e,o,head,he,ho,torch.device('cpu'),cursor,self.binding(),old['source_all_rank_states'],dict(cursor=cursor,plan_sha256='a'*64)))
 def hashes(self,v):return state_hashes(*v[:6],torch.device('cpu'))
 def test_durable_joint_CP_next_update_exact_and_clocks_distinct(self):
  values=setup();update(values);saved=self.blob(values,1);before=self.hashes(values)
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);receipt=save_joint(base,saved)
   self.assertEqual(self.hashes(values),before);self.assertEqual(len(receipt['accepted_hashes']),9)
   self.assertTrue(receipt['full_CPU_reload_passed']);self.assertEqual(receipt['legacy_Adam_steps'],10465);self.assertEqual(receipt['new_head_Adam_steps'],1)
   self.assertEqual(json.loads((base/'latest.json').read_text()),receipt)
   update(values);expected=self.hashes(values)
   raw=torch.load(receipt['path'],map_location='cpu',mmap=True,weights_only=False)
   status=restore_joint(raw,*values[:6],torch.device('cpu'),expected_binding=self.binding(),expected_cursor=1)
   self.assertEqual(status['data_cursor'],dict(cursor=1,plan_sha256='a'*64));update(values)
   self.assertEqual(self.hashes(values),expected);self.assertEqual(u.steps(values[2]),10466);self.assertEqual(h.optimizer_steps(values[5]),2)
   del raw;gc.collect()
 def test_new_head_corruption_rejected_before_restore_or_latest(self):
  values=setup();update(values);blob=self.blob(values,1);before=self.hashes(values)
  next(iter(blob['head']['raw'].values())).add_(1)
  with self.assertRaisesRegex(ValueError,'checksum'):
   restore_joint(blob,*values[:6],torch.device('cpu'))
  self.assertEqual(before,self.hashes(values))
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):save_joint(Path(d),blob)
   self.assertFalse((Path(d)/'latest.json').exists())
 def test_config_plan_cursor_and_full_source_archive_bindings(self):
  values=setup();update(values);good=self.blob(values,1)
  cases=[]
  bad=copy.deepcopy(good);bad['data_cursor']['plan_sha256']='b'*64;cases.append(bad)
  bad=copy.deepcopy(good);bad['head']['steps']=0;cases.append(bad)
  bad=copy.deepcopy(good);bad['legacy']['source_all_rank_states'].pop();cases.append(bad)
  for bad in cases:
   with self.assertRaises(ValueError):restore_joint(bad,*values[:6],torch.device('cpu'))
  with self.assertRaisesRegex(ValueError,'binding'):
   restore_joint(good,*values[:6],torch.device('cpu'),expected_binding=dict(other='config'))
 def test_full_file_SHA_and_nonpersistent_head_buffers(self):
  values=setup();values[3].register_buffer('native_head_clock',torch.tensor([3.]),persistent=False);values[4].register_buffer('native_head_clock',torch.tensor([4.]),persistent=False)
  update(values);saved=self.blob(values,1)
  with tempfile.TemporaryDirectory() as d:
   receipt=save_joint(Path(d),saved)
   with self.assertRaisesRegex(ValueError,'SHA'):
    verify_joint_file(receipt['path'],expected_sha256='0'*64)
   values[3].native_head_clock.add_(1);values[4].native_head_clock.add_(1)
   raw=torch.load(receipt['path'],map_location='cpu',mmap=True,weights_only=False)
   restore_joint(raw,*values[:6],torch.device('cpu'))
   self.assertEqual(values[3].native_head_clock.item(),3.);self.assertEqual(values[4].native_head_clock.item(),4.)
   del raw;gc.collect()

if __name__=='__main__':unittest.main()
