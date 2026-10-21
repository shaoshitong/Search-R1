"""Narrow fixed100 cohort binding and unchanged capture/device contracts."""
import copy,inspect,json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from . import peer_audited100_l26_replay as c
from . import deploy_peer_audited100_l26 as d
from . import peer_audited122_l26_replay as old
from .test_peer_audited122_l26 import fixture


def config():
 ref=lambda path:dict(path=str(path),bytes=1,sha256='a'*64)
 return dict(status='FROZEN_AUDITED100_AFTER366_ACCEPTED_G10_LAYER26_CONFIG',questions=12800,trajectories=128000,domains={'nq':128,'hotpotqa':12672},source_selection=dict(path='/public-small-selection.json',bytes=170253,sha256=c.SELECTION_SHA),raw_complete=ref(c.INGEST/'complete.json'),raw_groups=ref(c.INGEST/'raw-g10/groups.json'),source_events=ref(c.INGEST/'source-events.json'),independent_join_audit=ref(c.INGEST/'independent-final-audit.json'),compute_layers=[20,23,26],persist_layers=[26],prefix_tokens=20,question_max_tokens=273,shards=4,batch_size=32,max_context_tokens=16384,actor_updates=0,new_rollouts_generated=0,existing30016_changed=False)

class Audited100Tests(unittest.TestCase):
 def load(self,value):
  ref=dict(path=str(c.TARGET/'cohort-config.json'),bytes=10,sha256='b'*64)
  with patch.object(c.r,'remote',side_effect=lambda x:Path(x)),patch.object(c.r,'fingerprints',return_value=ref),patch.object(c.r,'read',return_value=value),patch.object(c.r,'checked_file',side_effect=lambda x:x):
   return c.cohort_config(c.TARGET/'cohort-config.json','b'*64)
 def test_exact_budget_real_sources_and_backend(self):
  self.assertEqual(self.load(config())[1]['questions'],12800)
  for key,value in [('questions',15616),('trajectories',64000),('prefix_tokens',5),('compute_layers',[26]),('persist_layers',[20,23,26]),('actor_updates',1),('existing30016_changed',True)]:
   a=config();a[key]=value
   with self.assertRaises(ValueError,msg=key):self.load(a)
  a=config();a['raw_complete']['path']=str(c.INGEST/'future-complete.json')
  with self.assertRaises(ValueError):self.load(a)
 def test_four_disjoint_3200_G10_no_label_selection(self):
  rows=fixture();a,parts=old.split_identity(rows,8);changed=copy.deepcopy(rows)
  for g in changed:g['labels']=[1-y for y in g['labels']]
  self.assertEqual((a,parts),old.split_identity(changed,8));self.assertEqual(len({t for part in parts for g in part for t in g['trajectory_ids']}),80)
  self.assertEqual(inspect.signature(old.capture).parameters['expected_questions'].default,3904)
  self.assertIn('expected_questions=QUESTIONS//4',inspect.getsource(c.worker))
 def test_normal_two_gpu_config_bound_safe_renderer(self):
  text=d.render_sbatch(1,c.TARGET/'pair-01',c.TARGET/'code-test','a'*64,'b'*64)
  subprocess.run(['bash','-n'],input=text,text=True,check=True)
  exports='\n'.join(x for x in text.splitlines() if x.startswith('export '));subprocess.run(['bash','-euc',exports+'\ntest "$QWEN7B_CODE_ROOT" = '+str(c.TARGET/'code-test')],check=True)
  for value in ('--ntasks=2','--gres=gpu:2','--gpus-per-task=1','--gpu-bind=single:1','--qos=normal','--partition=acd_u','--config-sha256 '+('b'*64),'peer_audited100_l26_replay'):self.assertIn(value,text)
  self.assertNotIn('afterok',text);self.assertNotIn('emergency',text);self.assertNotIn('CUDA_VISIBLE_DEVICES=0',text)
  with self.assertRaises(ValueError):d.render_sbatch(0,old.TARGET/'pair-00',old.TARGET/'code-test','a'*64,'b'*64)
if __name__=='__main__':unittest.main()
