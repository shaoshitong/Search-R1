"""Narrow CPU identity, geometry, and dual-device dispatch boundaries."""
import copy,hashlib,json,subprocess,unittest
from . import peer_audited122_l26_replay as r
from . import deploy_peer_audited122_l26 as d

def fixture():
 return [dict(qid=json.dumps(['nq' if i%2 else 'hotpotqa',i]),question_hash=hashlib.sha256(str(i).encode()).hexdigest(),trajectory_ids=['q%d-t%d'%(i,j) for j in range(10)],labels=[i%2]*10) for i in range(8)]

class Audited122ReplayTests(unittest.TestCase):
 def test_all_accepted_not_filtered_by_labels(self):
  a=fixture();ids,parts=r.split_identity(a,8)
  b=copy.deepcopy(a)
  for g in b:g['labels']=[1-x for x in g['labels']]
  self.assertEqual((ids,parts),r.split_identity(b,8));self.assertEqual([len(p) for p in parts],[2]*4)
  self.assertEqual(len({t for p in parts for g in p for t in g['trajectory_ids']}),80)
  self.assertTrue(all(set(g)=={'qid','question_hash','trajectory_ids'} for g in ids))
 def test_bad_identity_or_domain_rejected(self):
  for field in ('qid','question_hash','trajectory_ids'):
   a=fixture();a[1][field]=a[0][field]
   with self.assertRaises(ValueError):r.split_identity(a,8)
  a=fixture();a[0]['qid']=json.dumps(['final',0])
  with self.assertRaises(ValueError):r.split_identity(a,8)
 def test_no_silent_geometry_truncation(self):
  row=dict(initial_prompt_token_ids=list(range(273)),policy_token_ids=list(range(20)),prefix_segments=[dict(context_tokens=list(range(293)))])
  original=copy.deepcopy(row);self.assertEqual(r.geometry(row),dict(prompt=273,policy=20,context=293,overlength=False));self.assertEqual(original,row)
  row['initial_prompt_token_ids'].append(273);self.assertTrue(r.geometry(row)['overlength']);self.assertEqual(len(row['initial_prompt_token_ids']),274)
  row['initial_prompt_token_ids']=[];row['prefix_segments'][0]['context_tokens']=list(range(4097));self.assertTrue(r.geometry(row)['overlength'])
 def test_two_jobs_two_gpu_and_safe_exports(self):
  text=d.render_sbatch(1,r.TARGET/'pair-01',r.TARGET/'code-test','a'*64)
  subprocess.run(['bash','-n'],input=text,text=True,check=True)
  exports='\n'.join(x for x in text.splitlines() if x.startswith('export '))
  subprocess.run(['bash','-euc',exports+'\ntest "$QWEN7B_CODE_ROOT" = '+str(r.TARGET/'code-test')],check=True)
  for word in ('--ntasks=2','--gres=gpu:2','--gpus-per-task=1','--gpu-bind=single:1','--mem=128G','--partition=acd_u','--qos=normal','--pair 1'):self.assertIn(word,text)
  self.assertNotIn('afterok',text);self.assertNotIn('emergency',text)
  with self.assertRaises(ValueError):d.render_sbatch(2,r.TARGET/'pair-02',r.TARGET/'code-test','a'*64)
 def test_submit_uncertainty_and_feature_isolation(self):
  for word in ('submission-intent.json','Unknown submit outcome','Existing feature output','squeue','--hold'):self.assertIn(word,d.SUBMIT)
  self.assertEqual(len(d.FILES),len(set(d.FILES)));self.assertNotIn('CUDA_VISIBLE_DEVICES=0',d.render_sbatch(0,r.TARGET/'pair-00',r.TARGET/'code-test','a'*64))
if __name__=='__main__':unittest.main()
