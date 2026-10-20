"""Small identity/dispatch boundary checks; no model/features loaded."""
import copy,hashlib,json,subprocess,unittest
from .peer_expanded_l26_replay import split_new,TARGET
from . import deploy_peer_expanded_l26 as d

def fixture():
    groups=[dict(qid=json.dumps(['nq',i]),question_hash=hashlib.sha256(str(i).encode()).hexdigest(),trajectory_ids=['q%d-t%d'%(i,j) for j in range(10)]) for i in range(10)]
    return dict(role='train',groups=groups,remaining_groups=groups[2:],summary=dict(labels_consulted=False,scores_consulted=False,existing8192_preserved_exactly=True,existing_questions=2,new_questions=8))

class ExpandedReplayTest(unittest.TestCase):
    def test_only_additional_fixed_identities_disjoint(self):
        p=fixture();parts=split_new(p,existing=2,added=8)
        self.assertEqual([len(v) for v in parts],[2]*4)
        old={t for g in p['groups'][:2] for t in g['trajectory_ids']}
        new=[t for part in parts for g in part for t in g['trajectory_ids']]
        self.assertEqual(len(new),80);self.assertFalse(old.intersection(new));self.assertEqual(len(set(new)),80)
    def test_prior_or_order_mutation_rejected(self):
        p=fixture();p['remaining_groups']=list(reversed(p['remaining_groups']))
        with self.assertRaises(ValueError):split_new(p,existing=2,added=8)
    def test_actual_ten_ids_and_unique_hash(self):
        for field in ('trajectory_ids','question_hash'):
            p=fixture();p['groups'][3][field]=copy.deepcopy(p['groups'][0][field])
            with self.assertRaises(ValueError):split_new(p,existing=2,added=8)
    def test_labels_and_wrong_domain_rejected(self):
        p=fixture();p['groups'][3]['labels']=[1]*10
        with self.assertRaises(ValueError):split_new(p,existing=2,added=8)
        p=fixture();p['groups'][3]['qid']=json.dumps(['final',3])
        with self.assertRaises(ValueError):split_new(p,existing=2,added=8)
    def test_selection_flags_and_role(self):
        for key in ('labels_consulted','scores_consulted'):
            p=fixture();p['summary'][key]=True
            with self.assertRaises(ValueError):split_new(p,existing=2,added=8)
        p=fixture();p['role']='validation'
        with self.assertRaises(ValueError):split_new(p,existing=2,added=8)
    def test_render_exports_resources_and_independence(self):
        code=TARGET/'code-unit';text=d.render_sbatch(0,TARGET/'shard-00',code,TARGET/'sel','a'*64,'b'*64)
        subprocess.run(['bash','-n'],input=text,text=True,check=True)
        exports='\n'.join(x for x in text.splitlines() if x.startswith('export '))
        subprocess.run(['bash','-euc',exports+'\ntest "$QWEN7B_CODE_ROOT" = '+str(code)],check=True)
        self.assertIn('--partition=acd_u',text);self.assertIn('--qos=normal',text);self.assertIn('--gres=gpu:1',text)
        self.assertNotIn('afterok',text);self.assertNotIn('emergency',text);self.assertIn('peer_expanded_l26_replay',text)
        with self.assertRaises(ValueError):d.render_sbatch(0,TARGET.parent/'old',code,'x','a'*64,'b'*64)
    def test_guarded_unknown_submission_contract_preserved(self):
        self.assertIn('submission-intent.json',d.legacy.SUBMIT);self.assertIn('Unknown submit outcome',d.legacy.SUBMIT)
        self.assertIn('Existing feature output',d.legacy.SUBMIT);self.assertIn("['squeue'",d.legacy.SUBMIT)
        self.assertEqual(len(d.FILES),len(set(d.FILES)))

if __name__=='__main__':unittest.main()
