"""Only handoff/renderer and real toy fullstate next-update qualifications."""
import copy,tempfile,unittest
from pathlib import Path
import torch
from . import direct_hidden_resume as r,direct_hidden_checkpoint as cp
from .test_direct_hidden_count import setup,update


class ResumeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def fixtures(self):
        c=dict(source_cursor=0,source_adam_steps=10464,update_budget=938,plan=dict(sha256='a'*64))
        last=dict(cursor=64,accepted_hashes=dict(raw='x'))
        verified=dict(cursor=64,legacy_Adam_steps=10528,new_head_Adam_steps=64,data_cursor=dict(cursor=64,plan_sha256='a'*64),
            accepted_hashes=last['accepted_hashes'],source_all_four_RNG_preserved=True)
        return [c,dict(status='EXACT_JOINT_ACCEPTED_CURSOR_SAVED',cursor=64),last,
            dict(cursor=64,steps=10528,head_steps=64),dict(next_cursor=64),verified,'743619|COMPLETED|0:0|02:00:00']

    def test_current_single_rank_handoff_preserves_original_archive_and_clocks(self):
        x=r.validate_handoff(*self.fixtures())
        self.assertEqual(x['remaining_updates'],874);self.assertFalse(x['hardware_topology_changed'])
        self.assertEqual(x['current_source_world_size'],1);self.assertEqual(x['original_branch_source_world_size'],4)
        self.assertFalse(x['new_head_optimizer_reset'])

    def test_active_unresolved_missing_state_and_already_complete_fail_closed(self):
        cases=[];x=self.fixtures();x[-1]='743619|RUNNING|0:0|00:15:00';cases.append(x)
        x=self.fixtures();x[4]['next_cursor']=65;cases.append(x)
        x=self.fixtures();x[3]['cursor']=63;cases.append(x)
        x=self.fixtures();x[5]['new_head_Adam_steps']=0;cases.append(x)
        for x in cases:
            with self.assertRaises(ValueError):r.validate_handoff(*x)
        with self.assertRaises(ValueError):r.validate_handoff(*self.fixtures(),complete_exists=True)

    def test_full_checkpoint_next_update_replay_preserves_old_and_new_Adam(self):
        values=setup();update(values);m,e,o,h,he,ho,own,old,*_=values;device=torch.device('cpu')
        bound=dict(plan=dict(sha256='a'*64),same_scientific_config=True)
        saved=copy.deepcopy(cp.joint_blob(*values[:6],device,1,bound,old['source_all_rank_states'],dict(cursor=1,plan_sha256='a'*64)))
        update(values);expected=cp.state_hashes(*values[:6],device)
        cp.restore_joint(saved,*values[:6],device,expected_binding=bound,expected_cursor=1)
        update(values);self.assertEqual(expected,cp.state_hashes(*values[:6],device))
        blob=cp.joint_blob(*values[:6],device,2,bound,old['source_all_rank_states'],dict(cursor=2,plan_sha256='a'*64))
        blob['continuation_first_transaction']=dict(cursor=2,steps=10466,head_steps=2)
        with tempfile.TemporaryDirectory() as d:
            receipt=cp.save_joint(Path(d),blob)
            self.assertEqual(receipt['legacy_Adam_steps'],10466);self.assertEqual(receipt['new_head_Adam_steps'],2)
            self.assertEqual(receipt['accepted_hashes'],expected)

    def test_new_job_reuses_exact501_config_without_dependency_or_E1_retraining(self):
        text=r.render('/remote/code501','/remote/immutable/resume.py','b'*64,'c'*64)
        self.assertIn('--time=03:00:00',text);self.assertIn('--qos=normal',text);self.assertIn('--gres=gpu:1',text)
        self.assertNotIn('dependency',text);self.assertNotIn('emergency',text)
        self.assertIn('q7b-direct-count938-r1',text);self.assertIn('resume-%j.log',text)
        self.assertIn('m.launch(',text)
        from . import deploy_direct_hidden_resume as deploy
        for action in ('stage','qualify','prepare','submit','status'):
            compile(deploy.remote_source(dict(base='/remote/test',action=action),action),'remote-'+action,'exec')

if __name__=='__main__':unittest.main()
