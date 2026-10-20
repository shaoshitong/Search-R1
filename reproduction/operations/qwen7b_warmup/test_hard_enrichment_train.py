"""Actual tiny CPU Adam/replay/serialization tests, never a real-E4/GPU claim."""
import copy,gc,subprocess,tempfile,unittest
from pathlib import Path
import torch
from . import hard_enrichment_train as t
from . import probe_ablation_train as u
from .test_probe_ablation_train import toy,ToyReader,cpu_blob


def config():
    return dict(role=t.ROLE,variant='targeted',source_checkpoint=dict(sha256=t.SOURCE_SHA,steps=14418,epoch=4),
        global_groups=48,update_budget=477,fit_questions=22896,prefix_tokens=20,question_max_tokens=273,
        recipe=dict(loss_kwargs=copy.deepcopy(t.LOSS)),lr=5e-6,ema_decay=.99,weight_decay=0,actor_updates=0,
        four_rank_bitexact_resume_claim=False,micro_groups=2,save_every=32,E1_reset=False,
        source_raw_ema_adam_RNG_preserved=True,canonical_source_rng_rank=0,source_world_size=4,current_world_size=1)


class HardEnrichmentTrainTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_E4_source_budget_loss_and_state_continuity_guards(self):
        c=config();t.validate_config(c)
        for key,value in [('global_groups',51),('update_budget',128),('E1_reset',True),('lr',1e-5),
                          ('source_raw_ema_adam_RNG_preserved',False),('canonical_source_rng_rank',1)]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):t.validate_config(bad)
        bad=copy.deepcopy(c);bad['source_checkpoint']['sha256']=u.SOURCE_SHA
        with self.assertRaises(ValueError):t.validate_config(bad)
        bad=copy.deepcopy(c);bad['recipe']['loss_kwargs']['within_rank_weight']=1.
        with self.assertRaises(ValueError):t.validate_config(bad)

    def test_fixed_once_question_plan_and_external_hash_isolation(self):
        groups=[dict(qid=str(i),question_hash='%064x'%i,split='train',trajectory_ids=['%d-%d'%(i,j) for j in range(5)],labels=[i%2,0,1,0,0]) for i in range(12)]
        qids=[g['qid'] for g in groups]
        p=dict(cohort_qids=qids,batches=[qids[:6],qids[6:]],global_groups=6,update_budget=2,each_question_once=True,epochs=1,ordered_qid_sha256=u.digest(qids))
        self.assertEqual(len(t.validate_plan(p,groups,synthetic_fixture=True)),12)
        for kwargs in (dict(forbidden_qids=['0']),dict(forbidden_hashes={groups[0]['question_hash']})):
            with self.assertRaises(ValueError):t.validate_plan(p,groups,synthetic_fixture=True,**kwargs)
        bad=copy.deepcopy(p);bad['batches'][1][0]='0'
        with self.assertRaises(ValueError):t.validate_plan(bad,groups,synthetic_fixture=True)
        bad=copy.deepcopy(groups);bad[1]['labels'][0]=True
        with self.assertRaises(ValueError):t.validate_plan(p,bad,synthetic_fixture=True)

    def test_true_B48_first_next_update_replays_Adam_EMA_RNG_without_reset(self):
        model,ema,opt,groups,c=toy();c['recipe']['loss_kwargs']=copy.deepcopy(t.LOSS)
        qids=list(groups)[:48];source=cpu_blob(model,ema,opt);device=torch.device('cpu')
        metrics,proof=t.replay_next(source,model,ema,opt,ToyReader(),groups,qids,c,device)
        self.assertTrue(proof['passed']);self.assertEqual(proof['source_steps'],1);self.assertEqual(proof['accepted_steps'],2)
        self.assertEqual(metrics['global_questions'],48);self.assertEqual(metrics['inter_actual_pairs'],24*24)
        self.assertEqual(metrics['within_eligible_groups'],0)
        self.assertEqual(proof['expected'],proof['actual'])
        self.assertEqual(int(source['optimizer']['state'][0]['step']),1)
        self.assertNotEqual(u.optimizer_hash(opt),u.optimizer_blob_hash(source['optimizer']))

    def test_actual_complete_checkpoint_CPU_reload_and_next_update_match(self):
        model,ema,opt,groups,c=toy();c['recipe']['loss_kwargs']=copy.deepcopy(t.LOSS);device=torch.device('cpu');reader=ToyReader();qids=list(groups)[:48]
        source=cpu_blob(model,ema,opt);t.replay_next(source,model,ema,opt,reader,groups,qids,c,device)
        binding=dict(config_sha256='config',matched_binding_sha256='matched')
        accepted=copy.deepcopy(t.blob(model,ema,opt,device,1,binding,source['source_all_rank_states']))
        before=u.state_hashes(model,ema,opt,device)
        with tempfile.TemporaryDirectory() as d:
            receipt=u.save_checkpoint(Path(d),accepted)
            saved=torch.load(u.checked(receipt),map_location='cpu',mmap=True,weights_only=False)
            self.assertEqual(saved['role'],t.ROLE);self.assertEqual(saved['source_steps'],14418)
            self.assertEqual(len(saved['source_all_rank_states']),4);self.assertFalse(saved['four_rank_bitexact_resume_claim'])
            u.run_update(model,ema,opt,reader,groups,qids,c,device);expected=u.state_hashes(model,ema,opt,device)
            u.restore_blob(saved,model,ema,opt,device);self.assertEqual(before,u.state_hashes(model,ema,opt,device))
            u.run_update(model,ema,opt,reader,groups,qids,c,device);self.assertEqual(expected,u.state_hashes(model,ema,opt,device))
            del saved;gc.collect()

    def test_resume_rejects_unsaved_later_state_and_final_only_evaluation_position(self):
        c=config();b=dict(config_sha256='c',matched_binding_sha256='m')
        saved=dict(role=t.ROLE,cursor=32,steps=14450,source_steps=14418,stage='training',**b)
        t.check_resume(saved,32,c,b,dict(cursor=32),dict(next_cursor=32))
        for kwargs in (dict(progress=dict(cursor=33)),dict(intent=dict(next_cursor=33))):
            with self.assertRaises(ValueError):t.check_resume(saved,32,c,b,**kwargs)
        final=dict(saved,cursor=477,steps=14895,stage='evaluation_pending');t.check_resume(final,477,c,b)
        t.require_final_checkpoint(final)
        with self.assertRaises(ValueError):t.require_final_checkpoint(saved)

    def test_two_normal_independent_one_GPU_two_hour_shells(self):
        for variant in ('targeted','random_control'):
            s=t.render_sbatch(variant,'/data/x/'+variant,'/data/x/code')
            subprocess.run(['bash','-n'],input=s,text=True,check=True)
            for text in ('--partition=acd_u','--qos=normal','--gres=gpu:1','--time=02:00:00'):
                self.assertIn(text,s)
            self.assertNotIn('--dependency',s);self.assertNotIn('emergency',s)


if __name__=='__main__':unittest.main()
