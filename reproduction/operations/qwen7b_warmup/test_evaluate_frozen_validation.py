"""Narrow synthetic CPU contracts; no official model/checkpoint/remote IO."""
import copy
import subprocess
import unittest
from . import evaluate_frozen_validation as e
from .probe_g10_train import frozen_validation


def pool():
    groups = [dict(qid='q%d'%i, trajectory_ids=['q%d-t%d'%(i,j) for j in range(10)],
                   labels=[i%2,0,0,0,0,1,1,1,1,1]) for i in range(4)]
    selected = [dict(qid=g['qid'], trajectory_ids=g['trajectory_ids'][::2]) for g in groups]
    plan = dict(groups=selected, summary=dict(role='validation', labels_consulted=False,
                                            selected_g5_sha256=e.digest(selected)))
    return groups,plan


def balanced(name='E2'):
    step,epoch = e.EXPECTED[name]
    descriptor = dict(name=name, path='/remote/fake.pt', sha256='a'*64, bytes=10, steps=step,
        source_config=dict(sha256='b'*64), model_dir='/remote/model')
    receipt = dict(descriptor, epoch=epoch, cursor=0, completed_epoch=True,
                   independent_cpu_reload=True)
    config = dict(model_dir=descriptor['model_dir'],prefix_tokens=20,question_max_tokens=273,
                  native_head_layer=27,fit_ema=.99)
    state = dict(role='EPOCH1_BALANCED_RANK1_CONTINUATION',steps=step,epoch=epoch,
        cursor=0,completed_epoch=True,actor_updates=0,config_sha256='b'*64,
        rank_states=[dict(rank=i,steps=step,rng='opaque synthetic RNG') for i in range(4)])
    return descriptor,receipt,config,state


def prediction(qid,labels,score):
    return dict(qid=qid,trajectory_ids=['%s-t%d'%(qid,i) for i in range(5)],
                strict_em=labels,p_j=[.5]*5,predicted_mixed=score)


class FrozenValidationTests(unittest.TestCase):
    def test_exact_existing_frozen_consumer_and_no_label_selection(self):
        groups,plan=pool()
        self.assertEqual(e.fixed_groups(groups,plan),frozen_validation(groups,plan))
        before=e.fixed_groups(groups,plan)
        changed=copy.deepcopy(groups)
        for g in changed:g['labels']=[1-y for y in g['labels']]
        after=e.fixed_groups(changed,plan)
        self.assertEqual([g['trajectory_ids'] for g in before],[g['trajectory_ids'] for g in after])
        self.assertNotEqual([g['labels'] for g in before],[g['labels'] for g in after])

    def test_frozen_plan_rejects_labels_wrong_ids_and_mutated_hash(self):
        groups,plan=pool()
        for mutate in ('label','foreign','duplicate','role','hash'):
            own=copy.deepcopy(plan)
            if mutate=='label':own['groups'][0]['labels']=[0]*5
            if mutate=='foreign':own['groups'][0]['trajectory_ids'][0]='other-question'
            if mutate=='duplicate':own['groups'][0]['trajectory_ids'][1]=own['groups'][0]['trajectory_ids'][0]
            if mutate=='role':own['summary']['role']='untouched_final'
            if mutate=='hash':own['summary']['selected_g5_sha256']='changed'
            if mutate!='hash':own['summary']['selected_g5_sha256']=e.digest(own['groups'])
            with self.assertRaises(ValueError):e.fixed_groups(groups,own)

    def test_both_balanced_epoch_metadata_and_four_rank_lineage(self):
        for name in ('E2','E4'):
            values=balanced(name);e.validate_checkpoint_metadata(*values)
            for key,value in [('steps',1),('actor_updates',1),('epoch',1),('completed_epoch',False),('cursor',2)]:
                invalid=list(copy.deepcopy(values));invalid[-1][key]=value
                with self.assertRaises(ValueError):e.validate_checkpoint_metadata(*invalid)
            invalid=list(copy.deepcopy(values));invalid[-1]['rank_states'][0]['steps']=0
            with self.assertRaises(ValueError):e.validate_checkpoint_metadata(*invalid)
            invalid=list(copy.deepcopy(values));invalid[2]['prefix_tokens']=5
            with self.assertRaises(ValueError):e.validate_checkpoint_metadata(*invalid)

    def test_three_event_actual_budget_schema_not_completed_epoch_fiction(self):
        d,r,c,s=balanced();d.update(name='event128',steps=10592)
        r.update(path=d['path'],sha256=d['sha256'],bytes=d['bytes'],steps=10592,cursor=128,
                 matched_binding_sha256='binding',accepted_hashes={'ema':'ema'})
        c=dict(model_dir=d['model_dir'],update_budget=128,
               recipe=dict(id='event',loss_kwargs=dict(trajectory_weight=1.,three_event_weight=1.)))
        s=dict(role='NEW_SINGLE_GPU_MATCHED_PROBE_LOSS_ABLATION',steps=10592,cursor=128,
               actor_updates=0,config_sha256='b'*64,matched_binding_sha256='binding',
               accepted_hashes={'ema':'ema'},source_all_rank_states=[dict(rng='opaque') for i in range(4)])
        e.validate_checkpoint_metadata(d,r,c,s)
        for bad in (127,129):
            changed=copy.deepcopy(s);changed['cursor']=bad
            with self.assertRaises(ValueError):e.validate_checkpoint_metadata(d,r,c,changed)
        c['recipe']['loss_kwargs']['binary_mixed_weight']=1.
        with self.assertRaises(ValueError):e.validate_checkpoint_metadata(d,r,c,s)

    def test_group_ba_two_recalls_class_fprs_auc_k_and_tie_selection(self):
        rows=[prediction('d',[1,0,0,0,0],.9),prediction('b',[1,1,0,0,0],.4),
              prediction('a',[0]*5,.6),prediction('c',[1]*5,.1)]
        result=e.summarize(rows,2);m=result['pool']
        self.assertEqual(m['confusion'],dict(tp=1,fp=1,fn=1,tn=1))
        self.assertEqual(m['mixed_recall'],.5);self.assertEqual(m['nonmixed_recall'],.5)
        self.assertEqual(m['mixed_balanced_accuracy'],.5);self.assertEqual(m['mixed_roc_auc'],.75)
        self.assertEqual(m['allwrong_false_positive_rate'],1.)
        self.assertEqual(m['allcorrect_false_positive_rate'],0.)
        self.assertEqual(m['k_histogram'],[1,1,1,0,0,1])
        self.assertEqual(result['selected512']['mixed_fraction'],.5)
        self.assertEqual(result['selected_qids_sha256'],e.digest(['d','a']))
        tied=[prediction('z',[1]*5,.8),prediction('a',[0]*5,.8),prediction('m',[1,0,0,0,0],.8)]
        first=e.summarize(tied,1)
        self.assertEqual(first['selected_qids_sha256'],e.digest(['a']))
        self.assertEqual(first['pool']['mixed_roc_auc'],.5)
        for r in tied:r['strict_em']=[1-y for y in r['strict_em']]
        self.assertEqual(e.summarize(tied,1)['selected_qids_sha256'],first['selected_qids_sha256'])

    def test_point_five_collapse_reports_ba_chance_not_high_trajectory_success(self):
        q=1-.5**5-.5**5
        rows=[prediction('wrong',[0]*5,q),prediction('right',[1]*5,q),
              prediction('mixed',[1,0,0,0,0],q)]
        m=e.summarize(rows,2)['pool']
        self.assertEqual(m['mixed_recall'],1.);self.assertEqual(m['nonmixed_recall'],0.)
        self.assertEqual(m['mixed_balanced_accuracy'],.5);self.assertEqual(m['mixed_roc_auc'],.5)
        self.assertEqual(m['allwrong_false_positive_rate'],1.)
        self.assertEqual(m['allcorrect_false_positive_rate'],1.)

    def test_deployment_normal_one_gpu_no_dependency_safe_exports(self):
        from .deploy_frozen_validation_eval import render_sbatch, BASE
        code=BASE+'/code-0123456789ab';config=BASE+'/inputs/config.json'
        script=render_sbatch(BASE,code,config,'a'*64)
        subprocess.run(['bash','-n'],input=script,text=True,check=True)
        exports='\n'.join(line for line in script.splitlines() if line.startswith(('FV_','export ')))
        subprocess.run(['bash','-euc',exports+'\ntest "$QWEN7B_CODE_ROOT" = "'+code+'"\n'],check=True)
        for expected in ('--partition=acd_u','--qos=normal','--gres=gpu:1','--cpus-per-task=8','--mem=64G'):
            self.assertIn(expected,script)
        self.assertNotIn('--dependency',script);self.assertNotIn('afterok',script)
        self.assertNotIn('@@FV_',script)


if __name__=='__main__':unittest.main()
