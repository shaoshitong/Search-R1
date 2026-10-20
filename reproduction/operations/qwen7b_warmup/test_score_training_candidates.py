"""Synthetic identity/role/ownEM tests; no model, GPU, checkpoint or remote IO."""
import copy
import ast
import subprocess
import unittest
from . import score_training_candidates as s


def fixture(n=8):
    canonical = [dict(qid='q%d'%i, question_hash='question-hash%d'%i,
        trajectory_ids=['q%d-t%d'%(i,j) for j in range(5)], labels=[i%2,0,1,0,1]) for i in range(n)]
    groups=[dict(copy.deepcopy(g),split='train') for g in canonical[:4]]
    summary=dict(questions=len(groups),trajectories=5*len(groups),
        question_selection_uses_labels=False,question_selection_uses_scores=False,
        validation_or_final_questions_used=False,unseen_in_initial169028_warmup_claim=False,
        unseen_in_previous_finetuning_claim=False,excluded_current_matched6528_fit=True,
        ordered_qid_sha256=s.e.digest([g['qid'] for g in groups]))
    plan=dict(role=s.ROLE,groups=groups,summary=summary)
    audits=[dict(role='training_mining_exclusions',groups=[dict(qid='q7',question_hash='question-hash7')],
                 source_refs=[dict(path='/remote/final-identity-receipt',sha256='opaque')])]
    return plan,canonical,audits


class CandidateScoringTests(unittest.TestCase):
    def test_authentic_train_ids_labels_and_honest_exposure_accepted(self):
        plan,canonical,audits=fixture()
        accepted=s.validate_candidate_plan(plan,canonical,audits,4)
        self.assertEqual(accepted['questions'],4);self.assertEqual(accepted['trajectories'],20)
        self.assertFalse(accepted['unseen_in_initial169028_warmup_claim'])
        self.assertFalse(accepted['unseen_in_previous_finetuning_claim'])
        self.assertTrue(accepted['excluded_current_matched6528_fit'])

    def test_exclusion_by_either_qid_or_content_hash_is_enforced(self):
        for exclusion in (dict(qid='q0',question_hash='otherhash'),
                          dict(qid='otherqid',question_hash='question-hash0')):
            plan,canonical,audits=fixture();audits[0]['groups'].append(exclusion)
            with self.assertRaises(ValueError):s.validate_candidate_plan(plan,canonical,audits,4)

    def test_never_claims_unseen_warmup_or_uses_scores_or_validation(self):
        for key in ('question_selection_uses_labels','question_selection_uses_scores',
                    'validation_or_final_questions_used','unseen_in_initial169028_warmup_claim',
                    'unseen_in_previous_finetuning_claim'):
            plan,canonical,audits=fixture();plan['summary'][key]=True
            with self.assertRaises(ValueError):s.validate_candidate_plan(plan,canonical,audits,4)
        plan,canonical,audits=fixture();plan['role']='validation'
        with self.assertRaises(ValueError):s.validate_candidate_plan(plan,canonical,audits,4)

    def test_own_em_or_trajectory_swaps_are_rejected(self):
        for kind in ('label','trajectory','hash','split','score'):
            plan,canonical,audits=fixture()
            if kind=='label':plan['groups'][0]['labels'][0]=1-plan['groups'][0]['labels'][0]
            if kind=='trajectory':plan['groups'][0]['trajectory_ids'][0]='q1-t0'
            if kind=='hash':plan['groups'][0]['question_hash']='otherhash'
            if kind=='split':plan['groups'][0]['split']='final'
            if kind=='score':plan['groups'][0]['predicted_mixed']=.9
            with self.assertRaises(ValueError):s.validate_candidate_plan(plan,canonical,audits,4)

    def test_repeated_or_padded_candidate_and_wrong_order_refused(self):
        plan,canonical,audits=fixture();plan['groups'][1]=copy.deepcopy(plan['groups'][0])
        plan['summary']['ordered_qid_sha256']=s.e.digest([g['qid'] for g in plan['groups']])
        with self.assertRaises(ValueError):s.validate_candidate_plan(plan,canonical,audits,4)
        plan,canonical,audits=fixture();plan['groups'].reverse()
        with self.assertRaises(ValueError):s.validate_candidate_plan(plan,canonical,audits,4)
        plan,canonical,audits=fixture()
        with self.assertRaises(ValueError):s.validate_candidate_plan(plan,canonical,audits,32768)

    def test_exclusions_consume_only_identity_audits_never_raw_final_rows(self):
        plan,canonical,audits=fixture();audits[0]['groups'][0]['labels']=[1]*5
        with self.assertRaises(ValueError):s.validate_candidate_plan(plan,canonical,audits,4)
        audits[0]['groups'][0].pop('labels');audits[0]['role']='final-evaluation-output'
        with self.assertRaises(ValueError):s.validate_candidate_plan(plan,canonical,audits,4)

    def test_labels_cannot_reorder_identity_selected_candidates(self):
        plan,canonical,audits=fixture();before=s.validate_candidate_plan(plan,canonical,audits,4)
        for own in canonical:own['labels']=[1-y for y in own['labels']]
        for group in plan['groups']:group['labels']=[1-y for y in group['labels']]
        after=s.validate_candidate_plan(plan,canonical,audits,4)
        self.assertEqual(before,after)

    def test_scorer_sbatch_normal_one_gpu_one_hour_and_real_jobid_newline(self):
        from . import deploy_training_candidate_score as d
        script=d.render_sbatch(d.BASE,d.BASE+'/code-0123456789ab',d.BASE+'/inputs/config.json','a'*64)
        subprocess.run(['bash','-n'],input=script,text=True,check=True)
        setup='\n'.join(line for line in script.splitlines() if line.startswith(('FV_','export ')))
        subprocess.run(['bash','-euc',setup+'\ntest "$QWEN7B_CODE_ROOT" = "'+d.BASE+'/code-0123456789ab"\n'],check=True)
        for literal in ('--partition=acd_u','--qos=normal','--gres=gpu:1','--mem=64G','--time=01:00:00',
                        '-m reproduction.operations.qwen7b_warmup.score_training_candidates'):
            self.assertIn(literal,script)
        self.assertNotIn('afterok',script);self.assertNotIn('--dependency',script)
        writes=[node for node in ast.walk(ast.parse(d.SUBMIT)) if isinstance(node,ast.Call) and
                isinstance(node.func,ast.Attribute) and node.func.attr=='write' and node.args and
                isinstance(node.args[0],ast.BinOp)]
        self.assertEqual(len(writes),1)
        self.assertEqual(ast.literal_eval(writes[0].args[0].right),'\n')


if __name__=='__main__':unittest.main()
