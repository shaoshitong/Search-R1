"""Small deterministic CPU identity/ownEM/quota tests; no real data/GPU use."""
import copy
import unittest
from .hard_enrichment import (prepare_candidates,select_enrichment,validate_predictions,
    QuotaUnreachable,identity_digest,product,verify_scoring_receipt)


def g(q,k):
    return dict(qid='q'+str(q),question_hash='%064x'%q,
        trajectory_ids=['q%d:t%d'%(q,j) for j in range(5)],labels=[int(j<k) for j in range(5)])


def source():
    train=[g(i, i%6) for i in range(48)]
    bank=dict(train=train,diagnostic=[g(100,0)])
    core=dict(cohort_qids=['q0','q1'],batches=[['q0','q1']])
    audit=dict(role='training_mining_exclusions',groups=[{k:r[k] for k in ('qid','question_hash')} for r in train[:2]+bank['diagnostic']],
        source_refs=[dict(path='/data/source.json',sha256='a'*64,role=role) for role in ('matched_fit','diagnostic512','selection2048','frozen_final')])
    return bank,core,audit


def candidate():
    bank,plan,audit=source()
    result=prepare_candidates(bank,plan,[audit],source_bank=dict(path='/data/bank.json',sha256='a'*64),
        features=dict(path='/data/features.json',sha256='b'*64),checkpoint=dict(path='/data/epoch4.pt',sha256='c'*64),
        exclusion_refs=[dict(path='/data/exclusion.json',sha256='d'*64)],matched_plan_ref=dict(path='/data/matched.json',sha256='e'*64),
        seed=5,synthetic_fixture=True,candidate_count=36,expected_source=48,expected_core=2)
    return result,[bank['train'][0],bank['train'][1]]


def scored(plan):
    rows=[]
    for i,group in enumerate(plan['groups']):
        # Genuine numeric formula, not invented q independent of own p_j.
        p=.02+(i%20)*.04
        ps=[p]*5;q=1-product(ps)-product([1-p for p in ps])
        rows.append(dict(qid=group['qid'],trajectory_ids=group['trajectory_ids'],strict_em=group['labels'],p_j=ps,predicted_mixed=q))
    return rows


class HardEnrichmentTests(unittest.TestCase):
    def test_prepared_plan_is_consumable_by_separate_frozen_scorer(self):
        from .score_training_candidates import validate_candidate_plan
        plan,_=candidate();bank,_,audit=source()
        accepted=validate_candidate_plan(plan,bank['train'],[audit],36)
        self.assertEqual(accepted['ordered_qid_sha256'],plan['summary']['ordered_qid_sha256'])
        self.assertEqual(accepted['ordered_own_trajectory_ids_sha256'],plan['summary']['ordered_own_trajectory_ids_sha256'])

    def test_candidate_selection_labelblind_and_excludes_only_current_not_whole169k_source(self):
        plan,_=candidate();bank,matched,audit=source()
        selected={r['qid'] for r in plan['groups']}
        self.assertTrue(selected);self.assertFalse({'q0','q1','q100'}&selected)
        self.assertEqual(plan['summary']['questions'],36)
        self.assertFalse(plan['summary']['unseen_in_initial169028_warmup_claim'])
        self.assertFalse(plan['summary']['question_selection_uses_scores'])
        self.assertTrue(all(set(r)=={'qid','question_hash','trajectory_ids','labels','split'} and r['split']=='train' for r in plan['groups']))
        self.assertEqual(plan['checkpoint'],plan['source_checkpoint'])
        for r in bank['train']:r['labels']=[1-y for y in r['labels']]
        changed=prepare_candidates(bank,matched,[audit],source_bank=plan['source_bank'],features=plan['features'],checkpoint=plan['checkpoint'],
            exclusion_refs=plan['exclusion_refs'],matched_plan_ref=plan['source_matched_plan'],seed=5,
            synthetic_fixture=True,candidate_count=36,expected_source=48,expected_core=2)
        self.assertEqual(plan['summary']['ordered_qid_sha256'],changed['summary']['ordered_qid_sha256'])
        self.assertEqual([r['trajectory_ids'] for r in plan['groups']],[r['trajectory_ids'] for r in changed['groups']])
        self.assertEqual(plan['summary']['eligible_source_questions'],48)
        self.assertEqual(plan['summary']['eligible_after_exclusions'],46)

    def test_strict_full_scoring_coverage_ids_ownEM_probability_formula(self):
        plan,_=candidate();rows=scored(plan)
        actual,proof=validate_predictions(plan,rows,expected=36)
        self.assertEqual(len(actual),36);self.assertLessEqual(proof['max_q_roundoff'],1e-12)
        mutations=[lambda rs:rs.pop(),lambda rs:rs.append(copy.deepcopy(rs[0])),
            lambda rs:rs[0]['trajectory_ids'].__setitem__(0,'foreign'),
            lambda rs:rs[0]['strict_em'].__setitem__(0,1-rs[0]['strict_em'][0]),
            lambda rs:rs[0].__setitem__('strict_em',[bool(y) for y in rs[0]['strict_em']]),
            lambda rs:rs[0]['p_j'].__setitem__(0,float('nan')),
            lambda rs:rs[0].__setitem__('predicted_mixed',.12345678)]
        for mutate in mutations:
            bad=copy.deepcopy(rows);mutate(bad)
            with self.assertRaises(ValueError):validate_predictions(plan,bad,expected=36)
        bad=copy.deepcopy(plan);bad['summary']['ordered_own_trajectory_ids_sha256']='f'*64
        with self.assertRaises(ValueError):validate_predictions(bad,rows,expected=36)

    def test_targeted_hard_strata_and_random_macro_class_budget_match(self):
        plan,core=candidate();rows=scored(plan)
        result=select_enrichment(plan,rows,core,seed=13,quota=2,expected_candidates=36,expected_core=2,synthetic_fixture=True)
        again=select_enrichment(plan,list(reversed(rows)),core,seed=13,quota=2,expected_candidates=36,expected_core=2,synthetic_fixture=True)
        self.assertEqual(result,again)
        targeted=result['branches']['targeted'];random=result['branches']['random_control']
        for branch in (targeted,random):
            summary=branch['summary'];self.assertEqual(summary['total_unique_questions'],10)
            self.assertEqual(summary['supplement_class_counts'],dict(allwrong=2,allcorrect=2,Mixed=4))
            self.assertEqual(len({r['qid'] for r in branch['groups']}),10)
            self.assertEqual({r['qid'] for r in core}<={r['qid'] for r in branch['groups']},True)
            self.assertEqual(summary['supplement_K_histogram'][0],2);self.assertEqual(summary['supplement_K_histogram'][5],2)
            self.assertFalse(summary['Mixed_internal_K_distribution_matched_claim'])
        lookup={r['qid']:r for r in rows}
        expected0=sorted([r['predicted_mixed'] for r in rows if sum(r['strict_em'])==0],reverse=True)[:2]
        expected5=sorted([r['predicted_mixed'] for r in rows if sum(r['strict_em'])==5],reverse=True)[:2]
        expected_m=sorted([r['predicted_mixed'] for r in rows if 0<sum(r['strict_em'])<5])[:2]
        self.assertAlmostEqual(targeted['summary']['strata']['high_q_allwrong']['q_mean'],sum(expected0)/2)
        self.assertAlmostEqual(targeted['summary']['strata']['high_q_allcorrect']['q_mean'],sum(expected5)/2)
        self.assertAlmostEqual(targeted['summary']['strata']['low_q_Mixed']['q_mean'],sum(expected_m)/2)
        self.assertFalse(random['summary']['question_selection_uses_scores'])
        self.assertTrue(targeted['summary']['question_selection_uses_scores'])

    def test_random_control_ids_invariant_to_scores_but_targeted_can_change(self):
        plan,core=candidate();rows=scored(plan)
        first=select_enrichment(plan,rows,core,seed=7,quota=2,expected_candidates=36,expected_core=2,synthetic_fixture=True)
        changed=copy.deepcopy(rows)
        for i,row in enumerate(changed):
            ps=[.03+(len(changed)-i)%18*.04]*5
            row.update(p_j=ps,predicted_mixed=1-product(ps)-product([1-p for p in ps]))
        second=select_enrichment(plan,changed,core,seed=7,quota=2,expected_candidates=36,expected_core=2,synthetic_fixture=True)
        self.assertEqual(first['branches']['random_control']['plan'],second['branches']['random_control']['plan'])
        self.assertEqual(first['branches']['random_control']['groups'],second['branches']['random_control']['groups'])

    def test_insufficient_class_reports_reachable_not_fake_or_resample(self):
        plan,core=candidate();rows=scored(plan)
        for group in plan['groups']:group['labels']=[0]*5
        for row in rows:row['strict_em']=[0]*5
        with self.assertRaises(QuotaUnreachable) as ctx:
            select_enrichment(plan,rows,core,quota=2,expected_candidates=36,expected_core=2,synthetic_fixture=True)
        self.assertEqual(ctx.exception.details['maximum_common_reachable_quota'],0)
        self.assertEqual(ctx.exception.details['actual_allcorrect'],0)
        self.assertTrue(ctx.exception.details['no_resampling'])

    def test_production_budgets_and_external_exclusions_are_required(self):
        bank,plan,audit=source()
        with self.assertRaisesRegex(ValueError,'Production'):
            prepare_candidates(bank,plan,[audit],source_bank={},features={},checkpoint={},exclusion_refs=[],matched_plan_ref={},candidate_count=36)
        audit['groups'].pop()
        with self.assertRaisesRegex(ValueError,'fixed diagnostic'):
            prepare_candidates(bank,plan,[audit],source_bank={},features={},checkpoint={},exclusion_refs=[],matched_plan_ref={},synthetic_fixture=True,candidate_count=36,expected_source=48,expected_core=2)
        candidate_plan,core=candidate();candidate_plan['groups'][0]['qid']=core[0]['qid']
        candidate_plan['summary']['ordered_qid_sha256']=identity_digest([g['qid'] for g in candidate_plan['groups']])
        with self.assertRaises(ValueError):
            select_enrichment(candidate_plan,scored(candidate_plan),core,quota=2,expected_candidates=36,expected_core=2,synthetic_fixture=True)

    def test_scoring_receipt_pins_complete_E4_and_same_source(self):
        candidate_plan,_=candidate();ref=dict(path='/data/candidates.json',sha256='9'*64)
        receipt=dict(status='E4_PREFIX20_TRAIN_ONLY_CANDIDATE_SCORING_COMPLETE',questions=32768,trajectories=163840,
            binding=dict(role='training_hard_mining_candidates',actor_optimizer_updates=0,probe_optimizer_updates=0,
                hard_strata_selected=False,checkpoint=dict(name='E4',steps=14418,epoch=4,sha256='c'*64),
                candidate_plan=ref,source_bank=candidate_plan['source_bank'],features=candidate_plan['features']),
            predictions=dict(path='/data/predictions.jsonl.gz',sha256='8'*64))
        receipt['binding']['checkpoint']['path']=candidate_plan['checkpoint']['path']
        receipt['binding']['plan_acceptance']=candidate_plan['summary']
        receipt['binding']['exclusion_refs']=candidate_plan['exclusion_refs']
        self.assertEqual(verify_scoring_receipt(receipt,ref,candidate_plan),receipt['predictions'])
        bad=copy.deepcopy(receipt);bad['binding']['checkpoint']['epoch']=3
        with self.assertRaises(ValueError):verify_scoring_receipt(bad,ref,candidate_plan)
        bad=copy.deepcopy(receipt);bad['binding']['checkpoint']['path']='/data/other.pt'
        with self.assertRaises(ValueError):verify_scoring_receipt(bad,ref,candidate_plan)

    def test_exclusion_audit_is_identity_only_and_requires_declared_four_scopes(self):
        from .hard_enrichment import exclusion_sets
        _,_,audit=source();self.assertEqual(exclusion_sets([audit])[2]['declared_scopes'],
            ['diagnostic512','frozen_final','matched_fit','selection2048'])
        bad=copy.deepcopy(audit);bad['groups'][0]['labels']=[0]*5
        with self.assertRaises(ValueError):exclusion_sets([bad])
        bad=copy.deepcopy(audit);bad['source_refs'].pop()
        with self.assertRaises(ValueError):exclusion_sets([bad])


if __name__=='__main__':unittest.main()
