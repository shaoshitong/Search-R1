"""Authentic two-stage group selection and fixed-suffix/history recovery tests."""
import copy,json,tempfile,unittest,gzip
from pathlib import Path
from unittest.mock import patch
from .two_stage_selection import specification,ordinary,checked
from .periodic_data import select_actual_mixed_groups
from .adaptive_candidates import digest
from .rollout import question_count

class TwoStageTests(unittest.TestCase):
    def config(self):return dict(run_id='run',two_stage_selection=specification(62),actual_em_selection_fallback='probe_score_fill')
    def test_only_authorized_suffix_ordinary_uses_1024(self):
        c=self.config()
        self.assertFalse(ordinary(c,61));self.assertTrue(ordinary(c,62));self.assertFalse(ordinary(c,70));self.assertTrue(ordinary(c,200))
        self.assertFalse(ordinary(dict(c,diagnostic_only=True),62))
        for changed in ['prefix_questions','continuation_questions','actor_questions','periodic_questions']:
            wrong=copy.deepcopy(c);wrong['two_stage_selection'][changed]+=4
            with self.assertRaises(ValueError):ordinary(wrong,62)
    def test_fixed_budget_is_5120_not_a_mixed_feedback_forecast(self):
        c=self.config();r=dict(schema_version=2,run_id='run',outer=62,policy_sha256=digest(c['two_stage_selection']),questions=5120,mode=c['two_stage_selection']['mode'],previous_outer=61,previous_joint_sha256='a'*64,consumed_before=100000,consumed_after=105120,fixed_budget_not_mixed_feedback=True)
        self.assertEqual(checked(c,62,r),r)
        for key,v in [('questions',1024),('consumed_after',105124),('previous_outer',60)]:
            with self.assertRaises(ValueError):checked(c,62,dict(r,**{key:v}))
    def test_1024_full_g5_selects_512_actual_mixed_and_preserves_original_ordinals(self):
        groups=[];labels=[];prefixes={};provenance=dict(actor_version='a',prefix_scan_sha256='scan',pool_selection_sha256='top1024')
        # Top1024 can contain sparse original ordinals all the way to5119.
        for i in range(1024):
            ordinal=i*5+4;tids=[f'q{i}-g{j}' for j in range(5)]
            groups.append(dict(qid=f'q{i}',candidate_ordinal=ordinal,q=i/1024,question_hash=f'h{i}',trajectory_ids=tids))
            for j,tid in enumerate(tids):
                prefixes[tid]=tid+'-prefix';labels.append(dict(provenance,trajectory_id=tid,qid=f'q{i}',strict_em=int(i<600 and j==0),prefix_sha256=prefixes[tid],selection_sha256=provenance['pool_selection_sha256']))
                labels[-1].pop('pool_selection_sha256')
        out,stats=select_actual_mixed_groups(groups,labels,fallback_mode='probe_score_fill',provenance=provenance,expected_prefixes=prefixes,expected_questions=1024,ordinal_limit=5120)
        self.assertEqual(len(out),512);self.assertTrue(all(g['actual_mixed'] for g in out));self.assertEqual(stats['actual_mixed_candidate_count'],600)
        self.assertEqual(out[0]['candidate_ordinal'],599*5+4)
        labels[0]['strict_em']=False
        with self.assertRaises(ValueError):select_actual_mixed_groups(groups,labels,fallback_mode='probe_score_fill',provenance=provenance,expected_prefixes=prefixes,expected_questions=1024,ordinal_limit=5120)
    def test_end_to_end_projection_reuses_exact_1024_pool_tokens_and_logprobs(self):
        from .test_actual_em_selection import candidates,write_json
        from .periodic_data import actual_em_selection,project_top512
        from .rollout import file_item,iter_rows
        from reproduction.prefix20_online.pool import digest as file_sha
        groups,labels,prefixes,provenance=candidates(600)
        groups=groups[:1024];wanted={tid for g in groups for tid in g['trajectory_ids']}
        labels=[r for r in labels if r['trajectory_id'] in wanted]
        for g in groups:g['candidate_ordinal']+=2048
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'outers/outer-000062'
            common=dict(run_id='run',outer=62,actor_version='actor0',prefix_scan_sha256='scan')
            write_json(root/'prefix/complete.json',dict(common,scan_sha256='scan',questions=5120))
            pp=root/'probe-score/pool-selection.json';pred=root/'probe-score/selection.json'
            write_json(pp,dict(common,selected=groups,selected_questions=1024,outcome_labels_used=False))
            write_json(pred,dict(common,selected=groups[:512],selected_questions=512,outcome_labels_used=False))
            idx=root/'probe-score/index.json';write_json(idx,[dict(trajectory_id=k,prefix_sha256=v) for k,v in prefixes.items()])
            write_json(root/'probe-score/manifest.json',dict(index_sha256=file_sha(idx),selection_sha256=file_sha(pred),prefix_scan_sha256='scan'))
            original_pred=pred.read_bytes();full=root/'full-pool';full.mkdir()
            for row in labels:row.update(selection_sha256=file_sha(pp),response_token_ids=[7,9],response_log_probs=[-.3,-.4])
            own=full/'own-em.jsonl';own.write_text(''.join(json.dumps(r)+'\n' for r in labels));shards=[]
            for rank in range(4):
                path=full/('rank-%d-full.jsonl.gz'%rank)
                with gzip.open(path,'wt') as f:
                    for row in labels:
                        if int(row['qid'][1:])%4==rank:f.write(json.dumps(row)+'\n')
                shards.append(file_item(path,1280))
            write_json(full/'complete.json',dict(common,questions=1024,trajectories=5120,selection_sha256=file_sha(pp),own_em=file_item(own,len(labels)),shards=shards))
            conf=base/'config.json';write_json(conf,dict(self.config(),base=str(base),actor_version='actor0',actor_model_dir='/official'))
            with patch('reproduction.prefix_selection200.periodic_data.candidate_count',return_value=5120):
                final=actual_em_selection(conf,62,allow_two_stage=True);result=project_top512(conf,62,final)
            self.assertEqual(pred.read_bytes(),original_pred);self.assertEqual(result['actual_selected_mixed_questions'],512)
            self.assertEqual(result['projected_generation_requests'],0)
            projected=[r for shard in result['shards'] for r in iter_rows(shard['path'])]
            originals={r['trajectory_id']:r for r in labels}
            self.assertEqual(len(projected),2560)
            for row in projected:
                original=originals[row['trajectory_id']]
                for key in ('response_token_ids','response_log_probs','strict_em','prefix_sha256'):self.assertEqual(row[key],original[key])

    def test_1024_continuation_not_admitted_into_legacy_or_p10(self):
        c=dict(self.config(),rollout_continuation_questions=1024)
        with patch('reproduction.prefix_selection200.rollout.candidate_count',return_value=5120):
            self.assertEqual(question_count(c,'continue',62),1024)
            with self.assertRaises(ValueError):question_count(dict(c,two_stage_selection=None),'continue',62)
            with self.assertRaises(ValueError):question_count(c,'continue',70)

if __name__=='__main__':unittest.main()
