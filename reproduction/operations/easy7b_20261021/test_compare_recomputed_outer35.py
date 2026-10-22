"""Tiny CPU fixtures for identity-independent hashes, not model equivalence."""
import copy
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.operations.easy7b_20261021 import compare_recomputed_outer35 as m

OLD, NEW = 'a'*32, 'b'*32


def digest(value): return hashlib.sha256(value.encode()).hexdigest()


def fixture(aid, job, commit):
    groups=[]
    for ordinal in range(2):
        prefix=m.RUN+'-35-attempt-'+aid+'-'+str(ordinal)+'-'+digest('parent'+str(ordinal))
        groups.append(dict(ordinal=ordinal,question_id='q'+str(ordinal),
            content_sha256=digest('question'+str(ordinal)),
            strict_correctness=[0,1,0,1,0] if ordinal==0 else [1]*5,
            category='mixed' if ordinal==0 else 'all_correct', selected=ordinal==0,
            trajectory_ids=[prefix+'-sample-'+str(sample) for sample in range(5)]))
    ids=groups[0]['trajectory_ids']; snapshot=m.RUN+':35:policy-'+aid
    selection=dict(step=35,source_commit=commit,policy_snapshot_id=snapshot,
        fixed_candidate_budget=True,attempted_candidates=2,responses_per_group=5,
        selected_groups=1,selected_mixed=1,selected_all_correct_fallback=0,pool_retry=0,groups=groups)
    gate=dict(schema_version=1,role='common8_behavior_probability_gate',accepted=True,
        run_id=m.RUN,slurm_job_id=job,logical_update=35,attempt_id=aid,policy_snapshot_id=snapshot,
        trajectory_count=5,trajectory_ids=ids,
        row_tensor_sha256={tid:digest('row'+str(i)) for i,tid in enumerate(ids)},
        tensor_sha256={name:digest(name) for name in m.TENSOR_NAMES},actual_frozen_old_all_finite=True,
        behavior_probability_policy='sampling_journal_no_extra_hf_forward',
        hf_recomputation_performed=False,sampling_dtype='torch.float32',
        actor_token_counts=[1,2,3,4,5],included_actor_token_count=15)
    return selection,gate


class CompareOuter35Tests(unittest.TestCase):
    def setUp(self):
        for key,value in (('CANDIDATES',2),('SELECTED',1)):
            holder=patch.object(m,key,value);holder.start();self.addCleanup(holder.stop)
        self.old=fixture(OLD,m.OLD_JOB,m.OLD_COMMIT)
        self.new=fixture(NEW,m.NEW_JOB,m.NEW_COMMIT)

    def test_different_attempt_job_and_source_identity_preserve_content_comparison(self):
        result=m.compare_documents(*self.old,*self.new,OLD,NEW)
        self.assertEqual(result['mapped_combined_row_hash_equal'],5)
        self.assertTrue(result['all_selected_combined_row_hashes_equal'])
        self.assertTrue(all(result['whole_tensor_hash_equal'].values()))
        self.assertEqual(result['strict_label_vector_differences_at_ordinal'],0)
        self.assertFalse(result['training_gate']);self.assertFalse(result['limits']['old_outer35_training_complete'])
        self.assertFalse(result['limits']['new_outer35_training_completion_checked'])

    def test_hash_difference_is_reported_without_failure_or_token_only_claim(self):
        changed=copy.deepcopy(self.new);gate=changed[1]
        gate['row_tensor_sha256'][gate['trajectory_ids'][2]]=digest('changed-combined-row')
        gate['tensor_sha256']['sampling_log_probs']=digest('changed-probabilities')
        result=m.compare_documents(*self.old,*changed,OLD,NEW)
        self.assertEqual(result['status'],'COMPARISON_RECORDED')
        self.assertEqual(result['mapped_combined_row_hash_different'],1)
        self.assertFalse(result['all_selected_combined_row_hashes_equal'])
        self.assertFalse(result['whole_tensor_hash_equal']['sampling_log_probs'])
        self.assertTrue(result['whole_tensor_hash_equal']['actor_mask'])
        self.assertFalse(result['training_failure_inferred']);self.assertFalse(result['restart_or_rollback_requested'])
        self.assertFalse(result['limits']['separate_token_id_hash_available_in_small_gate'])

    def test_wrong_question_identity_cannot_produce_false_equal_row_result(self):
        changed=copy.deepcopy(self.new)
        changed[0]['groups'][0]['content_sha256']=digest('another-original-question')
        result=m.compare_documents(*self.old,*changed,OLD,NEW)
        self.assertEqual(result['question_content_hash_differences'],1)
        self.assertEqual(result['comparable_same_question_selected_rows'],0)
        self.assertFalse(result['all_selected_combined_row_hashes_equal'])
        self.assertFalse(result['whole_tensor_row_alignment_equal'])
        changed[1]['slurm_job_id']=m.OLD_JOB
        with self.assertRaisesRegex(ValueError,'GATE_SELECTION_OR_EXECUTION_IDENTITY'):
            m.compare_documents(*self.old,*changed,OLD,NEW)

    def test_missing_new_gate_is_pending_and_no_raw_file_is_read(self):
        with tempfile.TemporaryDirectory() as td:
            old,new=Path(td)/'old',Path(td)/'new'
            for base,aid,values in ((old,OLD,self.old),(new,NEW,self.new)):
                (base/'artifacts/mixed-selection').mkdir(parents=True);(base/'logs').mkdir()
                (base/'artifacts/mixed-selection'/(aid+'.json')).write_text(json.dumps(values[0]))
                if base==old:(base/'logs'/('behavior-probability-attempt-'+aid+'.json')).write_text(json.dumps(values[1]))
                (base/'must-not-open.pt').write_bytes(b'not a tensor')
            with patch.object(m,'OLD_BASE',old),patch.object(m,'NEW_BASE',new),patch.object(m,'_read') as read:
                result=m.compare(OLD,NEW)
                self.assertEqual(result['status'],'PENDING_SAVED_JSON');self.assertEqual(result['missing'],['new_gate'])
                read.assert_not_called()
            (new/'logs'/('behavior-probability-attempt-'+NEW+'.json')).write_text(json.dumps(self.new[1]))
            with patch.object(m,'OLD_BASE',old),patch.object(m,'NEW_BASE',new),redirect_stdout(io.StringIO()) as output:
                self.assertEqual(m.main(['--old-attempt',OLD,'--new-attempt',NEW]),0)
            result=json.loads(output.getvalue());self.assertEqual(len(result['evidence']),4)
            self.assertTrue(result['all_selected_combined_row_hashes_equal'])

    def test_invalid_attempt_and_incomplete_row_hash_schema_never_compare(self):
        with self.assertRaisesRegex(ValueError,'EXPLICIT_FULL_ATTEMPT'):m.compare('18ace...',NEW)
        changed=copy.deepcopy(self.new);changed[1]['row_tensor_sha256'].pop(changed[1]['trajectory_ids'][0])
        with self.assertRaisesRegex(ValueError,'COMPLETE_COMBINED_ROW_HASHES'):
            m.compare_documents(*self.old,*changed,OLD,NEW)


if __name__=='__main__':unittest.main()
