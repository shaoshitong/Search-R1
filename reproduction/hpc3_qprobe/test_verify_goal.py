"""CPU evidence reconstruction/missing200 guards, without a training launch."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from reproduction.hpc3_qprobe.verify_goal import verify_selection,verify
from search_g1.qprobe_selection import count_predictions,select_questions,prequential_metrics


class GoalTests(unittest.TestCase):
    def fixture(self):
        probabilities=np.full((5120,6),1/6)
        predictions=count_predictions(probabilities)
        questions=[SimpleNamespace(ordinal=i,qid=str(i),question_hash=hashlib.sha256(str(i).encode()).hexdigest(),content_hash=str(i)) for i in range(5120)]
        rng=np.random.default_rng(20261013);before=copy.deepcopy(rng.bit_generator.state)
        chosen=select_questions(questions,probabilities,rng=rng,tie_seed='20261013:1')
        sources={x.ordinal:x.selection_source for x in chosen}
        rows=[dict(ordinal=i,qid=str(i),question_hash=q.question_hash,content_sha256=q.content_hash,
            probabilities=probabilities[i].tolist(),predicted_mean=float(predictions['mean'][i]),
            predicted_std=float(predictions['expected_std'][i]),predicted_variance=float(predictions['expected_variance'][i]),
            predicted_mixed_probability=float(predictions['mixed_probability'][i]),selected=i in sources,
            selection_source=sources.get(i,'unselected')) for i,q in enumerate(questions)]
        selection=dict(attempted_candidates=5120,selected_groups=512,exploration_questions=64,
            prediction_before_label=True,post_rollout_em_filter=False,questions=rows,step=1,
            selection_rng_before=before,selection_rng_after=copy.deepcopy(rng.bit_generator.state))
        groups=[dict(ordinal=q.ordinal,source=q.selection_source,trajectory_ids=[str(q.ordinal)+'-'+str(j) for j in range(5)],
                     strict_correctness=[1]*(q.ordinal%6)+[0]*(5-q.ordinal%6)) for q in chosen]
        measured=prequential_metrics([probabilities[q.ordinal] for q in chosen],[g['strict_correctness'] for g in groups],
                                    [q.selection_source for q in chosen])
        rollout=dict(no_post_rollout_refill=True,raw_generated_trajectories=2560,responses_per_group=5,groups=groups,prequential=measured)
        return selection,rollout

    def test_real_saved_selection_labels_and_metrics_reconstruction(self):
        selection,rollout=self.fixture();proof=verify_selection(selection,rollout,20261013)
        self.assertEqual(proof['all']['questions'],512);self.assertEqual(proof['random_exploration']['questions'],64)

    def test_changed_prediction_refill_and_incomplete_actual_labels_fail(self):
        selection,rollout=self.fixture()
        bad=copy.deepcopy(selection);bad['questions'][0]['predicted_std']+=0.01
        with self.assertRaises(ValueError):verify_selection(bad,rollout,20261013)
        bad=copy.deepcopy(rollout);bad['no_post_rollout_refill']=False
        with self.assertRaises(ValueError):verify_selection(selection,bad,20261013)
        bad=copy.deepcopy(rollout);bad['groups'][0]['strict_correctness'].pop()
        with self.assertRaises(ValueError):verify_selection(selection,bad,20261013)

    def test_missing200_hardfails_before_queries_or_any_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);(base/'logs').mkdir();(base/'logs/run-manifest.json').write_text(json.dumps({'run_id':'fixture'}))
            with patch('reproduction.hpc3_qprobe.verify_goal.os.getuid',return_value=14090),\
                 patch('reproduction.hpc3_qprobe.verify_goal.subprocess.check_output') as query:
                for publish_receipt in (True,False):
                    with self.assertRaisesRegex(ValueError,'completed200 receipt missing'):
                        verify(base,base/'audit',publish_receipt=publish_receipt)
                query.assert_not_called()


if __name__=='__main__':unittest.main()
