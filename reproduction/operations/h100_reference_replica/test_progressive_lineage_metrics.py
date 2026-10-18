import json
from pathlib import Path
import tempfile
import unittest
from .progressive_lineage_metrics import MODE,metric_row,summarize


def joint(outer):
    return dict(run_id='b',outer=outer,selection_metrics=dict(questions=512,mixed=208,
        all_wrong=221,all_correct=83,mixed_fraction=208/512),
        actor=dict(actual_adam_steps_per_rank=[outer*20]*4),
        probe=dict(actual_adam_steps_per_rank=[outer*10]*4,ema_updates=outer*10),
        probe_metrics=dict(new_adam_steps_per_rank=10,before_update={},after_update={}))


class LineageMetricsTests(unittest.TestCase):
    def test_explicit_boundary_excludes_later_old_execution(self):
        with tempfile.TemporaryDirectory()as tmp:
            b=Path(tmp);(b/'preparation').mkdir();(b/'preparation/launch-config.json').write_text(json.dumps(dict(run_id='b')))
            (b/'latest-joint.json').write_text(json.dumps(dict(outer=3)))
            for n in range(1,4):
                p=b/'outers'/('outer-%06d'%n);p.mkdir(parents=True);(p/'complete.json').write_text(json.dumps(joint(n)))
            report=summarize(b,2)
            self.assertEqual([r['outer']for r in report['rows']],[1,2])
            self.assertFalse(report['uncommitted_or_later_history_included'])
            with self.assertRaises(ValueError):summarize(b,4)

    def test_stage_boundaries_periodic_and_frozen_are_distinct(self):
        c=dict(run_id='b',b_progressive=dict(mode=MODE))
        for n,prefix,admitted in [(51,20,768),(60,20,2560),(100,20,2560),(101,40,1024),(150,40,2560),(151,80,1536),(200,80,1536)]:
            row=metric_row(c,joint(n),n)
            self.assertEqual((row['prefix_tokens'],row['continued_questions']),(prefix,admitted))
        self.assertEqual(metric_row(dict(run_id='b'),joint(151),151)['prefix_tokens'],5)

    def test_paired_original_metrics_are_not_balanced_resample_accuracy(self):
        j=joint(52);j['probe_metrics'].update(sampling=dict(sampled_counts=[256,256],unique_questions=319),
            before_update={'ema':{'train':dict(count=512,accuracy=208/512,balanced_accuracy=.5)}},
            after_update={'raw':{'train':dict(count=512,accuracy=.5859375,balanced_accuracy=.6197356577763039)}})
        row=metric_row(dict(run_id='b'),j,52)
        self.assertEqual(row['paired_original_distribution']['before_update']['ema']['train']['accuracy'],208/512)
        self.assertEqual(row['sampling']['sampled_counts'],[256,256])
        self.assertNotIn('accuracy',row['sampling'])
        j['selection_metrics']['mixed']=209
        with self.assertRaises(ValueError):metric_row(dict(run_id='b'),j,52)


if __name__=='__main__':unittest.main()
