"""Projection must reuse the completed G5 data, including sampled oldLP."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .periodic_data import project_top512


class PoolProjectionTests(unittest.TestCase):
    def test_selected_ppo_rows_are_exact_pool_rows_not_new_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/'outers/outer-000020'
            (root/'full-pool').mkdir(parents=True);(root/'prefix').mkdir();(root/'probe-score').mkdir()
            c=dict(base=str(base),run_id='newrun',actor_model_dir='/mnt/current-actor',actor_version='actor19')
            conf=base/'config.json';conf.write_text(json.dumps(c))
            (root/'prefix/complete.json').write_text(json.dumps(dict(run_id='newrun',outer=20,
                actor_version='actor19',scan_sha256='scan')))
            groups=[dict(qid='q%d'%i,question_hash='h%d'%i,candidate_ordinal=i,
                trajectory_ids=['t%d-%d'%(i,j) for j in range(5)]) for i in range(0,5120,10)]
            selection=dict(run_id='newrun',outer=20,actor_version='actor19',prefix_scan_sha256='scan',selected=groups)
            (root/'probe-score/selection.json').write_text(json.dumps(selection))
            (root/'full-pool/complete.json').write_text(json.dumps(dict(run_id='newrun',outer=20,
                actor_version='actor19',trajectories=25600,questions=5120,shards=[dict(path='synthetic-pool')],
                native_generation_requests=999)))
            source=[dict(trajectory_id='t%d-%d'%(i,j),qid='q%d'%i,strict_em=int(j%2==0),
                actor_version='actor19',selection_sha256='pool-selection',prefix_sha256='actual-prefix',
                prefix_scan_sha256='scan',response_token_ids=[i,j,5],response_log_probs=[-.1,-.2,-.3])
                for i in range(5120) for j in range(5)]
            with patch('reproduction.prefix_selection200.periodic_data.iter_rows',return_value=iter(source)) as stream:
                result=project_top512(conf,20)
            stream.assert_called_once_with('synthetic-pool')
            self.assertEqual(result['trajectories'],2560)
            self.assertEqual(result['projected_generation_requests'],0)
            self.assertEqual(result['actual_selected_mixed_fraction'],1.)
            projected=[]
            for shard in result['shards']:
                with gzip.open(shard['path'],'rt') as f:projected.extend(json.loads(line) for line in f)
            origin={r['trajectory_id']:r for r in source}
            self.assertEqual(len({r['trajectory_id'] for r in projected}),2560)
            for row in projected:
                original=origin[row['trajectory_id']]
                for key in ('response_token_ids','response_log_probs','strict_em','prefix_sha256','actor_version'):
                    self.assertEqual(row[key],original[key])
                self.assertEqual(row['full_pool_origin_selection_sha256'],'pool-selection')


if __name__=='__main__':unittest.main()
