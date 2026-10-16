"""Schema3 composition with real CPU state files and synthetic actor HF bytes."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from search_g1.test_qprobe_checkpoint import PairFixture
from search_g1.test_prefix20_runtime import TinyFourRankTransport,settings
from search_g1.qprobe_checkpoint import bind_restore,read_source,digest
from search_g1.prefix20_runtime import Prefix20Runtime,validate_saved_state


def prefix_fixture(root,outer=50):
    fixture=PairFixture(root,outer=outer)
    transport=TinyFourRankTransport()
    runtime=Prefix20Runtime(settings(fixture.base),fixture.base,transport=transport)
    for _ in range(outer*10):
        transport.parameter.grad=torch.ones_like(transport.parameter);transport.optimizer.step()
    transport.steps=outer*10;transport.outer=outer
    runtime.completed_outer=outer;runtime.completed_attempts={f'{i:032x}' for i in range(1,outer+1)}
    (fixture.state/'question-probe-state.pt').unlink()
    (fixture.state/'question-probe-state.pt.head.pt').unlink()
    proof=runtime.save_checkpoint(fixture.state,outer,fixture.stream,fixture.run['initial_probe_version'],
        actor_snapshot_id=fixture.run_id+':'+str(outer))
    fixture.run.update(question_probe=settings(fixture.base),data_selection_mode='prefix20_probe')
    (fixture.base/'logs/run-manifest.json').write_text(json.dumps(fixture.run))
    manifest=json.loads((fixture.state/'manifest.json').read_text());manifest['question_probe']=proof
    (fixture.state/'manifest.json').write_text(json.dumps(manifest))
    complete=json.loads((fixture.cp/'complete.json').read_text())
    complete['files_sha256']={str(p.relative_to(fixture.cp)):digest(p) for p in fixture.cp.rglob('*')
        if p.is_file() and p.name!='complete.json'}
    from reproduction.common8.runtime import checkpoint_complete
    with patch.dict(os.environ,SEARCH_G1_BASE=str(fixture.base),SEARCH_G1_TRAINING_WORLD_SIZE='4'), \
            patch('reproduction.common8.runtime.hf_checkpoint_complete',return_value=complete):
        checkpoint_complete(outer,fixture.cp)
    fixture.trainer.qprobe_runtime=runtime
    return fixture,proof


class Prefix20CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_checkpoint_complete_pair_reader_and_restore_binding_cover_all_four_states(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture,proof=prefix_fixture(directory)
            complete=json.loads((fixture.cp/'complete.json').read_text())
            self.assertTrue(all('training_state/prefix20-rank-%d.pt'%r in complete['files_sha256'] for r in range(4)))
            pair=fixture.publish();_,receipt,_,_,parent=read_source(pair['path'])
            self.assertEqual(receipt['head_version'],'prefix20-epoch68-outer-0050')
            target=copy.deepcopy(parent);target.pop('question_probe')
            bound=bind_restore(target,pair['path'])
            self.assertEqual(bound['data_selection_mode'],'prefix20_probe')
            self.assertNotIn('initial_head_checkpoint',bound['question_probe'])
            self.assertEqual(bound['full_state_restore']['completed_outer'],50)

    def test_cached_reader_detects_rank3_byte_change_even_with_same_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture,proof=prefix_fixture(directory);pair=fixture.publish();read_source(pair['path'])
            path=fixture.state/'prefix20-rank-3.pt';before=path.stat();raw=bytearray(path.read_bytes())
            raw[-30]^=1;path.write_bytes(raw);os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
            self.assertEqual(path.stat().st_size,before.st_size)
            with self.assertRaisesRegex(ValueError,'RANK_FILE_SHA'):read_source(pair['path'])

    def test_saved_schema_rejects_missing_rank_and_changed_adam_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture,proof=prefix_fixture(directory,outer=5)
            state=torch.load(proof['path'],map_location='cpu',weights_only=False)
            missing=copy.deepcopy(state);missing['rank_files'].pop()
            with self.assertRaisesRegex(ValueError,'FOUR_RANK'):validate_saved_state(missing,proof['path'],fixture.run_id+':5')
            path=fixture.state/'prefix20-rank-3.pt';rank=torch.load(path,map_location='cpu',weights_only=False)
            next(iter(rank['optimizer']['state'].values()))['step']+=1;torch.save(rank,path)
            state['rank_files'][3].update(bytes=path.stat().st_size,sha256=digest(path))
            with self.assertRaisesRegex(ValueError,'ADAM_STEP'):validate_saved_state(state,proof['path'],fixture.run_id+':5')

    def test_completed200_pair_cannot_start_update201(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture,proof=prefix_fixture(directory,outer=200);pair=fixture.publish()
            with self.assertRaisesRegex(ValueError,'TRAINING_TARGET_ALREADY_COMPLETE'):
                bind_restore(copy.deepcopy(fixture.run),pair['path'])


if __name__=='__main__':unittest.main()
