import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.prefix20_joint4 import retire_old_checkpoints as r


class RetirementBoundaryTest(unittest.TestCase):
    def test_checkpoint_inventory_keeps_raw_evaluation_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();checkpoint=root/'checkpoint';checkpoint.mkdir()
            (checkpoint/'model.safetensors').write_bytes(b'weights')
            (checkpoint/'rng.pt').write_bytes(b'rng')
            (checkpoint/'complete.json').write_text('{}')
            (checkpoint/'eval.jsonl').write_text('{}\n')
            (root/'raw.npy').write_bytes(b'raw')
            with patch.object(r,'targets',return_value=[checkpoint]),patch.object(r.os,'getuid',return_value=14090):
                actual=r.identity
                def identity(path):
                    value=actual(path);value['uid']=14090;return value
                with patch.object(r,'identity',side_effect=identity):
                    _,files=r.inventory()
            self.assertEqual({Path(x['path']).name for x in files},{'model.safetensors','rng.pt'})
            self.assertTrue((root/'raw.npy').exists())

    def test_symlink_cannot_escape_exact_checkpoint_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();checkpoint=root/'checkpoint';checkpoint.mkdir()
            outside=root/'outside.pt';outside.write_bytes(b'keep')
            (checkpoint/'weights.pt').symlink_to(outside)
            with patch.object(r,'targets',return_value=[checkpoint]):
                with self.assertRaisesRegex(ValueError,'SYMLINK'):
                    r.inventory()
            self.assertEqual(outside.read_bytes(),b'keep')

    def test_open_reader_blocks_before_unlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();(root/'preparation').mkdir()
            path=root/'preparation/quiescence.json'
            path.write_text(json.dumps(dict(scope='obsolete_sigma3_checkpoint_retirement',
                checked_unix=r.time.time(),jxulogin_no_checkpoint_reader=True,
                moulogin_no_checkpoint_reader=True,hkust3_no_qprobe_process=True,
                actual_compute_ledger_births_absent=False)))
            with self.assertRaisesRegex(ValueError,'QUIESCENCE'):
                r.guard(root,path)


if __name__=='__main__':unittest.main()
