"""Recovery path binding and source/destination integrity, without remote IO."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from .fixed_assets_backup import admitted_rows, full_read, safe_original, fixed_namespace

RUN = '6cd6f043ef1641ae91ac177b4c27fe4a'


class FixedAssetTests(unittest.TestCase):
    def plan(self):
        return dict(status='FIXED_RECOVERY_ASSETS_INVENTORY_ONLY',run_id=RUN,bytes=17,
            files=[dict(original_path='/dataset/frozen/model',exported_source_path='/mnt/data/stage/model',
                bytes=17,expected_sha256='a'*64,export_mtime_ns=0)])

    def test_namespace_comes_from_actual_full_run_identity(self):
        for run_id in (RUN, 'f'*32):
            plan=self.plan();plan['run_id']=run_id
            self.assertEqual(len(list(admitted_rows(plan))),1)
            self.assertTrue(fixed_namespace(run_id).name.startswith(run_id[:12]))

    def test_original_paths_are_preserved_while_reading_export_alias(self):
        row,source,destination=list(admitted_rows(self.plan()))[0]
        self.assertEqual(str(source),'/mnt/shared/workspace/training_data_node_8/stage/model')
        self.assertEqual(str(destination),'payload/dataset/frozen/model')
        self.assertEqual(row['original_path'],'/dataset/frozen/model')

    def test_rejects_wrong_run_duplicate_traversal_and_mac_files(self):
        for mutate in (lambda p:p.update(run_id='another-run'),
                       lambda p:p['files'][0].update(original_path='/Users/local/model'),
                       lambda p:p['files'][0].update(exported_source_path='/mnt/data/../secret'),
                       lambda p:p['files'][0].update(expected_sha256='not-a-digest'),
                       lambda p:p.update(bytes=123)):
            plan=self.plan();mutate(plan)
            with self.assertRaises(ValueError):list(admitted_rows(plan))
        plan=self.plan();plan['files']*=2;plan['bytes']*=2
        with self.assertRaises(ValueError):list(admitted_rows(plan))
        with self.assertRaises(ValueError):safe_original('/dataset/../credentials')

    def test_corrupted_destination_cannot_match_actual_source_read(self):
        with tempfile.TemporaryDirectory()as folder:
            source=Path(folder)/'source';target=Path(folder)/'copy';source.write_bytes(b'remote-recovery-model'*113)
            with target.open('xb')as out:actual=full_read(source,out,rate=10**9)
            self.assertEqual(actual,full_read(target,rate=10**9))
            target.write_bytes(target.read_bytes()[:-1]+b'!')
            self.assertNotEqual(actual,full_read(target,rate=10**9))


if __name__=='__main__':unittest.main()
