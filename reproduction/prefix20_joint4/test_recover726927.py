import json
from pathlib import Path
import tempfile
import unittest

from reproduction.prefix20_joint4.recover726927 import relocate_global_storage
from reproduction.prefix20_joint4.storage_complete import validate_upper_bound
from reproduction.prefix20_joint4.test_storage_complete import StorageCompletionTests
from reproduction.prefix20_joint4.control import storage_plan
from reproduction.prefix20_joint4.assets import sha


class RecoveryStorageTests(unittest.TestCase):
    def test_relocation_preserves_measurement_age_and_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();original=root/'old';target=root/'fresh'
            original.mkdir();target.mkdir();(target/'preparation').mkdir()
            proof=StorageCompletionTests().fixture(original)
            source=original/'preparation/personal-storage-observation.json';source.write_text(json.dumps(proof))
            before=sha(source);when=proof['checked_unix']
            result=relocate_global_storage(original,target)
            actual=json.loads((target/'preparation/personal-storage-observation.json').read_text())
            self.assertEqual(sha(source),before)
            self.assertEqual(validate_upper_bound(actual,target),1200)
            self.assertEqual(actual['checked_unix'],when)
            self.assertFalse(result['new_root_du_performed'])
            self.assertEqual(actual['relocation_provenance']['original_aggregate_sha256'],before)
            checkpoint=dict(per_save_allowance_bytes=100,save_steps=[5,50,100,150,200])
            old=storage_plan(proof,checkpoint,2*10**12,now=when)
            new=storage_plan(actual,checkpoint,2*10**12,now=when)
            self.assertEqual(new['measured_allocated_bytes'],old['measured_allocated_bytes'])
            self.assertEqual(new['projected_bytes']-old['projected_bytes'],2*2**30)

    def test_tampered_original_storage_is_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();original=root/'old';target=root/'fresh'
            original.mkdir();target.mkdir();(target/'preparation').mkdir()
            proof=StorageCompletionTests().fixture(original)
            (original/'preparation/personal-storage-observation.json').write_text(json.dumps(proof))
            Path(proof['permission_subtree_complements'][0]['stdout_file']).write_text('1000\n')
            with self.assertRaisesRegex(ValueError,'FILE_SHA'):
                relocate_global_storage(original,target)
            self.assertFalse((target/'preparation/copied-global-storage-observation').exists())


if __name__=='__main__':unittest.main()
