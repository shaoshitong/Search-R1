"""The native entry admits the authenticated restored reward probe only."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from reproduction.common8.run_pipeline import validate_pipeline_entry


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PipelineRestoreEntryTests(unittest.TestCase):
    def test_fresh_version_guard_is_unchanged(self):
        run=dict(profile='common8',completed_updates=0,initial_probe_version=None)
        validate_pipeline_entry(run)
        run['initial_probe_version']='unbound-version'
        with self.assertRaises(AssertionError):validate_pipeline_entry(run)

    def test_exact_saved_pair_version_accepted_and_changed_binding_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary).resolve();path=base/'source-receipt.json'
            checkpoint=base/'checkpoint';checkpoint.mkdir()
            manifest=base/'logs/run-manifest.json';parent_sha=write(manifest,dict(run_id='same-learner'))
            receipt=dict(kind='question_probe_pair',status='QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS',
                run_id='same-learner',step=5,reward_probe_version='pinned-version',
                source_base=str(base),checkpoint=str(checkpoint),source_run_manifest=dict(
                    path=str(manifest),full_read_sha256=parent_sha))
            receipt_sha=write(path,receipt)
            restore=dict(kind='question_probe_pair',source_receipt=str(path),source_receipt_sha256=receipt_sha,
                source_checkpoint=str(checkpoint),source_base=str(base),source_run_manifest_sha256=parent_sha,
                completed_outer=5,reward_probe_version='pinned-version')
            run=dict(profile='common8',completed_updates=0,run_id='same-learner',
                     initial_probe_version='pinned-version',full_state_restore=restore)
            validate_pipeline_entry(run)
            for field,value in (('initial_probe_version',None),('initial_probe_version','wrong'),
                                ('run_id','new-learner')):
                changed=copy.deepcopy(run);changed[field]=value
                with self.subTest(field=field,value=value),self.assertRaises(AssertionError):
                    validate_pipeline_entry(changed)
            for field,value in (('reward_probe_version','wrong'),('source_receipt_sha256','0'*64),
                                ('source_checkpoint',str(base/'other')),('completed_outer',0),
                                ('source_run_manifest_sha256','0'*64)):
                changed=copy.deepcopy(run);changed['full_state_restore'][field]=value
                with self.subTest(field=field),self.assertRaises(AssertionError):validate_pipeline_entry(changed)


if __name__=='__main__':unittest.main()
