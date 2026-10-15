"""A queued source repair must never cancel a running or foreign experiment."""
from pathlib import Path
import tempfile
import unittest

from reproduction.hpc3_dynrpo_periodic.pending_replacement import validate_pending,never_started


class PendingReplacement(unittest.TestCase):
    def fields(self):
        return dict(JobId='42',UserId='mou176(21930)',JobName='dynrpo-periodic5-formal',
                    WorkDir='/data/own',JobState='PENDING',Priority='0')

    def test_actual_running_job_is_rejected(self):
        fields=self.fields();fields['JobState']='RUNNING'
        with self.assertRaisesRegex(ValueError,'NEVER_CANCELS_STARTED'):
            validate_pending(fields,job='42',source=Path('/data/own'),uid=21930)

    def test_foreign_owner_namespace_or_job_is_rejected(self):
        for key,value in [('UserId','other(14090)'),('WorkDir','/data/foreign'),('JobId','43')]:
            fields=self.fields();fields[key]=value
            with self.assertRaisesRegex(ValueError,'IDENTITY_DIFFERS'):
                validate_pending(fields,job='42',source=Path('/data/own'),uid=21930,require_held=True)

    def test_unheld_pending_job_cannot_be_retired(self):
        fields=self.fields();fields['Priority']='200'
        with self.assertRaisesRegex(ValueError,'HELD_JOB'):
            validate_pending(fields,job='42',source=Path('/data/own'),uid=21930,require_held=True)
        validate_pending(fields,job='42',source=Path('/data/own'),uid=21930)

    def test_runtime_state_is_not_reinitialized_as_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory);(source/'logs').mkdir()
            never_started(source)
            (source/'logs/run-manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'REAL_RUNTIME_STATE'):never_started(source)


if __name__=='__main__':unittest.main()
