"""Pure controller-script binding checks; these do not claim GPU execution."""
import copy
import unittest

from reproduction.common8.formal_inline_safety import native_birth
from reproduction.common8.native_allocation import fields, wrapped_batch_script_identity


class NativeBatchBirthTests(unittest.TestCase):
    def allocation(self, command='(null)'):
        detail = ('JobId=724245 SubmitTime=2026-10-16T03:34:54 '
                  'StartTime=2026-10-16T03:56:30 UserId=mou176(21930) '
                  f'Command={command} WorkDir=/owned/r5 NodeList=ACD1-50 JobState=RUNNING')
        return dict(actual_allocation_detail=detail, slurm_job_id='724245')

    def wrapped(self):
        allocation = self.allocation()
        content = '#!/bin/sh\n# This script was created by sbatch --wrap.\n\nbash /owned/run.sh formal\n'
        allocation['actual_batch_script'] = wrapped_batch_script_identity(
            fields(allocation['actual_allocation_detail']), content)
        return allocation

    def test_actual_wrapped_script_supplies_identity_without_rewriting_command(self):
        allocation = self.wrapped()
        birth = native_birth(allocation)
        self.assertEqual(birth['Command'], '(null)')
        self.assertEqual(birth['batch_script_sha256'], allocation['actual_batch_script']['sha256'])

    def test_null_without_script_or_nonwrapped_content_rejects(self):
        with self.assertRaises(KeyError):
            native_birth(self.allocation())
        for content in ('', '#!/bin/sh\nbash /owned/run.sh\n',
                        '#!/bin/sh\n# This script was created by sbatch --wrap.\n\n'):
            with self.subTest(content=content), self.assertRaises(AssertionError):
                wrapped_batch_script_identity(fields(self.allocation()['actual_allocation_detail']), content)

    def test_changed_script_or_copied_job_birth_rejects(self):
        for mutation in ('content', 'sha256', 'job_birth'):
            allocation = copy.deepcopy(self.wrapped())
            script = allocation['actual_batch_script']
            if mutation == 'content':
                script['content'] += 'echo changed\n'
            elif mutation == 'sha256':
                script['sha256'] = '0' * 64
            else:
                script['slurm_job_birth']['StartTime'] = '2026-10-16T03:56:31'
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                native_birth(allocation)

    def test_path_command_keeps_existing_birth_and_other_null_fields_reject(self):
        allocation = self.allocation('/owned/run.sh')
        self.assertEqual(native_birth(allocation), {
            key: value for key, value in fields(allocation['actual_allocation_detail']).items()
            if key != 'JobState'})
        allocation = self.wrapped()
        allocation['actual_allocation_detail'] = allocation['actual_allocation_detail'].replace(
            'NodeList=ACD1-50', 'NodeList=(null)')
        with self.assertRaises(AssertionError):
            native_birth(allocation)


if __name__ == '__main__':
    unittest.main()
