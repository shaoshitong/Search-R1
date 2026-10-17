"""CPU-only privacy and single-dispatch checks; no real credential/SSH access."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from reproduction.easy2560_joint4 import invoke_hk_sidecar_once as operator


class InvocationTests(unittest.TestCase):
    def forbidden(self, *args, **kwargs):
        raise AssertionError('dry-run must not access credentials or SSH')

    def test_default_dry_run_does_not_read_credential_or_ssh(self):
        result = operator.run_once(credential_reader=self.forbidden, runner=self.forbidden)
        self.assertEqual(result['status'], 'DRY_RUN_NO_CREDENTIAL_READ_NO_SSH')
        self.assertEqual(result['ssh_attempts'], 0)
        self.assertFalse(result['credential_read_attempted'])

    def invoke_fake(self, runner):
        with tempfile.TemporaryDirectory() as folder:
            receipt = Path(folder) / 'receipt.json'
            result = operator.run_once(5, execute=True, receipt=receipt, runner=runner,
                                       credential_reader=lambda: 'DUMMY-SECRET-ONLY-IN-RAM')
            raw = receipt.read_text()
            self.assertNotIn('DUMMY-SECRET-ONLY-IN-RAM', raw)
            self.assertNotIn('DUMMY-SECRET-ONLY-IN-RAM', json.dumps(result))
            self.assertEqual(json.loads(raw), result)
            return result

    def test_ssh255_never_retries_or_logs_untrusted_output(self):
        calls = []
        def runner(argv, **kwargs):
            calls.append(argv)
            self.assertEqual(argv[5], 'hkust3')
            self.assertNotIn('DUMMY-SECRET-ONLY-IN-RAM', repr(argv))
            self.assertEqual(json.loads(kwargs['input']), {'password': 'DUMMY-SECRET-ONLY-IN-RAM'})
            return subprocess.CompletedProcess(argv, 255, 'DUMMY-SECRET-ONLY-IN-RAM', 'DUMMY-SECRET-ONLY-IN-RAM')
        result = self.invoke_fake(runner)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['ssh_attempts'], 1)
        self.assertTrue(result['status'].startswith('EXECUTION_UNCERTAIN'))

    def test_timeout_never_retries_or_serializes_exception_output(self):
        calls = []
        def runner(argv, **kwargs):
            calls.append(argv)
            raise subprocess.TimeoutExpired(argv, 900, output='DUMMY-SECRET-ONLY-IN-RAM')
        result = self.invoke_fake(runner)
        self.assertEqual(len(calls), 1)
        self.assertIn('UNCERTAIN_TIMEOUT', result['status'])

    def test_confirmed_success_only_copies_whitelisted_identity(self):
        def runner(argv, **kwargs):
            destination = operator.command_document(5)[2]
            remote = dict(status='EASY2560_INDEPENDENT_SMALL_SIDECAR_SHA_FSYNC_PASS',
                          directory=destination, receipt=destination+'/sidecar-archive-receipt.json',
                          file_count=12, total_bytes=1234, actual_cuda_restore_accepted=False,
                          post_callback_rng_captured=False, extra='DUMMY-SECRET-ONLY-IN-RAM')
            return subprocess.CompletedProcess(argv, 0, json.dumps(remote), 'DUMMY-SECRET-ONLY-IN-RAM')
        result = self.invoke_fake(runner)
        self.assertEqual(result['status'], 'HK_SIDECAR_SINGLE_ATTEMPT_CONFIRMED')
        self.assertEqual(result['file_count'], 12)
        self.assertNotIn('extra', result)

    def test_existing_dispatch_receipt_prevents_credential_read_and_ssh(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt = Path(folder) / 'receipt.json'
            receipt.write_text('{"status":"prior uncertain dispatch"}\n')
            with self.assertRaises(FileExistsError):
                operator.run_once(5, execute=True, receipt=receipt,
                                  runner=self.forbidden, credential_reader=self.forbidden)


if __name__ == '__main__':
    unittest.main()
