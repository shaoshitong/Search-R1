"""Small CPU-only tests of observation receipts, without SSH or model calls."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from release.recursive_bran_a.ablation150_infra import task_transport_stdin_v3 as transport


class ObservationTests(unittest.TestCase):
    def invoke(self, result=None, error=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        reader = root / 'reader.py'
        reader.write_text('print("fixture")\n')
        output = root / 'node.json'
        argv = ['transport', '--management-socket', '/tmp/owned-fixture/control.sock',
                '--require-socket', '--target', 'h100-11', '--python-file', str(reader),
                '--output', str(output), '--timeout', '45']
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(transport.subprocess, 'run', return_value=result, side_effect=error) as run:
            code = transport.main()
        observation = json.loads(Path(str(output) + '.observation.json').read_text())
        command = run.call_args.args[0]
        self.assertIn('ProxyCommand=false', command)
        self.assertTrue(observation['socket_required'])
        self.assertFalse(observation['cold_fallback_allowed'])
        self.assertEqual(observation['remote_training_state'], 'UNKNOWN')
        return code, output, observation

    def test_success_preserves_reader_output(self):
        payload = '{"fixture": true}\n'
        envelope = json.dumps(dict(observed_utc='fixture', exit_code=0, stdout=payload, stderr=''))
        code, output, observation = self.invoke(subprocess.CompletedProcess([], 0, envelope, ''))
        self.assertEqual(code, 0)
        self.assertEqual(output.read_text(), payload)
        self.assertEqual(observation['status'], 'COMPLETE')
        self.assertEqual(observation['outer_command_actual_exit'], 0)
        self.assertEqual(observation['reader_actual_exit'], 0)

    def test_timeout_has_no_fabricated_ssh_exit_or_node_output(self):
        code, output, observation = self.invoke(
            error=subprocess.TimeoutExpired('ssh', 70, output=b'partial', stderr=b'timeout'))
        self.assertEqual(code, 124)
        self.assertFalse(output.exists())
        self.assertEqual(observation['status'], 'TRANSPORT_TIMEOUT')
        self.assertIsNone(observation['outer_command_actual_exit'])
        self.assertTrue(observation['outer_timed_out'])

    def test_outer_failure_does_not_create_node_output(self):
        code, output, observation = self.invoke(subprocess.CompletedProcess([], 255, '', 'closed'))
        self.assertEqual(code, 255)
        self.assertFalse(output.exists())
        self.assertEqual(observation['status'], 'MANAGER_COMMAND_FAILED')
        self.assertEqual(observation['outer_command_actual_exit'], 255)

    def test_reader_failure_is_distinct_from_transport_exit(self):
        envelope = json.dumps(dict(observed_utc='fixture', exit_code=1, stdout='partial', stderr='reader error'))
        code, output, observation = self.invoke(subprocess.CompletedProcess([], 0, envelope, ''))
        self.assertEqual(code, 1)
        self.assertFalse(output.exists())
        self.assertEqual(observation['status'], 'REMOTE_READER_FAILED')
        self.assertEqual(observation['outer_command_actual_exit'], 0)
        self.assertEqual(observation['reader_actual_exit'], 1)


if __name__ == '__main__':
    unittest.main()
