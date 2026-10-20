"""CPU bootstrap checks; no control transport or GPU imports are executed."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import relay_parallel_control as control


class ParallelBootstrapTest(unittest.TestCase):
    def test_bootstrap_borrows_v5_relay_and_bounds_runtime(self):
        for source_deadline, owner_deadline, expected_runtime in ((5000, 3300, 2000), (5000, 9000, 3400)):
            with self.subTest(source_deadline=source_deadline, owner_deadline=owner_deadline), tempfile.TemporaryDirectory() as folder:
                base = Path(folder)
                old_root = base / 'preparation' / control.RELAY_DIRECTORY
                old_root.mkdir(parents=True)
                source_context = old_root / 'source-context.json'
                source_context.write_text(json.dumps(dict(source_sampling_deadline_epoch=source_deadline)))
                borrowed_ready = base / 'pool/jobs/job-relay10-2371322/attempt-original/relay-ready.json'
                dependency = dict(unit=control.UNIT, owner_unit=control.OWNER_UNIT,
                    owner_main_pid=2371322, relay_pid=2371352, owner_deadline_epoch=owner_deadline,
                    source_sampling_deadline_epoch=source_deadline,
                    source_context_sha256=hashlib.sha256(source_context.read_bytes()).hexdigest(),
                    reuse_local_relay_ready=str(borrowed_ready), reuse_local_relay_sha256='a' * 64,
                    reuse_local_relay_unit=control.OWNER_UNIT)
                with patch.object(control, 'BASE', str(base)):
                    identity, files, manifest = control.bundle()
                stage = base / 'preparation' / control.DIRECTORY / identity
                stage.mkdir(parents=True)
                for name, value in files.items():
                    (stage / name).write_bytes(value)
                (stage / 'manifest.json').write_text(json.dumps(manifest))
                (stage.parent / 'submit-context.json').write_text(json.dumps(dependency))
                group = '/system.slice/' + control.UNIT
                detail = 'Id=%s\nMainPID=123\nActiveState=active\nControlGroup=%s\n' % (control.UNIT, group)
                real_stat, real_read_text = os.stat, Path.read_text

                def stat(path, *args, **kwargs):
                    if str(path) in ('/proc/self/ns/mnt', '/proc/1/ns/mnt', '/proc/self/ns/net', '/proc/1/ns/net'):
                        return SimpleNamespace(st_ino=2 if str(path) == '/proc/self/ns/mnt' else 1)
                    return real_stat(path, *args, **kwargs)

                def read_text(path, *args, **kwargs):
                    if str(path) == '/proc/self/cgroup':
                        return '0::' + group + '\n'
                    return real_read_text(path, *args, **kwargs)

                namespace = {'__file__': str(stage / 'relay_parallel_bootstrap.py')}
                try:
                    with patch('os.geteuid', return_value=0), patch('os.getpid', return_value=123), \
                         patch('socket.gethostname', return_value='h100-10'), patch('os.stat', side_effect=stat), \
                         patch.object(Path, 'read_text', read_text), patch('time.time', return_value=1000), \
                         patch('subprocess.check_output', return_value=detail), patch('os.dup2'), patch('os.chdir'), \
                         patch('builtins.print'), patch('os.execvpe') as execute:
                        exec(compile(files['relay_parallel_bootstrap.py'], namespace['__file__'], 'exec'), namespace)
                finally:
                    if 'log' in namespace:
                        namespace['log'].close()
                executable, command, env = execute.call_args.args
                option = lambda name: command[command.index(name) + 1]
                self.assertEqual(executable, '/dataset/shaoshitong/Search-G1-common8/env/bin/python')
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '4,5,6,7')
                self.assertEqual(option('--model-gpus'), '4,5,6,7')
                self.assertEqual(option('--run-seconds'), str(expected_runtime))
                self.assertEqual(option('--reuse-local-relay-ready'), str(borrowed_ready))
                self.assertEqual(option('--reuse-local-relay-sha256'), 'a' * 64)
                self.assertEqual(option('--reuse-local-relay-unit'), control.OWNER_UNIT)
                self.assertEqual(option('--token-file'), str(old_root / 'relay-secret'))
                self.assertEqual(option('--upstream-ready'), str(old_root / 'upstream-ready.json'))
                self.assertEqual(option('--model-max-seqs'), '8')
                self.assertEqual(option('--model-batch-tokens'), '4096')
                evidence = json.loads(Path(option('--allocation-evidence')).read_text())
                self.assertEqual(evidence['unit'], control.UNIT)
                self.assertEqual(manifest['model_gpus'], [4, 5, 6, 7])
                self.assertNotEqual(control.UNIT, control.OWNER_UNIT)
                self.assertNotEqual(control.DIRECTORY, control.RELAY_DIRECTORY)
                self.assertFalse((stage.parent / 'relay-secret').exists())
                self.assertIn('relay_consumer_control.py', manifest['source_files'])

    def test_changed_original_template_fails_closed(self):
        with patch.object(control.original, 'bootstrap', return_value=b'changed original template'):
            with self.assertRaisesRegex(ValueError, 'Original consumer bootstrap changed'):
                control.bootstrap()


if __name__ == '__main__':
    unittest.main()
