"""Execute the source5 bootstrap at mocked OS/exec boundaries, CPU only."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import source_tp1_control as control


class SourceBootstrapTest(unittest.TestCase):
    def test_source_namespace_context_gpu_group_and_deadline_reach_sampler(self):
        for group, model in control.GROUPS.items():
            with self.subTest(group=group), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                stage = base / 'preparation/control/code'
                stage.mkdir(parents=True)
                context_path = stage.parent / 'sublease-context.json'
                context_path.write_text(json.dumps(dict(run_id=control.RUN_ID, model_indices=model,
                    parent=dict(mntns_inode=20, netns_inode=30, unit=control.SOURCE_UNIT),
                    source_sampling_deadline_epoch=5000)))
                (stage / 'manifest.json').write_text(json.dumps(dict(run_id=control.RUN_ID, files={})))
                unit = 'sst-q7b-g5-source5-extra-' + group + '-v2-' + control.RUN_ID[:12] + '.service'
                cgroup = '/system.slice/' + unit
                detail = 'Id=%s\nMainPID=123\nActiveState=active\nControlGroup=%s\n' % (unit, cgroup)
                original_stat, original_read = os.stat, Path.read_text

                def stat(path, *args, **kwargs):
                    if str(path) == '/proc/self/ns/mnt':return SimpleNamespace(st_ino=20)
                    if str(path) == '/proc/self/ns/net':return SimpleNamespace(st_ino=30)
                    return original_stat(path, *args, **kwargs)

                def read(path, *args, **kwargs):
                    if str(path) == '/proc/self/cgroup':return '0::' + cgroup + '\n'
                    return original_read(path, *args, **kwargs)

                with patch.object(control, 'BASE', str(base)):
                    source = control.bootstrap(unit, group)
                namespace = dict(__file__=str(stage / 'source_sublease_bootstrap.py'))
                try:
                    with patch('os.geteuid', return_value=0), patch('os.getpid', return_value=123), \
                            patch('socket.gethostname', return_value='h100-5'), patch('os.stat', side_effect=stat), \
                            patch.object(Path, 'read_text', read), patch('time.time', return_value=1000), \
                            patch('subprocess.check_output', return_value=detail), patch('os.dup2'), \
                            patch('os.chdir'), patch('builtins.print'), patch('os.execvpe') as execute:
                        exec(compile(source, namespace['__file__'], 'exec'), namespace)
                finally:
                    if namespace.get('log'):namespace['log'].close()
                executable, argv, env = execute.call_args.args
                option = lambda flag: argv[argv.index(flag) + 1]
                self.assertTrue(executable.endswith('/env/bin/python'))
                self.assertEqual(option('--model-gpus'), ','.join(map(str, model)))
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], option('--model-gpus'))
                self.assertEqual(Path(env['Q7B_SOURCE_SUBLEASE_CONTEXT']).resolve(), context_path.resolve())
                self.assertEqual(option('--reuse-local-relay-unit'), control.SOURCE_UNIT)
                self.assertEqual(option('--reuse-local-relay-sha256'), hashlib.sha256(context_path.read_bytes()).hexdigest())
                self.assertEqual(option('--run-seconds'), '3400')
                self.assertEqual(option('--batch-reserve-seconds'), '1200')
                self.assertEqual(option('--tp'), '1')
                self.assertEqual(env['SEARCH_G1_PHYSICAL_WORLD_SIZE'], '1')
                self.assertEqual(option('--model-max-seqs'), str(control.LIMITS[group]['seqs']))
                self.assertEqual(option('--model-batch-tokens'), str(control.LIMITS[group]['tokens']))
                self.assertEqual(option('--model-kv-block-cap'), str(control.LIMITS[group]['blocks']))
                self.assertEqual(option('--model-reserve-gib'), str(control.LIMITS[group]['reserve']))
                self.assertEqual(env['NCCL_NVLS_ENABLE'], '0')
                self.assertFalse((stage.parent / 'relay-secret').exists())


if __name__ == '__main__':unittest.main()
