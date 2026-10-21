"""CPU fixtures for the explicit recovery publisher and its separate observer."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


local = load('future_recovery_local_fixture', HERE/'recover_post_callback_future_local.py')
observer = load('future_recovery_observer_fixture', HERE/'observe_post_callback_future_recovery_hk.py')


class RecoveryLocalTests(unittest.TestCase):
    def test_python38_and_receipt_is_durable_before_child_receives_envelope(self):
        for name in ('recover_post_callback_future_local.py', 'observe_post_callback_future_recovery_hk.py'):
            ast.parse((HERE/name).read_text(), feature_version=(3,8))
        ast.parse(local.REMOTE_LAUNCH, feature_version=(3,8))
        # The recovery child's first action authenticates launch-receipt.json.
        # Unlike the original runner, it must not receive stdin before publish.
        self.assertLess(local.REMOTE_LAUNCH.index("publish(root/'launch-receipt.json',receipt)"),
                        local.REMOTE_LAUNCH.index('child.stdin.write(payload)'))
        self.assertIn("recovery.recovery_prerequisites()", local.REMOTE_LAUNCH)
        self.assertIn("str(tools/'recover_post_callback_future_hk.py')", local.REMOTE_LAUNCH)
        self.assertNotIn('runner.launch_prerequisites()', local.REMOTE_LAUNCH)

    def test_published_four_file_bundle_and_git_rejections_without_network(self):
        commit = 'c'*40
        def git(*args):
            if args == ('status','--porcelain'): return b''
            if args == ('rev-parse','HEAD'): return commit.encode()
            if args == ('symbolic-ref','--quiet','--short','HEAD'): return local.BRANCH.encode()
            if args == ('ls-remote','origin','refs/heads/'+local.BRANCH):
                return (commit+'\trefs/heads/'+local.BRANCH+'\n').encode()
            self.assertEqual(args[0], 'show')
            revision, name = args[1].split(':', 1)
            self.assertIn(revision, (commit, local.MODULE_COMMIT, local.HELPER_COMMIT))
            return (REPO/name).read_bytes()
        head, boundary, dependencies, files = local.published_sources(REPO, git=git)
        self.assertEqual(head, commit)
        self.assertEqual(set(files), {local.RUNNER, local.OBSERVER, local.RECOVERY, local.RECOVERY_OBSERVER})
        envelope = dict(password='fixture-only', module_source=boundary.decode(), module_sha256=local.MODULE_SHA,
            published_commit=local.MODULE_COMMIT, dependencies_source=dependencies.decode(),
            dependencies_sha256=hashlib.sha256(dependencies).hexdigest(), dependencies_commit=head)
        bundle = dict(launcher_commit=head, remote_head=head, origin_ref='refs/heads/'+local.BRANCH,
                      files=files, runner_envelope=envelope)
        self.assertLess(len(json.dumps(bundle).encode()), local.MAX_PAYLOAD)
        for bad in ('dirty','remote','boundary'):
            def changed(*args):
                if bad == 'dirty' and args == ('status','--porcelain'): return b'?? uncommitted.py\n'
                if bad == 'remote' and args[0] == 'ls-remote':
                    return ('d'*40+'\trefs/heads/'+local.BRANCH).encode()
                value = git(*args)
                if bad == 'boundary' and args[0] == 'show' and args[1].startswith(local.MODULE_COMMIT+':'):
                    return value[:-1]+b'x'
                return value
            with self.assertRaises(ValueError): local.published_sources(REPO, git=changed)

    def test_observer_checks_recovery_published_path_sha_without_file_global(self):
        with tempfile.TemporaryDirectory() as name:
            hk = Path(name).resolve(); commit = 'c'*40
            tools = hk/'post-callback-future-recovery-tools'/commit; tools.mkdir(parents=True)
            source = tools/observer.RUNNER; shutil.copyfile(HERE/observer.RUNNER, source)
            proof = dict(path=str(source), bytes=source.stat().st_size,
                         sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            launch = dict(launcher_commit=commit, tools=str(tools), public_code={observer.RUNNER:proof})
            with patch.object(observer, 'HK', hk), patch.dict(observer.__dict__, {'__file__':None}):
                self.assertEqual(observer.load_published(launch, observer.RUNNER).STEPS, (50,100,150,200))
                proof['sha256'] = 'f'*64
                with self.assertRaisesRegex(ValueError, 'SOURCE_DIFFERS'):
                    observer.load_published(launch, observer.RUNNER)

    def test_observer_routes_new_root_and_actual_companion_birth(self):
        with tempfile.TemporaryDirectory() as name:
            hk = Path(name).resolve(); old = hk/'future'; root = old/'recovery-python38-r1'; root.mkdir(parents=True)
            ordinary = load('ordinary_recovery_routing_fixture', HERE/'observe_post_callback_future_hk.py')
            ordinary.HK = hk
            ordinary.socket = SimpleNamespace(gethostname=lambda:'xielab3')
            ordinary.os = SimpleNamespace(getuid=lambda:1002)
            common = dict(base=ordinary.BASE, source_job=ordinary.JOB, run_id=ordinary.RUN,
                          training_commit=ordinary.TRAINING)
            for path, value in ((old/'failure.json', dict(error='original failed Python38 archive')),
                (old/'claim.json', dict(common, birth=dict(pid=3844281))),
                (root/'launch-receipt.json', dict(common, birth=dict(pid=100))),
                (root/'claim.json', dict(common, birth=dict(pid=200))),
                (root/'companion-start-receipt.json', dict(birth=dict(pid=200)))):
                path.write_text(json.dumps(value))
            runner = SimpleNamespace(process_observation=lambda birth:dict(state='LIVE', actual=dict(pid=birth['pid'])))
            def published(launch, filename):
                return ordinary if filename == observer.OBSERVER else runner
            with patch.object(observer,'HK',hk), patch.object(observer,'OLD_ROOT',old), \
                    patch.object(observer,'ROOT',root), patch.object(observer,'load_published',side_effect=published):
                actual = observer.observe()
                self.assertEqual(actual['state'], 'RUNNING')
                self.assertEqual(actual['actual_process']['actual']['pid'], 200)
                self.assertEqual(actual['root'], str(root))
                self.assertEqual(actual['original_failed_root'], str(old))
                self.assertFalse(actual['original_failed_controller_modified'])
                (root/'failure.json').write_text(json.dumps(dict(error='new recovery failure')))
                self.assertEqual(observer.observe()['state'], 'FAILED')


if __name__ == '__main__': unittest.main()
