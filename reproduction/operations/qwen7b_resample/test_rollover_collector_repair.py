"""CPU argv/once-intent regression; no real systemd, mounts or transport."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import rollover_terminal_control as control


def write(path, value):path.write_text(json.dumps(value))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class CollectorRepairTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self, state='failed', receipt_status='TERMINAL_ONLY_COLLECTOR_STARTED_ONCE'):
        with tempfile.TemporaryDirectory() as folder:
            physical = Path(folder) / 'physical';physical.mkdir()
            base = Path(folder) / 'alias';base.symlink_to(physical, target_is_directory=True)
            identity, files, manifest = control.bundle('h100-10', collector_instance='repair1')
            original = control.bundle('h100-10')[2]
            root = physical / 'preparation' / manifest['directory'];stage = root / identity;stage.mkdir(parents=True)
            for name, raw in files.items():(stage / name).write_bytes(raw)
            write(stage / 'manifest.json', manifest)
            write(stage / 'stage-receipt.json', dict(status='EXACT_TERMINAL_COLLECTOR_STAGE_SHA_PASS'))
            plan = root / 'plan.json';write(plan, dict(run_id=control.RUN_ID, node='h100-10'));plan_sha = sha(plan)
            old_intent, old_receipt = root / 'collector-intent.json', root / 'collector-receipt.json'
            write(old_intent, dict(unit=original['unit'], plan_sha256=plan_sha, argv=['systemd-run', '--unit=' + original['unit']]))
            write(old_receipt, dict(status=receipt_status, returncode=0))
            originals = (old_intent.read_bytes(), old_receipt.read_bytes(), plan.read_bytes(), plan.stat().st_ino)
            commands = []
            def query(command, **kwargs):
                self.assertEqual(command[:2], ['systemctl', 'show'])
                if command[2] == manifest['unit']:return 'LoadState=not-found\n'
                self.assertEqual(command[2], original['unit'])
                return 'Id=%s\nLoadState=loaded\nMainPID=%s\nActiveState=%s\nResult=exit-code\nExecMainStatus=1\n' % (original['unit'], '987' if state == 'active' else '0', state)
            def launch(command, **kwargs):
                self.assertEqual(command[0], 'systemd-run');commands.append(command)
                return SimpleNamespace(returncode=0, stdout='fake start only', stderr='')
            def to_physical(path):return str(physical / Path(path).relative_to(base))
            runtime = SimpleNamespace(validate_plan=lambda value:value, core=SimpleNamespace(physical=to_physical))
            values = dict(BASE=str(base), PHYSICAL=str(physical), RUN_ID=control.RUN_ID, ACTION='collect',
                MANIFEST=manifest, IDENTITY=identity, PLAN_SHA=plan_sha, WATCH=7200,
                PUBLICATION=dict(status='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED'))
            with patch('os.geteuid', return_value=0), patch('socket.gethostname', return_value='h100-10'), \
                 patch.object(subprocess, 'check_output', side_effect=query), patch.object(subprocess, 'run', side_effect=launch), \
                 patch.dict(sys.modules, {'rollover_terminal': runtime}), patch.object(sys, 'path', list(sys.path)), contextlib.redirect_stdout(io.StringIO()):
                yield SimpleNamespace(values=values, root=root, base=base, physical=physical, manifest=manifest,
                    original=original, stage=stage, plan_sha=plan_sha, commands=commands, originals=originals)
            self.assertEqual((old_intent.read_bytes(), old_receipt.read_bytes(), plan.read_bytes(), plan.stat().st_ino), originals)

    def test_filename_argv_and_distinct_repair_intent_bind_exact_existing_plan(self):
        with self.fixture() as f:
            exec(control.REMOTE, f.values)
            command = f.commands[0]
            self.assertFalse(any('\n' in arg or '\r' in arg for arg in command))
            self.assertNotIn('-c', command)
            bash = command.index('/bin/bash')
            self.assertEqual(command[bash + 1], str(f.stage / 'native_paths.sh'))
            self.assertEqual(command[bash + 2], 'h100-10')
            self.assertEqual(sha(Path(command[bash + 1])), f.manifest['files']['native_paths.sh']['sha256'])
            self.assertIn('--setenv=Q7B_NATIVE_UNIT_INSTANCE=' + f.manifest['unit'], command)
            self.assertEqual(command[command.index('--plan') + 1], str(f.base / 'preparation' / f.manifest['directory'] / 'plan.json'))
            self.assertEqual(command[command.index('--plan-sha256') + 1], f.plan_sha)
            intent = json.loads((f.root / 'collector-repair1-intent.json').read_text())
            self.assertEqual(intent['collector_instance'], 'repair1')
            self.assertEqual(intent['prior_collectors_terminal'][0]['observed']['ActiveState'], 'failed')
            self.assertEqual(intent['prior_collectors_terminal'][0]['unit'], f.original['unit'])
            self.assertNotEqual(f.manifest['unit'], f.original['unit'])
            self.assertEqual(f.manifest['directory'], f.original['directory'])
            self.assertEqual(control.bundle('h100-5')[2]['collector_intent'], 'collector-intent.json')
            # A consumed repair intent is also never replayed.
            exec(control.REMOTE, dict(f.values));self.assertEqual(len(f.commands), 1)

    def test_active_prior_or_unknown_start_refuses_before_new_intent(self):
        for state, status in (('active', 'TERMINAL_ONLY_COLLECTOR_STARTED_ONCE'), ('failed', 'COLLECTOR_START_UNKNOWN_QUERY_SAME_UNIT')):
            with self.subTest(state=state, status=status), self.fixture(state, status) as f:
                with self.assertRaises(AssertionError):exec(control.REMOTE, f.values)
                self.assertFalse(f.commands)
                self.assertFalse((f.root / 'collector-repair1-intent.json').exists())
                self.assertFalse((f.root / 'collector-repair1-receipt.json').exists())


if __name__ == '__main__':unittest.main()
