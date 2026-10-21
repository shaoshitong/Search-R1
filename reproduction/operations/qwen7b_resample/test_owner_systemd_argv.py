"""CPU-only final-argv/decoder regression; no remote, mount or systemd calls."""
import ast
import base64
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import guarded_rollover_control as guarded
from . import queued_generation_control as queued


def assignments(source, name):
    return [node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]


def evaluate(expression, values):
    return eval(compile(ast.Expression(expression), '<final-command-expression>', 'eval'), values)


class OwnerSystemdArgvTests(unittest.TestCase):
    def test_three_owners_decode_only_after_systemd_and_keep_exact_shell_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            # Real multiline bytes include shell syntax, quotes and dollar
            # expansion. These bytes must never appear in systemd's argv.
            payload = Path(directory) / 'native_paths.sh'
            payload.write_text('#!/bin/bash\nset -euo pipefail\nexpected="$1"; shift\nprintf "%s\\n" "$expected"\nexec "$@"\n')
            encoded = base64.b64encode(payload.read_bytes()).decode()
            for control, role, node, dummy, bootstrap in (
                    (guarded, 'owner', 'h100-5', 'owner-generation-paths', 'generation_bootstrap.py'),
                    (queued, 'upstream', 'h100-5', 'relay-generation-paths', 'relay_generation_bootstrap.py'),
                    (queued, 'downstream', 'h100-10', 'relay-generation-paths', 'relay_generation_bootstrap.py')):
                with self.subTest(role=role):
                    source = control.remote_source()
                    original = control.original.REMOTE
                    stage = Path(directory) / 'stage'
                    unit = 'sst-q7b-g5-test-' + role + '.service'
                    values = dict(base64=base64, FILES={'native_paths.sh': encoded}, stage=stage, str=str,
                        node=node, ROLE=role, role=role, UNIT=unit, unit=unit)
                    def owner_target(program):
                        return next(expression for expression in assignments(program, 'target')
                            if isinstance(expression, ast.List) and expression.elts[0].value == 'unshare')
                    old_target = evaluate(owner_target(original), dict(values))
                    target = evaluate(owner_target(source), dict(values))
                    self.assertEqual(target[:6], ['unshare', '--mount', '--propagation', 'private', '/usr/bin/python3', '-c'])
                    trailing = [dummy, node, '/usr/bin/python3', str(stage / bootstrap)]
                    self.assertEqual(target[7:], trailing)
                    values['target'] = target
                    final_expression = next(expression for expression in assignments(source, 'command')
                        if isinstance(expression, ast.List) and expression.elts[0].value == 'systemd-run')
                    command = evaluate(final_expression, values)
                    self.assertIn('--setenv=Q7B_NATIVE_UNIT_INSTANCE=' + unit, command)
                    self.assertTrue(all('\n' not in argument and '\r' not in argument and '$' not in argument for argument in command))
                    # Execute the actual generated single-line launcher; only
                    # its final exec boundary is replaced. No shell is run.
                    with patch.object(sys, 'argv', ['-c', *target[7:]]), patch('os.execv') as execute:
                        exec(compile(target[6], '<generated-owner-launcher>', 'exec'), {})
                    execute.assert_called_once_with('/bin/bash', ['/bin/bash', '-c', payload.read_text(), *trailing])
                    self.assertEqual(execute.call_args.args[1], old_target[4:])

    def test_source_and_node10_model_slot_targets_are_unchanged(self):
        for control in (guarded, queued):
            with self.subTest(control=control.__name__):
                def slot_target(source):
                    return next(expression for expression in assignments(source, 'target')
                        if isinstance(expression, ast.List) and expression.elts[0].value == 'nsenter')
                before = slot_target(control.original.REMOTE)
                after = slot_target(control.remote_source())
                self.assertEqual(ast.dump(before), ast.dump(after))
                values = dict(owner={'pid': 123}, str=str, stage=Path('/same/stage'))
                target = evaluate(after, values)
                self.assertEqual(target[:7], ['nsenter', '--target', '123', '--mount', '--net', '--', '/usr/bin/python3'])
                self.assertTrue(target[-1].endswith('_bootstrap.py'))
                self.assertNotIn('-c', target)


if __name__ == '__main__': unittest.main()
