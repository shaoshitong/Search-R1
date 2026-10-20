"""Local CPU checks only; no service action or GPU query is executed."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from reproduction.operations.qwen7b_resample.operational_reads import node10_failed_cohort_terminal as terminal


class Node10FailedCohortTerminalTest(unittest.TestCase):
    def test_birth_sources_scope_and_missing_birth(self):
        own = sorted(terminal.UNITS)[0]
        entries = {123: dict(pid=123, birth=None, sources=[])}
        terminal.collect_births(dict(pid=123, birth='456', cgroup='0::/system.slice/' + own), Path('own.json'), entries)
        terminal.collect_births(dict(pid=999, birth='888', unit=own, node='h100-5'), Path('foreign.json'), entries)
        terminal.collect_births(dict(foreign_initial_contexts=[dict(pid=777)], source=dict(pid=1490465,
            birth='6808057', cgroup='0::/system.slice/source5.service')), Path('allocation.json'), entries)
        terminal.collect_births(dict(pid=124, unit=own), Path('own-no-birth.json'), entries)
        self.assertEqual(set(entries), {123})
        self.assertEqual(entries[123]['birth'], '456')
        self.assertEqual(entries[123]['sources'][0]['path'], 'own.json')
        with self.assertRaisesRegex(AssertionError, 'disagree'):
            terminal.collect_births(dict(pid=123, birth='457', unit=own), Path('other.json'), entries)

    def test_original_sha_inode_and_stability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sampler.log'
            path.write_bytes(b'original HTTPError\n')
            row, raw = terminal.original(path)
            self.assertEqual(row['sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(row['inode'], path.stat().st_ino)
            terminal.stable([row])
            path.write_bytes(b'changed\n')
            with self.assertRaises(AssertionError):terminal.stable([row])
            link = Path(directory) / 'link.log';link.symlink_to(path)
            with self.assertRaises(OSError):terminal.original(link)

    def test_metadata_inventory_keeps_bootstrap_and_excludes_data(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ('allocation.json', 'sampler.log', 'bootstrap.py', 'launch.sh'):
                (folder / name).write_text('small original')
            (folder / 'states').mkdir();(folder / 'states/large.json').write_text('do not read')
            (folder / 'trajectory.jsonl').write_text('do not read')
            self.assertEqual({p.name for p in terminal.originals([folder])},
                             {'allocation.json', 'sampler.log', 'bootstrap.py', 'launch.sh'})

    def test_immutable_publish_never_replaces_original_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'complete.json'
            terminal.publish(path, dict(status='actual'))
            inode = path.stat().st_ino
            terminal.publish(path, dict(status='actual'))
            self.assertEqual(path.stat().st_ino, inode)
            with self.assertRaisesRegex(AssertionError, 'differs'):
                terminal.publish(path, dict(status='invented'))
            self.assertEqual(json.loads(path.read_text()), dict(status='actual'))
            self.assertFalse(list(Path(directory).glob('*.tmp-*')))

    def test_loopback_port_uses_listeners_not_active_connections(self):
        def network_text(path):
            if path.name == 'tcp':
                return 'header\n0: 0100007F:709F 00000000:0000 0A\n1: 0100007F:709F 00000000:0000 01\n'
            return 'header\n0: 00000000000000000000000001000000:709F 00000000000000000000000000000000:0000 0A\n'
        with mock.patch.object(Path, 'read_text', network_text):
            rows = terminal.listeners()
        self.assertEqual([row['address'] for row in rows], ['127.0.0.1', '::1'])

    def test_release_queries_only_four_units_and_own_gpu_pids(self):
        queried = []
        def output(command, **unused):
            queried.append(command)
            if command[0] == 'nvidia-smi':return '99999\n'
            return '\n'.join(['Id=' + command[2], 'MainPID=0', 'ActiveState=failed',
                              'Result=exit-code', 'ExecMainStatus=1', 'ControlGroup='])
        with mock.patch.object(terminal.subprocess, 'check_output', output), \
             mock.patch.object(Path, 'exists', return_value=False), \
             mock.patch.object(terminal, 'listeners', return_value=[]):
            result = terminal.release({123: dict(pid=123, birth=None)})
        self.assertEqual({cmd[2] for cmd in queried if cmd[0] == 'systemctl'}, terminal.UNITS)
        self.assertEqual(result['own_gpu_contexts'], [])
        self.assertEqual(result['remaining_known_pids'], [])
        self.assertTrue(result['original_births_gone'])
        self.assertFalse(result['foreign_signalled'])
        with mock.patch.object(terminal.subprocess, 'check_output', output), \
             mock.patch.object(Path, 'exists', return_value=False), \
             mock.patch.object(terminal, 'listeners', return_value=[]):
            with self.assertRaises(AssertionError):terminal.release({99999: dict(pid=99999, birth=None)})

    def test_canonical_alias_maps_only_own_base(self):
        self.assertEqual(terminal.physical(str(terminal.RECORDED / 'pool/jobs/x')), terminal.BASE / 'pool/jobs/x')
        with self.assertRaises(AssertionError):terminal.physical('/tmp/foreign')


if __name__ == '__main__':
    unittest.main()
