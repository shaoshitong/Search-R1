"""Focused local CPU tests: no actual process, NVML or network action."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from reproduction.operations.qwen7b_resample.operational_reads import node10_primary_v1_pre_model_terminal as terminal


class PrimaryV1TerminalTest(unittest.TestCase):
    def test_binding_uses_real_controller_birth_and_exact_cpu_borrowers(self):
        slot, pid = terminal.COHORT[0]
        unit = 'sst-q7b-g5-primary10-' + slot + '-v1-' + terminal.RUN_ID[:12] + '.service'
        allocation = dict(run_id=terminal.RUN_ID, node='h100-10', controller_pid=pid, controller_birth=12345,
            boot_id='actual-boot', native_unit=unit, native_cgroup='/system.slice/' + unit,
            source_owner=dict(pid=2882823), local_relay_process=dict(pid=2882852),
            local_relay_owned=False, upstream_retriever_owned=False)
        evidence = dict(node='h100-10', main_pid=pid, controller_birth=12345, boot_id='actual-boot',
            unit=unit, cgroup='/system.slice/' + unit)
        self.assertEqual(terminal.binding(slot, pid, allocation, evidence), unit)
        for key, invalid in (('controller_birth', 12346), ('node', 'h100-5'), ('main_pid', pid + 1), ('boot_id', 'other')):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                terminal.binding(slot, pid, allocation, dict(evidence, **{key: invalid}))
        with self.assertRaises(AssertionError):terminal.binding(slot, pid, dict(allocation, controller_birth=True), evidence)

    def test_child_missing_birth_is_null_and_protected_pid_is_rejected(self):
        entries = terminal.own_pids(2886544, 12345, Path('/own/job'), dict(own_process_groups=[2886600]))
        self.assertEqual(entries[2886544]['birth'], '12345')
        self.assertIsNone(entries[2886600]['birth'])
        self.assertIn('not recorded', entries[2886600]['sources'][0]['limitation'])
        for pid in terminal.PROTECTED:
            with self.subTest(pid=pid), self.assertRaises(AssertionError):
                terminal.own_pids(2886544, 12345, Path('/own/job'), dict(own_process_groups=[pid]))

    def test_actual_premodel_error_and_zero_counts_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory);launch = job / 'sampler-launches/launch-00';launch.mkdir(parents=True)
            (launch / 'failure.json').write_text(json.dumps(dict(error_type='KeyError', error="'original_cuda_visible_devices'")))
            (job / 'failure.json').write_text(json.dumps(dict(foreign_processes_signalled=False, upstream_retriever_signalled=False)))
            (job / 'sampler.log').write_text("_configure_one_visible_gpu\nshared_vram_worker.py\nKeyError: 'original_cuda_visible_devices'\n")
            fact = terminal.pre_model(job)
            self.assertFalse(fact['actual_model_initialized']);self.assertFalse(fact['actual_typed_oom'])
            self.assertEqual(fact['new_trajectories'], 0)
            (job / 'progress.json').write_text(json.dumps(dict(run_id=terminal.RUN_ID, trajectories=5)))
            with self.assertRaises(AssertionError):terminal.pre_model(job)
            (job / 'progress.json').unlink()
            (launch / 'model-initialized.json').write_text('{}')
            with self.assertRaises(AssertionError):terminal.pre_model(job)

    def test_inode_sha_stability_and_immutable_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory);log = folder / 'sampler.log';log.write_bytes(b'original failure\n')
            row = terminal.original(log);terminal.stable([row])
            self.assertEqual(row['sha256'], hashlib.sha256(log.read_bytes()).hexdigest())
            self.assertEqual(row['inode'], log.stat().st_ino)
            log.write_bytes(b'changed\n')
            with self.assertRaises(AssertionError):terminal.stable([row])
            proof = folder / 'complete.json';terminal.publish(proof, dict(actual=True));inode=proof.stat().st_ino
            terminal.publish(proof, dict(actual=True));self.assertEqual(proof.stat().st_ino, inode)
            with self.assertRaises(AssertionError):terminal.publish(proof, dict(actual=False))
            self.assertFalse(list(folder.glob('*.tmp-*')))

    def test_release_is_exact_five_units_and_does_not_require_cpu_relay_port_free(self):
        calls = []
        def output(command, **unused):
            calls.append(command)
            if command[0] == 'nvidia-smi':return '2882823\n2882852\n99999\n'
            return '\n'.join(['Id=' + command[2], 'MainPID=0', 'ActiveState=failed',
                              'Result=exit-code', 'ExecMainStatus=1', 'ControlGroup='])
        with mock.patch.object(terminal.subprocess, 'check_output', output), \
             mock.patch.object(Path, 'exists', return_value=False):
            fact = terminal.release({2886544: dict(birth='12345'), 2886600: dict(birth=None)})
        self.assertEqual({c[2] for c in calls if c[0] == 'systemctl'}, terminal.UNITS)
        self.assertEqual(len([c for c in calls if c[0] == 'systemctl']), 5)
        self.assertEqual(fact['own_gpu_contexts'], [])
        self.assertEqual(fact['borrowed_cpu_relay_pids_excluded'], [2882823, 2882852])
        self.assertTrue(fact['port_28831_free_not_required'])
        self.assertNotIn('loopback_28831_listeners', fact)
        with mock.patch.object(terminal.subprocess, 'check_output', output), \
             mock.patch.object(Path, 'exists', return_value=False):
            with self.assertRaises(AssertionError):terminal.release({99999: dict(birth=None)})

    def test_metadata_does_not_read_sampling_or_secret_material(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory);(folder / 'sampling').mkdir();(folder / 'tmp').mkdir()
            for relative in ('allocation.json', 'bootstrap.log', 'sampling/full.json', 'tmp/cache.json', 'relay-secret.json'):
                (folder / relative).write_text('original')
            self.assertEqual({p.name for p in terminal.metadata(folder)}, {'allocation.json', 'bootstrap.log'})
        self.assertEqual(terminal.physical(str(terminal.RECORDED / 'pool/x')), terminal.BASE / 'pool/x')
        with self.assertRaises(AssertionError):terminal.physical('/tmp/foreign')


if __name__ == '__main__':unittest.main()
