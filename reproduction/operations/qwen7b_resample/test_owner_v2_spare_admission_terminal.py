"""Focused CPU checks; service/NVML calls are mocked and no real proof is sealed."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from reproduction.operations.qwen7b_resample import guarded_generation_read as reader
from reproduction.operations.qwen7b_resample.operational_reads import owner_v2_spare_admission_terminal as terminal


class SpareAdmissionTerminalTest(unittest.TestCase):
    def test_exact_node_split_and_alias_recording(self):
        source, node10 = terminal.selected('h100-5'), terminal.selected('h100-10')
        self.assertEqual([row['key'] for row in source], ['source:gpu%d-tp1' % i for i in (1, 3, 4, 6, 7)])
        self.assertEqual([row['key'] for row in node10], ['node10:gpu1-tp1', 'node10:gpu2-tp1'])
        self.assertFalse(set(row['unit'] for row in source) & set(row['unit'] for row in node10))
        view = terminal.View(terminal.ALIAS)
        self.assertEqual(view.path(terminal.BASE / 'pool/jobs/x'), terminal.ALIAS / 'pool/jobs/x')
        self.assertEqual(view.canonical(terminal.ALIAS / 'pool/jobs/x'), str(terminal.BASE / 'pool/jobs/x'))
        with self.assertRaises(ValueError):view.path('/tmp/foreign')

    def test_release_checks_only_node_owned_units_and_numeric_pid_absence(self):
        rows = terminal.selected('h100-10');calls = []
        known = {row['pid']: dict(pid=row['pid'], birth=None) for row in rows}
        def output(command, **unused):
            calls.append(command)
            if command[0] == 'nvidia-smi':return '2714543\n3221084\n99999\n'
            return '\n'.join(['Id=' + command[2], 'MainPID=0', 'ActiveState=failed', 'ControlGroup=', 'Result=exit-code', 'ExecMainStatus=1'])
        with mock.patch.object(terminal.subprocess, 'check_output', output), mock.patch.object(Path, 'exists', return_value=False):
            observed = terminal.release(rows, known)
        self.assertEqual({command[2] for command in calls if command[0] == 'systemctl'}, {row['unit'] for row in rows})
        self.assertTrue(observed['known_birth_not_stored'])
        self.assertTrue(observed['original_births_gone'])
        self.assertIn('no missing historical birth', observed['birth_claim_basis'])
        self.assertEqual(observed['own_gpu_contexts'], [])
        with mock.patch.object(terminal.subprocess, 'check_output', output), mock.patch.object(Path, 'exists', return_value=False):
            with self.assertRaises(AssertionError):terminal.release(rows, {99999: dict(birth=None)})

    def test_original_inode_hash_and_inventory_omit_data_and_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(terminal, 'BASE', root):
                view = terminal.View(root);log = root / 'sampler.log';log.write_bytes(b'capacity admission failure\n')
                record = terminal.original(view, log);terminal.stable(view, [record])
                self.assertEqual(record['sha256'], hashlib.sha256(log.read_bytes()).hexdigest())
                self.assertEqual(record['inode'], log.stat().st_ino)
                (root / 'worker.py').write_text('code');(root / 'relay-secret.json').write_text('excluded')
                (root / 'sampling').mkdir();(root / 'sampling/body.json').write_text('excluded')
                self.assertEqual({p.name for p in terminal.metadata([root])}, {'sampler.log', 'worker.py'})
                log.write_bytes(b'changed\n')
                with self.assertRaises(AssertionError):terminal.stable(view, [record])

    def test_real_historical_index_consumer_accepts_numeric_absence_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory);failure = base / 'pool/jobs/job-old/attempt-one/failure.json';failure.parent.mkdir(parents=True)
            failure.write_text('{"error_type":"OutOfMemoryError"}')
            failure_sha = hashlib.sha256(failure.read_bytes()).hexdigest();proof = base / 'preparation/proof/complete.json';proof.parent.mkdir(parents=True)
            release = dict(original_births_gone=True, own_cgroups_empty=True, remaining_known_pids=[],
                remaining_original_births={}, own_gpu_contexts=[], known_birth_not_stored=True,
                birth_claim_basis='Every evidenced numeric PID is absent; missing historical births were not inferred')
            payload = dict(status=terminal.STATUS, run_id=terminal.RUN, controlled_stop=False,
                release_observations=[dict(release, epoch=1), dict(release, epoch=4)],
                original_files=[dict(path=str(failure), sha256=failure_sha)])
            terminal.publish(proof, payload)
            proof_sha = hashlib.sha256(proof.read_bytes()).hexdigest();r = reader.Reader(base, base, [base])
            index, proofs, errors = reader.historical_index(r, [dict(path=str(proof), sha256=proof_sha)], terminal.RUN)
            self.assertFalse(errors);self.assertEqual(len(proofs), 1)
            self.assertEqual(index[(str(failure), failure_sha)]['proof_status'], terminal.STATUS)
            self.assertFalse(json.loads(proof.read_text())['controlled_stop'])
            terminal.publish(proof, payload)
            with self.assertRaises(AssertionError):terminal.publish(proof, dict(payload, controlled_stop=True))


if __name__ == '__main__':unittest.main()
