"""Read-only regression against retained real first-five worker receipts.

Set SEARCH_G1_TEST_ACTUAL_ZERO_REFERENCE_BASE on HPC3. No model/Adam tensor is
loaded and no receipt is changed; the negative case modifies an in-memory copy.
"""
import copy
import json
import os
from pathlib import Path
import unittest

from reproduction.common8.formal_contract import monitor_reference_contract
from reproduction.common8.formal_attempt_audit import _audit_monitor_rows
from reproduction.common8.policy_calibration_plan import document_sha256
from reproduction.common8.safety_calibration_evidence import EvidenceError, _reference_required


ACTUAL_BASE = os.environ.get('SEARCH_G1_TEST_ACTUAL_ZERO_REFERENCE_BASE')


@unittest.skipUnless(ACTUAL_BASE, 'Requires retained actual HPC3 first-five receipts')
class ActualHpc3ZeroReferenceMonitor(unittest.TestCase):
    def test_real_first_five_and_changed_valid_hash(self):
        from search_g1.qprobe_saved_checkpoint import _first_five_safety_audit
        base = Path(ACTUAL_BASE).resolve()
        run = json.loads((base / 'logs/run-manifest.json').read_text())
        self.assertEqual((run['execution_backend'], run['execution_site'], run['training_world_size']),
                         ('slurm', 'hpc3', 4))
        proof = _first_five_safety_audit(base, run)
        self.assertEqual(proof['actual_optimizer_writes_per_rank'], 100)
        self.assertEqual(proof['actual_complete_outers'], 5)
        plan = json.loads((base / 'logs/policy-safety-calibration.json').read_text())
        binding = dict(run_id=run['run_id'], calibration_sha256=document_sha256(plan),
                       monitor_reference_contract=monitor_reference_contract(base, run))
        self.assertFalse(_reference_required(binding))
        attempt = json.loads((base / 'logs/actual-attempts.jsonl').read_text().splitlines()[0])
        from reproduction.common8.behavior_runtime import read_gate_receipt
        gate = read_gate_receipt(base, run, attempt['logical_update'],
                                attempt_id=attempt['attempt_id'], attempt_index=attempt['attempt_index'])
        attempt['behavior_probability_gate_sha256'] = gate['proof_sha256']
        rows = {}
        for rank in range(4):
            all_rows = [json.loads(line) for line in
                        (base / 'logs' / f'optimizer-rank-{rank}.jsonl').read_text().splitlines()]
            rows[rank] = [row for row in all_rows
                          if (row.get('context') or row)['attempt_id'] == attempt['attempt_id']]
        _audit_monitor_rows(rows, binding, attempt, 0, run=run)
        changed = copy.deepcopy(rows)
        monitor = next(row for row in changed[0] if row['kind'] == 'POLICY_MONITOR')
        hashes = monitor['monitor_identity']['tensor_sha256']
        self.assertNotIn('ref_log_prob', hashes)
        original = hashes['old_log_probs']
        hashes['old_log_probs'] = ('0' if original[0] != '0' else '1') + original[1:]
        with self.assertRaises((EvidenceError, AssertionError)):
            _audit_monitor_rows(changed, binding, attempt, 0, run=run)


if __name__ == '__main__':
    unittest.main()
