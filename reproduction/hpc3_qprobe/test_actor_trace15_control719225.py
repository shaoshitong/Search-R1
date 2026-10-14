"""Negative consumer cases: no remote calls, child launch, tensor or model imports."""
import copy
import unittest
from reproduction.hpc3_qprobe import actor_trace15_control719225 as control


class TraceResultConsumer(unittest.TestCase):
    def setUp(self):
        self.process = dict(pid=1, start_ticks=2, sid=1, uid=14090)
        self.child = dict(pid=3, start_ticks=4, sid=1, uid=14090)
        self.result = dict(process=self.process, child=self.child, actual_child_exit_code=0, stderr=dict(bytes=0))
        self.claim = dict(producer=dict(producer_commit='source', source_manifest_sha256='manifest',
                                        native_allocation_sha256='allocation'))
        self.output = dict(status='ACTUAL_RAW15_PRIVATE_TRACE_ALLTOKENS_AND_ACTOR_ROWS_EQUAL', step=15,
                           job='719225', raw_file=dict(sha256=control.RAW_SHA), source_commit='source',
                           source_manifest=dict(sha256='manifest'), allocation_file=dict(sha256='allocation'),
                           actual_private_trace_byte_equal=True, actual_selected_logprob_projection_equal=True,
                           hf_forward_performed=False, model_weights_loaded=False, generated_trajectories=0,
                           actor_or_head_updates_performed=0, evidence_files_written=0)

    def check(self, result=None, process=None, child=None, live=False, output=None):
        control.validate_completed(result if result is not None else self.result,
                                   process if process is not None else self.process,
                                   child if child is not None else self.child, live,
                                   output if output is not None else self.output, self.claim)

    def test_complete_bound_wait_and_reader_can_be_consumed(self):
        self.check()

    def test_missing_birth_header_cannot_be_consumed(self):
        value = dict(self.result, process=None, child=None)
        with self.assertRaises(ValueError):
            control.validate_completed(value, None, None, False, self.output, self.claim)

    def test_missing_or_reused_or_wrong_owner_or_foreign_session_child_rejected(self):
        for changed in ({}, dict(self.child, start_ticks=0), dict(self.child, uid=9999),
                        dict(self.child, sid=5), dict(self.child, pid=self.process['pid'])):
            with self.subTest(child=changed), self.assertRaises(ValueError):
                self.check(result=dict(self.result, child=changed), child=changed)

    def test_live_child_or_worker_error_or_nonzero_wait_or_stderr_rejected(self):
        with self.assertRaises(ValueError):
            self.check(live=True)
        for changed in (dict(self.result, worker_error={}), dict(self.result, actual_child_exit_code=1),
                        dict(self.result, stderr=dict(bytes=1))):
            with self.subTest(result=changed), self.assertRaises(ValueError):
                self.check(result=changed)

    def test_claim_manifest_and_allocation_anchors_must_match(self):
        for key in ('source_manifest', 'allocation_file'):
            output = copy.deepcopy(self.output)
            output[key]['sha256'] = 'changed'
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.check(output=output)

    def test_actor_write_or_hf_forward_cannot_pass_readonly_result(self):
        for key, value in (('actor_or_head_updates_performed', 1), ('hf_forward_performed', True)):
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.check(output=dict(self.output, **{key: value}))


if __name__ == '__main__':
    unittest.main()
