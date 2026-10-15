"""Explicit final routes and real-file checksum boundaries; no remote jobs."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from reproduction.hpc3_qprobe import independent_goal_acceptance719225 as reader
from reproduction.hpc3_qprobe import independent_goal_control719225 as holder
from reproduction.hpc3_qprobe import raw_terminal_reaggregation719225 as recon
from reproduction.hpc3_qprobe import test_independent_goal_acceptance719225 as fixtures
from reproduction.hpc3_qprobe import test_independent_goal_control719225 as holder_fixtures

write = fixtures.write


class ReaderRouteTests(unittest.TestCase):
    def fixture(self, root):
        base, ea, _, accounting = fixtures.EATerminalTests().fixture(root)
        native_base = Path(json.loads((ea.DIRECTORY/'final-consumer-binding.json').read_text())['audit_base'])
        original_goal = base/'logs/qprobe-final-goal-acceptance.json'; goal = json.loads(original_goal.read_text()); original_goal.unlink()
        old_result = write(ea.DIRECTORY/'result.json', dict(exit_code=1, error='original producer failed'))
        directory = root/'followthrough/recovery719225-finalcpu-20261015'; directory.mkdir(parents=True)
        stage1 = 'b'*40; stage3 = 'f'*40; source = dict(commit=stage3, publication_sha256='e'*64)
        reference = dict(commit=stage1, original_producer_state='FAILED', original_producer_exit_code='1:0')
        for name in ('result', 'output', 'structural_acceptance'):
            reference[name] = write(root/'structural'/name, {'original_failure_preserved':True})
        goal.update(producer_final_cpu_recovery=reference, original_producer_accounting_remains_failed=True)
        goal_file = write(directory/'goal-acceptance.json', goal)
        process = dict(pid=100, sid=100, start_ticks=101, uid=reader.OWNER)
        child = dict(pid=101, sid=100, start_ticks=102, uid=reader.OWNER)
        claim_file = write(directory/'claim.json', dict(source=source, recovery_commit=stage1,
            admission=dict(producer_final_cpu_recovery=reference)))
        write(directory/'process.json', process); write(directory/'child.json', child)
        result_file = write(directory/'result.json', dict(process=process, child=child, source=source,
            claim_sha256=claim_file['sha256'], actual_child_exit_code=0, supervisor_exit_code=0))
        binding_file = write(directory/'final-consumer-binding.json', dict(actual_process=child, helper_commit=stage3,
            recovery_commit=stage1, full_cpu_consumer_returned=True, goal_sha256=goal_file['sha256'],
            audit_base=str(native_base), actual_native_audit_accounting=accounting))
        output_file = write(directory/'consumer.stdout', dict(status='stage3 intermediate pass', goal_complete=False))
        held = dict(status='ACTUAL_HELD_RECOVERY_NATIVE4_AND_GOAL_REVALIDATION_COMPLETED0', source=source,
            actual_child_exit_code=0, producer_final_cpu_recovery=reference, original_accounting_remains_failed=True,
            goal_complete=False, full_objective_accepted=False, independent_teacher_raw_final_consumer_still_required=True,
            result_path=result_file['path'], result_sha256=result_file['sha256'], goal_acceptance=goal_file,
            consumer_binding=binding_file, audit_base=str(native_base), output=output_file,
            original_ea=dict(fingerprint='d'*64, files={'result.json':old_result}))
        helper = SimpleNamespace(DIRECTORY=directory, HELD_STATUS=held['status'], consume=Mock(return_value=held))
        return base, helper, source, reference, stage1, stage3, accounting

    def test_normal_failed_producer_never_loads_typed_recovery(self):
        with patch.object(reader, 'completed', side_effect=ValueError('actual producer FAILED')), \
                patch('reproduction.hpc3_qprobe.final_control.final_producer') as recovery:
            with self.assertRaisesRegex(ValueError, 'FAILED'):
                reader.producer_acceptance()
            recovery.assert_not_called()

    def test_both_exact_recovery_pins_required(self):
        for pair in (('a'*40, None), (None, 'b'*40), ('wrong', 'b'*40)):
            with self.assertRaises(ValueError):
                reader.recovery_options(*pair)
        reader.recovery_options(None, None); reader.recovery_options('a'*40, 'b'*40)

    def test_separate_held_recovery_keeps_original_ea_failure_and_same_native_gates(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(reader, 'OWNER', os.getuid()):
            root = Path(folder).resolve()/'control'
            base, helper, source, ref, stage1, stage3, accounting = self.fixture(root)
            with patch.object(reader, 'ROOT', root), patch.object(reader, 'BASE', base), \
                    patch.object(reader, 'load_fixed', return_value=(helper, source)), \
                    patch.object(reader, 'completed', return_value=accounting):
                audit_base, native, proof = reader.recovery_evidence(stage1, stage3, ref)
                self.assertEqual(native['slurm_job_id'], '800000')
                self.assertEqual(proof['kind'], 'ACTUAL_SEPARATE_HELD_RECOVERY_CPU_CONSUMER')
                self.assertTrue(proof['original_producer_accounting_remains_failed'])
                self.assertFalse((base/'logs/qprobe-final-goal-acceptance.json').exists())
                self.assertEqual(json.loads((root/'followthrough/719225/result.json').read_text())['exit_code'], 1)
                bad = copy.deepcopy(helper.consume.return_value); bad['source'] = dict(source, commit='c'*40)
                helper.consume.return_value = bad
                with self.assertRaisesRegex(ValueError, 'typed held0'):
                    reader.recovery_evidence(stage1, stage3, ref)

    def test_changed_recovery_goal_or_native_bytes_cannot_match_a_held_receipt(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(reader, 'OWNER', os.getuid()):
            root = Path(folder).resolve()/'control'
            base, helper, source, ref, stage1, stage3, accounting = self.fixture(root)
            with patch.object(reader, 'ROOT', root), patch.object(reader, 'BASE', base), \
                    patch.object(reader, 'load_fixed', return_value=(helper, source)), \
                    patch.object(reader, 'completed', return_value=accounting):
                path = helper.DIRECTORY/'goal-acceptance.json'; value = json.loads(path.read_text())
                write(path, dict(value, completed_outer=201))
                with self.assertRaisesRegex(ValueError, 'bytes differ'):
                    reader.recovery_evidence(stage1, stage3, ref)

    def test_typed_route_accepts_original_raw_pass_without_reaggregation(self):
        raw = SimpleNamespace(DIRECTORY=Path('/original'), STATUS='original RAW PASS')
        pub = dict(commit=reader.RAW); proof = dict(source=pub)
        with patch.object(reader, 'document', return_value=({'status':raw.STATUS}, {})), \
                patch.object(reader, 'raw_evidence', return_value=(proof, ['original tasks'])) as original, \
                patch.object(recon, 'verify') as fresh:
            self.assertEqual(reader.raw_final_evidence(raw, pub, 'a'*40, 'b'*40), (proof, ['original tasks']))
            original.assert_called_once_with(raw); fresh.assert_not_called()

    def test_only_explicit_failed_raw_route_can_freshly_reaggregate(self):
        raw = SimpleNamespace(DIRECTORY=Path('/original'), STATUS='original RAW PASS')
        pub = dict(commit=reader.RAW)
        proof = dict(status=recon.STATUS, source={'commit':'a'*40}, original_raw_source=pub,
            producer_final_cpu_recovery={'commit':'b'*40}, original_raw_status_remains_failed=True,
            goal_complete=False, full_objective_accepted=False, independent_final_actual_held_wait0_still_required=True,
            precise_original_aggregation_call_failure_proven=False, actor_model_loaded=False,
            decode_reexecuted=0, generated_trajectories=0, actor_head_or_scheduler_updates=0,
            receipt_or_cache_or_claim_writes=0, process_or_job_submissions=0, tasks=['full original tasks'])
        with patch.object(reader, 'document', return_value=({'status':recon.FAILED_STATUS}, {})), \
                patch.object(recon, 'verify', return_value=proof) as fresh, patch.object(reader, 'raw_evidence') as original:
            self.assertEqual(reader.raw_final_evidence(raw, pub, 'a'*40, 'b'*40)[1], proof['tasks'])
            fresh.assert_called_once_with('a'*40, 'b'*40); original.assert_not_called()
            fresh.return_value = dict(proof, decode_reexecuted=1)
            with self.assertRaisesRegex(ValueError, 'another source, scope'):
                reader.raw_final_evidence(raw, pub, 'a'*40, 'b'*40)
        with patch.object(reader, 'document', return_value=({'status':'UNKNOWN'}, {})), patch.object(recon, 'verify') as fresh:
            with self.assertRaisesRegex(ValueError, 'live/unknown'):
                reader.raw_final_evidence(raw, pub, 'a'*40, 'b'*40)
            fresh.assert_not_called()


class HolderRouteTests(unittest.TestCase):
    def fixture(self, directory):
        commit, result = holder_fixtures.IndependentFinalControlTests().fixture(directory)
        stage1='b'*40; stage3='c'*40
        binding=dict(producer_final_cpu_recovery={'commit':stage1}, followthrough_source={'commit':stage3},
            followthrough_result_sha256='d'*64, goal_sha256='e'*64, consumer_binding_sha256='f'*64,
            raw_result_sha256='1'*64, raw_claim_sha256='2'*64)
        claim=json.loads((directory/'claim.json').read_text())
        claim.update(recovery_commit=stage1,recovery_followthrough_commit=stage3,
            admission=dict(recovery_binding=binding,raw_original_failure_anchors={}))
        write(directory/'claim.json',claim)
        proof=json.loads((directory/'consumer.stdout').read_text())
        proof.update(recovery_commit=stage1,recovery_followthrough_commit=stage3,
            producer_final_cpu_recovery=binding['producer_final_cpu_recovery'],
            producer_accounting=dict(state='FAILED',exit_code='1:0'),original_producer_accounting_remains_failed=True,
            original_EA_status_remains_failed=True,actual_recovery_full_CPU_consumer=dict(source=binding['followthrough_source'],
                result={'sha256':binding['followthrough_result_sha256']},goal={'sha256':binding['goal_sha256']},
                binding={'sha256':binding['consumer_binding_sha256']}),independent_all_normal_return_raw=dict(
                result={'sha256':binding['raw_result_sha256']},claim={'sha256':binding['raw_claim_sha256']}))
        write(directory/'consumer.stdout',proof)
        result['claim_sha256']=holder.ctl.digest(directory/'claim.json')
        result['stdout'].update(sha256=holder.ctl.digest(directory/'consumer.stdout'),bytes=(directory/'consumer.stdout').stat().st_size)
        write(directory/'result.json',result)
        return commit,stage1,stage3,result,proof,binding

    def test_recovered_full_output_can_be_consumed_only_under_same_once_intent(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder).resolve();commit,stage1,stage3,result,proof,binding=self.fixture(directory)
            with patch.object(holder,'DIRECTORY',directory),patch.object(holder.ctl,'source',return_value=(directory,result['source'])), \
                    patch.object(holder.ctl,'read',side_effect=lambda p:json.loads(Path(p).read_text())), \
                    patch.object(holder.ctl,'live',return_value=False),patch.object(holder,'preserved_recovery') as preserved:
                accepted=holder.consume(commit,recovery_commit=stage1,recovery_followthrough_commit=stage3)
                self.assertEqual(accepted['actual_child_exit_code'],0);self.assertFalse(accepted['goal_complete'])
                preserved.assert_called_once()
                with self.assertRaisesRegex(ValueError,'another route'):
                    holder.consume(commit)
                bad=copy.deepcopy(proof);bad['actual_EA_full_CPU_consumer']={'pretend':'old EA passed'}
                with self.assertRaisesRegex(ValueError,'recovered final stdout'):
                    holder.accept_output(bad,commit,recovery_commit=stage1,recovery_followthrough_commit=stage3,recovery_binding=binding)

    def test_recovery_missing200_refuses_before_scope_or_any_process(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder).resolve()/'not-started'
            with patch.object(holder,'DIRECTORY',directory),patch.object(holder,'caps'), \
                    patch.object(reader,'early_history',side_effect=ValueError('missing actual200')), \
                    patch.object(holder.ctl,'source') as source,patch.object(holder.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(ValueError,'missing actual200'):
                    holder.start('a'*40,recovery_commit='b'*40,recovery_followthrough_commit='c'*40)
                source.assert_not_called();launch.assert_not_called();self.assertFalse(directory.exists())

    def test_live_supervisor_cannot_be_consumed_even_after_child_wait0(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder).resolve();commit,stage1,stage3,result,_,_=self.fixture(directory)
            with patch.object(holder,'DIRECTORY',directory),patch.object(holder.ctl,'source',return_value=(directory,result['source'])), \
                    patch.object(holder.ctl,'read',side_effect=lambda p:json.loads(Path(p).read_text())), \
                    patch.object(holder.ctl,'live',side_effect=lambda row:row==result['process']):
                with self.assertRaisesRegex(ValueError,'held final consumer wait'):
                    holder.consume(commit,recovery_commit=stage1,recovery_followthrough_commit=stage3)


if __name__=='__main__':
    unittest.main()
