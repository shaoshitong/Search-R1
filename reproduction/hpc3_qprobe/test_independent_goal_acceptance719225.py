"""Real file consumers and negative terminal gates; no model/remote/job calls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from reproduction.hpc3_qprobe import independent_goal_acceptance719225 as audit


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value, *, binary=False, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if binary else json.dumps(value).encode()
    path.write_bytes(raw); path.chmod(mode)
    return dict(path=str(path), sha256=sha(raw), bytes=len(raw))


class FileTests(unittest.TestCase):
    def test_changed_same_size_and_symlink_private_file_rejected(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(audit, 'OWNER', os.getuid()):
            path = Path(folder).resolve()/'raw.pt'; original = write(path, b'abcd', binary=True)
            audit.bound_record(original, path)
            path.write_bytes(b'efgh')
            with self.assertRaisesRegex(ValueError, 'full SHA'):
                audit.bound_record(original, path)
            link = Path(folder).resolve()/'trace.json'; link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, 'linked evidence'):
                audit.read_file(link)

    def test_shared_full_read_is_reused_but_metadata_or_expected_hash_change_rejected(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(audit, 'OWNER', os.getuid()):
            path = Path(folder).resolve()/'train'; record = write(path, b'data', binary=True)
            reader = audit.SharedReads()
            with patch.object(audit, 'read_file', wraps=audit.read_file) as full_read:
                reader.read(path, record['sha256']); reader.read(path, record['sha256'])
                self.assertEqual(full_read.call_count, 1)
            with self.assertRaisesRegex(ValueError, 'changed'):
                reader.read(path, 'a'*64)
            path.write_bytes(b'edit')
            with self.assertRaisesRegex(ValueError, 'changed'):
                reader.finish()


class EarlyTests(unittest.TestCase):
    def test_missing200_fails_before_import_query_or_source_or_any_write(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(audit, 'OWNER', os.getuid()), \
             patch.object(audit, 'BASE', Path(folder).resolve()), patch.object(audit, 'publication') as source, \
             patch.object(audit, 'completed') as accounting, patch.object(audit, 'load_fixed') as pinned, \
             patch.dict(os.environ, dict(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                 **{k:'1' for k in audit.CAPS})), patch.object(sys, 'dont_write_bytecode', True):
            logs = Path(folder).resolve()/'logs'; logs.mkdir()
            (logs/'actual-updates.jsonl').write_text(json.dumps(dict(step=33, run_id=audit.RUN))+'\n')
            before = sorted(str(p) for p in Path(folder).resolve().rglob('*'))
            with self.assertRaisesRegex(ValueError, 'current6..200'):
                audit.verify('a'*40)
            source.assert_not_called(); accounting.assert_not_called(); pinned.assert_not_called()
            self.assertEqual(before, sorted(str(p) for p in Path(folder).resolve().rglob('*')))

    def test_state_only_cannot_be_an_actual_EA_return(self):
        ea = SimpleNamespace(inspect=lambda: dict(actual_live_birth=False,
            process=dict(actual_process=dict(pid=2, start_ticks=5, sid=2, uid=audit.OWNER)),
            result=None, state=dict(full_objective_accepted=True)))
        with self.assertRaisesRegex(ValueError, 'state alone'):
            audit.ea_evidence(ea, {})


class EATerminalTests(unittest.TestCase):
    def fixture(self, root):
        base = root.parent/'current'; directory = root/'followthrough/719225'; native_base = root.parent/'native'
        process = dict(pid=100, start_ticks=1000, sid=100, uid=audit.OWNER)
        native = dict(run_id=audit.RUN, producer_commit=audit.PRODUCER, world_size=4, actual_completed_outer=200,
            actual_cuda_restore_accepted=True, update201_performed=False, fit_called=False, audit_id='audit',
            actual_actor_optimizer_write_delta=0, actual_actor_scheduler_advance_delta=0,
            actual_head_update_delta=0, actual_head_optimizer_write_delta=0, generated_trajectories=0,
            slurm_job_id='800000')
        native_file = write(native_base/'logs/final-restore-readback.json', native)
        accepted_file = write(native_base/'logs/final-restore-audit-accepted.json', dict(accepted=True,
            actual_cuda_restore_accepted=True, owned_cleanup=True, readback_sha256=native_file['sha256'], audit_id='audit'))
        goal = dict(status=audit.GOAL_STATUS, run_id=audit.RUN, completed_outer=200, update201_performed=False,
            actual_native_restore_accepted_sha256=accepted_file['sha256'])
        goal_file = write(base/'logs/qprobe-final-goal-acceptance.json', goal)
        accounting = dict(job='800000', state='COMPLETED', exit_code='0:0', raw='authentic accounting')
        binding = dict(actual_process=process, helper_commit=audit.EA, full_cpu_consumer_returned=True,
            goal_sha256=goal_file['sha256'], audit_base=str(native_base), actual_native_audit_accounting=accounting)
        write(directory/'final-consumer-binding.json', binding)
        publication = dict(commit=audit.EA, publication_path='/proof', publication_sha256='a'*64)
        claim = dict(source=publication, binding=dict(fixed='intent'), training_submission_permitted=False,
            actor_model_reload_permitted=False, direct_gpu_access_permitted=False, native_final_restore_only_submission_permitted=True)
        write(directory/'claim.json', claim)
        result = dict(exit_code=0, error=None, actual_process=process, training_submission_performed=False,
            actor_model_loaded=False, direct_gpu_device_access_performed=False,
            native_final_restore_only_submission_permitted=True, result=dict(full_objective_accepted=True,
                actual_final_native_restore_accepted=True, phase='ACTUAL_FULL_CPU_GOAL_CONSUMER_PASS',
                final_goal_path=goal_file['path'], final_goal_sha256=goal_file['sha256']))
        write(directory/'result.json', result)
        process_record = dict(actual_process=process, commit=audit.EA)
        write(directory/'process.json', process_record)
        ea = SimpleNamespace(DIRECTORY=directory, fixed_binding=lambda:claim['binding'], inspect=lambda:dict(
            actual_live_birth=False, process=process_record, result=result))
        return base, ea, publication, accounting

    def test_real_returned_EA_receipt_binding_and_changed_goal_or_birth_rejected(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(audit, 'OWNER', os.getuid()):
            root = Path(folder).resolve()/'control'; base, ea, pub, accounting = self.fixture(root)
            with patch.object(audit, 'ROOT', root), patch.object(audit, 'BASE', base), \
                 patch.object(audit, 'completed', return_value=accounting):
                native, _, proof = audit.ea_evidence(ea, pub)
                self.assertEqual(proof['accounting']['job'], '800000')
                path = ea.DIRECTORY/'final-consumer-binding.json'
                value = json.loads(path.read_text()); value['actual_process']['start_ticks'] += 1
                write(path, value)
                with self.assertRaisesRegex(ValueError, 'consumer/goal binding'):
                    audit.ea_evidence(ea, pub)


class AttemptTests(unittest.TestCase):
    def fixture(self, root):
        base = root.parent/'current'; original = root/'source'/audit.PRODUCER
        aid = 'a'*32; folder = base/'artifacts/question-probe/attempts'/aid
        files = {}
        for key, path, data in (
            ('raw_file', folder/'real-rollout.pt', b'actual raw'),
            ('private_trace_file', base/'artifacts/behavior-traces'/('attempt-'+aid+'.json'), b'actual trace'),
            ('gate_file', base/'logs'/('behavior-probability-attempt-'+aid+'.json'), b'actual gate'),
            ('allocation_file', base/'logs/native-allocation-admission.json', b'actual allocation'),
            ('selection_file', folder/'selection.json', b'actual selection'),
            ('rollout_file', folder/'rollout-audit.json', b'actual rollout'),
            ('head_file', base/'artifacts/question-probe/completed'/(aid+'.json'), b'actual commit')):
            files[key] = write(path, data, binary=True)
        (base/'artifacts/behavior-traces').chmod(0o700)
        train = write(base/'data/train.parquet', b'canonical gold', binary=True)
        model = root/'model'; model_shas = {}
        for name in audit.TOKENIZERS:
            model_shas[name] = write(model/name, name.encode(), binary=True)['sha256']
        source_name = 'search_g1/signals.py'
        source = write(original/source_name, b'original scorer', binary=True)
        run = dict(run_id=audit.RUN, source_commit=audit.PRODUCER, initial_model=str(model),
            initial_model_sha256=model_shas, input_sha256={'train.parquet':train['sha256']},
            source_sha256={source_name:source['sha256']})
        files['source_manifest'] = write(base/'logs/run-manifest.json', run)
        attempt = dict(attempt_id=aid, logical_update=6, actual='normal return')
        (base/'logs/actual-attempts.jsonl').write_text(json.dumps(attempt)+'\n')
        task = dict(step=6, attempt_id=aid, complete_outer=True, record_sha256=audit.canonical_sha(attempt),
            producer=dict(base=str(base), producer_commit=audit.PRODUCER, producer_job='719225', run_id=audit.RUN,
                source_manifest_sha256=files['source_manifest']['sha256'],
                native_allocation_sha256=files['allocation_file']['sha256']))
        output = {k:files[k] for k in ('raw_file','private_trace_file','gate_file','allocation_file','source_manifest')}
        output['canonical_proof'] = dict(train_sha256=train['sha256'], tokenizer_sha256=model_shas,
            **{k:files[k] for k in ('selection_file','rollout_file','head_file')})
        output['actual_attempt_transaction'] = dict(attempt_record_sha256=task['record_sha256'])
        return base, task, output

    def test_actual_original_readers_accept_ledger_append_and_reject_raw_or_private_changes(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(audit, 'OWNER', os.getuid()):
            root = Path(folder).resolve()/'control'; base, task, output = self.fixture(root)
            with patch.object(audit, 'ROOT', root), patch.object(audit, 'BASE', base):
                proof = audit.fresh_attempt(task, output)
                self.assertEqual(proof['step'], 6)
                with (base/'logs/actual-attempts.jsonl').open('a') as ledger:
                    ledger.write(json.dumps(dict(attempt_id='b'*32, logical_update=7))+'\n')
                audit.fresh_attempt(task, output)
                for key in ('raw_file', 'private_trace_file'):
                    path = Path(output[key]['path']); original = path.read_bytes()
                    path.write_bytes(b'x'*len(original))
                    with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'full SHA'):
                        audit.fresh_attempt(task, output)
                    path.write_bytes(original)

    def test_original_path_relabel_rejected(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(audit, 'OWNER', os.getuid()):
            root = Path(folder).resolve()/'control'; base, task, output = self.fixture(root)
            output['raw_file'] = dict(output['raw_file'], path=str(base/'different.pt'))
            with patch.object(audit, 'ROOT', root), patch.object(audit, 'BASE', base), \
                 self.assertRaisesRegex(ValueError, 'path was relabeled'):
                audit.fresh_attempt(task, output)


class TeacherTests(unittest.TestCase):
    def fixture(self, root, repository):
        old_base = root.parent/'old'; cache = root/'teacher'; model = root/'model'
        questions = [dict(qid='q'+str(i), question_hash=sha(str(i).encode())) for i in range(5120)]
        model_shas = {name:write(model/name, name.encode(), binary=True)['sha256'] for name in audit.TOKENIZERS}
        source_name = 'search_g1/signals.py'
        source_sha = write(root/'source'/audit.OLD_PRODUCER/source_name, b'original scorer', binary=True)['sha256']
        train_sha = write(old_base/'data/train.parquet', b'original canonical gold', binary=True)['sha256']
        run = dict(question_probe=dict(teacher_warmup_cache=str(cache), seed=17), model_revision=audit.OFFICIAL_MODEL_REVISION,
            source_commit=audit.OLD_PRODUCER, initial_model=str(model), initial_model_sha256=model_shas,
            input_sha256={'train.parquet':train_sha}, source_sha256={source_name:source_sha})
        identity = dict(questions=questions, training_world_size=4, seed=17, model_revision=run['model_revision'],
            model_files_sha256=model_shas, train_parquet_sha256=train_sha,
            scorer_sources={source_name:source_sha}, rollout_sources={source_name:source_sha})
        identity_sha = write(cache/'identity.json', identity)['sha256']
        workers = [dict(rank=i, parameter_sha256='p', optimizer_sha256='o', scheduler_sha256='s',
                        actual_optimizer_writes=0, completed_outer=0) for i in range(4)]
        measured = []; chunks = []; all_counts = {str(k):0 for k in range(6)}
        from search_g1.qprobe_warmup import moments
        for index in range(10):
            teacher_run = '0742d8e346284d6da9656a826ce14395' if index < 2 else audit.RUN
            groups = []; counts = {str(k):0 for k in range(6)}
            for offset, question in enumerate(questions[index*512:(index+1)*512]):
                k = offset % 6; vector = [1]*k+[0]*(5-k); count, mean, std = moments(vector)
                groups.append(dict(question, run_id=teacher_run, source_outer=0, strict_em=vector,
                    correct_count=count, mean=mean, std=std,
                    trajectory_ids=['%d-%d-%d'%(index,offset,s) for s in range(5)]))
                counts[str(k)] += 1
            raw = write(cache/('raw-%02d.pt'%index), ('raw'+str(index)).encode(), binary=True)
            chunk = dict(index=index, groups=groups,
                raw_rollout=dict(relative_path=Path(raw['path']).name, sha256=raw['sha256'], bytes=raw['bytes']),
                frozen_workers=dict(before=workers, after=copy.deepcopy(workers)))
            chunk_sha = write(cache/('chunk-%02d.json'%index), chunk)['sha256']
            chunks.append(dict(relative_path='chunk-%02d.json'%index, sha256=chunk_sha))
            measured.append(dict(chunk=index, uid=audit.OWNER, actual_cpu_only=True, cache_writer_constructed=False,
                cache_lock_taken=False, model_weights_loaded=False, generated_trajectories=0,
                actor_or_head_updates_performed=0, questions=512, trajectories=2560, identity_sha256=identity_sha,
                canonical_train_sha256=train_sha, canonical_all5120_seed_order_qid_text_rebuilt_equal=True,
                actual_all512_raw_strict_terminal_EM_K_mean_std_ddof1_equal=True,
                actual_frozen_four_rank_and_raw_provenance_equal=True, chunk_sha256=chunk_sha,
                raw_sha256=raw['sha256'], raw_bytes=raw['bytes'], teacher_run_id=teacher_run,
                K_distribution=counts, strict_correct=sum(k*counts[str(k)] for k in range(6))))
            for k, value in counts.items(): all_counts[k] += value
        write(cache/'complete.json', dict(identity_sha256=identity_sha, questions=5120, trajectories=25600, chunks=chunks))
        baseline = dict(teacher_all10_actual_raw_redecode=measured, teacher_cross_chunk_uniqueness=dict(
            questions=5120, trajectories=25600, actual_all_teacher_global_qid_trajectory_uniqueness=True,
            K_distribution=all_counts))
        write(repository/audit.BASELINE, baseline)
        return old_base, cache, run, audit.canonical_sha(baseline)

    def test_full_synthetic_ten_chunk_baseline_and_current_raw_binding(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(audit, 'OWNER', os.getuid()):
            root = Path(folder).resolve()/'control'; repository = root/'source'/'future'
            old_base, cache, run, payload_sha = self.fixture(root, repository)
            with patch.object(audit, 'ROOT', root), patch.object(audit, 'OLD_BASE', old_base), \
                 patch.object(audit, 'BASELINE_PAYLOAD_SHA', payload_sha):
                proof = audit.teacher_evidence(repository, run)
                self.assertEqual((proof['questions'], proof['trajectories']), (5120, 25600))
                self.assertGreater(proof['K_distribution']['0'], 0); self.assertGreater(proof['K_distribution']['5'], 0)
                raw = cache/'raw-00.pt'; raw.write_bytes(b'edit')
                with self.assertRaisesRegex(ValueError, 'full SHA'):
                    audit.teacher_evidence(repository, run)

    def test_self_reported_replacement_baseline_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(audit, 'OWNER', os.getuid()):
            repository = Path(folder).resolve()
            write(repository/audit.BASELINE, {k:[] for k in audit.BASELINE_FIELDS})
            with self.assertRaisesRegex(ValueError, 'baseline changed'):
                audit.teacher_evidence(repository, {})


if __name__ == '__main__':
    unittest.main()
