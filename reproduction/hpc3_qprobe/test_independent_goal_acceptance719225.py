"""Real file consumers and negative terminal gates; no model/remote/job calls."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback
from contextlib import ExitStack
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
    def test_final_full_sha_rejects_equal_metadata_same_size_rewrite(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(audit, 'OWNER', os.getuid()):
            path = Path(folder).resolve()/'train'; record = write(path, b'data', binary=True)
            # Model indistinguishable remote metadata without sleeping or
            # weakening the expected content assertion.
            with patch.object(audit, 'signature', return_value=(1, 2, 4, 5, 6)):
                reader = audit.SharedReads(); reader.read(path, record['sha256'])
                path.write_bytes(b'edit')
                with self.assertRaisesRegex(ValueError, 'changed during final full SHA'):
                    reader.finish()

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


class SyntheticFinalProtocol:
    """Small original-byte fixtures, never evidence of actual training/CUDA.

    Run the real independent verify, goal/evaluation/transport consumers,
    teacher baseline reader, all201 raw child/file consumers and native two
    boundary/prefix readers together. Mock only unavailable publication/controller
    handles, Slurm/process births, the prior frozen-head/PPO-journal CPU proof,
    and decoding/model state already measured outside this pure final reader.
    The toy raw bytes and synthetic native reports do not claim a real decode,
    optimizer write, model reload or successful experiment.
    """
    COMMIT = 'f'*40
    STAGE1 = 'b'*40
    STAGE3 = 'e'*40

    def __init__(self, folder, route):
        from reproduction.hpc3_qprobe import (final_control as ctl, final_restore_audit as native_audit,
            raw_audit_followthrough719225 as raw, raw_terminal_reaggregation719225 as recon,
            verify_goal as goal)
        from reproduction.common8 import validation_identity
        from search_g1 import qprobe_checkpoint
        self.ctl, self.native_audit, self.raw, self.recon = ctl, native_audit, raw, recon
        self.goal, self.validation_identity, self.checkpoint = goal, validation_identity, qprobe_checkpoint
        self.route = route
        self.root = Path(folder).resolve()/'control'
        self.repository = self.root/'source'/self.COMMIT
        with patch.object(audit,'OWNER',os.getuid()):
            self.base, self.ea, self.ea_source, self.native_accounting = EATerminalTests().fixture(self.root)
            self.old, self.teacher_cache, self.old_run, self.baseline_sha = TeacherTests().fixture(self.root, self.repository)
        self.native_accounting['raw'] = '800000|COMPLETED|0:0|jxu957|14090|jxu957|formal-user|emergency_acd\n'
        self.own_file = self.repository/'reproduction/hpc3_qprobe/independent_goal_acceptance719225.py'
        write(self.own_file, Path(audit.__file__).read_bytes(), binary=True)
        self.raw_file = self.repository/'reproduction/hpc3_qprobe/raw_terminal_reaggregation719225.py'
        write(self.raw_file, Path(recon.__file__).read_bytes(), binary=True)
        self.raw_directory = self.root/'observations/raw-all200-719225-20261015'
        self.process = dict(pid=10000, sid=10000, start_ticks=101, uid=os.getuid())
        self.raw_source = dict(commit=audit.RAW, publication_path='/synthetic/raw-publication', publication_sha256='c'*64)
        self.stage3_source = dict(commit=self.STAGE3, publication_path='/synthetic/stage3-publication', publication_sha256='d'*64)
        self.feature_root = self.root/'features'
        features = write(self.feature_root/'features.npy', b'synthetic feature bytes', binary=True)
        index = write(self.feature_root/'index.jsonl', b'synthetic feature index\n', binary=True)
        write(self.feature_root/'manifest.json', dict(spec_sha='feature-spec', count=169028,
            spec={'feature_dim':1}, files={'features.npy':features, 'index.jsonl':index}))
        self.old_run.update(run_id=audit.RUN, training_world_size=4, actor_micro_batch_per_rank=4,
            budget={'common_manifest_sha256':'synthetic-common'},
            question_probe=dict(self.old_run['question_probe'], cache_dir=str(self.feature_root)))
        source_name = next(iter(self.old_run['source_sha256']))
        write(self.root/'source'/audit.PRODUCER/source_name, b'original scorer', binary=True)
        self.run = copy.deepcopy(self.old_run)
        self.run.update(source_commit=audit.PRODUCER,
            full_state_restore=dict(kind='question_probe_saved_checkpoint', completed_outer=5))
        self.manifests = {}
        for base, run in ((self.old, self.old_run), (self.base, self.run)):
            self.manifests[base] = write(base/'logs/run-manifest.json', run)
            write(base/'logs/native-allocation-admission.json', {'synthetic_original_allocation':str(base)})
            write(base/'data/train.parquet', b'original canonical gold', binary=True)
            (base/'artifacts/behavior-traces').mkdir(parents=True, mode=0o700)
        self.parts = [dict(base=self.old, run=self.old_run, first=1, last=5),
                      dict(base=self.base, run=self.run, first=6, last=200)]
        self.bindings = {}
        for part, job in zip(self.parts, ('718650','719225')):
            base = part['base']; self.bindings[base] = dict(base=str(base), run_id=audit.RUN,
                producer_commit=part['run']['source_commit'], producer_job=job,
                source_manifest_sha256=self.manifests[base]['sha256'],
                native_allocation_sha256=ctl.digest(base/'logs/native-allocation-admission.json'))
        self.raw_parts = [dict(first=p['first'], last=p['last'], producer=self.bindings[p['base']]) for p in self.parts]
        self._attempts()
        self._evaluations()
        self._archives()
        self._native(features)
        self._cpu_route()
        publication_files = {
            str(p.relative_to(self.repository)):sha(p.read_bytes()) for p in self.repository.rglob('*') if p.is_file()}
        write(self.root/'preparation'/('source-'+self.COMMIT+'-verification.json'),
              dict(commit=self.COMMIT, status='PUBLISHED_SOURCE_FULL_SHA_PASS', files=publication_files))

    @staticmethod
    def lines(path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row)+'\n' for row in values))

    def _attempts(self):
        # One synthetic normal-return guard at logical6 must survive all consumers.
        ledgers = {self.old:[], self.base:[]}; updates = {self.old:[], self.base:[]}
        completed = 0
        for index in range(1, 202):
            complete = index != 6; step = completed+1
            base = self.old if step <= 5 else self.base
            row = dict(run_id=audit.RUN, slurm_job_id=self.bindings[base]['producer_job'],
                attempt_id='%032x'%index, attempt_index=index, logical_update=step,
                optimizer_writes_this_attempt=20 if complete else 0, complete_outer=complete,
                opportunities_processed=20, scheduler_advances=int(complete),
                completed_outer=completed+int(complete), status='complete' if complete else 'recoverable_guard')
            ledgers[base].append(row)
            if complete:
                updates[base].append(dict(step=step, run_id=audit.RUN, attempt_id=row['attempt_id']))
            completed += int(complete)
        for base in ledgers:
            self.lines(base/'logs/actual-attempts.jsonl', ledgers[base])
            self.lines(base/'logs/actual-updates.jsonl', updates[base])
        self.attempts = ledgers[self.old]+ledgers[self.base]
        self.tasks = self.raw.tasks_from_rows(self.raw_parts, [ledgers[self.old], ledgers[self.base]])
        self.raw_claim = write(self.raw_directory/'claim.json', dict(source=self.raw_source, parts=self.raw_parts,
            command=[str(self.recon.ORIGINAL_RAW_EXECUTABLE),'-m',self.raw.MODULE,'--commit',audit.RAW,'--worker'],
            training_control_permitted=False, goal_complete=False, created_unix=1.))
        write(self.raw_directory/'process.json', self.process)
        self.records = []
        for task in self.tasks:
            base = Path(task['producer']['base']); run = self.old_run if base == self.old else self.run
            aid = task['attempt_id']; directory = base/'artifacts/question-probe/attempts'/aid
            canonical = dict(train_sha256=run['input_sha256']['train.parquet'],
                tokenizer_sha256=run['initial_model_sha256'], actual_head_committed=task['complete_outer'],
                actual_raw_decode_strict_terminal_em_equal=True, actual_512_qid_text_gold_equal=True,
                actual_selection_G5_E64_top448_prequential_head_equal=True, questions=512, trajectories=2560)
            output = dict(status='ACTUAL_RAW_CANONICAL_TRACE_ALLTOKENS_AND_ACTOR_ROWS_EQUAL', step=task['step'],
                job=task['producer']['producer_job'], source_commit=task['producer']['producer_commit'],
                source_manifest=self.manifests[base], allocation_file=dict(path=str(base/'logs/native-allocation-admission.json'),
                    sha256=self.ctl.digest(base/'logs/native-allocation-admission.json'),
                    bytes=(base/'logs/native-allocation-admission.json').stat().st_size), canonical_proof=canonical,
                actual_attempt_transaction=dict(attempt_id=aid, attempt_index=task['attempt_index'],
                    complete_outer=task['complete_outer'], actual_optimizer_writes=task['actual_optimizer_writes'],
                    attempt_record_sha256=task['record_sha256']), included_actor_token_count=3,
                hf_forward_performed=False, model_weights_loaded=False, generated_trajectories=0,
                actor_or_head_updates_performed=0, evidence_files_written=0)
            for name in ('actual_private_trace_byte_equal','actual_all2560_ordered_ids_equal','actual_512_parent_digests_equal',
                         'actual_2560_actor_tensor_rows_equal','actual_all_retained_token_contexts_equal',
                         'actual_selected_logprob_projection_equal'):
                output[name] = True
            for name, path in (
                ('raw_file',directory/'real-rollout.pt'),
                ('private_trace_file',base/'artifacts/behavior-traces'/('attempt-'+aid+'.json')),
                ('gate_file',base/'logs'/('behavior-probability-attempt-'+aid+'.json'))):
                output[name] = write(path, (name+aid).encode(), binary=True)
            for name, path in (('selection_file',directory/'selection.json'), ('rollout_file',directory/'rollout-audit.json'),
                ('head_file',base/'artifacts/question-probe'/('completed' if task['complete_outer'] else 'aborted')/(aid+'.json'))):
                canonical[name] = write(path, {'synthetic_attempt':aid})
            retained = self.raw_directory/'attempts'/aid
            child = dict(pid=20000+task['attempt_index'], sid=self.process['sid'], start_ticks=200+task['attempt_index'], uid=os.getuid())
            claim = write(retained/'claim.json', dict(task=task, source=self.raw_source, process=self.process, created_unix=2.))
            write(retained/'child.json',child)
            stdout = write(retained/'audit.stdout',output); stderr = write(retained/'audit.stderr', b'',binary=True)
            result = write(retained/'result.json', dict(task=task, source=self.raw_source, process=self.process,
                child=child, actual_child_exit_code=0, claim_sha256=claim['sha256'], started_unix=3., completed_unix=4.,
                stdout=stdout, stderr=stderr))
            record = dict(task=task, result_path=result['path'], result_sha256=result['sha256'], output_sha256=stdout['sha256'],
                raw_sha256=output['raw_file']['sha256'], private_trace_sha256=output['private_trace_file']['sha256'],
                included_actor_token_count=output['included_actor_token_count'])
            self.records.append(record); write(self.raw_directory/'accepted'/('%06d.json'%task['attempt_index']),record)
        self.raw_result = dict(status=self.raw.STATUS, source=self.raw_source, process=self.process,
            claim_sha256=self.raw_claim['sha256'], normal_return_attempts=201, successful_formal_outers=list(range(1,201)),
            records=self.records, record_list_sha256=audit.canonical_sha(self.records), goal_complete=False,
            started_unix=2., completed_unix=5.)
        if self.route == 'typed_failed_raw':
            self.raw_result = dict(status=self.recon.FAILED_STATUS, source=self.raw_source, process=self.process,
                claim_sha256=self.raw_claim['sha256'], completed_attempt_audits=201, error_type='FileNotFoundError',
                error='[Errno 2] No such file or directory', goal_complete=False, started_unix=2., completed_unix=5.)
        write(self.raw_directory/'result.json',self.raw_result)
        write(self.raw_directory/'state.json',dict(process=self.process))
        write(self.raw_directory/'supervisor.log', (json.dumps(self.raw_result)+'\n').encode(),binary=True)

    def _evaluations(self):
        self.canonical = {}; ledgers = {self.old:[], self.base:[]}
        def make(split, step, counts, base):
            sources = [source for source,count in counts.items() for _ in range(count)]
            ids = [split+str(i) for i in range(len(sources))]
            record = dict(sha256=sha(split.encode()), source_counts=counts)
            self.canonical[(base,split)] = (record, {'data_source':SimpleNamespace(tolist=lambda:sources)}, ids)
            em = [int(i%3 == 0) for i in range(len(ids))]; f1 = [1. if x else .25 for x in em]
            return dict(run_id=audit.RUN, split=split, step=step, final=split=='final', question_ids=ids,
                data_sources=sources, em=em, f1=f1, counts=counts, correct=sum(em), weighted_em=sum(em)/len(em),
                weighted_f1=math.fsum(f1)/len(f1), validation_sha256=record['sha256'])
        for step in (5,50,100,150,200):
            base = self.old if step == 5 else self.base
            ledgers[base].append(make('validation',step,{'nq':256,'hotpotqa':256},base))
        for base, rows in ledgers.items(): self.lines(base/'logs/validation-identities.jsonl',rows)
        counts = dict(nq=3610,triviaqa=11313,popqa=14267,hotpotqa=7405,**{'2wikimultihopqa':12576,'musique':2417,'bamboogle':125})
        final = make('final',200,counts,self.base); write(self.base/'logs/final-identity.json',final)
        source_em = {}; source_f1 = {}
        for source,count in counts.items():
            indices = [i for i,name in enumerate(final['data_sources']) if name == source]
            source_em[source] = sum(final['em'][i] for i in indices)/count
            source_f1[source] = math.fsum(final['f1'][i] for i in indices)/count
        self.full = dict(structural_accepted=True, actual_outer_updates=200, run_id=audit.RUN,
            final_source_em=source_em, final_source_f1=source_f1, final_counts=counts,
            final_weighted_em=final['weighted_em'], final_weighted_f1=final['weighted_f1'])
        write(self.base/'logs/full-run-acceptance.json',self.full)
        write(self.base/'logs/pipeline-status.json',dict(run_id=audit.RUN,
            stage='FINISHED_FULL200_FINAL51713_AND_OWNED_CLEANUP' if self.route == 'normal' else 'FAILED'))
        write(self.base/'logs/cleanup-status.json',dict(run_id=audit.RUN,cleaned=True))

    def _archives(self):
        self.transports = {}; self.target_receipt = None
        for ordinal, step in enumerate((5,50,100,150,200),1):
            directory = self.root/'archives'/audit.RUN/('global_step_'+str(step))
            sources = [write(self.base/'synthetic-checkpoints'/str(step)/('state-'+str(i)),
                             ('synthetic state '+str(step)+':'+str(i)).encode(),binary=True) for i in range(ordinal)]
            sources.append(self.manifests[self.base]); receipts = {}; paths = {}; records = {}
            for role in ('source','archive','target'):
                root = self.base if role == 'source' else directory/role
                rows = []
                for source in sources:
                    path = Path(source['path']) if role == 'source' else root/Path(source['path']).relative_to(self.base)
                    if role != 'source':write(path,Path(source['path']).read_bytes(),binary=True)
                    st = path.stat();rows.append(dict(path=str(path),inode=st.st_ino,bytes=st.st_size,
                        mtime_ns=st.st_mtime_ns,full_read_sha256=source['sha256']))
                progress = dict(completed_outer=step,cumulative_writes=step*20,attempt_index=step+int(step>=6),
                    aborted_attempts=int(step>=6))
                receipt = dict(schema_version=1,kind='question_probe_pair',run_id=audit.RUN,step=step,world_size=4,
                    source_base=str(root),files=rows,source_run_manifest=rows[-1],boundary_driver=rows[0],
                    boundary_worker_files=[],reward_probe_manifest=rows[0],reward_files=[],historical_artifacts={},
                    attempt_progress=progress,driver_state=dict(source_consumed=1024000),hf_saved_tensor_count=435,
                    ranks=[dict(rank=i,used_attempt_ids=[t['attempt_id'] for t in self.tasks],policy_rejections=0) for i in range(4)])
                path = self.base/'artifacts/question-probe/completed-pairs'/('global_step_'+str(step))/'source-receipt.json' if role=='source' else root/'relocated-source-receipt.json'
                paths[role] = write(path,receipt); receipts[role] = receipt; records[role] = rows
            files = [dict(source_path=s['path'],archive_path=a['path'],target_path=t['path'],bytes=s['bytes'],sha256=s['full_read_sha256'])
                     for s,a,t in zip(records['source'],records['archive'],records['target'])]
            transport = dict(schema_version=1,status='QPROBE_PAIR_SOURCE_ARCHIVE_TARGET_FULL_SHA_PASS',run_id=audit.RUN,
                step=step,world_size=4,source_manifest_sha256=self.manifests[self.base]['sha256'],files=files,
                **{role+'_receipt':{key:row[key] for key in ('path','sha256')} for role,row in paths.items()})
            for role in ('archive','target'):
                transport[role+'_file_set'] = sorted([r['path'] for r in records[role]]+[paths[role]['path']])
            write(directory/'complete.json',transport);self.transports[step]=directory/'complete.json'
            if step == 200:self.target_receipt = receipts['target'];self.target_path = Path(paths['target']['path'])

    def _native(self, features):
        self.audit_base = self.root.parent/'native'
        uuids = ['synthetic-GPU-'+str(i) for i in range(4)]
        allocation = dict(accepted=True,training_world_size=4,run_id=audit.RUN,slurm_job_id='800000',actual_cuda_runtime_uuids=uuids)
        write(self.audit_base/'logs/native-allocation-admission.json',allocation)
        write(self.audit_base/'logs/run-manifest.json',dict(run_id=audit.RUN,source_commit=audit.EA,
            final_restore_audit={'audit_id':'audit'},full_state_restore={'source_receipt':str(self.target_path)}))
        workers = [dict(rank=i,phase='first',slurm_job_id='800000',actual_cuda_uuid=uuids[i],
            process_identity=dict(pid=30000+i,start_ticks=300+i,uid=os.getuid(),sid=30000),
            counters=dict(completed_outer=200,actual_optimizer_writes=4000),actual_adam_scheduler_readback_equal=True,
            actual_rng_readback_equal=True,actual_optimizer_writes=4000,scheduler_last_epoch=200,last_attempt_index=201,
            model_hf_exact_tensors_checked_on_rank0=435,model_shard_sha256='model',optimizer_sha256='adam',
            scheduler_sha256='scheduler',rng_sha256='rng') for i in range(4)]
        second = copy.deepcopy(workers)
        for row in second:row['phase']='second'
        driver = dict(phase='first',completed_outer=200,head_update_count=200,head_scheduler=None,head_schedule='constant_lr_Adam',
            actual_head_model_adam_private_rng_replay_readback_equal=True,actual_selector_rng_readback_equal=True,
            actual_candidate_stream_readback_equal=True,actual_driver_rng_readback_equal=True)
        driver2 = dict(driver,phase='second')
        write(self.audit_base/'logs/full-state-restored-workers.json',dict(ranks=workers))
        write(self.audit_base/'logs/full-state-restored-driver.json',dict(actual_driver_rng_readback_equal=True,
            completed_outer=200,actual_candidate_source_cursor=1024000))
        prefix = dict(completed_outer=200,writes=4000,attempt_index=201,rejections=0,aborted=1,
            used_attempt_ids=[t['attempt_id'] for t in self.tasks])
        native_path = self.audit_base/'logs/final-restore-readback.json'
        native = json.loads(native_path.read_text())
        native.update(audit_commit=audit.EA,actual_loaded_target_receipt=str(self.target_path),workers_first=workers,
            workers_second=second,driver_first=driver,driver_second=driver2,head_scheduler=None,
            actual_worker_restore_prefix=prefix,source_receipt_sha256=self.ctl.digest(self.target_path),
            actual_feature_cache=dict(spec_sha='feature-spec',count=169028,feature_dim=1,feature_file_sha256=features['sha256']))
        native_file = write(native_path,native)
        self.accepted = write(self.audit_base/'logs/final-restore-audit-accepted.json',dict(accepted=True,
            actual_cuda_restore_accepted=True,owned_cleanup=True,readback_sha256=native_file['sha256'],audit_id='audit',
            source_receipt_sha256=self.ctl.digest(self.target_path),transport_proof_sha256=self.ctl.digest(self.transports[200])))
        goal_path = self.base/'logs/qprobe-final-goal-acceptance.json'
        value = json.loads(goal_path.read_text());value['actual_native_restore_accepted_sha256']=self.accepted['sha256']
        goal_file = write(goal_path,value)
        binding_path = self.ea.DIRECTORY/'final-consumer-binding.json'
        binding = json.loads(binding_path.read_text());binding.update(goal_sha256=goal_file['sha256'],
            actual_native_audit_accounting=self.native_accounting);write(binding_path,binding)
        ea_result = json.loads((self.ea.DIRECTORY/'result.json').read_text());ea_result['result']['final_goal_sha256']=goal_file['sha256']
        write(self.ea.DIRECTORY/'result.json',ea_result)
        self.ea.inspect = lambda:dict(actual_live_birth=False,process=json.loads((self.ea.DIRECTORY/'process.json').read_text()),
            result=json.loads((self.ea.DIRECTORY/'result.json').read_text()))

    def _cpu_route(self):
        self.reference = None;self.helper = None
        if self.route == 'normal':return
        self.reference = dict(commit=self.STAGE1,original_producer_state='FAILED',original_producer_exit_code='1:0')
        for name in ('result','output','structural_acceptance'):
            self.reference[name] = write(self.root/'structural'/name,self.full if name=='structural_acceptance' else {'synthetic_typed_boundary':True})
        original = write(self.ea.DIRECTORY/'result.json',dict(exit_code=1,error='synthetic preserved original EA failure'))
        goal_path = self.base/'logs/qprobe-final-goal-acceptance.json';goal=json.loads(goal_path.read_text());goal_path.unlink()
        goal.update(producer_final_cpu_recovery=self.reference,original_producer_accounting_remains_failed=True)
        directory = self.root/'followthrough/recovery719225-finalcpu-20261015'
        goal_file = write(directory/'goal-acceptance.json',goal)
        parent = dict(pid=40000,sid=40000,start_ticks=400,uid=os.getuid());child=dict(parent,pid=40001,start_ticks=401)
        claim = write(directory/'claim.json',dict(source=self.stage3_source,recovery_commit=self.STAGE1,
            admission={'producer_final_cpu_recovery':self.reference}))
        write(directory/'process.json',parent);write(directory/'child.json',child)
        result = write(directory/'result.json',dict(process=parent,child=child,source=self.stage3_source,
            claim_sha256=claim['sha256'],actual_child_exit_code=0,supervisor_exit_code=0))
        binding = write(directory/'final-consumer-binding.json',dict(actual_process=child,helper_commit=self.STAGE3,
            recovery_commit=self.STAGE1,full_cpu_consumer_returned=True,goal_sha256=goal_file['sha256'],
            audit_base=str(self.audit_base),actual_native_audit_accounting=self.native_accounting))
        output = write(directory/'consumer.stdout',dict(status='synthetic prior held CPU proof',goal_complete=False))
        held = dict(status='SYNTHETIC_PRIOR_HELD_BOUNDARY',source=self.stage3_source,actual_child_exit_code=0,
            producer_final_cpu_recovery=self.reference,original_accounting_remains_failed=True,goal_complete=False,
            full_objective_accepted=False,independent_teacher_raw_final_consumer_still_required=True,
            result_path=result['path'],result_sha256=result['sha256'],goal_acceptance=goal_file,consumer_binding=binding,
            audit_base=str(self.audit_base),output=output,original_ea=dict(fingerprint='synthetic-original-failure',files={'result.json':original}))
        self.helper = SimpleNamespace(DIRECTORY=directory,HELD_STATUS=held['status'],consume=Mock(return_value=held))

    def contexts(self):
        stack = ExitStack()
        for module,name,value in ((audit,'ROOT',self.root),(audit,'BASE',self.base),(audit,'OLD_BASE',self.old),
            (audit,'OWNER',os.getuid()),(audit,'BASELINE_PAYLOAD_SHA',self.baseline_sha),(audit,'__file__',str(self.own_file)),
            (self.native_audit,'OWNER',os.getuid()),(self.ctl,'OWNER',os.getuid()),(self.ctl,'ROOT',self.root),
            (self.raw,'BASE',self.base),(self.raw,'DIRECTORY',self.raw_directory),(self.recon,'BASE',self.base),
            (self.recon,'ROOT',self.root),(self.recon,'ORIGINAL',self.raw_directory),(self.recon,'__file__',str(self.raw_file))):
            stack.enter_context(patch.object(module,name,value))
        stack.enter_context(patch.dict(os.environ,dict(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
            QPROBE_SOURCE=str(self.repository),**{k:'1' for k in audit.CAPS})))
        stack.enter_context(patch.object(sys,'dont_write_bytecode',True))
        def pinned(commit,filename,attribute):
            if commit == audit.EA:return self.ea,self.ea_source
            if commit == audit.RAW:return self.raw,self.raw_source
            if commit == self.STAGE3:return self.helper,self.stage3_source
            raise AssertionError('unexpected pinned controller '+commit)
        stack.enter_context(patch.object(audit,'load_fixed',side_effect=pinned))
        stack.enter_context(patch.object(self.raw,'validate',return_value=(self.repository,self.raw_source)))
        stack.enter_context(patch.object(self.ctl,'live',return_value=False))
        def producer(base,complete=False):
            base=Path(base);run=self.old_run if base==self.old else self.run
            return run,self.bindings[base],'synthetic original accounting'
        stack.enter_context(patch.object(self.ctl,'producer',side_effect=producer))
        stack.enter_context(patch.object(self.ctl,'final_producer',return_value=(self.run,self.bindings[self.base],
            '719225|FAILED|1:0|jxu957|14090|jxu957|formal-user|emergency_acd\n',self.reference)))
        def accounting(command,**kwargs):
            job=command[command.index('-j')+1]
            if any(value in ('JobID,State,ExitCode,UID','--format=JobID,State,ExitCode,UID') for value in command):
                return job+'|COMPLETED|0:0|14090\n'
            return job+'|COMPLETED|0:0|jxu957|14090|jxu957|formal-user|emergency_acd\n'
        stack.enter_context(patch.object(audit.subprocess,'check_output',side_effect=accounting))
        # Prior external CPU/GPU receipts, not local optimizer/model execution.
        stack.enter_context(patch.object(self.goal,'history_parts',return_value=self.parts))
        stack.enter_context(patch.object(self.goal,'verify_attempt_parts',return_value=(self.attempts,[],[])))
        stack.enter_context(patch.object(self.goal,'verify_teacher',return_value=dict(questions=5120,trajectories=25600)))
        stack.enter_context(patch.object(self.goal,'os',SimpleNamespace(getuid=lambda:14090)))
        remote_archives='/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/archives'
        stack.enter_context(patch.object(self.goal,'Path',side_effect=lambda value: self.root/'archives' if str(value)==remote_archives else Path(value)))
        stack.enter_context(patch.object(self.validation_identity,'expected',side_effect=lambda base,split:self.canonical[(Path(base),split)]))
        stack.enter_context(patch.object(self.checkpoint,'read_source',return_value=(self.target_path,self.target_receipt,
            self.target_path.parent,self.target_path.parent,self.run)))
        return stack

    def verify(self):
        options = {} if self.route == 'normal' else dict(recovery_commit=self.STAGE1,recovery_followthrough_commit=self.STAGE3)
        return audit.verify(self.COMMIT,**options)


class SyntheticFinalProtocolTests(unittest.TestCase):
    """Compatibility only: these PASS results never accept actual200 or CUDA."""
    def positive(self,route):
        with tempfile.TemporaryDirectory() as folder:
            fixture=SyntheticFinalProtocol(folder,route)
            with fixture.contexts():
                before={str(p):p.read_bytes() for p in Path(folder).rglob('*') if p.is_file()}
                value=fixture.verify()
                self.assertEqual(value['status'],audit.STATUS)
                self.assertEqual(value['fresh_full_goal']['actual_attempts'],201)
                self.assertEqual(value['independent_all_normal_return_raw']['fully_skipped_attempts'],1)
                self.assertEqual(value['authentic_teacher']['trajectories'],25600)
                self.assertEqual(set(value['fresh_full_goal']['archives']),{'5','50','100','150','200'})
                self.assertEqual([value['fresh_full_goal']['archives'][str(s)]['files'] for s in (5,50,100,150,200)],[2,3,4,5,6])
                self.assertEqual(value['fresh_full_goal']['final_native_questions'],51713)
                self.assertFalse(value['goal_complete']);self.assertFalse(value['update201_performed'])
                if route=='normal':
                    self.assertIn('actual_EA_full_CPU_consumer',value)
                    self.assertNotIn('producer_final_cpu_recovery',value)
                else:
                    self.assertNotIn('actual_EA_full_CPU_consumer',value)
                    self.assertTrue(value['original_EA_status_remains_failed'])
                    self.assertEqual(value['producer_accounting']['state'],'FAILED')
                if route=='typed_failed_raw':
                    raw=value['independent_all_normal_return_raw']
                    self.assertTrue(raw['original_raw_status_remains_failed'])
                    self.assertFalse(raw['precise_original_aggregation_call_failure_proven'])
                    self.assertEqual(raw['decode_reexecuted'],0)
                self.assertEqual(before,{str(p):p.read_bytes() for p in Path(folder).rglob('*') if p.is_file()})

    def test_synthetic_full_verify_normal_completed0_protocol(self):self.positive('normal')
    def test_synthetic_full_verify_typed_recovery_original_raw_pass_protocol(self):self.positive('typed_raw_pass')
    def test_synthetic_full_verify_typed_recovery_fresh_failed_raw_protocol(self):self.positive('typed_failed_raw')

    def test_full_protocol_rejects_same_size_archive_tamper_and_native_second_readback_drift(self):
        for change in ('archive','native'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
                fixture=SyntheticFinalProtocol(folder,'normal')
                if change=='archive':
                    transport=json.loads(fixture.transports[100].read_text())
                    target=Path(transport['files'][0]['target_path']);original=target.read_bytes()
                    target.write_bytes(bytes([original[0]^1])+original[1:])
                    self.assertEqual(target.stat().st_size,len(original))
                else:
                    path=fixture.audit_base/'logs/final-restore-readback.json';native=json.loads(path.read_text())
                    native['workers_second'][0]['optimizer_sha256']='changed';native_file=write(path,native)
                    accepted=json.loads((fixture.audit_base/'logs/final-restore-audit-accepted.json').read_text())
                    accepted['readback_sha256']=native_file['sha256'];accepted_file=write(fixture.audit_base/'logs/final-restore-audit-accepted.json',accepted)
                    goal_path=fixture.base/'logs/qprobe-final-goal-acceptance.json';goal=json.loads(goal_path.read_text())
                    goal['actual_native_restore_accepted_sha256']=accepted_file['sha256'];goal_file=write(goal_path,goal)
                    binding_path=fixture.ea.DIRECTORY/'final-consumer-binding.json';binding=json.loads(binding_path.read_text())
                    binding['goal_sha256']=goal_file['sha256'];write(binding_path,binding)
                    result_path=fixture.ea.DIRECTORY/'result.json';result=json.loads(result_path.read_text())
                    result['result']['final_goal_sha256']=goal_file['sha256'];write(result_path,result)
                with fixture.contexts():
                    if change=='archive':
                        with self.assertRaisesRegex(ValueError,
                            '^FINAL_RESTORE_AUDIT: fresh source/archive/target full SHA differs$'):
                            fixture.verify()
                    else:
                        # The production equal_tree uses an intentionally bare
                        # assert. Prove its exact call chain, rather than count
                        # any unrelated AssertionError as this negative PASS.
                        try:fixture.verify()
                        except AssertionError as error:
                            self.assertEqual(str(error),'')
                            frames=traceback.extract_tb(error.__traceback__)
                            self.assertEqual(frames[-1].name,'equal_tree')
                            self.assertTrue(frames[-1].filename.endswith('/search_g1/full_state_restore.py'))
                            self.assertIn('verify_two_boundaries',[row.name for row in frames])
                        else:self.fail('the real second Adam readback comparison accepted a drift')


if __name__ == '__main__':
    unittest.main()
