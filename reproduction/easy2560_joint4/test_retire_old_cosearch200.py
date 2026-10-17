"""Tiny real-file destructive fixtures; no SSH, Slurm, Torch or model assets."""
import copy
import ast
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.easy2560_joint4 import retire_old_cosearch200 as op
from reproduction.easy2560_joint4 import recover_cosearch200_source_audit as recovery


class OldCoSearchRetirementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve(); self.patches = ExitStack()
        self.base = self.root/'source'; self.cp = self.base/'checkpoints/global_step_200'
        self.archive = self.root/'hk/checkpoints/global_step_200'; self.uid = os.getuid(); self.operation = 'a'*32
        for name, value in dict(BASE=self.base, CP=self.cp, ARCHIVE=self.archive, SOURCE_UID=self.uid,
             OTHER_UID=self.uid+1, HK_UID=self.uid, HPC_AUDIT=self.root/'source-audit', HK_AUDIT=self.root/'hk-audit').items():
            self.patches.enter_context(patch.object(op, name, value))
        self.sroot = op.HPC_AUDIT/self.operation; self.hroot = op.HK_AUDIT/self.operation
        self.sroot.mkdir(parents=True); self.hroot.mkdir(parents=True)
        self.rows = []
        for name in op.CORE_NAMES:
            raw = (json.dumps({'rank_state': name})+'\n').encode() if name in op.RECOVERY_NAMES else ('tiny-tensor-'+name).encode()
            self.put(self.cp/name, raw); self.put(self.archive/name, raw)
            self.rows.append(dict(relative_path=name, source_path=str(self.cp/name), archive_path=str(self.archive/name),
                bytes=len(raw), historical_receipt_sha256=op.digest(raw), source_uid=self.uid,
                source_mtime_ns=(self.cp/name).stat().st_mtime_ns))
        core = [dict(path=r['source_path'], bytes=r['bytes'], sha256=r['historical_receipt_sha256']) for r in self.rows]
        source = dict(status='BOTH_ROLE_FULL_CHECKPOINT_FILES_SHA_PASS', step=200, files=core)
        archive = dict(status='HKUST3_ALL_CHECKPOINT_SHA_PASS', step=200, path=str(self.archive), files=core)
        source_sha = self.put(self.base/'checkpoint-receipt-200.json', source)
        copy_sha = self.put(self.base/'archive-verified-200.json', archive)
        hk_sha = self.put(self.archive/'hkust3-archive-receipt.json', archive)
        for name, value in dict(SOURCE_RECEIPT_SHA=source_sha, ARCHIVE_COPY_SHA=copy_sha, HK_RECEIPT_SHA=hk_sha,
             TOTAL_BYTES=sum(r['bytes'] for r in self.rows),
             TARGET_BYTES=sum(r['bytes'] for r in self.rows if r['relative_path'] in op.TARGET_NAMES)).items():
            self.patches.enter_context(patch.object(op, name, value))
        for name in op.ROOT_METADATA:
            if not (self.base/name).exists(): self.put(self.base/name, dict(live_metadata=name))
        for name in op.CP_METADATA: self.put(self.cp/name, dict(retain_metadata=name))
        self.put(self.cp/'README.txt', b'keep every other small metadata file, even plain text')
        self.candidate = dict(base=str(self.base), checkpoint_step=200, source_job=op.SOURCE_JOB,
            source_job_state='FAILED', exit_code='1:0', full_checkpoint_receipt=dict(files=33, bytes=op.TOTAL_BYTES,
                source_receipt_sha256=source_sha, hk_receipt_copy_sha256=copy_sha, archive_directory=str(self.archive)),
            conservative_tensor_only_candidate=dict(files=25, bytes=op.TARGET_BYTES,
                source_paths=[r for r in self.rows if r['relative_path'] in op.TARGET_NAMES]),
            preserve_small_recovery_proofs=[r for r in self.rows if r['relative_path'] in op.RECOVERY_NAMES])
        self.candidate_path = self.root/'candidate.json'; sha = self.put(self.candidate_path, self.candidate)
        self.patches.enter_context(patch.object(op, 'CANDIDATE_SHA', sha))
        self.patches.enter_context(patch.object(op, 'birth', side_effect=self.birth))
        self.plan = op.load_plan(self.candidate_path)

    def tearDown(self): self.patches.close(); self.temp.cleanup()

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()
        path.write_bytes(raw); return op.digest(raw)

    def birth(self, pid=None):
        return dict(host=op.socket.gethostname(), uid=os.getuid(), pid=os.getpid() if pid is None else pid,
            start_ticks=12345, boot_id='fixture-kernel-boot', pid_namespace='pid:[fixture]')

    @contextmanager
    def site(self, site):
        with patch.object(op.socket, 'gethostname', return_value=op.HK_HOST if site == 'hk' else op.HPC_HOST): yield

    def guard(self, uid):
        actor = self.birth(); actor.update(host=op.HPC_HOST, uid=uid)
        return dict(status='OWN_UID_READER_SCAN', candidate_sha256=op.CANDIDATE_SHA, operation_id=self.operation,
            host=op.HPC_HOST, uid=uid, process_identity=actor, checked_unix=time.time(),
            scanned_own_processes=1, readers=[], inaccessible=[])

    def proofs(self):
        with self.site('hk'): archive = op.validate(self.plan, self.hroot, self.operation, 'hk')
        with self.site('source'), patch.object(op, 'terminal', return_value={'exact_failed_job': op.SOURCE_JOB}):
            source = op.validate(self.plan, self.sroot, self.operation, 'source')
        return source, archive, {str(uid):self.guard(uid) for uid in (op.SOURCE_UID, op.OTHER_UID)}

    def execute(self, source, archive, guards):
        with self.site('source'), patch.object(op, 'terminal', return_value={'exact_failed_job': op.SOURCE_JOB}), \
             patch.object(op, 'readers', return_value=self.guard(op.SOURCE_UID)):
            return op.execute(self.plan, source, archive, guards, self.sroot, self.operation)

    def test_exact_scope25_full33_and_both_fresh_sha_passes_have_real_byte_progress(self):
        self.assertEqual(len(op.TARGET_NAMES), 25); self.assertEqual(len(op.CORE_NAMES), 33)
        self.assertTrue(all(n.endswith('.pt') for n in op.TARGET_NAMES))
        source, archive, _ = self.proofs()
        self.assertEqual(set(source['files']), set(op.CORE_NAMES)); self.assertEqual(source['hashed_bytes'], op.TOTAL_BYTES)
        self.assertEqual({n:r['sha256'] for n,r in source['files'].items()}, {n:r['sha256'] for n,r in archive['files'].items()})
        for root in (self.sroot, self.hroot):
            progress = json.loads((root/'progress.json').read_text())
            self.assertEqual(progress['hashed_bytes'], op.TOTAL_BYTES); self.assertEqual(progress['completed_files'], 33)

    def test_positive_only25_unlinked_all_metadata_and_hk33_remain_and_two_postchecks_finalize(self):
        before = {str(p.relative_to(self.root)):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        source, archive, guards = self.proofs(); expected = sum(source['files'][n]['identity']['allocated_bytes'] for n in op.TARGET_NAMES)
        result = self.execute(source, archive, guards)
        self.assertEqual(result['status'], 'EXACT25_UNLINKED_PENDING_INDEPENDENT_POSTCHECKS')
        self.assertEqual(result['confirmed_unlinked_allocated_bytes'], expected)
        self.assertFalse(result['filesystem_quota_free_bytes_measured'])
        removed = {str((self.cp/n).relative_to(self.root)) for n in op.TARGET_NAMES}
        for rel, raw in before.items():
            p = self.root/rel
            if rel in removed: self.assertFalse(p.exists())
            else: self.assertEqual(p.read_bytes(), raw, rel)
        self.assertFalse((self.base/'archive-delete-safe-200').exists())
        with self.site('source'): first = op.verify_after(self.plan, result, self.operation, 'source')
        with self.site('hk'): second = op.verify_after(self.plan, result, self.operation, 'hk')
        with self.site('source'):
            with self.assertRaisesRegex(ValueError, 'TWO_INDEPENDENT_POSTCHECKS_REQUIRED'):
                op.finalize(result, [first, first], self.sroot, self.operation)
            final = op.finalize(result, [first, second], self.sroot, self.operation)
        self.assertEqual(final['status'], 'AUTHORIZED_EXACT25_HPC_PT_RETIRED_FULL_HK33_RETAINED')
        self.assertTrue((self.sroot/'complete.json').is_file())

    def test_candidate_hash_and_structural_scope_reject_active_or_extra_paths(self):
        self.put(self.candidate_path, dict(self.candidate, base='/active/730226'))
        with self.assertRaisesRegex(ValueError, 'FIXED_CANDIDATE_SHA_REQUIRED'): op.load_plan(self.candidate_path)
        attack = copy.deepcopy(self.candidate)
        attack['conservative_tensor_only_candidate']['source_paths'][0]['source_path'] = '/active/730226/model.pt'
        sha = self.put(self.candidate_path, attack)
        with patch.object(op, 'CANDIDATE_SHA', sha), self.assertRaisesRegex(ValueError, 'EXACT_CORE_PATH_OR_SHA_DIFFERS'):
            op.load_plan(self.candidate_path)
        self.assertTrue(all((self.cp/n).exists() for n in op.TARGET_NAMES))

    def test_target_symlink_to_active_run_rejected_before_sha_or_any_unlink(self):
        outside = self.root/'active730226/model.pt'; self.put(outside, b'active sentinel untouched')
        target = self.cp/op.TARGET_NAMES[0]; target.unlink(); target.symlink_to(outside)
        with self.site('source'), patch.object(op, 'terminal', return_value={}), self.assertRaisesRegex(ValueError, 'SYMLINK'):
            op.validate(self.plan, self.sroot, self.operation, 'source')
        self.assertEqual(outside.read_bytes(), b'active sentinel untouched')
        self.assertTrue(all((self.cp/n).exists() for n in op.TARGET_NAMES[1:]))

    def test_hk_missing_recovery_or_corrupt_tensor_never_produces_full33_proof(self):
        victim = self.archive/op.RECOVERY_NAMES[0]; original = victim.read_bytes(); victim.unlink()
        with self.site('hk'), self.assertRaises(FileNotFoundError): op.validate(self.plan, self.hroot, self.operation, 'hk')
        victim.write_bytes(original); (self.archive/op.TARGET_NAMES[0]).write_bytes(b'wrong backup')
        with self.site('hk'), self.assertRaisesRegex(ValueError, 'FRESH_CORE33_SHA_DIFFERS'):
            op.validate(self.plan, self.hroot, self.operation, 'hk')
        self.assertTrue(all((self.cp/n).exists() for n in op.TARGET_NAMES))

    def test_hardlinked_source_target_is_preserved(self):
        os.link(self.cp/op.TARGET_NAMES[0], self.root/'extra-hardlink')
        with self.site('source'), patch.object(op, 'terminal', return_value={}), self.assertRaisesRegex(ValueError, 'HARDLINK'):
            op.validate(self.plan, self.sroot, self.operation, 'source')
        self.assertEqual((self.cp/op.TARGET_NAMES[0]).stat().st_nlink, 2)

    def test_changed_source_identity_after_sha_blocks_before_first_unlink(self):
        source, archive, guards = self.proofs(); target = self.cp/op.TARGET_NAMES[-1]
        target.write_bytes(target.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'SOURCE_IDENTITY_CHANGED_BEFORE_UNLINK'):
            self.execute(source, archive, guards)
        self.assertFalse((self.sroot/'actions.jsonl').exists()); self.assertTrue(all((self.cp/n).exists() for n in op.TARGET_NAMES))

    def test_same_content_live_receipt_inode_rewrite_and_controller_update_do_not_block(self):
        source, archive, guards = self.proofs(); receipt = self.base/'archive-verified-200.json'; old_inode = receipt.stat().st_ino
        replacement = receipt.with_name('copy.tmp'); replacement.write_bytes(receipt.read_bytes()); replacement.replace(receipt)
        self.assertNotEqual(receipt.stat().st_ino, old_inode)
        self.put(self.base/'controller-state.json', dict(live_writer_epoch=time.time()))
        result = self.execute(source, archive, guards)
        self.assertEqual(len(result['confirmed_unlinked_files']), 25)

    def test_missing_stale_other_uid_or_inaccessible_reader_guard_blocks_before_unlink(self):
        source, archive, original = self.proofs()
        for mutation in ('reader', 'inaccessible', 'stale', 'wrong_uid', 'wrong_namespace'):
            guards = copy.deepcopy(original); g = guards[str(op.OTHER_UID)]
            if mutation == 'reader': g['readers'] = [{'pid':99, 'matches':['target.pt']}]
            elif mutation == 'inaccessible': g['inaccessible'] = [{'pid':99, 'error':'permission'}]
            elif mutation == 'stale': g['checked_unix'] -= 301
            elif mutation == 'wrong_uid': g['process_identity']['uid'] = op.SOURCE_UID
            else: g['process_identity']['pid_namespace'] = 'other namespace'
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'BOTH_UID_READER_PROOFS_REQUIRED'):
                self.execute(source, archive, guards)
        self.assertFalse((self.sroot/'actions.jsonl').exists()); self.assertTrue(all((self.cp/n).exists() for n in op.TARGET_NAMES))

    def test_nth_unlink_error_preserves_prepared_journal_and_claims_only_confirmed_actual_bytes(self):
        source, archive, guards = self.proofs(); original = op.os.unlink; count = 0
        def failing(name, *args, **kwargs):
            nonlocal count
            if kwargs.get('dir_fd') is not None and str(name).endswith('.pt'):
                count += 1
                if count == 3: raise OSError('constructed third unlink failure')
            return original(name, *args, **kwargs)
        with patch.object(op.os, 'unlink', side_effect=failing), self.assertRaisesRegex(OSError, 'third unlink failure'):
            self.execute(source, archive, guards)
        journal = [json.loads(line) for line in (self.sroot/'actions.jsonl').read_text().splitlines()]
        self.assertEqual([r['stage'] for r in journal], ['PREPARED','UNLINKED','PREPARED','UNLINKED','PREPARED'])
        failed = json.loads((self.sroot/'deletion-failure.json').read_text())
        expected = sum(source['files'][n]['identity']['allocated_bytes'] for n in op.TARGET_NAMES[:2])
        self.assertEqual(failed['confirmed_unlinked_allocated_bytes'], expected)
        self.assertEqual(failed['confirmed_unlinked_files'], list(op.TARGET_NAMES[:2]))
        self.assertFalse((self.sroot/'deletion-result.json').exists()); self.assertFalse((self.sroot/'complete.json').exists())
        self.assertEqual(sum((self.cp/n).exists() for n in op.TARGET_NAMES), 23)
        self.assertTrue(all((self.archive/n).is_file() for n in op.CORE_NAMES))
        self.assertTrue(all((self.cp/n).is_file() for n in op.RECOVERY_NAMES + op.CP_METADATA))
        with self.assertRaises(FileExistsError): self.execute(source, archive, guards)

    def test_symlink_swap_after_prepared_cannot_unlink_active_sentinel(self):
        source, archive, guards = self.proofs(); sentinel = self.root/'active730226/model.pt'
        self.put(sentinel, b'active model sentinel stays intact'); original = op.journal
        def swap(root, record):
            original(root, record)
            if record['stage'] == 'PREPARED':
                path = self.cp/record['relative_path']; path.unlink(); path.symlink_to(sentinel)
        with patch.object(op, 'journal', side_effect=swap), self.assertRaisesRegex(ValueError, 'TARGET_REPLACED_BEFORE_UNLINK'):
            self.execute(source, archive, guards)
        failure = json.loads((self.sroot/'deletion-failure.json').read_text())
        self.assertEqual(failure['confirmed_unlinked_files'], []); self.assertEqual(failure['confirmed_unlinked_allocated_bytes'], 0)
        self.assertEqual(sentinel.read_bytes(), b'active model sentinel stays intact')
        self.assertFalse((self.sroot/'deletion-result.json').exists())

    def test_foreign_operation_or_missing_core_archive_proof_has_no_unlink_authority(self):
        source, archive, guards = self.proofs()
        for mutation in ('operation', 'missing_recovery', 'operator_sha'):
            wrong = copy.deepcopy(archive)
            if mutation == 'operation': wrong['operation_id'] = 'b'*32
            elif mutation == 'missing_recovery': del wrong['files'][op.RECOVERY_NAMES[0]]
            else: wrong['operator_source_sha256'] = '0'*64
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'FRESH33_BOUND_PROOF_REQUIRED|FRESH_PROOF_OWNER_HOST_DIFFERS'):
                self.execute(source, wrong, guards)
        self.assertFalse((self.sroot/'deletion-claim.json').exists()); self.assertTrue(all((self.cp/n).exists() for n in op.TARGET_NAMES))

    def test_large_nontarget_is_identity_preserved_without_content_sha_and_only25_removed(self):
        large = self.cp/'actor/huggingface/tokenizer.json'
        self.put(large, b'x'*(2*1024**2+17)); before = op.identity(large)
        original = op.small_raw
        def small_only(path):
            self.assertNotEqual(Path(path), large, 'large retained tokenizer must not be read for SHA')
            return original(path)
        with patch.object(op, 'small_raw', side_effect=small_only):
            source, archive, guards = self.proofs()
            row = source['preserved_checkpoint_tree']['files']['actor/huggingface/tokenizer.json']
            self.assertFalse(row['content_sha256_read']); self.assertNotIn('sha256', row)
            result = self.execute(source, archive, guards)
        self.assertEqual(op.identity(large), before)
        self.assertEqual(len(result['confirmed_unlinked_files']), 25)

    def test_large_nontarget_replaced_identity_rejected_before_unlink(self):
        large = self.cp/'actor/huggingface/tokenizer.json'
        self.put(large, b'x'*(2*1024**2+17)); source, archive, guards = self.proofs()
        replacement = large.with_suffix('.replacement'); replacement.write_bytes(large.read_bytes()); replacement.replace(large)
        with self.assertRaisesRegex(ValueError, 'NONTARGET_METADATA_CHANGED_OR_MISSING'):
            self.execute(source, archive, guards)
        self.assertFalse((self.sroot/'actions.jsonl').exists())
        self.assertTrue(all((self.cp/n).exists() for n in op.TARGET_NAMES))

    def test_exact_legacy_hk_sha_allowed_but_old_source_or_other_hk_sha_rejected(self):
        source, archive, _ = self.proofs(); archive['operator_source_sha256'] = op.LEGACY_HK_OPERATOR_SHA
        op.proof_bind(self.plan, archive, self.operation, 'hk')
        source['operator_source_sha256'] = op.LEGACY_HK_OPERATOR_SHA
        with self.assertRaisesRegex(ValueError, 'FRESH_PROOF_OWNER_HOST_DIFFERS'):
            op.proof_bind(self.plan, source, self.operation, 'source')
        archive['operator_source_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'FRESH_PROOF_OWNER_HOST_DIFFERS'):
            op.proof_bind(self.plan, archive, self.operation, 'hk')

    def test_published_d45_hk_worker_source_functions_remain_ast_identical(self):
        repo = Path(__file__).resolve().parents[2]
        path = 'reproduction/easy2560_joint4/retire_old_cosearch200.py'
        raw = subprocess.check_output(['git', 'show', 'd45b76fb66b6c5ef9f90552f843cbf551aa51e3f:'+path], cwd=str(repo))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), op.LEGACY_HK_OPERATOR_SHA)
        def functions(data):
            return {n.name:ast.dump(n, include_attributes=False) for n in ast.parse(data).body if isinstance(n, ast.FunctionDef)}
        old, current = functions(raw), functions(Path(op.__file__).read_bytes())
        for name in ('need','canonical','digest','no_links','identity','small_raw','small','fsync_dir',
                     'write','birth','context','receipt_map','progress','measured_sha','validate'):
            with self.subTest(name=name): self.assertEqual(old[name], current[name])

    @contextmanager
    def failed_audit(self):
        failed_birth = dict(self.birth(), host=op.HPC_HOST, pid=456789)
        with patch.object(recovery, 'OPERATION_ID', self.operation), \
             patch.object(recovery, 'FAILED_BIRTH', failed_birth), self.site('source'):
            self.put(self.sroot/'plan.json', self.plan)
            self.put(self.sroot/'source-launch-claim.json', dict(operation_id=self.operation,
                candidate_sha256=op.CANDIDATE_SHA, phase='source', operator_source_sha256=op.LEGACY_HK_OPERATOR_SHA))
            self.put(self.sroot/'source-process.json', dict(operation_id=self.operation, phase='source', child=failed_birth))
            self.put(self.sroot/'source-worker.json', failed_birth)
            self.put(self.sroot/'source-exit.json', dict(returncode=1, error=recovery.FAILED_ERROR))
            self.put(self.sroot/'source.log', b'original failed source traceback remains byte-identical\n')
            self.put(self.sroot/'source.lock', b'')
            yield failed_birth

    def test_failed_source_audit_exact_six_renamed_unchanged_plan_and_cp_untouched(self):
        with self.failed_audit():
            before = {name:(op.identity(self.sroot/name), (self.sroot/name).read_bytes())
                      for name in recovery.MOVE_NAMES+('plan.json',)}
            cp_before = {n:op.identity(self.cp/n) for n in op.CORE_NAMES}
            preview = recovery.inspect(self.sroot, self.root/'no-proc')
            self.assertEqual(preview['status'], 'EXACT_FAILED_SOURCE_AUDIT_READY_TO_PRESERVE')
            self.assertFalse((self.sroot/'source-recovery-claim.json').exists())
            result = recovery.preserve(self.sroot, self.root/'no-proc')
            self.assertEqual(result['confirmed_moved_names'], list(recovery.MOVE_NAMES))
            self.assertFalse(result['worker_launched'])
            for name, (identity, raw) in before.items():
                target = self.sroot/name if name == 'plan.json' else self.sroot/recovery.ATTEMPT_NAME/name
                self.assertEqual(op.identity(target), identity); self.assertEqual(target.read_bytes(), raw)
                if name != 'plan.json': self.assertFalse((self.sroot/name).exists())
            self.assertEqual({n:op.identity(self.cp/n) for n in op.CORE_NAMES}, cp_before)
            with self.assertRaisesRegex(ValueError, 'RECOVERY_ALREADY_CLAIMED'):
                recovery.preserve(self.sroot, self.root/'no-proc')

    def test_failed_source_audit_live_exact_birth_blocks_preservation_but_reused_pid_is_exited(self):
        with self.failed_audit() as failed:
            proot = self.root/'proc'; proc = proot/str(failed['pid']); (proc/'ns').mkdir(parents=True)
            self.put(proot/'sys/kernel/random/boot_id', failed['boot_id'].encode())
            (proc/'ns/pid').symlink_to(failed['pid_namespace'])
            fields = ['S']+['0']*19; fields[19] = str(failed['start_ticks'])
            (proc/'stat').write_text(str(failed['pid'])+' (worker) '+' '.join(fields))
            with self.assertRaisesRegex(ValueError, 'ORIGINAL_BIRTH_STILL_LIVE'):
                recovery.preserve(self.sroot, proot)
            self.assertFalse((self.sroot/'source-recovery-claim.json').exists())
            fields[19] = str(failed['start_ticks']+1)
            (proc/'stat').write_text(str(failed['pid'])+' (reused) '+' '.join(fields))
            self.assertEqual(recovery.inspect(self.sroot, proot)['exited_observation']['reason'], 'BIRTH_REUSED')

    def test_failed_source_audit_wrong_exit_birth_deletion_or_symlink_never_moves(self):
        with self.failed_audit():
            exit_path = self.sroot/'source-exit.json'; raw = exit_path.read_bytes()
            self.put(exit_path, dict(returncode=0, error=recovery.FAILED_ERROR))
            with self.assertRaisesRegex(ValueError, 'EXACT_FAILED_SOURCE_ATTEMPT_REQUIRED'):
                recovery.preserve(self.sroot, self.root/'no-proc')
            exit_path.write_bytes(raw)
            worker_path = self.sroot/'source-worker.json'; worker_raw = worker_path.read_bytes()
            self.put(worker_path, dict(recovery.FAILED_BIRTH, start_ticks=recovery.FAILED_BIRTH['start_ticks']+1))
            with self.assertRaisesRegex(ValueError, 'EXACT_FAILED_SOURCE_ATTEMPT_REQUIRED'):
                recovery.preserve(self.sroot, self.root/'no-proc')
            worker_path.write_bytes(worker_raw)
            for name in ('deletion-claim.json', 'actions.jsonl', 'source-validation.json'):
                self.put(self.sroot/name, b'{}')
                with self.assertRaisesRegex(ValueError, 'FAILED_SOURCE_AUDIT_HAS_'):
                    recovery.preserve(self.sroot, self.root/'no-proc')
                (self.sroot/name).unlink()
            sentinel = self.root/'active730226/sentinel'; self.put(sentinel, b'never moved or changed')
            path = self.sroot/'source.log'; path.unlink(); path.symlink_to(sentinel)
            with self.assertRaisesRegex(ValueError, 'SYMLINK'):
                recovery.preserve(self.sroot, self.root/'no-proc')
            self.assertEqual(sentinel.read_bytes(), b'never moved or changed')
            self.assertFalse((self.sroot/'source-recovery-claim.json').exists())

    def test_failed_source_audit_third_rename_failure_is_durable_partial_and_not_retryable(self):
        with self.failed_audit():
            rename = os.rename; count = 0
            def fail_third(source, dest):
                nonlocal count
                count += 1
                if count == 3: raise OSError('constructed third phase-rename failure')
                return rename(source, dest)
            with patch.object(recovery.os, 'rename', side_effect=fail_third), self.assertRaisesRegex(OSError, 'third phase-rename'):
                recovery.preserve(self.sroot, self.root/'no-proc')
            destination = self.sroot/recovery.ATTEMPT_NAME
            failure = json.loads((destination/'preserve-failure.json').read_text())
            self.assertEqual(failure['confirmed_moved_names'], list(recovery.MOVE_NAMES[:2]))
            self.assertFalse(failure['auto_retry_allowed']); self.assertFalse((destination/'preserved.json').exists())
            journal = [json.loads(row) for row in (destination/'actions.jsonl').read_text().splitlines()]
            self.assertEqual([r['stage'] for r in journal], ['PREPARED_RENAME','RENAMED','PREPARED_RENAME','RENAMED','PREPARED_RENAME'])
            self.assertTrue(all((self.sroot/name).exists() for name in recovery.MOVE_NAMES[2:]))
            self.assertTrue((self.sroot/'plan.json').exists())
            with self.assertRaisesRegex(ValueError, 'RECOVERY_ALREADY_CLAIMED'):
                recovery.preserve(self.sroot, self.root/'no-proc')

    def test_reader_scan_checks_real_fd_and_maps_including_current_uid_process(self):
        proc = self.root/'proc/123'; (proc/'fd').mkdir(parents=True)
        (proc/'fd/7').symlink_to(self.cp/op.TARGET_NAMES[0]); (proc/'maps').write_text('000-001 r--p 0 00:00 0 '+str(self.cp/op.TARGET_NAMES[1])+'\n')
        with self.site('source'): observation = op.readers(self.plan, self.root/'proc')
        self.assertEqual(observation['scanned_own_processes'], 1)
        self.assertEqual(len(observation['readers']), 1); self.assertEqual(len(observation['readers'][0]['matches']), 2)
        self.assertEqual(observation['inaccessible'], [])

    def test_reader_scan_inaccessible_own_maps_is_reported_as_incomplete(self):
        proc = self.root/'proc/123'; (proc/'fd').mkdir(parents=True); (proc/'maps').write_text('')
        original = Path.read_text
        def denied(path, *args, **kwargs):
            if path == proc/'maps': raise PermissionError('constructed own process maps denial')
            return original(path, *args, **kwargs)
        with self.site('source'), patch.object(Path, 'read_text', denied): observation = op.readers(self.plan, self.root/'proc')
        self.assertEqual(observation['readers'], []); self.assertEqual(len(observation['inaccessible']), 1)

    def test_terminal_requires_exact_failed_main_closed_extern_and_no_queue_entry(self):
        def run(responses):
            calls = []
            def execute(argv, **kwargs):
                self.assertIn(op.SOURCE_JOB, argv); calls.append(argv)
                return SimpleNamespace(returncode=responses[len(calls)-1][0], stdout=responses[len(calls)-1][1], stderr='')
            return execute
        closed = '713379|FAILED|1:0|2026-10-14T14:02:58\n713379.extern|COMPLETED|0:0|2026-10-14T14:02:58\n'
        self.assertEqual(op.terminal(run([(1,''),(0,closed)]))['accounting']['returncode'], 0)
        for queue, accounting in (('713379|RUNNING\n',closed), ('',closed.replace('713379|FAILED','713379|COMPLETED')),
                ('',closed.splitlines()[0]+'\n'), ('',closed.replace('713379|FAILED','713379.batch|FAILED'))):
            with self.subTest(queue=queue, accounting=accounting), self.assertRaises(ValueError):
                op.terminal(run([(0,queue),(0,accounting)]))

    def test_persistent_launch_claim_never_overwrites_even_after_failure(self):
        claim = self.sroot/'source-launch-claim.json'; op.write(claim, {'old_claim':True})
        args = SimpleNamespace(phase='source',root=str(self.sroot),operation_id=self.operation,candidate=str(self.candidate_path))
        with self.site('source'), patch.object(op.subprocess, 'Popen') as spawn, self.assertRaises(FileExistsError):
            op.launch(args, self.plan)
        spawn.assert_not_called(); self.assertEqual(json.loads(claim.read_text()), {'old_claim':True})


if __name__ == '__main__': unittest.main()
