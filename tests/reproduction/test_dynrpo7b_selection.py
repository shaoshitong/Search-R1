"""Small CPU checks for the new selection and recoverable phase boundaries."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from reproduction.dynrpo7b.selection import Candidate, candidate_budget, group_counts, select_actual_mixed, select_top_prefix
from reproduction.dynrpo7b.mixed_backstop import BackstopLedger
from reproduction.dynrpo7b.checkpoint import RuntimeCheckpoint
from reproduction.dynrpo7b.rolling import MANAGER, _producer_reference, _retire_old
from reproduction.dynrpo7b.restore import source as restore_source, validate_rank_state
from reproduction.dynrpo7b.runtime import ProbeRuntime


MIXED = (1, 0, 0, 0, 0)
WRONG = (0, 0, 0, 0, 0)
CORRECT = (1, 1, 1, 1, 1)


def pool_with_mixed(count, total=640):
    return [Candidate(str(i), i, 1 - i / 1000,
                      MIXED if i < count else WRONG if i < 400 else CORRECT)
            for i in range(total)]


class SelectionTests(unittest.TestCase):
    def test_mixed_first_nonmixed_score_and_same_payload(self):
        a, b, c = object(), object(), object()
        rows = [Candidate("wrong", 0, .9, WRONG, a),
                Candidate("correct", 1, .8, CORRECT, b),
                Candidate("mixed", 2, .1, MIXED, c)]
        self.assertEqual([row.question_id for row in select_top_prefix(rows, 2)], ["wrong", "correct"])
        selected = select_actual_mixed(rows, 2)
        self.assertEqual([row.question_id for row in selected], ["mixed", "wrong"])
        self.assertIs(selected[0].payload, c)
        self.assertIs(selected[1].payload, a)
        self.assertEqual(candidate_budget(10), 2560)
        self.assertEqual(candidate_budget(200), 5120)

    def test_backstop_153_triggers_but154_does_not(self):
        for count, triggered in ((153, True), (154, False)):
            rows = pool_with_mixed(count)
            # Outside the initial512, only the first unfinished G5 is Mixed.
            rows = rows[:512] + [replace(row, strict_em=None) for row in rows[512:]]
            with tempfile.TemporaryDirectory() as directory:
                ledger = BackstopLedger(directory, rows, rows[:512], outer=1, actor_version="a0", probe_version="p0")
                self.assertEqual(ledger.summary()["triggered"], triggered)
                batch = ledger.next_batch()
                self.assertEqual(len(batch), 128 if triggered else 0)
                if triggered:
                    done = [replace(row, strict_em=MIXED if index == 0 else WRONG)
                            for index, row in enumerate(batch)]
                    ledger.commit_batch(done)
                    self.assertEqual(group_counts(ledger.selected())["mixed"], 154)
                    self.assertEqual(ledger.state["replacements"][0]["out"], "399")
                    self.assertEqual(ledger.next_batch(), [])

    def test_backstop_partial_resume_does_not_repeat_finished_g5(self):
        rows = pool_with_mixed(153)
        rows = rows[:512] + [replace(row, strict_em=None) for row in rows[512:]]
        with tempfile.TemporaryDirectory() as directory:
            args = dict(outer=1, actor_version="a0", probe_version="p0")
            first = BackstopLedger(directory, rows, rows[:512], **args)
            batch = first.next_batch()
            done = replace(batch[0], strict_em=MIXED)
            self.assertFalse(first.commit_batch([done], {done.question_id: "/remote/g5-512.json"}))
            resumed = BackstopLedger(directory, rows, rows[:512], **args)
            remaining = resumed.next_batch()
            self.assertEqual(len(remaining), 127)
            self.assertNotIn(done.question_id, [row.question_id for row in remaining])
            self.assertTrue(resumed.commit_batch([replace(row, strict_em=WRONG) for row in remaining]))
            self.assertEqual(resumed.summary()["mixed"], 154)
            self.assertEqual(next(row for row in resumed.completed() if row.question_id == done.question_id).payload,
                             "/remote/g5-512.json")

    def test_exhaustion_and_already_completed_mixed(self):
        rows = pool_with_mixed(0, 512)
        with tempfile.TemporaryDirectory() as directory:
            ledger = BackstopLedger(directory, rows, rows, outer=10, actor_version="a9", probe_version="p0")
            self.assertEqual(ledger.next_batch(), [])
            self.assertEqual(ledger.summary()["shortfall"], 154)
        rows = pool_with_mixed(153)
        rows[512] = replace(rows[512], strict_em=MIXED)
        with tempfile.TemporaryDirectory() as directory:
            ledger = BackstopLedger(directory, rows, rows[:512], outer=10, actor_version="a9", probe_version="p0")
            self.assertEqual(ledger.next_batch(), [])
            self.assertEqual(ledger.summary()["mixed"], 154)

    def test_actor100_probe_resume_and_prefix_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = RuntimeCheckpoint(directory, "r")
            journal.actor_complete(100, {"ranks": ["r0", "r1", "r2", "r3"]},
                                   "driver100", "probe90", prefix_tokens=80,
                                   rollout_ref="actor99-G5", grounding_version="g50")
            self.assertEqual(journal.next_action(), "fit_probe")
            self.assertEqual(journal.state["fitting_prefix_tokens"], 160)
            journal.probe_epoch(1, "probe100-E1")
            resumed = RuntimeCheckpoint(directory, "r")
            self.assertEqual(resumed.state["completed_actor_outer"], 100)
            self.assertEqual(resumed.state["probe_completed_epoch"], 1)
            resumed.probe_epoch(2, "probe100-E2")
            resumed.probe_epoch(3, "probe100-E3")
            resumed.probe_published()
            self.assertEqual(resumed.next_action(), "refresh_grounding")
            resumed.grounding_published("g100", {"adopted_ranks": [0, 1, 2, 3]})
            self.assertEqual(resumed.next_action(), "next_outer")
            self.assertEqual(resumed.state["probe_effective_from_outer"], 101)
            resumed.actor_complete(101, "actor101", "driver101", "probe100-E3", prefix_tokens=160,
                                   rollout_ref="actor100-G5", grounding_version="g100")
            self.assertEqual(resumed.state["phase"], "outer_complete")

    def test_rolling_uses_producer_receipts_and_owned_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "actor/global_step_9"
            state_dir = checkpoint / "training_state"
            state_dir.mkdir(parents=True)
            names = ["model.safetensors", "training_state/driver.pt"] + [
                "training_state/rank-%d.pt" % rank for rank in range(4)]
            for name in names:
                (checkpoint / name).write_bytes(b"producer wrote these bytes")
            index = checkpoint / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"w": "model.safetensors"}}))
            workers = [dict(rank=rank, path=str(state_dir / ("rank-%d.pt" % rank)), sha256="native-rank-%d" % rank)
                       for rank in range(4)]
            driver = dict(path=str(state_dir / "driver.pt"), sha256="native-driver")
            native = dict(completed_outer=9, worker_files=workers, driver_file=driver,
                          question_probe={"path": "probe9.pt"})
            (state_dir / "manifest.json").write_text(json.dumps(native))
            hashes = {name: "producer-file-sha" for name in names}
            hashes.update({str(Path(row["path"]).relative_to(checkpoint)): row["sha256"]
                           for row in [*workers, driver]})
            hashes.update({"model.safetensors.index.json": "producer-index-sha",
                           "training_state/manifest.json": "producer-state-sha"})
            (checkpoint / "complete.json").write_text(json.dumps(dict(
                run_id="r", step=9, finite_all_saved_tensors=True, files_sha256=hashes)))
            actor, saved = _producer_reference(checkpoint, "r", 9)
            self.assertEqual(actor["files_sha256"], hashes)
            self.assertEqual(saved["question_probe"], {"path": "probe9.pt"})
            metadata, actor_root = root / "metadata", root / "actor"
            for step in range(1, 10):
                managed = metadata / ("global_step_%d" % step)
                managed.mkdir(parents=True)
                owned = actor_root / ("global_step_%d" % step)
                owned.mkdir(exist_ok=True)
                (managed / "boundary.json").write_text(json.dumps(dict(
                    manager=MANAGER, run_id="r", completed_outer=step, checkpoint=str(owned))))
            foreign = actor_root / "global_step_777"
            foreign.mkdir()
            self.assertEqual(_retire_old(metadata, actor_root, "r"), [1, 2, 3, 4, 6, 7])
            self.assertTrue(foreign.exists())
            self.assertEqual(sorted(path.name for path in metadata.iterdir()),
                             ["global_step_5", "global_step_8", "global_step_9"])


    def test_pending_restore_reads_committed_epoch_without_replaying_actor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "logs").mkdir()
            (root / "logs/run-manifest.json").write_text(json.dumps(dict(run_id="r")))
            reward_dir = root / "artifacts/round_0"
            reward_dir.mkdir(parents=True)
            (reward_dir / "version-manifest.json").write_text(json.dumps(dict(version="g0",round=0)))
            checkpoint = root / "actor/global_step_10"
            checkpoint.mkdir(parents=True)
            actual = checkpoint / "rank0.pt"
            actual.write_bytes(b"actual persisted state")
            actor = dict(file_bytes={"rank0.pt": actual.stat().st_size},
                         file_mtime_ns={"rank0.pt": actual.stat().st_mtime_ns},
                         worker_files=[dict(rank=rank, actual_optimizer_writes=197) for rank in range(4)])
            phase = dict(run_id="r", completed_actor_outer=10, probe_pending=True,
                         probe_completed_epoch=0, probe_ref="probe0", grounding_version="g0")
            phase_path = root / "formal-state.json"
            phase_path.write_text(json.dumps(dict(phase, probe_completed_epoch=1, probe_ref="actual-E1")))
            receipt = dict(kind="dynrpo7b_native4", status="DYNRPO7B_ACTOR_SAVED_CALLBACKS_PENDING",
                           run_id="r", completed_outer=10, source_base=str(root), checkpoint=str(checkpoint),
                           actor_ref=actor, phase=phase, phase_path=str(phase_path))
            path = root / "actor-pending.json"
            path.write_text(json.dumps(receipt))
            _, restored, _, _, _ = restore_source(path)
            self.assertEqual(restored["step"], 10)
            self.assertEqual(restored["actual_optimizer_writes"], 197)
            self.assertEqual(restored["phase"]["probe_completed_epoch"], 1)
            self.assertEqual(restored["phase"]["probe_ref"], "actual-E1")
            receipt.update(status="DYNRPO7B_ROLLING_READY", phase=dict(phase, probe_pending=False, probe_completed_epoch=3))
            path.write_text(json.dumps(receipt))
            _, restored, _, _, _ = restore_source(path)
            self.assertEqual(restored["phase"]["probe_completed_epoch"], 3)
            self.assertFalse(restored["phase"]["probe_pending"])
            saved = dict(rank=0, world_size=4, completed_outer=10, actual_optimizer_writes=197,
                         scheduler={"last_epoch": 10}, optimizer={"state": {0: {"step": 197.0}}})
            self.assertEqual(validate_rank_state(saved, 0, 10, 197)["actual_optimizer_writes"], 197)

    def test_driver_resumes_old_fit_directory_without_duplicate_tensor_load(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / 'logs').mkdir()
            (base / 'logs/run-manifest.json').write_text(json.dumps(dict(run_id='r')))
            old_fit = base / 'original-execution/update-0010'
            old_fit.mkdir(parents=True)
            refs = [dict(path='real-E%d.pt' % epoch) for epoch in (1,2,3)]
            (old_fit / 'epoch-01.json').write_text(json.dumps(dict(epoch=1,checkpoint=refs[0])))
            journal = RuntimeCheckpoint(base/'preparation/dynrpo7b-checkpoints','r')
            journal.actor_complete(10,'actor10','driver10','probe0',prefix_tokens=80,
                                   rollout_ref='same-G5',grounding_version='g0')
            calls=[]
            def rpc(operation,payload):
                calls.append(operation)
                if operation=='init':
                    return [dict(initialized=True)]
                self.assertEqual(operation,'fit')
                self.assertEqual(payload['output_dir'],str(old_fit))
                self.assertEqual(payload['actor_checkpoint_ref'],dict(path='/old/actor10'))
                return [dict(checkpoint=refs[2],epochs=[dict(epoch=i,checkpoint=refs[i-1]) for i in (2,3)])]
            runtime=ProbeRuntime(dict(output_dir=str(base/'new-output')),
                                 base,SimpleNamespace(dynrpo7b_probe_operation=rpc))
            runtime.completed_outer=10
            runtime.pending=dict(fit_groups=['originalG5'],fit_output_dir=str(old_fit),
                                 actor_checkpoint_ref=dict(path='/old/actor10'))
            runtime.fit_after_checkpoint(None,'/old/actor10')
            self.assertEqual(calls,['init','fit'])
            self.assertEqual(runtime._probe_ref,refs[2])
            self.assertIsNone(runtime.pending)
            self.assertEqual(RuntimeCheckpoint(base/'preparation/dynrpo7b-checkpoints','r').state['probe_completed_epoch'],3)


if __name__ == "__main__":
    unittest.main()
