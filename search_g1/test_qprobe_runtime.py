"""Successful-outer and real DataProto collector tests; no GPUs or network.

The collector uses the native framework DataProto with a fake generation
manager, not a fake tensor protocol. Native tests skip when that framework's
optional dependencies are absent; transaction tests still execute locally.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from search_g1.qprobe_model import OnlineQProbe, QProbeConfig
from search_g1.qprobe_runtime import (
    QProbeRuntime, QProbeCandidateStream, collect, validate_config, _retriever_contract,
    _derived_policy_snapshot_id,
)
from search_g1.qprobe_selection import PredictQuestion

try:
    from verl import DataProto
except ModuleNotFoundError:
    DataProto = None


class QuestionProbeSnapshotSealTests(unittest.TestCase):
    def setUp(self):
        self.identity = dict(run_id="run", logical_update=1, policy_snapshot_id="actor:1",
                             trajectory_ids=["row-0", "row-1"], source_sha256={"source.py": "sha"})
        self.meta = dict(behavior_identity=copy.deepcopy(self.identity))
        self.turns = [[dict(policy_snapshot_id="actor:1")],
                      [dict(policy_snapshot_id="actor:1"), dict(policy_snapshot_id="actor:1")]]

    def test_missing_top_level_is_derived_from_all_turns_without_mutating_metadata(self):
        before = copy.deepcopy((self.meta, self.turns, self.identity))
        self.assertEqual(_derived_policy_snapshot_id(self.meta, self.turns, self.identity), "actor:1")
        self.assertEqual((self.meta, self.turns, self.identity), before)
        self.assertNotIn("policy_snapshot_id", self.meta)
        self.meta["policy_snapshot_id"] = "actor:1"
        self.assertEqual(_derived_policy_snapshot_id(self.meta, self.turns, self.identity), "actor:1")

    def test_entire_returned_behavior_identity_must_match(self):
        for fault in ("missing", "none", "missing_field", "extra_field", "changed_source", "changed_order"):
            with self.subTest(fault=fault):
                meta = copy.deepcopy(self.meta)
                if fault == "missing": del meta["behavior_identity"]
                elif fault == "none": meta["behavior_identity"] = None
                elif fault == "missing_field": del meta["behavior_identity"]["run_id"]
                elif fault == "extra_field": meta["behavior_identity"]["unexpected"] = True
                elif fault == "changed_source": meta["behavior_identity"]["source_sha256"]["source.py"] = "other"
                elif fault == "changed_order": meta["behavior_identity"]["trajectory_ids"].reverse()
                with self.assertRaisesRegex(RuntimeError, "QPROBE_BEHAVIOR_IDENTITY_CHANGED"):
                    _derived_policy_snapshot_id(meta, self.turns, self.identity)

    def test_missing_empty_or_malformed_actual_turn_evidence_is_rejected(self):
        faults = (None, [], [[], self.turns[1]], [{"policy_snapshot_id": "actor:1"}],
                  [[None]], [["actor:1"]], [[{}]], [[{"policy_snapshot_id": None}]],
                  [[{"policy_snapshot_id": ""}]], [[{"policy_snapshot_id": " "}]])
        for turns in faults:
            with self.subTest(turns=turns), self.assertRaisesRegex(RuntimeError, "MISSING_REAL_QPROBE_TURN"):
                _derived_policy_snapshot_id(self.meta, turns, self.identity)

    def test_every_actual_turn_must_observe_the_same_expected_snapshot(self):
        faults = ([[dict(policy_snapshot_id="actor:2")]],
                  [self.turns[0], [dict(policy_snapshot_id="actor:2")]],
                  [[dict(policy_snapshot_id="actor:1"), dict(policy_snapshot_id="actor:2")]])
        for turns in faults:
            with self.subTest(turns=turns), self.assertRaisesRegex(RuntimeError, "QPROBE_POLICY_SNAPSHOT_CHANGED"):
                _derived_policy_snapshot_id(self.meta, turns, self.identity)
        # Even a correct top-level claim cannot substitute for actual turn evidence.
        self.meta["policy_snapshot_id"] = "actor:1"
        with self.assertRaisesRegex(RuntimeError, "QPROBE_POLICY_SNAPSHOT_CHANGED"):
            _derived_policy_snapshot_id(self.meta, faults[0], self.identity)

    def test_present_top_level_none_empty_or_conflicting_snapshot_is_rejected(self):
        for snapshot in (None, "", " ", "actor:2"):
            with self.subTest(snapshot=snapshot), self.assertRaisesRegex(RuntimeError, "QPROBE_POLICY_SNAPSHOT_CHANGED"):
                _derived_policy_snapshot_id(dict(self.meta, policy_snapshot_id=snapshot), self.turns, self.identity)

    def test_expected_snapshot_must_be_nonempty(self):
        for snapshot in (None, "", " "):
            identity = dict(self.identity, policy_snapshot_id=snapshot)
            with self.subTest(snapshot=snapshot), self.assertRaisesRegex(RuntimeError, "QPROBE_POLICY_SNAPSHOT_CHANGED"):
                _derived_policy_snapshot_id(dict(behavior_identity=copy.deepcopy(identity)), self.turns, identity)


class MemoryCache:
    feature_dim = 4
    spec_sha = "a" * 64

    def __init__(self):
        self.calls = []

    def get_features(self, qids, questions):
        # A strict two-field API makes accidental ground_truth forwarding fail.
        self.calls.append((list(qids), list(questions)))
        self.assertions = all(isinstance(x, str) for x in questions)
        return torch.tensor([[i/1000, 1, i % 3, i % 7] for i in range(len(qids))], dtype=torch.float32)


def runtime(base, exploration=64, candidate_questions=5120):
    cache = MemoryCache()
    config = dict(candidate_questions=candidate_questions, selected_questions=512,
                  exploration_questions=exploration, responses_per_question=5, seed=42,
                  head_batch_size=256, head_epochs_per_outer=1)
    head = OnlineQProbe(4, QProbeConfig.from_mapping(config), feature_spec_sha=cache.spec_sha)
    return QProbeRuntime(config, base, feature_cache=cache, head=head)


def stage_pending(value, outer=1, attempt_id="test-attempt", count=512):
    features = torch.rand((count, 4), generator=torch.Generator().manual_seed(1))
    prediction = value.head.predict(features)
    labels = [dict(complete=True, qid="q" + str(i),
                   question_hash=hashlib.sha256(("q" + str(i)).encode()).hexdigest(),
                   strict_em=[int(j < i % 6) for j in range(5)],
                   selection_source="random_exploration" if i < 64 else "predicted_mixed")
              for i in range(count)]
    value.begin_outer(outer=outer, attempt_id=attempt_id, selected=list(range(count)), features=features,
        predictions={key:prediction[key] for key in ("mixed_probability","expected_variance")}, prediction_version=value.version,
        policy_snapshot_id="pre-update-actor:" + str(outer),
        contracts=dict(run_id="run", sampling_contract_sha="sampling", strict_scoring_contract_sha="strict",
                       retriever_version="retriever"))
    value.observe_labels(labels, [x["strict_em"] for x in labels])
    return labels


class PublishedHeadCheckpointReceiptTests(unittest.TestCase):
    """Real tiny head publication consumed by the unchanged native checkpoint guard."""
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def checkpoint(self, base):
        from search_g1.training_state import atomic_save
        checkpoint = base / "checkpoint"
        directory = checkpoint / "training_state"
        directory.mkdir(parents=True)
        value = runtime(base / "artifacts")
        stage_pending(value)
        value.complete_outer(1, "run:1")
        question = value.save_checkpoint(directory, 1, {"consumed_source_rows": 512}, "reward0",
                                         actor_snapshot_id="run:1")
        workers = []
        for rank in range(4):
            state = dict(run_id="run", rank=rank, world_size=4, scheduler={"last_epoch": 1},
                         actual_optimizer_writes=20, optimizer={"state": {0: {
                             "step": torch.tensor(20.), "exp_avg": torch.zeros(1),
                             "exp_avg_sq": torch.zeros(1)}}})
            workers.append(dict(atomic_save(directory / f"rank-{rank}.pt", state), rank=rank))
        driver = atomic_save(directory / "driver.pt", dict(completed_outer=1,
                             candidate_stream={"consumed_source_rows": 512}))
        (directory / "manifest.json").write_text(json.dumps(dict(completed_outer=1,
            worker_files=workers, driver_file=driver, question_probe=question)))
        (base / "logs").mkdir()
        (base / "logs/run-manifest.json").write_text(json.dumps(dict(profile="common8", run_id="run",
            training_world_size=4, question_probe={})))
        return checkpoint, question

    def consume(self, base, checkpoint):
        from reproduction.common8 import runtime as native
        # Only HF model validation is substituted; the actual worker/driver/head
        # byte, hash and CPU-state checkpoint checks all execute unchanged.
        with patch.dict(os.environ, SEARCH_G1_BASE=str(base), SEARCH_G1_TRAINING_WORLD_SIZE="4"), \
                patch.object(native, "hf_checkpoint_complete", return_value={"files_sha256": {}}):
            return native.checkpoint_complete(1, checkpoint)

    def test_real_published_head_passes_native_receipt_consumer(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            checkpoint, question = self.checkpoint(base)
            result = self.consume(base, checkpoint)
            self.assertEqual(result["question_probe"]["actor_snapshot_id"], "run:1")
            self.assertEqual(result["actual_optimizer_writes_per_rank"], 20)
            self.assertEqual(result["files_sha256"]["training_state/question-probe-state.pt.head.pt"],
                             question["head"]["checkpoint_sha"])

    def test_modified_head_bytes_are_rejected_before_complete_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            checkpoint, question = self.checkpoint(base)
            path = Path(question["head"]["path"])
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 1  # Keep file size unchanged: the native hash check must reject it.
            path.write_bytes(raw)
            with self.assertRaises(AssertionError):
                self.consume(base, checkpoint)
            self.assertFalse((checkpoint / "complete.json").exists())

class QuestionProbeTransactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_failed_actor_outer_never_fits_and_success_fits_once_with_original_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            value = runtime(temp)
            version = value.version
            stage_pending(value, attempt_id="failed")
            self.assertEqual(value.version, version)
            before_rng = copy.deepcopy(value.selection_rng.bit_generator.state)
            value.abort_outer(1, {"attempt_id": "failed"}, "only partial Adam writes")
            self.assertEqual(value.version, version)
            self.assertEqual(value.head.update_count, 0)
            self.assertEqual(value.selection_rng.bit_generator.state, before_rng)
            stage_pending(value, attempt_id="successful")
            event = value.complete_outer(1, "run:1", {"attempt_id": "successful"})
            self.assertEqual(value.head.update_count, 1)
            self.assertIsNone(value.pending)
            self.assertEqual(event["label_policy_snapshot_id"], "pre-update-actor:1")
            self.assertEqual(event["successful_actor_snapshot_id"], "run:1")
            self.assertEqual(event["prequential"]["all"]["questions"], 512)
            self.assertTrue(all(x["policy_snapshot_id"] == "pre-update-actor:1" for x in value.head.replay_labels))
            with self.assertRaises(RuntimeError): value.complete_outer(1, "run:1")

    def test_incomplete_labels_or_wrong_attempt_cannot_update_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            value = runtime(temp)
            with self.assertRaises(ValueError): stage_pending(value, count=511)
            self.assertEqual(value.head.update_count, 0)
            with self.assertRaises(RuntimeError): value.save(Path(temp)/"state.pt", "run:0")
            value.abort_outer(1)
            stage_pending(value)
            with self.assertRaises(ValueError): value.complete_outer(1, "run:1", {"attempt_id": "wrong"})
            self.assertEqual(value.head.update_count, 0)

    def test_failed_commit_publication_rolls_back_head_optimizer_rng_and_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            value = runtime(temp)
            stage_pending(value)
            version = value.version
            state = value.head.state_dict()
            with patch("search_g1.qprobe_runtime._write_json", side_effect=OSError("disk write failed")):
                with self.assertRaises(OSError): value.complete_outer(1, "run:1")
            self.assertEqual(value.version, version)
            self.assertEqual(value.head.update_count, 0)
            self.assertEqual(value.head.replay_labels, [])
            self.assertTrue(torch.equal(value.head.rng.get_state(), state["rng"]))
            self.assertIsNotNone(value.pending)
            value.complete_outer(1, "run:1")
            self.assertEqual(value.head.update_count, 1)

    def test_head_and_rng_restore_bound_to_same_actor_not_future_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            value = runtime(temp)
            stage_pending(value)
            value.complete_outer(1, "run:1")
            value.selection_rng.permutation(5120)
            receipt = value.save_checkpoint(Path(temp)/"cp1", 1, {"candidate_cursor": 5120}, "reward-probe0", "run:1")
            path = receipt["path"]
            version = value.version
            expected_draw = value.selection_rng.permutation(5120)
            stage_pending(value, outer=2, attempt_id="second")
            value.complete_outer(2, "run:2")
            self.assertNotEqual(value.version, version)
            with self.assertRaises(ValueError): value.restore(path, "run:2")
            state = value.restore(path, "run:1")
            self.assertEqual(value.version, version)
            self.assertEqual(value.completed_outer, 1)
            np.testing.assert_array_equal(value.selection_rng.permutation(5120), expected_draw)
            self.assertEqual(state["candidate_state"]["candidate_cursor"], 5120)
            # Reject corrupt head before deserializing it.
            head_path = Path(path).with_name(Path(path).name + ".head.pt")
            head_path.write_bytes(b"corrupt")
            with self.assertRaises(ValueError): value.restore(path, "run:1")

    def test_restored_configuration_may_relocate_paths_but_not_exploration_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            value = runtime(temp)
            receipt = value.save(Path(temp)/"state.pt", "run:0")
            other = runtime(temp, exploration=128)
            with self.assertRaises(ValueError): other.restore(receipt["path"], "run:0")

    def test_strict_feature_api_and_zero_exploration_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            value = runtime(temp, exploration=0)
            question = PredictQuestion(0, "question-id", "just the question")
            _, output, version = value.score([question])
            self.assertEqual(value.config["exploration_questions"], 0)
            self.assertEqual(value.cache.calls, [(["question-id"], ["just the question"])])
            self.assertEqual(version, value.version)
            self.assertEqual(output["mixed_probability"].shape, (1,))
        with self.assertRaises(ValueError): validate_config({"post_rollout_em_filter": True})

    def test_bootstrap_head_configuration_cannot_silently_override_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            value = runtime(temp)
            head_path = Path(temp)/"bootstrap.pt"
            value.head.save(head_path, actor_snapshot_id="run:0")
            config = dict(value.config, initial_head_checkpoint=str(head_path))
            exact = QProbeRuntime(config, temp, feature_cache=MemoryCache())
            self.assertEqual(exact.head.config, value.head.config)
            with self.assertRaisesRegex(ValueError, "head config differs"):
                QProbeRuntime(dict(config, head_learning_rate=.001), temp, feature_cache=MemoryCache())
            with self.assertRaisesRegex(ValueError, "head config differs"):
                QProbeRuntime(dict(config, seed=43), temp, feature_cache=MemoryCache())

    def test_retriever_semantics_survive_transport_relocation_but_not_index_or_topk_change(self):
        config = SimpleNamespace(retriever=SimpleNamespace(url="http://old-host:8100", topk=3))
        run = {"retriever_version": "immutable-index-sha"}
        original = _retriever_contract(config, run)
        config.retriever.url = "http://new-task-local-host:9200"
        relocated = _retriever_contract(config, run)
        self.assertEqual(original["retriever_version"], relocated["retriever_version"])
        self.assertEqual(original["retriever_contract"], relocated["retriever_contract"])
        self.assertNotEqual(original["retriever_url"], relocated["retriever_url"])
        self.assertNotIn("url", original["retriever_contract"])
        config.retriever.topk = 4
        self.assertNotEqual(original["retriever_version"], _retriever_contract(config, run)["retriever_version"])
        config.retriever.topk = 3
        self.assertNotEqual(original["retriever_version"],
                            _retriever_contract(config, {"retriever_version": "other-index"})["retriever_version"])
        self.assertEqual(original["retriever_version"],
                         _retriever_contract(config, {"retriever_manifest_sha256": "immutable-index-sha"})["retriever_version"])
        paired = dict(run, full_state_restore={"kind": "question_probe_pair"}, retriever_topk=3)
        self.assertEqual(original["retriever_version"], _retriever_contract(config, paired)["retriever_version"])
        config.retriever.topk = 4
        with self.assertRaisesRegex(RuntimeError, "pinned retrieval topk"):
            _retriever_contract(config, paired)
        with self.assertRaisesRegex(RuntimeError, "pinned retriever"):
            _retriever_contract(config, {})


class CharTokenizer:
    pad_token_id = 0

    def decode(self, tokens):
        return "".join(chr(int(x)) for x in tokens)


class NativeMemoryStream:
    def __init__(self, size=6000):
        self.loader = SimpleNamespace(dataset=list(range(size)))
        self.consumed = 0
        self.offset = 0
        self.pending = None

    def state(self):
        return dict(candidate_order=list(range(len(self.loader.dataset))), candidate_cursor=self.consumed,
                    epoch=0, consumed_source_rows=self.consumed, pending_source=None, pending_offset=0)

    def one(self):
        index = self.consumed
        self.consumed += 1
        return DataProto.from_dict(tensors=dict(input_ids=torch.tensor([[index+1000, 65]]),
            attention_mask=torch.ones((1, 2), dtype=torch.long), position_ids=torch.tensor([[0, 1]])),
            non_tensors=dict(candidate_stable_id=np.array([str(index)], dtype=object),
                candidate_original_question=np.array(["question " + str(index)], dtype=object),
                index=np.array([index], dtype=object),
                reward_model=np.array([{"ground_truth": {"target": ["yes"]}}], dtype=object)))


class NativeLightDataset:
    prompt_key = "prompt"

    def __init__(self, size=6000):
        self.size = size
        self.materialized = []
        # Provide the same .dataframe.iloc interface without an extra test package.
        self.dataframe = SimpleNamespace(iloc=self)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Native dataset.__getitem__ does tokenization; direct .iloc must not.
        self.materialized.append(index)
        return dict(input_ids=torch.tensor([index+1000, 65]), attention_mask=torch.ones(2, dtype=torch.long),
            position_ids=torch.tensor([0, 1]), index=index,
            reward_model={"ground_truth": {"target": ["yes"]}})

    def row(self, index):
        return dict(prompt=[{"role": "user", "content": "question " + str(index)}],
                    data_source="nq", extra_info={"index": index})


class RawRows:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset.row(index)


class FixedSampler:
    def __init__(self, size):
        self.size = size
        self.draws = 0

    def __iter__(self):
        self.draws += 1
        return iter(list(range(self.size)) if self.draws == 1 else list(reversed(range(self.size))))


def light_loader(size=6000):
    dataset = NativeLightDataset(size)
    dataset.dataframe = SimpleNamespace(iloc=RawRows(dataset))
    def collate(records):
        result = {}
        for key in records[0]:
            values = [x[key] for x in records]
            result[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else np.array(values, dtype=object)
        return result
    return SimpleNamespace(dataset=dataset, sampler=FixedSampler(size), collate_fn=collate)


def fake_prepare(parent, outer, *, repeats, **attempt):
    ids = [str(int(i)) for i in parent.non_tensor_batch["index"]]
    return dict(run_id="run", logical_update=outer, policy_snapshot_id="sampling-actor:"+str(outer),
        parent_group_ids=ids, trajectory_ids=[qid+"-sample-"+str(i) for qid in ids for i in range(5)])


def fake_attach(batch, identity, *, do_sample):
    batch.non_tensor_batch["behavior_row_id"] = np.array(identity["trajectory_ids"], dtype=object)
    batch.meta_info.update(behavior_identity=identity, policy_snapshot_id=identity["policy_snapshot_id"], do_sample=do_sample)


class NativeFakeManager:
    config = SimpleNamespace(max_start_length=2,max_prompt_length=4096,max_response_length=500,max_obs_length=500)

    def __init__(self, missing_row=False):
        self.calls = []
        self.missing_row = missing_row
        self.timing_raw = {}
        self.original_probabilities = None
        self.returned_meta_keys = None

    def run_llm_loop(self, gen_batch, initial_input_ids):
        self.calls.append(len(gen_batch))
        size = len(gen_batch) - int(self.missing_row)
        responses = []
        for i, qindex in enumerate(gen_batch.batch["input_ids"][:size, 0]):
            correct_count = (int(qindex)-1000) % 6
            text = "<answer>yes</answer>" if i % 5 < correct_count else "<answer>no</answer>"
            responses.append([ord(c) for c in text])
        width = max(map(len, responses))
        tensor = torch.zeros((size, width), dtype=torch.long)
        mask = torch.zeros_like(tensor)
        probabilities = torch.zeros((size, width), dtype=torch.float32)
        for i, row in enumerate(responses):
            tensor[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
            probabilities[i, :len(row)] = -.25 - (i % 99)/1000
        self.original_probabilities = probabilities.clone()
        prompts = gen_batch.batch["input_ids"][:size].clone()
        full_mask = torch.cat((torch.ones_like(prompts), mask), dim=1)
        turns = np.empty(size, dtype=object)
        turns[:] = [[dict(real_generated_turn=True,
                         policy_snapshot_id=gen_batch.meta_info["behavior_identity"]["policy_snapshot_id"])]
                    for _ in range(size)]
        output = DataProto.from_dict(tensors=dict(prompts=prompts, responses=tensor,
            input_ids=torch.cat((prompts, tensor), dim=1), attention_mask=full_mask,
            info_mask=full_mask.clone(), sampling_mask=mask.bool(), sampling_log_probs=probabilities,
            position_ids=(full_mask.cumsum(-1)-1)*full_mask),
            non_tensors=dict(behavior_trajectory_id=gen_batch.non_tensor_batch["behavior_row_id"][:size],
                             behavior_turns=turns),
            meta_info=dict(behavior_identity=copy.deepcopy(gen_batch.meta_info["behavior_identity"]),
                behavior_sampling={"temperature": 1., "top_p": 1.},
                turns_stats=[1]*size, active_mask=[False]*size, valid_action_stats=[1]*size,
                valid_search_stats=[0]*size, actual_retrieval_stats=[0]*size))
        self.returned_meta_keys = frozenset(output.meta_info)
        return output


@unittest.skipIf(DataProto is None, "Native DataProto dependencies are unavailable")
class QuestionProbeNativeCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def _fixture(self, temp, *, exploration=64):
        base = Path(temp); (base/"logs").mkdir()
        (base/"logs/run-manifest.json").write_text(json.dumps(dict(run_id="run", source_commit="commit",
            model_revision="official-revision", budget={"common_manifest_sha256": "common-data"},
            retriever_version="immutable-index-version")))
        value = runtime(temp, exploration=exploration)
        snapshots = []
        def capture(directory):
            snapshots.append(directory)
            return [{"rank": 0, "path": directory+"/rank-0.pt", "sha256": "worker-rng"}]
        trainer = SimpleNamespace(global_steps=1, tokenizer=CharTokenizer(),
            actor_rollout_wg=SimpleNamespace(capture_mixed_state=capture),
            reward_fn=SimpleNamespace(expected_probe_version="old-reward-probe0"),
            config=SimpleNamespace(max_turns=4, data=SimpleNamespace(max_start_length=2,max_prompt_length=4096,max_response_length=500,max_obs_length=500),
                                   retriever=SimpleNamespace(url="http://private-retriever", topk=3)))
        return trainer, value, snapshots

    def test_native_5120_question_pool_generates_only_2560_actual_trajectories(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"SEARCH_G1_BASE": temp}):
            trainer, value, snapshots = self._fixture(temp)
            manager, stream = NativeFakeManager(), QProbeCandidateStream(light_loader())
            with patch("reproduction.common8.behavior_runtime.prepare_identity", fake_prepare), \
                 patch("reproduction.common8.behavior_runtime.attach_generation_identity", fake_attach):
                parents, output, identity, audit = collect(trainer, manager, stream,
                    {"attempt_id": "native-test", "attempt_index": 1}, value)
            self.assertEqual(manager.calls, [2560])
            self.assertNotIn("policy_snapshot_id", manager.returned_meta_keys)
            self.assertEqual(output.meta_info["behavior_identity"], identity)
            self.assertEqual(output.meta_info["policy_snapshot_id"], identity["policy_snapshot_id"])
            self.assertEqual(stream.consumed, 5120)
            self.assertEqual(len(stream.loader.dataset.materialized), 512)
            self.assertEqual(len(set(stream.loader.dataset.materialized)), 512)
            self.assertEqual((len(parents), len(output)), (2560, 2560))
            self.assertEqual(audit["raw_generated_trajectories"], 2560)
            self.assertEqual(audit["selected_random_exploration"], 64)
            self.assertEqual(audit["selected_predicted_mixed"], 448)
            self.assertEqual(audit["retriever_url"], "http://private-retriever")
            self.assertEqual(audit["retriever_contract"], dict(topk=3, recorded_index_version="immutable-index-version"))
            self.assertEqual(audit["sampling_contract"]["temperature"], 1.)
            self.assertEqual(audit["strict_scoring_contract"]["target"], "strict_terminal_em")
            self.assertTrue(all(len(audit[key]) == 64 for key in (
                "sampling_contract_sha", "strict_scoring_contract_sha", "retriever_version")))
            self.assertGreater(audit["selected_all_wrong"], 0)
            self.assertNotIn("selection_fallback", parents.batch)
            self.assertEqual(output.non_tensor_batch["behavior_trajectory_id"].tolist(), identity["trajectory_ids"])
            self.assertTrue(torch.equal(output.batch["sampling_log_probs"], manager.original_probabilities))
            self.assertEqual(len(value.cache.calls[0][0]), 5120)
            self.assertTrue(value.cache.assertions)
            self.assertEqual(len(snapshots), 2)
            self.assertEqual(value.head.update_count, 0)
            self.assertIsNotNone(value.pending)
            selected = json.loads((value.output_dir/"attempts/native-test/selection.json").read_text())
            self.assertTrue(selected["prediction_before_label"])
            self.assertEqual(len(selected["questions"]), 5120)
            self.assertTrue(all("strict_em" not in q and "ground_truth" not in q for q in selected["questions"]))
            value.complete_outer(1, "run:1", {"attempt_id": "native-test"})
            self.assertEqual(value.head.update_count, 1)
            self.assertEqual(len(value.head.replay_labels), 512)

    def test_light_stream_partial_restore_and_epoch_boundary_do_not_redraw_sampler(self):
        from search_g1.training_state import compact_stream_state
        from search_g1.qprobe_runtime import _collect_pool
        loader = light_loader(6000)
        stream = QProbeCandidateStream(loader)
        self.assertEqual(loader.sampler.draws, 1)
        first = [stream.one().non_tensor_batch["candidate_stable_id"][0] for _ in range(17)]
        self.assertEqual(len(loader.dataset.materialized), 0)
        state = compact_stream_state(stream.state())
        self.assertEqual(state["pending_offset"], 17)
        restored = QProbeCandidateStream(loader, restored_state=copy.deepcopy(state))
        self.assertEqual(loader.sampler.draws, 1)
        self.assertEqual(stream.one().non_tensor_batch["candidate_stable_id"].tolist(),
                         restored.one().non_tensor_batch["candidate_stable_id"].tolist())
        self.assertEqual(stream.consumed, restored.consumed)
        self.assertEqual(len(loader.dataset.materialized), 0)
        # Deliberately cross the permutation boundary near the end. The reverse
        # next order overlaps the current tail; one source-length cap is too short.
        while restored.consumed < 5900: restored.one()
        parents, pool = _collect_pool(restored, 6000)
        self.assertEqual(len(pool), 6000)
        self.assertEqual(len({q.qid for q in pool}), 6000)
        self.assertNotIn("input_ids", parents.batch)
        self.assertEqual(len(loader.dataset.materialized), 0)
        self.assertEqual(loader.sampler.draws, 2)

    def test_legacy_encoded_pending_batch_restores_to_same_light_source_rows(self):
        from search_g1.mixed_runtime import CandidateStream
        loader = light_loader()
        encoded = CandidateStream(loader)
        encoded.one()
        self.assertEqual(len(loader.dataset.materialized), 512)
        state = copy.deepcopy(encoded.state())
        restored = QProbeCandidateStream(loader, restored_state=state)
        question = restored.one()
        self.assertEqual(int(question.batch["qprobe_source_index"][0]), 1)
        self.assertEqual(question.non_tensor_batch["candidate_original_question"].tolist(), ["question 1"])
        self.assertEqual(restored.consumed, 2)
        self.assertEqual(loader.sampler.draws, 1)
        self.assertEqual(len(loader.dataset.materialized), 512)

    def test_partial_infrastructure_output_is_not_labelled_all_wrong(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"SEARCH_G1_BASE": temp}):
            trainer, value, _ = self._fixture(temp, exploration=0)
            with patch("reproduction.common8.behavior_runtime.prepare_identity", fake_prepare), \
                 patch("reproduction.common8.behavior_runtime.attach_generation_identity", fake_attach):
                with self.assertRaisesRegex(RuntimeError, "INCOMPLETE_QPROBE_GENERATION"):
                    collect(trainer, NativeFakeManager(missing_row=True), NativeMemoryStream(),
                            {"attempt_id": "partial-test", "attempt_index": 1}, value)
            self.assertIsNone(value.pending)
            self.assertEqual(value.head.update_count, 0)
            directory = value.output_dir/"attempts/partial-test"
            self.assertTrue((directory/"selection.json").exists())
            self.assertFalse((directory/"rollout-audit.json").exists())

    def test_native_collector_rejects_corrupt_snapshot_evidence_before_labels(self):
        for fault, error in (
            ("identity", "QPROBE_BEHAVIOR_IDENTITY_CHANGED"),
            ("missing_turn_snapshot", "MISSING_REAL_QPROBE_TURN_POLICY_SNAPSHOT"),
            ("nondict_turn", "MISSING_REAL_QPROBE_TURN_POLICY_SNAPSHOT"),
            ("mixed_turns", "QPROBE_POLICY_SNAPSHOT_CHANGED"),
            ("top_none", "QPROBE_POLICY_SNAPSHOT_CHANGED"),
            ("top_conflict", "QPROBE_POLICY_SNAPSHOT_CHANGED"),
        ):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp, \
                    patch.dict(os.environ, {"SEARCH_G1_BASE": temp}):
                trainer, value, snapshots = self._fixture(temp, exploration=0)
                class CorruptingManager(NativeFakeManager):
                    def run_llm_loop(self, gen_batch, initial_input_ids):
                        output = super().run_llm_loop(gen_batch, initial_input_ids)
                        if fault == "identity": output.meta_info["behavior_identity"]["run_id"] = "different-run"
                        elif fault == "missing_turn_snapshot":
                            del output.non_tensor_batch["behavior_turns"][0][0]["policy_snapshot_id"]
                        elif fault == "nondict_turn": output.non_tensor_batch["behavior_turns"][0] = [None]
                        elif fault == "mixed_turns":
                            output.non_tensor_batch["behavior_turns"][-1][0]["policy_snapshot_id"] = "different-actor"
                        elif fault == "top_none": output.meta_info["policy_snapshot_id"] = None
                        elif fault == "top_conflict": output.meta_info["policy_snapshot_id"] = "different-actor"
                        return output
                with patch("reproduction.common8.behavior_runtime.prepare_identity", fake_prepare), \
                     patch("reproduction.common8.behavior_runtime.attach_generation_identity", fake_attach), \
                     self.assertRaisesRegex(RuntimeError, error):
                    collect(trainer, CorruptingManager(), NativeMemoryStream(),
                            {"attempt_id": "corrupt-test", "attempt_index": 1}, value)
                self.assertIsNone(value.pending)
                self.assertEqual(value.head.update_count, 0)
                self.assertEqual(len(snapshots), 1)
                directory = value.output_dir/"attempts/corrupt-test"
                self.assertTrue((directory/"selection.json").exists())
                self.assertFalse((directory/"after-generation.pt").exists())
                self.assertFalse((directory/"rollout-audit.json").exists())


if __name__ == "__main__": unittest.main()
