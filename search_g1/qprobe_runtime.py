"""Native Search-G1 question-probe collection and successful-outer transactions.

The probe never receives a DataProto or gold answer. Only the selected 512
questions are repeated and sent to the existing agentic generation manager.
The head is fitted only by an explicit successful-outer hook in the trainer.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from search_g1.qprobe_selection import (
    QuestionPool, count_predictions, prequential_metrics, select_questions, validate_budget,
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".partial-", dir=path.parent)
    published = False
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Immutable, no race that overwrites a prior receipt.
        published = True
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    except Exception:
        if published and path.exists() and os.stat(path).st_ino == os.stat(temporary).st_ino:
            path.unlink()  # Remove only this failed publication, never an existing receipt.
        raise
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _atomic_save(path, value):
    from search_g1.training_state import atomic_save
    return atomic_save(path, value)


def _plain_config(config):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True)
    except ImportError:
        pass
    return dict(config)


def validate_config(config):
    config = _plain_config(config)
    validate_budget(config.get("candidate_questions", 5120), config.get("selected_questions", 512),
                    config.get("exploration_questions", 64), config.get("responses_per_question", 5))
    # The native behavior identity contract and eight-rank actor budget are 512/G5.
    if config.get("selected_questions", 512) != 512:
        raise ValueError("Native question-probe runtime requires selected_questions=512")
    if config.get("post_rollout_em_filter", False) is not False:
        raise ValueError("Question-probe mode forbids post-rollout filtering or refill")
    if config.get("ranking", "expected_sample_std") != "expected_sample_std":
        raise ValueError("The implemented acquisition ranks expected sample std")
    if config.get("exploration_order", "uniform_first") != "uniform_first":
        raise ValueError("Question-probe exploration must be drawn uniformly before Top selection")
    if type(config.get("std_ddof", 1)) is not int or config.get("std_ddof", 1) != 1 or \
            config.get("target", "strict_terminal_em") != "strict_terminal_em":
        raise ValueError("Question-probe labels require strict terminal EM and sample std ddof=1")
    if type(config.get("update_every_completed_outer", 1)) is not int or config.get("update_every_completed_outer", 1) != 1:
        raise ValueError("The new count head must update every successful outer iteration")
    if type(config.get("seed", 42)) is not int:
        raise ValueError("Question-probe seed must be an integer")
    return config


def _prediction_cpu(prediction):
    if "probabilities" not in prediction:
        raise ValueError("Question-probe head omitted its six-count probabilities")
    return count_predictions(prediction["probabilities"])


class QProbeRuntime:
    def __init__(self, config, base_dir, tokenizer=None, *, feature_cache=None, head=None):
        self.config = validate_config(config)
        self.base = Path(base_dir).resolve()
        self.output_dir = Path(self.config.get("output_dir") or self.base / "artifacts/question-probe").resolve()
        if not self.output_dir.is_relative_to(self.base):
            raise ValueError("Question-probe output must be inside this task's run directory")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        if feature_cache is None:
            from search_g1.qprobe_features import FeatureCache
            cache_dir = self.config.get("cache_dir")
            if not cache_dir:
                raise ValueError("A prepared immutable remote question feature cache is required")
            feature_cache = FeatureCache(cache_dir)
        self.cache = feature_cache
        if not getattr(self.cache, "spec_sha", None) or not getattr(self.cache, "feature_dim", 0):
            raise ValueError("Feature cache lacks its immutable spec or feature dimension")
        self.feature_spec_sha = self.cache.spec_sha
        if head is None:
            from search_g1.qprobe_model import OnlineQProbe, QProbeConfig
            initial = self.config.get("initial_head_checkpoint")
            if initial:
                head = OnlineQProbe.load(initial, expected_feature_spec_sha=self.feature_spec_sha)
                expected_head_config = QProbeConfig.from_mapping(self.config)
                if head.config != expected_head_config:
                    raise ValueError("Initial question-probe head config differs from the effective runtime config")
            else:
                head = OnlineQProbe(self.cache.feature_dim, QProbeConfig.from_mapping(self.config),
                                   feature_spec_sha=self.feature_spec_sha)
        if getattr(head, "feature_spec_sha", self.feature_spec_sha) != self.feature_spec_sha:
            raise ValueError("Probe head belongs to a different feature namespace")
        if getattr(head, "feature_dim", self.cache.feature_dim) != self.cache.feature_dim:
            raise ValueError("Probe head and fixed feature cache dimensions differ")
        self.head = head
        self.selection_rng = np.random.default_rng(self.config.get("seed", 42))
        self.pending = None
        self.completed_outer = int(getattr(self.head, "completed_outer", 0))
        self.completed_attempts = set()

    @property
    def version(self):
        return self.head.version

    def score(self, questions):
        """A narrow qid/question API: no labels or complete DataProto enter the cache/head."""
        features = self.cache.get_features([q.qid for q in questions], [q.question for q in questions])
        if features.ndim != 2 or features.shape != (len(questions), self.cache.feature_dim):
            raise ValueError("Question cache returned the wrong feature layout")
        if not torch.isfinite(features).all():
            raise ValueError("Question features are nonfinite")
        frozen_version = self.version
        prediction = _prediction_cpu(self.head.predict(features))
        if self.version != frozen_version:
            raise RuntimeError("Probe changed during frozen question prediction")
        return features.detach().cpu(), prediction, frozen_version

    def begin_outer(self, *, outer, attempt_id, selected, features, probabilities,
                    prediction_version, policy_snapshot_id, contracts):
        if self.pending is not None:
            raise RuntimeError("Previous question-probe attempt has not been completed or aborted")
        if attempt_id in self.completed_attempts:
            raise RuntimeError("Question-probe attempt was already committed")
        self.pending = dict(outer=outer, attempt_id=attempt_id, selected=selected,
            features=features.detach().cpu().clone(), probabilities=np.asarray(probabilities).copy(),
            prediction_version=prediction_version, policy_snapshot_id=policy_snapshot_id,
            contracts=dict(contracts), labels=None, prequential=None)

    def observe_labels(self, labels, strict_vectors):
        if self.pending is None or self.pending["labels"] is not None:
            raise RuntimeError("Missing or repeated question-probe label transaction")
        if len(labels) != 512 or len(strict_vectors) != 512:
            raise ValueError("Incomplete selected question labels")
        sources = [label["selection_source"] for label in labels]
        measured = prequential_metrics(self.pending["probabilities"], strict_vectors, sources)
        # This is measured against the immutable pre-rollout predictions, before fit.
        self.pending["labels"] = copy.deepcopy(labels)
        self.pending["prequential"] = measured
        return measured

    def complete_outer(self, outer, actor_snapshot_id, attempt_identity=None):
        """Commit labels after the actual complete actor outer, never on partial Adam writes."""
        pending = self.pending
        if pending is None or pending["outer"] != outer or pending["labels"] is None:
            raise RuntimeError("No complete real rollout labels for this successful outer")
        if attempt_identity is not None and attempt_identity["attempt_id"] != pending["attempt_id"]:
            raise ValueError("Successful actor attempt differs from pending probe labels")
        if pending["attempt_id"] in self.completed_attempts or outer <= self.completed_outer:
            raise RuntimeError("Repeated or future/rewound question-probe outer commit")
        if not isinstance(actor_snapshot_id, str) or not actor_snapshot_id:
            raise ValueError("Successful actor snapshot identity is required")
        old_head = self.head.state_dict()
        try:
            result = self.head.update_completed_outer(outer, pending["labels"], pending["features"],
                policy_snapshot_id=pending["policy_snapshot_id"],
                sampling_contract_sha=pending["contracts"]["sampling_contract_sha"],
                strict_scoring_contract_sha=pending["contracts"]["strict_scoring_contract_sha"],
                retriever_version=pending["contracts"]["retriever_version"],
                run_id=pending["contracts"]["run_id"], attempt_id=pending["attempt_id"])
            event = dict(stage="question_probe_outer_committed", outer=outer,
                attempt_id=pending["attempt_id"], label_policy_snapshot_id=pending["policy_snapshot_id"],
                successful_actor_snapshot_id=actor_snapshot_id, prediction_version=pending["prediction_version"],
                head_version=self.version, feature_spec_sha=self.feature_spec_sha,
                prequential=pending["prequential"], head_update=result)
            _write_json(self.output_dir / "completed" / (pending["attempt_id"] + ".json"), event)
        except Exception:
            # Publication failure cannot leave a fitted-but-uncommitted future head.
            self.head.load_state_dict(old_head)
            raise
        self.completed_outer = outer
        self.completed_attempts.add(pending["attempt_id"])
        self.pending = None
        return event

    def abort_outer(self, outer, attempt_identity=None, reason="actor outer was incomplete"):
        if self.pending is None:
            return None
        pending = self.pending
        if pending["outer"] != outer or (attempt_identity is not None and
                pending["attempt_id"] != attempt_identity["attempt_id"]):
            raise ValueError("Aborted actor attempt differs from pending probe transaction")
        event = dict(stage="question_probe_attempt_aborted", outer=outer,
            attempt_id=pending["attempt_id"], reason=str(reason), head_updated=False,
            label_policy_snapshot_id=pending["policy_snapshot_id"], prediction_version=pending["prediction_version"],
            labels_observed=pending["labels"] is not None)
        _write_json(self.output_dir / "aborted" / (pending["attempt_id"] + ".json"), event)
        # A recoverable failed actor attempt keeps its real consumed source/RNG
        # and weights, just like the existing mixed-first retry. Do not rewind them.
        self.pending = None
        return event

    def save(self, path, actor_snapshot_id, candidate_state=None, reward_probe_version=None):
        if self.pending is not None:
            raise RuntimeError("Cannot publish a completed actor/head snapshot with pending labels")
        path = Path(path)
        head_path = path.with_name(path.name + ".head.pt")
        head_receipt = self.head.save(head_path, actor_snapshot_id=actor_snapshot_id)
        state = dict(schema_version=1, config=self.config, feature_spec_sha=self.feature_spec_sha,
            actor_snapshot_id=actor_snapshot_id, completed_outer=self.completed_outer,
            completed_attempts=sorted(self.completed_attempts), head_version=self.version,
            selection_rng=copy.deepcopy(self.selection_rng.bit_generator.state),
            head_file=head_path.name, head_sha256=_sha(head_path),
            candidate_state=candidate_state, reward_probe_version=reward_probe_version,
            head_receipt=head_receipt)
        receipt = _atomic_save(path, state)
        return dict(receipt, actor_snapshot_id=actor_snapshot_id, completed_outer=self.completed_outer,
                    feature_spec_sha=self.feature_spec_sha, head_version=self.version,
                    head=head_receipt)

    def save_checkpoint(self, directory, completed_outer, candidate_state, reward_probe_version,
                        actor_snapshot_id=None):
        if completed_outer != self.completed_outer:
            raise ValueError("Actor checkpoint and question-probe completed outer differ")
        actor_snapshot_id = actor_snapshot_id or "completed_outer:" + str(completed_outer)
        return self.save(Path(directory) / "question-probe-state.pt", actor_snapshot_id,
                         candidate_state, reward_probe_version)

    def restore(self, path, expected_actor_snapshot_id):
        if self.pending is not None:
            raise RuntimeError("Cannot restore over an active probe transaction")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("schema_version") != 1 or state["actor_snapshot_id"] != expected_actor_snapshot_id:
            raise ValueError("Question-probe snapshot does not belong to the restored actor")
        if state["feature_spec_sha"] != self.feature_spec_sha:
            raise ValueError("Question-probe restore feature namespace differs")
        # Runtime paths may change on a new node. Acquisition/training settings cannot.
        ignore = {"cache_dir", "output_dir", "initial_head_checkpoint", "snapshot_path", "restore_state"}
        old = {k: v for k, v in state["config"].items() if k not in ignore}
        new = {k: v for k, v in self.config.items() if k not in ignore}
        if old != new:
            raise ValueError("Question-probe configuration changed across exact restore")
        head_path = Path(path).parent / state["head_file"]
        if head_path.name != state["head_file"] or _sha(head_path) != state["head_sha256"]:
            raise ValueError("Question-probe head file/hash differs from paired snapshot")
        from search_g1.qprobe_model import OnlineQProbe
        head = OnlineQProbe.load(head_path, expected_actor_snapshot_id=expected_actor_snapshot_id,
                               expected_feature_spec_sha=self.feature_spec_sha)
        if head.version != state["head_version"] or int(head.completed_outer) != state["completed_outer"]:
            raise ValueError("Question-probe head state is ahead of/behind the paired actor")
        self.head = head
        self.completed_outer = state["completed_outer"]
        self.completed_attempts = set(state["completed_attempts"])
        self.selection_rng.bit_generator.state = state["selection_rng"]
        return state


class QProbeCandidateStream:
    """The existing resumable sampler, with tokenization only for selected rows.

    Composition keeps optional native dependencies lazy for CPU-only head use.
    Sampler order/cursor/epoch/consumed/pending fields have the exact existing
    CandidateStream state schema, including restoration of legacy pending rows.
    """
    def __init__(self, loader, restored_state=None):
        from search_g1.mixed_runtime import CandidateStream
        self._source = CandidateStream(loader, restored_state=restored_state)
        self.loader = loader

    @property
    def consumed(self):
        return self._source.consumed

    @property
    def order(self):
        return self._source.order

    @property
    def cursor(self):
        return self._source.cursor

    @property
    def epoch(self):
        return self._source.epoch

    @property
    def pending(self):
        return self._source.pending

    @property
    def offset(self):
        return self._source.offset

    def state(self):
        return self._source.state()

    def _fill(self):
        from verl import DataProto
        from search_g1.qprobe_features import original_question_and_id
        source = self._source
        if source.cursor == len(source.order):
            source.order = list(iter(self.loader.sampler))
            source.cursor, source.epoch = 0, source.epoch + 1
        indices = source.order[source.cursor:source.cursor + 512]
        source.cursor += len(indices)
        questions, ids = [], []
        for index in indices:
            original = self.loader.dataset.dataframe.iloc[index]
            question, qid = original_question_and_id(original, self.loader.dataset.prompt_key)
            questions.append(question)
            ids.append(qid)
        source.pending = DataProto.from_dict(
            tensors={"qprobe_source_index": torch.tensor(indices, dtype=torch.long)},
            non_tensors={"candidate_original_question": np.array(questions, dtype=object),
                         "candidate_stable_id": np.array(ids, dtype=object)})
        source.offset = 0

    def one(self):
        from verl import DataProto
        from search_g1.qprobe_features import original_question_and_id
        source = self._source
        if source.pending is None or source.offset == len(source.pending):
            self._fill()
        pending, offset = source.pending, source.offset
        if "qprobe_source_index" in pending.batch:
            index = int(pending.batch["qprobe_source_index"][offset])
        else:
            # A legacy checkpoint can contain a partly consumed encoded batch.
            # Its source indexes are the just-read slice of the restored order.
            begin = source.cursor - len(pending)
            if begin < 0 or begin + offset >= len(source.order):
                raise ValueError("Legacy pending candidate rows cannot be bound to source order")
            index = source.order[begin + offset]
        original = self.loader.dataset.dataframe.iloc[index]
        question, qid = original_question_and_id(original, self.loader.dataset.prompt_key)
        if str(pending.non_tensor_batch["candidate_stable_id"][offset]) != qid or \
                pending.non_tensor_batch["candidate_original_question"][offset] != question:
            raise ValueError("Restored candidate question differs from the actual training source")
        source.offset += 1
        source.consumed += 1
        return DataProto.from_dict(tensors={"qprobe_source_index": torch.tensor([index], dtype=torch.long)},
            non_tensors={"candidate_original_question": np.array([question], dtype=object),
                         "candidate_stable_id": np.array([qid], dtype=object)})

    def materialize(self, candidate_parents, ordinals):
        """Apply the unchanged native dataset tokenizer/collator to only 512 selections."""
        from verl import DataProto
        from search_g1.qprobe_features import original_question_and_id
        records = []
        for ordinal in ordinals:
            index = int(candidate_parents.batch["qprobe_source_index"][ordinal])
            actual_question, actual_qid = original_question_and_id(
                self.loader.dataset.dataframe.iloc[index], self.loader.dataset.prompt_key)
            if actual_question != candidate_parents.non_tensor_batch["candidate_original_question"][ordinal] or \
                    actual_qid != candidate_parents.non_tensor_batch["candidate_stable_id"][ordinal]:
                raise ValueError("Selected question changed between probe prediction and actor materialization")
            record = dict(self.loader.dataset[index])
            record["candidate_original_question"] = candidate_parents.non_tensor_batch["candidate_original_question"][ordinal]
            record["candidate_stable_id"] = candidate_parents.non_tensor_batch["candidate_stable_id"][ordinal]
            records.append(record)
        return DataProto.from_single_dict(self.loader.collate_fn(records))


def _collect_pool(stream, count):
    from verl import DataProto
    pool, rows = QuestionPool(count), []
    # The current partial epoch plus at most one full next epoch. At an epoch
    # boundary the two sampler permutations can overlap; a one-length cap would
    # incorrectly report exhaustion even for a fully unique source dataset.
    dataset_size = len(stream.loader.dataset)
    if dataset_size < count:
        raise RuntimeError("UNIQUE_QPROBE_CANDIDATE_SOURCE_EXHAUSTED")
    examined = 0
    while len(rows) < count:
        if examined >= 2 * dataset_size:
            raise RuntimeError("UNIQUE_QPROBE_CANDIDATE_SOURCE_EXHAUSTED")
        row = stream.one()
        examined += 1
        key = row.non_tensor_batch["candidate_stable_id"][0]
        question = row.non_tensor_batch["candidate_original_question"][0]
        ordinal = pool.reserve(key, question)
        if ordinal is not None:
            rows.append(row)
    return DataProto.concat(rows), pool.questions


def _complete_generation(output, identity):
    """Infrastructure/provenance failures cannot silently become five incorrect EMs.

    A completed model rollout that exhausts its action budget or emits an illegal
    answer is retained and scored zero by the original strict parser.
    """
    from search_g1.mixed_runtime import ROW_META
    expected_ids = identity["trajectory_ids"]
    if len(output) != 2560 or len(expected_ids) != 2560 or len(set(expected_ids)) != 2560:
        raise RuntimeError("INCOMPLETE_QPROBE_GENERATION")
    actual_ids = output.non_tensor_batch.get("behavior_trajectory_id")
    turns = output.non_tensor_batch.get("behavior_turns")
    if actual_ids is None or list(actual_ids) != expected_ids or turns is None or len(turns) != 2560 or not all(turns):
        raise RuntimeError("MISSING_OR_MISORDERED_REAL_QPROBE_TRAJECTORIES")
    if output.meta_info.get("policy_snapshot_id") != identity["policy_snapshot_id"]:
        raise RuntimeError("QPROBE_POLICY_SNAPSHOT_CHANGED_DURING_GENERATION")
    for key in ROW_META:
        if key not in output.meta_info or len(output.meta_info[key]) != 2560:
            raise RuntimeError("INCOMPLETE_QPROBE_TRAJECTORY_METADATA: " + key)
    for key in ("responses", "prompts", "attention_mask", "info_mask", "sampling_mask", "sampling_log_probs"):
        if key not in output.batch:
            raise RuntimeError("MISSING_QPROBE_GENERATION_TENSOR: " + key)
    response = output.batch["responses"]
    probabilities = output.batch["sampling_log_probs"]
    mask = output.batch["sampling_mask"]
    if response.shape != mask.shape or response.shape != probabilities.shape:
        raise RuntimeError("MALFORMED_QPROBE_SAMPLING_JOURNAL")
    if not torch.isfinite(probabilities[mask.bool()]).all():
        raise RuntimeError("NONFINITE_QPROBE_SAMPLING_JOURNAL")
    prompt_width = output.batch["prompts"].shape[-1]
    if output.batch["attention_mask"].shape[-1] != prompt_width + response.shape[-1] or \
            not torch.equal(mask.bool(), output.batch["info_mask"][:, -response.shape[-1]:].bool()):
        raise RuntimeError("MALFORMED_QPROBE_PROVENANCE_MASK")


def _retriever_contract(config, run):
    """Bind label meaning to the index and retrieval budget, not its endpoint.

    A paired restore may relocate an identical immutable index behind a new
    hostname or port. The URL remains observable transport evidence but must
    not turn those unchanged labels into a different head-training protocol.
    """
    index_version = run.get("retriever_version") or run.get("retriever_manifest_sha256")
    if not isinstance(index_version, str) or not index_version.strip():
        raise RuntimeError("Question-probe labels require a pinned retriever index version")
    topk = int(config.retriever.topk)
    if (run.get("full_state_restore") or {}).get("kind") in ("question_probe_pair","question_probe_saved_checkpoint") and \
            topk != run.get("retriever_topk", 3):
        raise RuntimeError("Paired question-probe restore changed the pinned retrieval topk")
    semantic = dict(topk=topk, recorded_index_version=index_version)
    return dict(retriever_version=_document_sha(semantic), retriever_contract=semantic,
                retriever_url=str(config.retriever.url))


def _contracts(trainer, output, run):
    from search_g1 import signals
    from search_r1 import qa_protocol
    from verl.utils.reward_score import qa_em
    if not output.meta_info.get("behavior_sampling"):
        raise RuntimeError("Question-probe labels lack the actual generation sampling contract")
    sampling = dict(output.meta_info["behavior_sampling"])
    config = trainer.config
    sampling.update(max_turns=config.max_turns,
        **{name: getattr(config.data, name, None) for name in
           ("max_start_length", "max_prompt_length", "max_response_length", "max_obs_length")})
    # Pin the scorer itself, rather than labelling shaped reward as strict EM.
    scoring = {name: _sha(module.__file__) for name, module in
               (("signals", signals), ("qa_protocol", qa_protocol), ("qa_em", qa_em))}
    scoring.update(target="strict_terminal_em", responses_per_question=5, std_ddof=1)
    scoring["common_data_manifest_sha256"] = run["budget"]["common_manifest_sha256"]
    return dict(run_id=run["run_id"], sampling_contract_sha=_document_sha(sampling),
        strict_scoring_contract_sha=_document_sha(scoring), sampling_contract=sampling,
        strict_scoring_contract=scoring, **_retriever_contract(config, run))


def collect(trainer, manager, stream, attempt_identity, runtime=None):
    """Return the same native repeated-parents/output/identity/audit contract as mixed-first."""
    from reproduction.common8.behavior_runtime import prepare_identity, attach_generation_identity
    from search_g1.mixed_runtime import subset, strict_vectors
    from search_g1.training_state import compact_stream_state, driver_rng

    base = Path(os.environ["SEARCH_G1_BASE"])
    if runtime is None:
        runtime = getattr(trainer, "qprobe_runtime", getattr(trainer, "question_probe_runtime", None))
    if runtime is None:
        raise RuntimeError("The trainer must own a paired question-probe runtime")
    if runtime.pending is not None:
        raise RuntimeError("Previous question-probe attempt was not resolved")
    outer = int(trainer.global_steps)
    tag = attempt_identity["attempt_id"] if attempt_identity is not None else str(outer)
    if not isinstance(tag, str) or not tag or Path(tag).name != tag:
        raise ValueError("Unsafe question-probe attempt identity")
    directory = runtime.output_dir / "attempts" / tag
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    before_selection_rng = copy.deepcopy(runtime.selection_rng.bit_generator.state)
    _atomic_save(directory / "before-selection.pt", dict(stream=compact_stream_state(stream.state()),
        driver_rng=driver_rng(), selection_rng=before_selection_rng, head_version=runtime.version,
        feature_spec_sha=runtime.feature_spec_sha))
    candidate_parents, questions = _collect_pool(stream, runtime.config.get("candidate_questions", 5120))
    features, predictions, prediction_version = runtime.score(questions)
    members = select_questions(questions, predictions["probabilities"],
        selected_questions=512, exploration_questions=runtime.config.get("exploration_questions", 64),
        rng=runtime.selection_rng, tie_seed=str(runtime.config.get("seed", 42)) + ":" + str(outer))
    ordinals = [m.ordinal for m in members]
    parent = stream.materialize(candidate_parents, ordinals) if isinstance(stream, QProbeCandidateStream) \
        else subset(candidate_parents, ordinals)
    parent.non_tensor_batch["selection_source"] = np.array([m.selection_source for m in members], dtype=object)
    identity = prepare_identity(parent, outer, repeats=5,
        **({k: attempt_identity[k] for k in ("attempt_id", "attempt_index")}
           if attempt_identity is not None else {}))
    chosen = set(ordinals)
    acquisition = dict(stage="question_probe_selection_frozen", step=outer,
        attempt_id=tag, feature_spec_sha=runtime.feature_spec_sha, head_version=prediction_version,
        policy_snapshot_id=identity["policy_snapshot_id"], attempted_candidates=len(questions),
        selected_groups=512, exploration_questions=runtime.config.get("exploration_questions", 64),
        prediction_before_label=True, post_rollout_em_filter=False, ranking="expected_sample_std",
        selection_rng_before=before_selection_rng,
        selection_rng_after=copy.deepcopy(runtime.selection_rng.bit_generator.state),
        questions=[dict(ordinal=q.ordinal, qid=q.qid, question_hash=q.question_hash,
            content_sha256=q.content_hash, probabilities=predictions["probabilities"][q.ordinal].tolist(),
            predicted_mean=float(predictions["mean"][q.ordinal]),
            predicted_std=float(predictions["expected_std"][q.ordinal]),
            predicted_variance=float(predictions["expected_variance"][q.ordinal]),
            predicted_mixed_probability=float(predictions["mixed_probability"][q.ordinal]),
            selected=q.ordinal in chosen,
            selection_source=next((m.selection_source for m in members if m.ordinal == q.ordinal), "unselected"))
            for q in questions])
    # Durable prediction/selection evidence exists before the first real generated token.
    _write_json(directory / "selection.json", acquisition)
    selected_features = features[ordinals].clone()
    selected_probabilities = predictions["probabilities"][ordinals].copy()
    repeated = parent.repeat(repeat_times=5, interleave=True)
    gen = repeated.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
    attach_generation_identity(gen, identity, do_sample=True)
    before_workers = trainer.actor_rollout_wg.capture_mixed_state(str(directory / "before-generation-worker-rng"))
    _atomic_save(directory / "before-generation.pt", dict(stream=compact_stream_state(stream.state()),
        driver_rng=driver_rng(), selection_rng=copy.deepcopy(runtime.selection_rng.bit_generator.state),
        identity=identity, parents=parent, selected_features=selected_features,
        head_version=prediction_version, worker_files=before_workers))
    manager.timing_raw = {}
    output = manager.run_llm_loop(gen_batch=gen,
        initial_input_ids=gen.batch["input_ids"][:, -manager.config.max_start_length:].clone().long())
    _complete_generation(output, identity)
    after_workers = trainer.actor_rollout_wg.capture_mixed_state(str(directory / "after-generation-worker-rng"))
    _atomic_save(directory / "after-generation.pt", dict(stream=compact_stream_state(stream.state()),
        driver_rng=driver_rng(), selection_rng=copy.deepcopy(runtime.selection_rng.bit_generator.state),
        identity=identity, worker_files=after_workers))
    vectors = strict_vectors(parent, output, trainer.tokenizer)
    if len(vectors) != 512 or any(len(v) != 5 or any(type(x) is not int or x not in (0, 1) for x in v) for v in vectors):
        raise RuntimeError("INCOMPLETE_STRICT_QPROBE_LABEL_GROUPS")
    run = json.loads((base / "logs/run-manifest.json").read_text())
    contracts = _contracts(trainer, output, run)
    labels = [dict(complete=True, qid=questions[m.ordinal].qid, question_hash=questions[m.ordinal].question_hash,
        strict_em=values, selection_source=m.selection_source)
        for m, values in zip(members, vectors)]
    runtime.begin_outer(outer=outer, attempt_id=tag, selected=ordinals, features=selected_features,
        probabilities=selected_probabilities, prediction_version=prediction_version,
        policy_snapshot_id=identity["policy_snapshot_id"], contracts=contracts)
    measured = runtime.observe_labels(labels, vectors)
    _atomic_save(directory / "real-rollout.pt", dict(parents=parent, output=output, identity=identity,
        selected_ordinals=ordinals, strict_correctness=vectors, probe_transaction=runtime.pending,
        stream=compact_stream_state(stream.state()), selection_rng=copy.deepcopy(runtime.selection_rng.bit_generator.state)))
    categories = ["all_wrong" if sum(v) == 0 else "all_correct" if sum(v) == 5 else "mixed" for v in vectors]
    audit = dict(selection_mode="question_probe", attempted_candidates=len(questions), selected_groups=512,
        selected_mixed=categories.count("mixed"), selected_all_correct=categories.count("all_correct"),
        selected_all_wrong=categories.count("all_wrong"), selected_all_correct_fallback=0,
        selected_random_exploration=runtime.config.get("exploration_questions", 64),
        selected_predicted_diversity=512-runtime.config.get("exploration_questions", 64),
        responses_per_group=5, raw_generated_trajectories=2560,
        raw_generated_policy_tokens=int(output.batch["sampling_mask"].sum().item()),
        raw_actual_retrieval_calls=sum(output.meta_info["actual_retrieval_stats"]),
        candidate_fill_seconds=time.monotonic()-started,
        candidate_generation_timings={k: float(v) for k, v in manager.timing_raw.items()},
        step=outer, attempt_id=tag, policy_snapshot_id=identity["policy_snapshot_id"],
        source_consumed=stream.consumed, source_commit=run["source_commit"],
        model_revision=run["model_revision"], data_manifest_sha256=run["budget"]["common_manifest_sha256"],
        frozen_probe_version=getattr(trainer.reward_fn, "expected_probe_version", None),
        question_probe_version=prediction_version, feature_spec_sha=runtime.feature_spec_sha,
        no_post_rollout_refill=True, prequential=measured,
        groups=[dict(ordinal=m.ordinal, question_id=m.qid, content_sha256=questions[m.ordinal].content_hash,
            trajectory_ids=identity["trajectory_ids"][i*5:(i+1)*5], strict_correctness=vectors[i],
            category=categories[i], selected=True, source=m.selection_source)
            for i, m in enumerate(members)])
    # Keep the semantic hashes and their exact payloads alongside transport
    # evidence. All 5120 candidates were predicted, but only these 512 selected
    # groups have actual labels; the audit never presents them as an unbiased
    # all-candidate bootstrap corpus.
    audit.update(contracts)
    _write_json(directory / "rollout-audit.json", audit)
    # No reordering/subsetting of generated rows: all 512/G5 groups, including K=0,
    # remain in the native actor batch with their actual sampling journal intact.
    output.meta_info["behavior_identity"] = identity
    # Do not synthesize the legacy selection_fallback bit: its optional actor
    # diagnostic calls every False row "mixed", which is false for K=0/K=5
    # question-probe groups. The original PPO loss never depends on that bit.
    print(json.dumps(dict(stage="question_probe_collection_complete", step=outer,
        candidates=len(questions), actual_rollout_questions=512, actual_rollout_trajectories=2560,
        probe_version=prediction_version)), flush=True)
    return repeated, output, identity, audit
