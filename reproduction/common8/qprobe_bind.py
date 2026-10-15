"""Prospectively opt a new, unlaunched common8 run into question-probe selection."""
import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from reproduction.common8.qprobe_configuration import arguments
from reproduction.common8.source_files import runtime_files


def bind(base, cache_dir, *, exploration_questions=64, retriever_version,
         initial_head_checkpoint=None, seed=20261013, teacher_warmup_cache=None, head_overrides=None):
    base = Path(base).resolve()
    manifest = base / "logs/run-manifest.json"
    run = json.loads(manifest.read_text())
    if run.get("completed_updates") != 0 or run.get("profile") != "common8":
        raise ValueError("Only a prospectively prepared common8 run can be bound")
    forbidden = ("training-launch-claim.json", "actual-attempts.jsonl",
                 "actual-updates.jsonl", "resolved-training-config.json",
                 "policy-safety-calibration.json")
    if any((base / "logs" / name).exists() for name in forbidden):
        raise ValueError("Cannot change a claimed, admitted or already trained run")
    if run.get("question_probe") is not None:
        raise ValueError("Question-probe configuration is already immutable for this run")
    if run.get("full_state_restore"):
        raise ValueError("A restored run must use an exact actor/head paired receipt")
    if not isinstance(retriever_version, str) or not retriever_version.strip():
        raise ValueError("An immutable retrieval-index version/manifest hash is required")
    repo = base / "repo"
    if not repo.is_dir():
        raise ValueError("Stage this branch into the new run's repo before binding")
    cache = Path(cache_dir).resolve()
    from search_g1.qprobe_features import FeatureCache
    from search_g1.qprobe_model import OnlineQProbe, QProbeConfig
    features = FeatureCache(cache)
    features.verify_all()
    if features.manifest['count'] != 169028 or features.spec.get('train_parquet_sha256') != run['input_sha256']['train.parquet']:
        raise ValueError("Question feature cache does not cover the verified canonical training artifact")
    if teacher_warmup_cache and initial_head_checkpoint:
        raise ValueError('Teacher warmup and historical initial head are distinct initialization routes')
    overrides = {} if head_overrides is None else dict(head_overrides)
    if set(overrides) - {'loss_mode', 'std_loss_weight', 'mean_loss_weight', 'auxiliary_count_loss_weight'}:
        raise ValueError('Only explicit loss controls may be overridden')
    head_config = QProbeConfig(seed=seed, bootstrap_max_groups_per_outer=5120 if teacher_warmup_cache else 512,
                               **overrides)
    settings = dict(cache_dir=str(cache), output_dir=str(base / "artifacts/question-probe"),
                    candidate_questions=5120, selected_questions=512,
                    responses_per_question=5, exploration_questions=exploration_questions,
                    seed=seed)
    settings.update({("head_hidden_size" if key == "hidden_size" else key): value
                     for key,value in asdict(head_config).items()
                     if key != "bootstrap_max_groups_per_outer" or teacher_warmup_cache})
    if teacher_warmup_cache:
        path = Path(teacher_warmup_cache)
        if not path.is_absolute() or not str(path.resolve()).startswith(('/data/', '/mnt/', '/backup/')):
            raise ValueError('Teacher warmup cache must reside on explicit remote storage')
        settings['teacher_warmup_cache'] = str(path.resolve())
    if initial_head_checkpoint:
        initial_path = Path(initial_head_checkpoint).resolve()
        head = OnlineQProbe.load(initial_path, expected_feature_spec_sha=features.spec_sha)
        if head.completed_outer != 0:
            raise ValueError("A fresh actor requires a head at outer 0; later actors require a completed paired receipt")
        if head.config != head_config:
            raise ValueError("Bootstrap head configuration differs from the effective runtime configuration")
        settings["initial_head_checkpoint"] = str(initial_path)
    proposed = dict(run, question_probe=settings, retriever_version=retriever_version,
                    mixed_first_selection=False)
    arguments(proposed, base)
    current_source = {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in runtime_files(repo)
    }
    if run.get("source_sha256") != current_source:
        raise ValueError("Prepared source differs from staged branch; prepare a new run from this branch before binding")
    proposed["source_sha256"] = current_source
    original = manifest.read_bytes()
    backup = base / "preparation/question-probe-pre-bind.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    with backup.open("xb") as stream:
        stream.write(original); stream.flush(); os.fsync(stream.fileno())
    temp = manifest.with_name("run-manifest.question-probe.partial")
    with temp.open("x") as stream:
        json.dump(proposed, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, manifest)
    descriptor = os.open(manifest.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return proposed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--exploration-questions", type=int, choices=(0, 64, 128, 256), default=64)
    parser.add_argument("--retriever-version", required=True)
    parser.add_argument("--initial-head-checkpoint")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument('--teacher-warmup-cache')
    args = parser.parse_args()
    run = bind(args.base, args.cache_dir, exploration_questions=args.exploration_questions,
               retriever_version=args.retriever_version,
               initial_head_checkpoint=args.initial_head_checkpoint, seed=args.seed,
               teacher_warmup_cache=args.teacher_warmup_cache)
    print(json.dumps({"run_id": run["run_id"], "selection_mode": "question_probe",
                      "question_probe": run["question_probe"], "training_started": False}))


if __name__ == "__main__":
    main()
