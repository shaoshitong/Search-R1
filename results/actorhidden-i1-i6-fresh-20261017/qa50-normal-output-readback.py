"""CPU-only normal QA50 output/identity/scoring readback; no generation."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time


def inspect(base):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("This normal output readback is CPU-only")
    from reproduction.prefix_selection200.eval import load_questions, score_tokens, CONTRACT
    from transformers import AutoTokenizer
    base = Path(base)
    proof = base / "preparation/qa0050-original-output-independent-readback-once-v1.json"
    if proof.exists():
        return dict(status="EXISTING_QA_PROOF_NO_REPLAY", path=str(proof),
                    sha256=hashlib.sha256(proof.read_bytes()).hexdigest(), proof=json.loads(proof.read_text()))
    read = lambda path: json.loads(Path(path).read_text())
    config = read(base / "preparation/launch-config.json")
    root = base / "outers/outer-000050"
    complete = read(root / "validation/complete.json")
    joint = read(root / "complete.json")
    assert complete["run_id"] == joint["run_id"] == config["run_id"]
    assert complete["actor_version"] == joint["actor"]["actor_version"]
    assert complete["actor_model_dir"] == joint["actor"]["hf_model_dir"]
    assert complete["qa_count"] == 512 and complete["evaluation_contract"] == CONTRACT
    assert complete["actual_worker_exit_codes"] == {str(rank): 0 for rank in range(8)}
    source = Path(config["validation_questions"])
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_sha == complete["question_source_sha256"] == "3d6318365b5bb98039f1e28ca6bc7d1de98eccc5cc833600e39ce1c2306f9ceb"
    canonical = load_questions(source, 512)
    tokenizer = AutoTokenizer.from_pretrained(complete["actor_model_dir"], local_files_only=True)
    records, files = [], []
    for rank, reference in enumerate(complete["rank_receipts"]):
        path = Path(reference["path"])
        body = path.read_bytes()
        assert hashlib.sha256(body).hexdigest() == reference["sha256"]
        receipt = json.loads(body)
        assert receipt["rank"] == rank and receipt["run_id"] == config["run_id"]
        assert receipt["actor_version"] == complete["actor_version"]
        assert receipt["actual_compute_dtype"] == "torch.bfloat16"
        assert receipt["evaluation_contract"] == CONTRACT and receipt["qa_count"] == 64
        shard = receipt["full_output"]; payload = Path(shard["path"])
        raw = payload.read_bytes()
        assert len(raw) == shard["bytes"] and hashlib.sha256(raw).hexdigest() == shard["sha256"]
        rows = [json.loads(line) for line in gzip.decompress(raw).splitlines()]
        assert len(rows) == len(receipt["rows"]) == 64
        for row, recorded in zip(rows, receipt["rows"]):
            assert all(row[key] == value for key, value in recorded.items())
            question = canonical[row["evaluation_ordinal"]]
            assert (row["qid"], row["data_source"]) == (question["qid"], question["data_source"])
            assert row["evaluation_ordinal"] % 8 == rank
            score = score_tokens(tokenizer, row["initial_prompt_token_ids"], row["response_token_ids"],
                                 row["response_info_mask"], question["reward_model"]["ground_truth"]["target"])
            assert score["em"] == row["em"] and math.isclose(score["f1"], row["f1"], abs_tol=1e-12)
            records.append(recorded)
        files.extend([dict(path=str(path), sha256=reference["sha256"], bytes=len(body)),
                      dict(path=str(payload), sha256=shard["sha256"], bytes=len(raw))])
    records.sort(key=lambda row: row["evaluation_ordinal"])
    assert [row["evaluation_ordinal"] for row in records] == list(range(512))
    assert len({row["qid"] for row in records}) == 512
    assert Counter(row["data_source"] for row in records) == dict(nq=256, hotpotqa=256)
    em, f1 = (sum(row[key] for row in records) / 512 for key in ("em", "f1"))
    assert em == complete["em"] and math.isclose(f1, complete["f1"], abs_tol=1e-12)
    result = dict(status="QA50_ORIGINAL_OUTPUT_IDENTITY_SCORE_AGGREGATE_READBACK_PASS",
        run_id=config["run_id"], actor_version=complete["actor_version"], qa_count=512,
        correct=sum(row["em"] for row in records), em=em, f1=f1,
        original_receipts_and_outputs_full_sha=files, question_source_sha256=source_sha,
        canonical_qids_ordinals_sources_verified=True, actual_worker_exit_codes=complete["actual_worker_exit_codes"],
        strict_scores_recomputed_from_actual_tokens_and_original_gold=True,
        gpu_forward_performed=False, new_generation_performed=False,
        full_inputs_or_outputs_transferred_to_Mac=False, checked_unix=time.time(),
        control_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with proof.open("x") as stream:
        json.dump(result, stream, sort_keys=True); stream.flush(); os.fsync(stream.fileno())
    return dict(path=str(proof), sha256=hashlib.sha256(proof.read_bytes()).hexdigest(), proof=result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--base", required=True)
    print(json.dumps(inspect(parser.parse_args().base)))
