"""Offline remote preparation; no training, actor changes or agentic rollouts.

Examples (all asset/output paths must be remote, not the controller Mac)::

  python -m search_g1.qprobe_prepare precompute --data /shared/run/data \
    --model /shared/official-qwen --official-manifest /shared/official-files.json \
    --cache-dir /shared/newrun/qprobe/features --device cuda --batch-size 4
  python -m search_g1.qprobe_prepare bootstrap --data /shared/run/data \
    --cache-dir /shared/newrun/qprobe/features --audit old/audit1.json old/audit2.json \
    --provenance /shared/lineage.json --output /shared/newrun/qprobe/initial.pt

Bootstrap provenance is explicit, not inferred from a filename or conversation.
It binds actor_outer/actor_snapshot_id, run_id, source_commit,
sampling_contract_sha, strict_scoring_contract_sha, retriever_version and
source_policies [{source_outer, policy_snapshot_id, audit_sha256}]. An audit must
contain every generated candidate, including excluded K=0 and K=5 groups.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from search_g1.qprobe_features import (
    COMMON_MANIFEST_SHA, OFFICIAL_ENCODER, OFFICIAL_REVISION, QUESTION_EXTRACTION_RULE,
    FeatureCache, FeatureCacheWriter, extract_question_features, original_question_and_id,
    question_text_sha, sha_file, sha_json,
)
from search_g1.qprobe_model import OnlineQProbe, QProbeConfig


def train_questions(data_dir, prompt_key='prompt'):
    # Verify existing canonical artifacts, rather than constructing a split by
    # row count. Scoring answers are never returned to encoder or prediction.
    from reproduction.common8.verify_data import verify
    import pyarrow.parquet as pq
    proof = verify(Path(data_dir))
    table = pq.read_table(Path(data_dir)/'train.parquet', columns=[prompt_key, 'data_source', 'extra_info'])
    rows = [original_question_and_id(row, prompt_key) for row in table.to_pylist()]
    ids = [qid for _question, qid in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('QPROBE_CANONICAL_TRAIN_HAS_DUPLICATE_STABLE_IDS')
    return rows, proof


def precompute(args):
    from reproduction.common8.prepare_run import verify_model
    from transformers import AutoModel, AutoTokenizer
    rows, data_proof = train_questions(args.data, args.prompt_key)
    model_dir = Path(args.model)
    # Verify both complete official-file SHA/LFS records and revision before
    # loading. local_files_only prevents accidental model downloads or caches.
    hashes = verify_model(model_dir, Path(args.official_manifest))
    config = json.loads((model_dir/'config.json').read_text())
    if args.feature_layer is None:
        args.feature_layer = config['num_hidden_layers']-1 if getattr(args,'sequence_probe',False) else 27
    if not 0 <= args.feature_layer < config['num_hidden_layers']:
        raise ValueError('QPROBE_FEATURE_LAYER_OUT_OF_RANGE')
    template = dict(question_extraction=QUESTION_EXTRACTION_RULE,
                    prompt_key=args.prompt_key, added_template=None,
                    tokenizer_add_special_tokens=True, truncation=False)
    tokenizer_files = {name:sha for name, sha in hashes.items()
                       if name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')}
    spec = dict(encoder_id=OFFICIAL_ENCODER, encoder_revision=OFFICIAL_REVISION,
                encoder_files_sha256=sha_json(hashes), tokenizer_sha256=sha_json(tokenizer_files),
                prompt_template_sha256=sha_json(template), feature_layer=args.feature_layer,
                pooling='last_nonpadding_prompt_token', feature_dtype='float16',
                feature_dim=int(config['hidden_size']), max_prompt_tokens=args.max_prompt_tokens,
                question_extraction=QUESTION_EXTRACTION_RULE, split='train',
                train_manifest_sha256=COMMON_MANIFEST_SHA,
                train_parquet_sha256=data_proof['products']['train']['sha256'],
                official_manifest_sha256=sha_file(args.official_manifest), prompt_spec=template)
    # FP32 CPU and BF16 CUDA extraction can differ numerically even after FP16
    # storage. They must not share a feature/head namespace silently.
    spec['encoder_compute_dtype'] = 'float32' if args.device == 'cpu' else 'bfloat16'
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        raise ValueError('QPROBE_TOKENIZER_MISSING_PAD_TOKEN')
    dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
                                     torch_dtype=dtype).to(args.device)
    if getattr(args, 'sequence_probe', False):
        from search_g1.qprobe_sequence import SequenceCacheWriter, extract_question_sequences
        if args.feature_layer != config['num_hidden_layers']-1:
            raise ValueError('QPROBE_SEQUENCE_REQUIRES_LAST_LAYER')
        spec.update(representation='frozen_token_sequence_v2', pooling='all_nonpadding_prompt_tokens')
        writer = SequenceCacheWriter(args.cache_dir, spec, len(rows))
        # Copy the actual official decoder layer, not random config initialization.
        block = model.layers[-1]
        source_state = {k:v.detach().cpu().clone() for k,v in block.state_dict().items()}
        norm_state = {k:v.detach().cpu().clone() for k,v in model.norm.state_dict().items()}
        from search_g1.qprobe_transformer import state_digest
        initialization = dict(schema_version=2, architecture='qwen2_decoder3_count6',
            official_revision=OFFICIAL_REVISION, encoder_files_sha256=spec['encoder_files_sha256'],
            qwen_config=model.config.to_dict(), attention_implementation=model.config._attn_implementation, source_layer_index=args.feature_layer,
            source_layer_state=source_state, source_layer_sha256=state_digest(block),
            source_norm_state=norm_state, source_norm_sha256=state_digest(model.norm))
        torch.save(initialization, writer.partial/'initialization.pt')
        extract = extract_question_sequences
    else:
        writer = FeatureCacheWriter(args.cache_dir, spec, len(rows))
        extract = extract_question_features
    started = time.monotonic()
    try:
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset+args.batch_size]
            questions = [q for q, _qid in batch]; qids = [qid for _q, qid in batch]
            features = extract(model, tokenizer, questions, feature_layer=args.feature_layer,
                max_prompt_tokens=args.max_prompt_tokens, device=args.device)
            writer.append(qids, questions, features)
            print(json.dumps(dict(stage='question_feature_precompute', completed=writer.written,
                                  total=len(rows), elapsed_seconds=time.monotonic()-started)), flush=True)
        manifest = writer.finish()
    except Exception:
        # Only this command's incomplete feature staging directory is removed;
        # source model, canonical data and other tasks are untouched.
        writer.abort(); raise
    if getattr(args, "sequence_probe", False):
        from search_g1.qprobe_sequence import SequenceCache
        verification = SequenceCache(args.cache_dir).verify_all()
    else:
        verification = FeatureCache(args.cache_dir).verify_all()
    return dict(stage='question_feature_cache_complete', cache_dir=args.cache_dir,
                manifest_sha256=sha_file(Path(args.cache_dir)/'manifest.json'),
                elapsed_seconds=time.monotonic()-started, **verification)


def bootstrap_labels(audit_paths, provenance, question_by_id):
    required = ('actor_outer', 'actor_snapshot_id', 'run_id', 'source_commit', 'sampling_contract_sha',
                'strict_scoring_contract_sha', 'retriever_version', 'source_policies')
    if any(key not in provenance for key in required):
        raise ValueError('QPROBE_BOOTSTRAP_LINEAGE_PROOF_INCOMPLETE')
    if type(provenance['actor_outer']) is not int or provenance['actor_outer'] < 0:
        raise ValueError('QPROBE_BOOTSTRAP_BAD_ACTOR_OUTER')
    for key in required[1:-1]:
        if not isinstance(provenance[key], str) or not provenance[key]:
            raise ValueError('QPROBE_BOOTSTRAP_BAD_LINEAGE_VALUE: ' + key)
    proofs = {}
    for item in provenance['source_policies']:
        key = item['audit_sha256']
        if key in proofs: raise ValueError('QPROBE_DUPLICATE_LINEAGE_AUDIT')
        if type(item['source_outer']) is not int or not 0 <= item['source_outer'] <= provenance['actor_outer']:
            raise ValueError('QPROBE_BOOTSTRAP_FUTURE_LABELS')
        proofs[key] = item
    labels, questions, qids = [], [], []
    consumed_proofs = set()
    for path in audit_paths:
        digest = sha_file(path)
        if digest not in proofs: raise ValueError('QPROBE_AUDIT_NOT_IN_ACTOR_LINEAGE_PROOF')
        if digest in consumed_proofs: raise ValueError('QPROBE_DUPLICATE_BOOTSTRAP_AUDIT')
        consumed_proofs.add(digest); proof = proofs[digest]
        audit = json.loads(Path(path).read_text())
        for key in ('sampling_contract_sha','strict_scoring_contract_sha','retriever_version'):
            if audit.get(key) != provenance[key]:
                raise ValueError('QPROBE_BOOTSTRAP_AUDITED_CONTRACT_MISSING_OR_MISMATCH: ' + key)
        if audit.get('source_commit') != provenance['source_commit']:
            raise ValueError('QPROBE_BOOTSTRAP_SOURCE_COMMIT_MISMATCH')
        if audit.get('model_revision') != OFFICIAL_REVISION or audit.get('data_manifest_sha256') != COMMON_MANIFEST_SHA:
            raise ValueError('QPROBE_BOOTSTRAP_MODEL_OR_DATA_MISMATCH')
        if audit.get('step') != proof['source_outer'] or audit.get('policy_snapshot_id') != proof['policy_snapshot_id']:
            raise ValueError('QPROBE_BOOTSTRAP_SAMPLING_ACTOR_MISMATCH')
        groups = audit.get('groups', [])
        if (not groups or audit.get('responses_per_group') != 5 or
            audit.get('attempted_candidates') != len(groups) or
            audit.get('raw_generated_trajectories') != 5*len(groups)):
            raise ValueError('QPROBE_BOOTSTRAP_REQUIRES_COMPLETE_ALL_CANDIDATE_POOL')
        if sorted(x['ordinal'] for x in groups) != list(range(len(groups))):
            raise ValueError('QPROBE_BOOTSTRAP_INCOMPLETE_OR_DUPLICATE_ORDINALS')
        attempt_id = Path(path).stem
        seen_qids, seen_trajectories = set(), set()
        for group in groups:
            qid = group['question_id']
            if qid in seen_qids or qid not in question_by_id:
                raise ValueError('QPROBE_BOOTSTRAP_QID_NOT_UNIQUE_CANONICAL_TRAIN')
            seen_qids.add(qid); question = question_by_id[qid]
            from search_g1.mixed_selection import question_identity
            if group.get('content_sha256') != question_identity(question):
                raise ValueError('QPROBE_BOOTSTRAP_ORIGINAL_QUESTION_MISMATCH')
            trajectory_ids = group.get('trajectory_ids', [])
            if len(trajectory_ids) != 5 or len(set(trajectory_ids)) != 5 or any(x in seen_trajectories for x in trajectory_ids):
                raise ValueError('QPROBE_BOOTSTRAP_INCOMPLETE_OR_DUPLICATE_TRAJECTORIES')
            seen_trajectories.update(trajectory_ids)
            em = group.get('strict_correctness')
            if not isinstance(em, list) or len(em) != 5 or any(type(x) is not int or x not in (0,1) for x in em):
                raise ValueError('QPROBE_BOOTSTRAP_MISSING_REAL_STRICT_EM_VECTOR')
            k = sum(em)
            category = 'all_wrong' if k == 0 else 'all_correct' if k == 5 else 'mixed'
            if group.get('category') != category:
                raise ValueError('QPROBE_BOOTSTRAP_CATEGORY_LABEL_MISMATCH')
            labels.append(dict(qid=qid, question_hash=question_text_sha(question), strict_em=em,
                correct_count=k, complete=True, selection_source='bootstrap_all_candidate',
                run_id=provenance['run_id'], attempt_id=attempt_id, source_outer=proof['source_outer'],
                policy_snapshot_id=proof['policy_snapshot_id'],
                sampling_contract_sha=provenance['sampling_contract_sha'],
                strict_scoring_contract_sha=provenance['strict_scoring_contract_sha'],
                retriever_version=provenance['retriever_version']))
            qids.append(qid); questions.append(question)
    if consumed_proofs != set(proofs):
        raise ValueError('QPROBE_LINEAGE_PROOF_REFERENCES_MISSING_AUDIT')
    return labels, qids, questions


def bootstrap(args):
    rows, _proof = train_questions(args.data, args.prompt_key)
    manifest = json.loads((Path(args.cache_dir)/'manifest.json').read_text())
    if manifest.get('schema_version') == 2:
        raise ValueError('QPROBE_SEQUENCE_BOOTSTRAP_REQUIRES_SEQUENCE_ONLINE_HEAD')
    else:
        cache = FeatureCache(args.cache_dir)
    provenance = json.loads(Path(args.provenance).read_text())
    labels, qids, questions = bootstrap_labels(args.audit, provenance, {qid:q for q,qid in rows})
    config = QProbeConfig.from_mapping(json.loads(Path(args.config).read_text()) if args.config else {})
    probe = OnlineQProbe(cache.feature_dim, config, feature_spec_sha=cache.spec_sha)
    metrics = probe.bootstrap(labels, cache.get_features(qids, questions),
                              actor_outer=provenance['actor_outer'], actor_snapshot_id=provenance['actor_snapshot_id'])
    receipt = probe.save(args.output, actor_snapshot_id=provenance['actor_snapshot_id'])
    # Read back the actual bytes/binding before reporting ready.
    reloaded = OnlineQProbe.load(args.output, expected_actor_snapshot_id=provenance['actor_snapshot_id'],
                                expected_feature_spec_sha=cache.spec_sha,
                                expected_checkpoint_sha=receipt['checkpoint_sha'])
    if reloaded.version != probe.version: raise ValueError('QPROBE_BOOTSTRAP_READBACK_MISMATCH')
    result = dict(stage='question_probe_bootstrap_complete', provenance_sha256=sha_file(args.provenance),
                  complete_historical_groups=len(labels), metrics=metrics, checkpoint=receipt)
    with open(str(args.output)+'.receipt.json', 'x') as f:
        json.dump(result, f, indent=2); f.flush(); os.fsync(f.fileno())
    return result


def parser():
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = root.add_subparsers(dest='mode', required=True)
    make = modes.add_parser('precompute', help='Encode canonical train questions once; no rollout')
    make.add_argument('--data', required=True); make.add_argument('--model', required=True)
    make.add_argument('--official-manifest', required=True); make.add_argument('--cache-dir', required=True)
    make.add_argument('--sequence-probe', action='store_true', help='Store frozen full last-layer tokens and actual official decoder initialization')
    make.add_argument('--device', default='cpu'); make.add_argument('--batch-size', type=int, default=4)
    make.add_argument('--feature-layer', type=int, default=None)
    make.add_argument('--max-prompt-tokens', type=int, default=4096)
    make.add_argument('--prompt-key', default='prompt'); make.set_defaults(func=precompute)
    boot = modes.add_parser('bootstrap', help='Initialize head from complete all-candidate strict labels')
    boot.add_argument('--data', required=True); boot.add_argument('--cache-dir', required=True)
    boot.add_argument('--audit', nargs='+', required=True); boot.add_argument('--provenance', required=True)
    boot.add_argument('--output', required=True); boot.add_argument('--config')
    boot.add_argument('--prompt-key', default='prompt'); boot.set_defaults(func=bootstrap)
    check = modes.add_parser('verify-cache', help='Read back complete feature SHA and finite values')
    check.add_argument('--cache-dir', required=True)
    check.set_defaults(func=lambda args: FeatureCache(args.cache_dir).verify_all())
    return root


def main():
    args = parser().parse_args()
    if getattr(args, 'batch_size', 1) <= 0: raise ValueError('QPROBE_NONPOSITIVE_PRECOMPUTE_BATCH')
    print(json.dumps(args.func(args), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
