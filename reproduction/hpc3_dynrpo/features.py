"""Resumable all-token extraction using the release's unchanged encoder hook."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from .common import BASE, DATA, MODEL, own, read, sha, publish, progress, teacher_labels, normalize_preparation_gpu


def durable_cursor(writer):
    from search_g1.qprobe_features import sha_json
    writer.packed.flush()
    writer.index.flush(); os.fsync(writer.index.fileno())
    with (writer.partial/'packed_features.npy').open('rb') as f:
        os.fsync(f.fileno())
    publish(BASE/'preparation/feature-cursor.json', dict(
        partial=str(writer.partial), written=writer.written, index_bytes=writer.index.tell(),
        spec_sha=sha_json(writer.spec),
        checked_unix=time.time()))


def reopen_writer(path, spec, records):
    import numpy as np
    from search_g1.qprobe_token_features import TokenFeatureCacheWriter
    from search_g1.qprobe_features import sha_json
    cursor = BASE/'preparation/feature-cursor.json'
    if not cursor.exists():
        if list(path.parent.glob(path.name+'.partial-*')):
            raise ValueError('Unclaimed partial bank exists; inspect before resuming')
        result = TokenFeatureCacheWriter(path, spec, records)
        durable_cursor(result)
        return result
    c = read(cursor); partial = Path(c['partial'])
    if partial.parent != path.parent or not partial.name.startswith(path.name+'.partial-') or c['spec_sha'] != sha_json(spec):
        raise ValueError('Feature cursor does not belong to this bank')
    with (partial/'index.jsonl').open('r+b') as f:
        f.truncate(c['index_bytes'])
    accepted = [json.loads(x) for x in (partial/'index.jsonl').read_text().splitlines()]
    if len(accepted) != c['written'] or any(row['qid'] != records[n]['qid'] for n, row in enumerate(accepted)):
        raise ValueError('Committed bank prefix changed')
    result = TokenFeatureCacheWriter.__new__(TokenFeatureCacheWriter)
    result.path, result.spec, result.records = path, spec, records
    result.partial, result.written = partial, c['written']
    result.offsets = np.load(partial/'offsets.npy', allow_pickle=False)
    result.packed = np.lib.format.open_memmap(partial/'packed_features.npy', mode='r+')
    result.index = (partial/'index.jsonl').open('a', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args(); own(BASE)
    gpu = normalize_preparation_gpu()
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from reproduction.hpc3_qprobe.model_manifest import create
    from search_g1.qprobe_prepare import train_questions
    from search_g1.qprobe_features import QUESTION_EXTRACTION_RULE, sha_json, question_text_sha
    from search_g1.qprobe_token_features import scan_token_lengths, extract_question_token_features, TokenFeatureCache
    if torch.cuda.device_count() != 1:
        raise ValueError('Actual preparation CUDA visibility differs')
    destination = BASE/'token-features'
    if (BASE/'preparation/features-ready.json').is_file():
        previous = read(BASE/'preparation/features-ready.json')
        if sha(destination/'manifest.json') != previous['manifest_sha256']:
            raise ValueError('Existing accepted bank changed')
        progress('FULL_TOKEN_BANK_ALREADY_ACCEPTED', rows=169028); return
    progress('VERIFYING_CANONICAL_MODEL_DATA_TEACHER', gpu_uuid=gpu)
    manifest_path = BASE/'preparation/official-model-manifest.json'
    if manifest_path.exists():
        from reproduction.hpc3_qprobe.model_manifest import verify
        official = read(manifest_path); verify(MODEL, official)
    else:
        official = create(MODEL, manifest_path)
    hashes = {r['file']:r['sha256'] for r in official['files']}
    rows, data_proof = train_questions(DATA)
    labels, provenance = teacher_labels()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    scan_path = BASE/'preparation/global-scan.json'
    records_path = BASE/'preparation/global-token-bindings.jsonl'
    if scan_path.exists():
        scan = read(scan_path); records = [json.loads(x) for x in records_path.read_text().splitlines()]
        if len(records) != 169028 or any(r['qid'] != rows[n][1] or r['question_hash'] != question_text_sha(rows[n][0]) for n,r in enumerate(records)):
            raise ValueError('Accepted canonical token scan changed')
    else:
        progress('CANONICAL_TOKEN_SCAN', rows=0, total=len(rows))
        lengths, tokens, scan = scan_token_lengths(tokenizer, rows)
        records = [dict(qid=qid, question_hash=question_text_sha(question), length=int(length), token_ids_sha256=token)
                   for (question,qid),length,token in zip(rows,lengths,tokens)]
        with records_path.open('x') as f:
            for record in records:
                f.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+'\n')
            f.flush(); os.fsync(f.fileno())
        publish(scan_path, scan, once=True)
    if scan['global_max_tokens'] != 244 or scan['total_tokens'] != 18367021:
        raise ValueError('The real canonical global token scan differs from the runbook')
    template = dict(question_extraction=QUESTION_EXTRACTION_RULE,prompt_key='prompt',added_template=None,
                    tokenizer_add_special_tokens=True,truncation=False)
    token_files = {k:v for k,v in hashes.items() if k in ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt')}
    spec = dict(encoder_id='Qwen/Qwen2.5-3B-Instruct',encoder_revision=official['revision'],
        encoder_files_sha256=sha_json(hashes),tokenizer_sha256=sha_json(token_files),prompt_template_sha256=sha_json(template),
        train_parquet_sha256=data_proof['products']['train']['sha256'],train_manifest_sha256=data_proof['manifest_sha256'],
        split='train',question_extraction=QUESTION_EXTRACTION_RULE,representation='all_nonpadding_question_tokens_packed_v1',
        feature_dim=2048,feature_layer=27,encoder_compute_dtype='bfloat16',feature_dtype='bfloat16',truncation=False,
        global_max_tokens=244,global_train_rows=169028,global_scan_sha256=sha(scan_path),
        token_feature_producer_sha256=sha(Path(__file__)),teacher_identity_sha256=provenance['teacher_identity_sha256'],
        teacher_complete_sha256=provenance['teacher_complete_sha256'],original_label_feature_spec_sha256=provenance['original_feature_spec_sha256'])
    if not destination.exists():
        writer = reopen_writer(destination,spec,records)
        progress('ENCODER_LOADING', rows=writer.written,total=169028,packed_bytes=scan['total_tokens']*2048*2,gpu_uuid=gpu)
        model = AutoModel.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
                    torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0')
        model.requires_grad_(False); model.eval(); started = time.monotonic(); first = writer.written
        try:
            for start in range(writer.written,len(rows),args.batch_size):
                chunk = rows[start:start+args.batch_size]
                values = extract_question_token_features(model,tokenizer,[x[0] for x in chunk],
                            feature_layer=27,max_tokens=244,device='cuda:0')
                writer.append([x[1] for x in chunk],values)
                if writer.written % 256 == 0 or writer.written == len(rows):
                    durable_cursor(writer)
                    progress('TOKEN_FEATURE_ENCODING',rows=writer.written,total=169028,
                             real_tokens=int(writer.offsets[writer.written]),seconds=time.monotonic()-started,
                             new_rows=writer.written-first)
            durable_cursor(writer)
            del model
            torch.cuda.empty_cache()
            progress('TOKEN_BANK_FULL_SHA',rows=writer.written,total=169028)
            writer.finish()
        except BaseException:
            # Only the accepted fsynced prefix may be resumed; every staged file is retained.
            raise
    cache = TokenFeatureCache(destination); cache.verify_all()
    if cache.manifest['count'] != 169028 or cache.manifest['total_tokens'] != 18367021:
        raise ValueError('Full canonical bank is incomplete')
    for label in labels:
        if cache.index[label['qid']]['question_hash'] != label['question_hash']:
            raise ValueError('Warmup label is not the canonical token row')
    receipt = dict(stage='HPC3_FULL_TOKEN_BANK_FULL_SHA_PASS',cache_dir=str(destination),manifest_sha256=sha(destination/'manifest.json'),
        spec_sha=cache.spec_sha,count=169028,total_tokens=18367021,global_max_tokens=244,model_revision=official['revision'],
        model_files_sha256=hashes,teacher_provenance=provenance,files=cache.manifest['files'],checked_unix=time.time(),
        shared_inputs_modified=False,trajectories_generated=0,formal_actor_outers=0)
    publish(BASE/'preparation/features-ready.json',receipt,once=True)
    progress('FULL_TOKEN_BANK_ACCEPTED',rows=169028,manifest_sha256=receipt['manifest_sha256'])


if __name__ == '__main__':
    main()
