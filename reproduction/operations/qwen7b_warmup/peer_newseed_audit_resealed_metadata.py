"""Explicit alternate attestation for metadata-resealed donor G5 batches.

The original byte receipt chain is NOT asserted equal to the current scan and
selection. Both chains are retained. Fresh prefix/full/own-EM payload SHA,
current own G5 selection, original per-full-row chain, prefix token/context/
position reconstruction, seed/model/identity and all-file fstat are still hard
requirements. No source or producer file is changed. Target ingestion must
independently recompute tokenizer strict EM before use.
"""
import hashlib
from pathlib import Path
from . import peer_newseed_audit_reference as original
from .peer_newseed_audit_reference import *

class Audit(original.Audit):
    def __init__(self,args):
        super().__init__(args)
        if hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()!='3def055adebfdb38e9326b9db385338c44800160c1b0a64e1ee78d8c3b68b906':
            raise ValueError('Pinned original auditor differs')
        self.source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def batch(self, inp, batch, input_hash):
        dest = self.base / 'sampling' / inp.parent.name / 'batches' / inp.stem
        seal = dest / 'complete.json'
        if not seal.exists():
            return None
        outer = dest / 'outers/outer-000001'
        files = dict(batch_seal=seal, config=dest / 'config.json', questions=outer / 'questions.json',
                     prefix_seal=outer / 'prefix/complete.json', prefix_scan=outer / 'prefix/scan.json',
                     prefix_rank=outer / 'prefix/rank-0-complete.json',
                     full_seal=outer / 'full/complete.json', full_rank=outer / 'full/rank-0-complete.json',
                     selection=outer / 'probe-score/selection.json', prefix=outer / 'prefix/rank-0-prefix.jsonl.gz',
                     full=outer / 'full/rank-0-full.jsonl.gz', own_em=outer / 'full/rank-0-own-em.jsonl')
        before = {key: stat(path) for key, path in files.items()}
        cache_path = self.directory / 'batches' / inp.parent.name / (inp.stem + '.json')
        binding = dict(cache_version=CACHE_VERSION, auditor_sha256=self.source_sha,
                       run_id=RUN_ID, seed=SEED, root_config_sha256=self.root_config_hash,
                       input_sha256=input_hash, files=before)
        if cache_path.exists():
            cached = read(cache_path)
            if cached.get('binding') == binding and cached.get('status') == 'VERIFIED':
                require(cached['proof']['questions'] == len(batch['rows'])
                        and cached['proof']['trajectories'] == cached['proof']['labels'] == G * len(batch['rows']),
                        'Cached proof counts differ from bound input')
                self.reused += 1
                return cached['proof']
        if self.max_new and self.attempted >= self.max_new:
            return dict(deferred=True)
        self.attempted += 1
        count = G * len(batch['rows'])
        expected = {row['qid']: row for row in batch['rows']}
        require(len(expected) == len(batch['rows']), 'Duplicate input qid within batch')
        config, sealed = read(files['config']), read(seal)
        for key, value in dict(run_id=RUN_ID, seed=SEED, model_revision=REVISION, warmup_only=True,
                               actor_hidden_native=False, actor_optimizer_updates=0,
                               actor_version='official7b-' + REVISION,
                               sampling_attempt_id=self.root_config.get('sampling_attempt_id')).items():
            require(config.get(key) == value, 'Batch config differs: ' + key)
        require(self.path(config['base']).resolve() == dest.resolve(), 'Batch config BASE differs')
        for key, value in dict(run_id=RUN_ID, input_sha256=input_hash, batch=inp.stem,
                               questions=count // G, trajectories=count, hidden_cached=False,
                               model_revision=REVISION, actor_optimizer_updates=0).items():
            require(sealed.get(key) == value, 'Batch seal differs: ' + key)
        require(read(files['questions']) == dict(outer=1, rows=batch['rows']), 'Sampling question rows differ from input')
        config_hash = sha(files['config'])
        prefix_rank, full_rank = read(files['prefix_rank']), read(files['full_rank'])
        self.receipt(prefix_rank, 'prefix', config_hash, count, config)
        self.receipt(full_rank, 'continue', config_hash, count, config)
        require(sealed['prefix'] == prefix_rank and sealed['full'] == full_rank, 'Batch/phase rank receipts differ')
        prefix_seal, scan, full_seal = read(files['prefix_seal']), read(files['prefix_scan']), read(files['full_seal'])
        scan_hash, selection_hash = sha(files['prefix_scan']), sha(files['selection'])
        require(prefix_seal['scan_sha256'] == scan_hash and scan['ranks'] == [prefix_rank], 'Prefix scan seal differs')
        require(all(prefix_seal.get(k) == v for k, v in scan.items()), 'Prefix complete/scan differs')
        require(scan['questions'] == count // G and scan['trajectories'] == count
                and scan['head_prefix_tokens'] == PREFIX and scan['model_revision'] == REVISION,
                'Prefix scan metadata differs')
        require(all(full_seal.get(k) == v for k, v in full_rank.items()) and full_seal['labels'] == count,
                'Full completion seal differs')
        old_scan_hash,old_selection_hash=full_rank['prefix_scan_sha256'],full_rank['selection_sha256']
        require(all(isinstance(h,str) and len(h)==64 and set(h)<=set('0123456789abcdef') for h in (old_scan_hash,old_selection_hash)), 'Original continuation chain hashes malformed')
        metadata_repacked=(old_scan_hash!=scan_hash or old_selection_hash!=selection_hash)
        selection = read(files['selection'])
        require(selection['run_id'] == RUN_ID and selection['outer'] == 1
                and selection['actor_version'] == config['actor_version']
                and selection['prefix_scan_sha256'] == scan_hash
                and selection['kind'] == 'ALL_G5_NO_SELECTION', 'G5 selection binding differs')
        items = dict(prefix=prefix_rank['prefix'], full=full_rank['full'], own_em=full_rank['own_em'])
        independent = {}
        for key, item in items.items():
            path = self.artifact(item, files[key], count)
            independent[key] = sha(path)
            require(independent[key] == item['sha256'], 'Independent SHA differs: ' + str(path))
        prefixes = self.prefix_rows(files['prefix'], expected, config, count)
        selected = {}
        for group in selection['selected']:
            qid = group['qid']
            require(qid in expected and qid not in selected, 'Unexpected/duplicate selected qid')
            require(group['question_hash'] == expected[qid]['question_hash']
                    and group['candidate_ordinal'] == expected[qid]['candidate_ordinal'], 'Selection question identity differs')
            tids = group['trajectory_ids']
            want = {prefix['trajectory_id'] for prefix in prefixes.values() if prefix['qid'] == qid}
            require(len(tids) == G and len(set(tids)) == G and set(tids) == want, 'Selection lacks exact own G5')
            selected[qid] = tids
        require(selected.keys() == expected.keys(), 'Selection misses input questions')
        labels, tokens = self.full_rows(files['full'], expected, config, prefixes, count, old_selection_hash, old_scan_hash)
        seen_labels = set()
        for label in rows(files['own_em']):
            tid = label['trajectory_id']
            require(tid not in seen_labels and tid in labels, 'Unexpected/duplicate own-em trajectory')
            require(label == labels[tid], 'Own strictEM label does not match full trajectory')
            seen_labels.add(tid)
        require(seen_labels == labels.keys(), 'Own strictEM labels miss full trajectories')
        require(before == {key: stat(path) for key, path in files.items()}, 'Sealed file changed during audit')
        masks = {qid: 0 for qid in expected}
        for prefix in prefixes.values():
            masks[prefix['qid']] |= 1 << prefix['response_index']
        require(all(value == (1 << G) - 1 for value in masks.values()), 'Question does not have response indices 0..4')
        proof = dict(questions=len(expected), trajectories=len(labels), labels=len(seen_labels),
                     independent_sha256=independent, producer_sha256={key: value['sha256'] for key, value in items.items()},
                     actual_trajectory_ids_sha256=hashlib.sha256(canonical(sorted(labels))).hexdigest(),
                     strict_em_ones=sum(row['strict_em'] for row in labels.values()),
                     metadata_reseal_attestation=dict(original_receipt_chain_equal_current=not metadata_repacked,
                         original_continuation_scan_sha256=old_scan_hash,original_continuation_selection_sha256=old_selection_hash,
                         fresh_current_scan_sha256=scan_hash,fresh_current_selection_sha256=selection_hash,
                         same_prefix_rank_receipt_and_payload=True,same_all_G5_IDs=True,
                         full_rows_original_chain_and_current_prefix_causal_tokens_exact=True,
                         strict_EM_tokenizer_recomputed=False,producer_metadata_modified=False), **tokens)
        atomic(cache_path, dict(status='VERIFIED', binding=binding, proof=proof, verified_epoch=time.time()))
        self.fresh += 1
        return proof

