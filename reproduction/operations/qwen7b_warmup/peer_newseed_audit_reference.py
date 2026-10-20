"""CPU-only, resumable audit of the sealed seed20261021 G5 data on remote storage.

Run on NAS10, for example::

    python3 audit_data.py --base /mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020

The original recorded BASE is translated to --base automatically. Additional
mount aliases can be supplied as --path-map OLD=NEW. Only BASE/audit (or the
explicit --audit-dir) is written. No sampler, service, model or GPU is imported.
Exit codes: COMPLETE=0, INCOMPLETE=2, FAILED=1. A FAILED audit never stops sampling.
"""

import argparse
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


RUN_ID = '70188b8e50fd4b45bc420ce58014f46e'
SEED = 20261021
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
RECORDED_BASE = '/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020'
QUESTIONS, TRAJECTORIES, BATCHES, G, PREFIX = 169028, 845140, 1322, 5, 20
CACHE_VERSION = 1


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open(encoding='utf8') as stream:
        return json.load(stream)


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-%d' % os.getpid())
    with tmp.open('wb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stat(path):
    s = path.stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                inode=s.st_ino, device=s.st_dev)


def token_ids(value, name, nonempty=False):
    require(isinstance(value, list) and (value or not nonempty)
            and all(type(x) is int and x >= 0 for x in value), name + ': invalid token IDs')
    return value


def log_probs(value, name):
    require(isinstance(value, list) and all(type(x) in (int, float)
            and math.isfinite(x) and x <= 1e-6 for x in value), name + ': invalid log probabilities')
    return value


def rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf8') as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()), '%s:%d: empty record' % (path, number))
            yield json.loads(line)


def seed_for(run_id, trajectory_id, turn, chunk, seed, attempt):
    # Exact CPU formula in prefix_selection200.rollout.seed_for; no GPU imports.
    value = [run_id, 1, trajectory_id, int(turn), int(chunk), int(seed)]
    if attempt is not None:
        value.append(attempt)
    return int.from_bytes(hashlib.sha256(canonical(value)).digest()[:8], 'big') % (2 ** 31 - 1)


class Audit:
    def __init__(self, args):
        self.base = Path(args.base).resolve()
        require(sys.platform != 'darwin', 'Run this data audit on remote Linux storage, not the Mac')
        self.directory = Path(args.audit_dir).resolve() if args.audit_dir else self.base / 'audit'
        self.maps = [(RECORDED_BASE, str(self.base))]
        for item in args.path_map:
            old, separator, new = item.partition('=')
            require(separator and old.startswith('/') and new.startswith('/'),
                    '--path-map requires absolute OLD=NEW prefixes')
            self.maps.append((old.rstrip('/'), new.rstrip('/')))
        self.maps.sort(key=lambda item: len(item[0]), reverse=True)
        self.source_sha = sha(Path(__file__))
        require(args.max_new_batches >= 0, '--max-new-batches must be nonnegative')
        self.max_new = args.max_new_batches
        self.attempted = 0
        self.reused = self.fresh = 0
        self.errors = []

    def path(self, recorded):
        value = str(recorded)
        for old, new in self.maps:
            if value == old or value.startswith(old + '/'):
                return Path(new + value[len(old):])
        return Path(value)

    def artifact(self, item, expected, count):
        require(isinstance(item, dict), 'Missing producer file receipt: ' + str(expected))
        path = self.path(item['path'])
        require(path.resolve() == expected.resolve(), 'Unexpected artifact path: ' + str(path))
        require(type(item['bytes']) is int and item['bytes'] == path.stat().st_size,
                'Producer byte count differs: ' + str(path))
        require(item.get('count') == count, 'Producer record count differs: ' + str(path))
        require(isinstance(item.get('sha256'), str) and len(item['sha256']) == 64,
                'Missing producer SHA256: ' + str(path))
        return path

    def receipt(self, receipt, phase, config_hash, count, config):
        for key, expected in dict(run_id=RUN_ID, outer=1, rank=0, phase=phase,
                                  actor_version='official7b-' + REVISION,
                                  config_sha256=config_hash, questions=count // G,
                                  trajectories=count, sampling_attempt_id=config.get('sampling_attempt_id')).items():
            require(receipt.get(key) == expected, 'Phase receipt differs: ' + key)
        require(receipt.get('engine_seed') == seed_for(RUN_ID, 'engine', 0, 0,
                SEED, config.get('sampling_attempt_id')), 'Engine seed binding differs')

    def identity(self, row, expected, config):
        qid = row['qid']
        require(qid in expected, 'Unexpected qid: ' + str(qid))
        source = expected[qid]
        index = row['response_index']
        require(type(index) is int and 0 <= index < G, 'Response index outside 0..4')
        tid = '%s-outer000001-c%04d-g%d' % (RUN_ID, source['candidate_ordinal'], index)
        for key, value in dict(run_id=RUN_ID, outer=1, trajectory_id=tid, split='train',
                               question_hash=source['question_hash'],
                               candidate_ordinal=source['candidate_ordinal'],
                               actor_version=config['actor_version']).items():
            require(row.get(key) == value, 'Trajectory identity differs: ' + key)
        prompt = token_ids(row['initial_prompt_token_ids'], 'initial prompt', nonempty=True)
        require(prompt == source['initial_prompt_token_ids'], 'Initial prompt IDs differ from canonical input')
        return tid, source

    def prefix_rows(self, path, expected, config, count):
        found = {}
        for row in rows(path):
            tid, _ = self.identity(row, expected, config)
            require(tid not in found, 'Duplicate prefix trajectory: ' + tid)
            claimed = row['prefix_sha256']
            material = dict(row)
            del material['prefix_sha256']
            require(hashlib.sha256(canonical(material)).hexdigest() == claimed, 'Prefix record SHA differs')
            ids = token_ids(row['policy_token_ids'], 'prefix policy')
            positions = row['policy_position_ids']
            require(isinstance(positions, list) and all(type(x) is int and x >= 0 for x in positions)
                    and len(positions) == len(ids) == row['retained_policy_length'], 'Prefix positions/length differ')
            require(len(ids) <= PREFIX and (len(ids) == PREFIX or row['terminal'] is True),
                    'Short prefix lacks its natural terminal')
            require(row['future_information_included'] is False, 'Prefix contains future information')
            require(isinstance(row['prefix_segments'], list), 'Missing prefix segments')
            found[tid] = row
        require(len(found) == count, 'Prefix record count differs')
        return found

    def full_rows(self, path, expected, config, prefixes, count, selection_hash, scan_hash):
        found = {}
        total_policy = total_response = 0
        short_prefixes = 0
        for row in rows(path):
            tid, source = self.identity(row, expected, config)
            require(tid not in found, 'Duplicate full trajectory: ' + tid)
            require(tid in prefixes, 'Full trajectory has no prefix')
            prefix = prefixes[tid]
            for key, value in dict(policy_snapshot_id=config['actor_version'],
                                   prefix_sha256=prefix['prefix_sha256'],
                                   prefix_scan_sha256=scan_hash, selection_sha256=selection_hash,
                                   reward_model=source['reward_model'], data_source=source['data_source']).items():
                require(row.get(key) == value, 'Full/prefix/input binding differs: ' + key)
            require(type(row['strict_em']) is int and row['strict_em'] in (0, 1), 'Invalid full strict_em')
            response = token_ids(row['response_token_ids'], 'response')
            mask = row['response_info_mask']
            probabilities = log_probs(row['response_log_probs'], 'response')
            require(isinstance(mask, list) and all(type(x) is int and x in (0, 1) for x in mask)
                    and len(response) == len(mask) == len(probabilities), 'Response/mask/logprob lengths differ')
            turns, journal = row['behavior_turns'], row['sampling_journal']
            require(isinstance(turns, list) and turns and isinstance(journal, list) and journal,
                    'Missing actual behavior turns or sampling journal')
            rebuilt, rebuilt_mask, rebuilt_lp, flattened, segments = [], [], [], [], []
            retained, retained_positions = [], []
            for turn in turns:
                request = token_ids(turn['actual_request_token_ids'], 'turn request', nonempty=True)
                require(request == row['initial_prompt_token_ids'] + rebuilt, 'Turn request context differs')
                raw = token_ids(turn['raw_generated_token_ids'], 'turn raw policy')
                lp = log_probs(turn['selected_logprobs'], 'turn policy')
                kept = turn['kept_token_count']
                require(type(kept) is int and 0 <= kept <= len(raw) == len(lp), 'Turn kept/raw/logprob lengths differ')
                require(turn['trajectory_id'] == tid and turn['policy_snapshot_id'] == config['actor_version'],
                        'Turn trajectory/policy identity differs')
                require(turn['prefix_cut']['kept_token_ids'] == raw[:kept]
                        and turn['prefix_cut']['kept_token_count'] == kept, 'Turn cut token IDs differ')
                chunks = turn['actual_sampling_chunks']
                require(isinstance(chunks, list) and chunks, 'Missing actual sampling chunks')
                chunk_ids, chunk_lp = [], []
                for chunk in chunks:
                    ids = token_ids(chunk['raw_generated_token_ids'], 'chunk raw policy')
                    logs = log_probs(chunk['selected_logprobs'], 'chunk policy')
                    require(len(ids) == len(logs), 'Chunk tokens/logprobs differ')
                    require(chunk['actual_request_token_ids'] == request + chunk_ids, 'Native chunk request context differs')
                    turn_index, chunk_index = chunk['logical_turn_index'], chunk['native_chunk_index']
                    require(chunk['request_id'] == '%s-t%d-chunk%d' % (tid, turn_index, chunk_index)
                            and turn_index == turn['turn_index'], 'Native chunk identity differs')
                    require(chunk['seed'] == seed_for(RUN_ID, tid, turn_index, chunk_index,
                            SEED, config.get('sampling_attempt_id'))
                            and chunk.get('sampling_attempt_id') == config.get('sampling_attempt_id'),
                            'Native chunk run/seed binding differs')
                    require(isinstance(chunk['sampling'], dict) and chunk['sampling'], 'Missing actual sampling metadata')
                    chunk_ids.extend(ids)
                    chunk_lp.extend(logs)
                require(chunk_ids == raw and chunk_lp == lp, 'Sampling chunks differ from turn policy')
                flattened.extend(chunks)
                take = min(kept, PREFIX - len(retained))
                if take:
                    segments.append(dict(turn_index=turn['turn_index'], policy_start=len(retained),
                                         policy_length=take, request_length=len(request),
                                         position_start=len(request), context_tokens=request + raw[:take]))
                    retained.extend(raw[:take])
                    retained_positions.extend(range(len(request), len(request) + take))
                observation = token_ids(turn['observation_token_ids'], 'turn observation')
                correction = token_ids(turn['correction_token_ids'], 'turn correction')
                require(not (observation and correction), 'Turn has two environment suffixes')
                environment = observation + correction
                rebuilt.extend(raw[:kept] + environment)
                rebuilt_mask.extend([1] * kept + [0] * len(environment))
                rebuilt_lp.extend(lp[:kept] + [0.] * len(environment))
            require(rebuilt == response and rebuilt_mask == mask and rebuilt_lp == probabilities,
                    'Full response does not reconstruct from actual behavior turns')
            require(flattened == journal, 'Full sampling journal differs from behavior chunks')
            require(prefix['policy_token_ids'] == retained and prefix['policy_position_ids'] == retained_positions
                    and prefix['prefix_segments'] == segments, 'Prefix replay tokens differ from full trajectory')
            require(row['terminal_reason'] in ('natural_eos', 'answer', 'final_action_budget'), 'Missing full terminal reason')
            if prefix['terminal']:
                require(prefix['terminal_reason'] == row['terminal_reason'], 'Early terminal prefix/full reason differs')
            short_prefixes += int(len(retained) < PREFIX)
            total_policy += sum(mask)
            total_response += len(response)
            found[tid] = {key: row[key] for key in ('trajectory_id', 'qid', 'strict_em',
                'actor_version', 'prefix_sha256', 'prefix_scan_sha256', 'selection_sha256')}
        require(len(found) == count and found.keys() == prefixes.keys(), 'Full/prefix trajectory membership differs')
        return found, dict(policy_tokens=total_policy, response_tokens=total_response,
                           short_terminal_prefixes=short_prefixes)

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
        require(full_rank['prefix_scan_sha256'] == scan_hash and full_rank['selection_sha256'] == selection_hash,
                'Continuation selection/scan binding differs')
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
        labels, tokens = self.full_rows(files['full'], expected, config, prefixes, count, selection_hash, scan_hash)
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
                     strict_em_ones=sum(row['strict_em'] for row in labels.values()), **tokens)
        atomic(cache_path, dict(status='VERIFIED', binding=binding, proof=proof, verified_epoch=time.time()))
        self.fresh += 1
        return proof

    def run(self):
        config = read(self.base / 'config.json')
        self.root_config = config
        self.root_config_hash = sha(self.base / 'config.json')
        for key, value in dict(run_id=RUN_ID, seed=SEED, model_revision=REVISION,
                               train_questions=QUESTIONS, train_trajectories=TRAJECTORIES,
                               warmup_only=True, actor_hidden_native=False, actor_optimizer_updates=0).items():
            require(config.get(key) == value, 'Root config differs: ' + key)
        require(self.path(config['base']).resolve() == self.base, 'Root config BASE differs')
        manifest = read(self.base / 'inputs/manifest.json')
        require(manifest['seed'] == SEED and manifest['questions'] == QUESTIONS
                and manifest['trajectories'] == TRAJECTORIES, 'Input manifest run dimensions differ')
        inputs = sorted((self.base / 'inputs').glob('shard-*/batch*.json'))
        declared = {}
        for item in manifest['batches']:
            path = self.path(item['path']).resolve()
            require(path not in declared, 'Duplicate manifest input path')
            declared[path] = item
        require(set(declared) == {p.resolve() for p in inputs}, 'Input manifest/files membership differs')
        require(len(inputs) == BATCHES, 'Expected all 1322 canonical input batches')
        input_qids, ordinals, candidates, validated_qids = {}, set(), set(), set()
        verified, incomplete, deferred = [], [], []
        for inp in inputs:
            try:
                batch = read(inp)
                digest = sha(inp)
                require(digest == declared[inp.resolve()]['sha256'], 'Input SHA differs from manifest')
                require(batch['split'] == 'train' and batch['model_revision'] == REVISION
                        and len(batch['rows']) == declared[inp.resolve()]['questions'], 'Input batch metadata differs')
                for row in batch['rows']:
                    qid, ordinal, candidate = row['qid'], row['source_ordinal'], row['candidate_ordinal']
                    require(qid not in input_qids and ordinal not in ordinals and candidate not in candidates,
                            'Canonical input qid/source ordinal/candidate ordinal repeats')
                    require(row['split'] == 'train' and type(ordinal) is int and type(candidate) is int,
                            'Input question split/ordinal differs')
                    token_ids(row['initial_prompt_token_ids'], 'canonical input prompt', nonempty=True)
                    input_qids[qid] = ordinal
                    ordinals.add(ordinal)
                    candidates.add(candidate)
                proof = self.batch(inp, batch, digest)
                if proof is None:
                    incomplete.append(str(inp.relative_to(self.base)))
                elif proof.get('deferred'):
                    deferred.append(str(inp.relative_to(self.base)))
                else:
                    validated_qids.update(row['qid'] for row in batch['rows'])
                    verified.append(proof)
            except Exception as error:
                self.errors.append(dict(batch=str(inp.relative_to(self.base)),
                                        error_type=type(error).__name__, error=str(error)))
            if (len(verified) + len(incomplete) + len(deferred) + len(self.errors)) % 25 == 0:
                print(json.dumps(dict(event='AUDIT_PROGRESS', verified_batches=len(verified),
                      verified_questions=len(validated_qids), unsealed_batches=len(incomplete),
                      sealed_pending_batches=len(deferred), failed_batches=len(self.errors))), flush=True)
        if not self.errors:
            require(len(input_qids) == QUESTIONS and ordinals == set(range(QUESTIONS)), 'Canonical input questions/ordinals incomplete')
            ordered = [qid for qid, _ in sorted(input_qids.items(), key=lambda item: item[1])]
            require(hashlib.sha256(json.dumps(ordered).encode()).hexdigest() == manifest['ordered_qid_sha256'],
                    'Canonical ordered qid SHA differs')
        expected_seals = {self.base / 'sampling' / p.parent.name / 'batches' / p.stem / 'complete.json' for p in inputs}
        extra = set((self.base / 'sampling').glob('shard-*/batches/batch*/complete.json')) - expected_seals
        if extra:
            self.errors.append(dict(error='Sealed sampling batches absent from canonical inputs', paths=sorted(map(str, extra))))
        totals = {key: sum(item[key] for item in verified) for key in ('questions', 'trajectories', 'labels',
                  'strict_em_ones', 'policy_tokens', 'response_tokens', 'short_terminal_prefixes')}
        complete = (not self.errors and not incomplete and not deferred and len(verified) == BATCHES
                    and len(validated_qids) == QUESTIONS and totals['trajectories'] == TRAJECTORIES
                    and totals['labels'] == TRAJECTORIES)
        return dict(status='FAILED' if self.errors else 'COMPLETE' if complete else 'INCOMPLETE',
                    run_id=RUN_ID, seed=SEED, model_revision=REVISION, base=str(self.base),
                    expected=dict(questions=QUESTIONS, trajectories=TRAJECTORIES, batches=BATCHES),
                    discovered_input_batches=len(inputs), canonical_unique_input_qids=len(input_qids),
                    verified_batches=len(verified), verified_unique_qids=len(validated_qids),
                    verified_unique_trajectories=totals['trajectories'],
                    uniqueness_basis='Per-batch exact actual trajectory IDs and G5 indices; globally unique canonical qid and candidate ordinal',
                    **totals, unsealed_batches=len(incomplete), unsealed_inputs=incomplete,
                    sealed_pending_batches=len(deferred), sealed_pending_inputs=deferred,
                    cache_reused_batches=self.reused, independently_read_batches=self.fresh,
                    failed_batches=len(self.errors), errors=self.errors,
                    strict_em_recomputed=False,
                    strict_em_verification='Exact own-em/full label correspondence only; tokenizer-based strict_label was not rerun',
                    strict_em_source='reproduction/prefix20_online/rollout.py:strict_label -> search_g1.signals.decode_segments/contexts -> verl.utils.reward_score.qa_em.em_check',
                    states_shards_independently_audited=False, auditor_sha256=self.source_sha,
                    epoch=time.time(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', required=True, help='Actual remote directory containing config/inputs/sampling')
    parser.add_argument('--path-map', action='append', default=[], metavar='OLD=NEW', help='Additional recorded absolute path prefix translation')
    parser.add_argument('--audit-dir', help='Remote proof/cache directory; default BASE/audit')
    parser.add_argument('--max-new-batches', type=int, default=0, help='Audit at most N new/changed sealed batches this invocation; 0 means unlimited')
    args = parser.parse_args()
    auditor = None
    try:
        auditor = Audit(args)
        result = auditor.run()
    except Exception as error:
        result = dict(status='FAILED', run_id=RUN_ID, seed=SEED,
                      error_type=type(error).__name__, error=str(error),
                      strict_em_recomputed=False, epoch=time.time())
    if auditor is not None:
        atomic(auditor.directory / 'latest.json', result)
    summary = {key: value for key, value in result.items()
               if key not in ('unsealed_inputs', 'sealed_pending_inputs', 'errors')}
    if result.get('errors'):
        summary['first_errors'] = result['errors'][:10]
    if auditor is not None:
        summary['proof_path'] = str(auditor.directory / 'latest.json')
    print(json.dumps(summary, sort_keys=True), flush=True)
    return dict(COMPLETE=0, INCOMPLETE=2, FAILED=1)[result['status']]


if __name__ == '__main__':
    raise SystemExit(main())
