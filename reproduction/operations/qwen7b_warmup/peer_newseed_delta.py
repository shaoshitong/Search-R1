"""Read-only sealed metadata delta against the actually imported peer snapshot.

No rollout/EM/prefix payload, archive, model or tensor is opened. Only canonical
input JSON, seals/configs, prior membership identities and existing audit proof
metadata are read. Payload SHA claims are inherited explicitly, never rerun.
"""
import hashlib
import json
from pathlib import Path
import stat
import time

RUN = '70188b8e50fd4b45bc420ce58014f46e'
SEED = 20261021
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
ATTEMPT = 'independent-seed20261021-g5-20261020'
AUDITOR = '3def055adebfdb38e9326b9db385338c44800160c1b0a64e1ee78d8c3b68b906'
NAME = 'qwen7b-resample-seed20261021-g5-20261020'
BASE = Path('/mnt/shared/workspace/training_data_node_10/shaoshitong') / NAME
SNAPSHOT = Path('/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/newseed-20261021-185747-445fc691')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def fstat(path):
    s = path.stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                inode=s.st_ino, device=s.st_dev)


def load(path):
    before = fstat(path); data = path.read_bytes(); after = fstat(path)
    if before != after:
        raise ValueError('Metadata changed during read: ' + str(path))
    return json.loads(data), dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def regular(base, rel):
    rel = Path(rel)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe source-relative metadata path')
    p = base
    for part in rel.parts:
        p = p / part
        if p.is_symlink():
            raise ValueError('Symlink source metadata forbidden')
    if not stat.S_ISREG(p.stat().st_mode):
        raise ValueError('Regular source metadata required')
    return p


def relative(base, value):
    value = str(value)
    for root in (str(BASE), '/mnt/data/shaoshitong/' + NAME, str(base)):
        if value.startswith(root + '/'):
            rel = value[len(root)+1:]; regular(base, rel); return rel
    raise ValueError('Source path escapes the approved run')


def contract(config):
    values = dict(run_id=RUN, seed=SEED, source_seed=1729, model_revision=REVISION,
                  sampling_attempt_id=ATTEMPT, capture_prefix_tokens=20, feature_layer=26,
                  question_max_tokens=273, actor_optimizer_updates=0, warmup_only=True,
                  actor_hidden_native=False)
    if any(config.get(k) != v for k, v in values.items()):
        raise ValueError('Peer root/batch run/model/seed/Prefix20 contract differs')


def partition(current, imported):
    """Metadata-only identity partition, also used by small CPU fixtures."""
    cur = {x['input']: x for x in current}; old = {x['input']: x for x in imported}
    if len(cur) != len(current) or len(old) != len(imported) or not set(old) <= set(cur):
        raise ValueError('Duplicate batches or an imported seal disappeared')
    for name, item in old.items():
        if any(cur[name][k] != item[k] for k in ('input_sha256', 'seal_sha256', 'questions', 'trajectories')):
            raise ValueError('Previously imported batch binding changed')
    return [cur[name] for name in sorted(set(cur)-set(old))]


def cached_audit(base, rel, config_sha, input_sha, questions):
    inp = Path(rel); dest = base/'sampling'/inp.parent.name/'batches'/inp.stem
    outer = dest/'outers/outer-000001'
    files = dict(batch_seal=dest/'complete.json', config=dest/'config.json', questions=outer/'questions.json',
                 prefix_seal=outer/'prefix/complete.json', prefix_scan=outer/'prefix/scan.json',
                 prefix_rank=outer/'prefix/rank-0-complete.json', full_seal=outer/'full/complete.json',
                 full_rank=outer/'full/rank-0-complete.json', selection=outer/'probe-score/selection.json',
                 prefix=outer/'prefix/rank-0-prefix.jsonl.gz', full=outer/'full/rank-0-full.jsonl.gz',
                 own_em=outer/'full/rank-0-own-em.jsonl')
    cache = base/'audit/batches'/inp.parent.name/(inp.stem+'.json')
    if not cache.exists():
        return dict(status='SEALED_NO_EXISTING_INDEPENDENT_BATCH_PROOF')
    proof, ref = load(regular(base, cache.relative_to(base)))
    expected = dict(cache_version=1, auditor_sha256=AUDITOR, run_id=RUN, seed=SEED,
                    root_config_sha256=config_sha, input_sha256=input_sha,
                    files={key: fstat(path) for key, path in files.items()})
    counts = proof.get('proof', {})
    if proof.get('status') != 'VERIFIED' or proof.get('binding') != expected:
        return dict(status='EXISTING_PROOF_NOT_CURRENTLY_FSTAT_BOUND', proof=ref)
    if (counts.get('questions') != questions or counts.get('trajectories') != 5*questions
            or counts.get('labels') != 5*questions):
        raise ValueError('Existing independent own-G5 proof counts differ')
    return dict(status='EXISTING_INDEPENDENT_AUDIT_CURRENT_FSTAT_BOUND', proof=ref,
                verified_epoch=proof['verified_epoch'], own_trajectory_ids_sha256=counts['actual_trajectory_ids_sha256'],
                bulk_payload_SHA_recomputed=False, strict_EM_tokenizer_recomputed=False)


def check_receiver(binding, snapshot):
    source = binding['source_snapshot']; receiver = binding['receiver']; ing = binding['raw_ingest']
    if (receiver['status'] != 'PEER_NEWSEED_RAW_ALL_ARCHIVES_AND_FILES_FRESH_SHA256_VERIFIED'
            or ing['status'] != 'RECEIVED_NEWSEED5_PLUS_CANONICAL_OLD5_RAW_G10_STRICT_EM_CAUSAL_SEEDS_ACCEPTED'
            or receiver['questions'] != 31232 or receiver['trajectories'] != 156160
            or ing['questions'] != 31232 or ing['trajectories'] != 312320
            or ing['receiver_complete'] != receiver['complete'] or receiver['groups'] != 16):
        raise ValueError('Actual imported16/archive and raw-G10 receipt binding required')
    for key in ('snapshot_id', 'source_run_id', 'seed', 'revision', 'questions', 'trajectories', 'files'):
        if source[key] != snapshot[key]:
            raise ValueError('Source frozen snapshot differs from imported catalog')
    group_keys = ('group','archive_sha256','archive_bytes','questions','trajectories','cohort_sha256')
    if source['groups'] != [{k:g[k] for k in group_keys} for g in snapshot['groups']]:
        raise ValueError('Actual imported16 archive identities differ')


def inventory(imported_binding, base=BASE, snapshot=SNAPSHOT):
    base, snapshot = Path(base), Path(snapshot)
    root, root_ref = load(regular(base, 'config.json')); contract(root)
    manifest, manifest_ref = load(regular(base, 'inputs/manifest.json'))
    previous, previous_ref = load(snapshot/'plan.json'); complete, complete_ref = load(snapshot/'complete.json')
    check_receiver(imported_binding, complete)
    if (complete_ref['sha256'] != imported_binding['previous_independent_source_export_complete_sha256']
            or any(previous_ref[k] != complete['plan'][k] for k in ('bytes','sha256'))):
        raise ValueError('Prior source complete/plan differs from accepted export audit')
    if previous['source_config']['sha256'] != root_ref['sha256'] or previous['input_manifest']['sha256'] != manifest_ref['sha256']:
        raise ValueError('Source canonical config/manifest changed since imported snapshot')
    if manifest['seed'] != SEED or manifest['questions'] != 169028 or len(manifest['batches']) != 1322:
        raise ValueError('Canonical all-train manifest scope differs')
    current, unsealed, declared = [], [], set()
    for item in manifest['batches']:
        rel = relative(base, item['path']); inp = Path(rel)
        if rel in declared:
            raise ValueError('Repeated canonical input batch')
        declared.add(rel); seal_rel = str(Path('sampling')/inp.parent.name/'batches'/inp.stem/'complete.json')
        if not (base/seal_rel).exists():
            unsealed.append(rel); continue
        seal, seal_ref = load(regular(base, seal_rel))
        expected = dict(run_id=RUN, model_revision=REVISION, input_sha256=item['sha256'], batch=inp.stem,
                        questions=item['questions'], trajectories=5*item['questions'], hidden_cached=False, actor_optimizer_updates=0)
        if any(seal.get(k) != v for k, v in expected.items()):
            raise ValueError('Claimed seal differs from canonical source run')
        audit = cached_audit(base, rel, root_ref['sha256'], item['sha256'], item['questions'])
        current.append(dict(input=rel, input_sha256=item['sha256'], seal=seal_rel,
                            seal_sha256=seal_ref['sha256'], questions=item['questions'], trajectories=5*item['questions'], audit=audit))
    if declared != {str(p.relative_to(base)) for p in (base/'inputs').glob('shard-*/batch*.json')}:
        raise ValueError('Canonical input membership differs')
    known_seals = {str(Path('sampling')/Path(name).parent.name/'batches'/Path(name).stem/'complete.json') for name in declared}
    if {str(p.relative_to(base)) for p in (base/'sampling').glob('shard-*/batches/batch*/complete.json')} - known_seals:
        raise ValueError('Unmanifested sealed source batch')
    delta = partition(current, previous['selected_batches'])
    member, member_ref = load(snapshot/'canonical-membership.json')
    if any(member_ref[k] != complete['canonical_membership'][k] for k in ('bytes','sha256')):
        raise ValueError('Accepted source identity membership SHA differs')
    if member['source_run_id'] != RUN or member['questions'] != 31232 or len(member['groups']) != 31232:
        raise ValueError('Imported snapshot membership differs')
    old_ids = {g['qid'] for g in member['groups']}; old_hashes = {g['question_hash'] for g in member['groups']}
    old_tids = {t for g in member['groups'] for t in g['trajectory_ids']}
    ids, hashes, tids, candidates = set(), set(), set(), set(); domains = {}; overlap_hashes = set()
    for item in delta:
        batch, input_ref = load(regular(base, item['input']))
        if input_ref['sha256'] != item['input_sha256'] or batch['split'] != 'train' or batch['model_revision'] != REVISION:
            raise ValueError('Fresh delta input SHA/train identity differs')
        rows = batch['rows']; inp = Path(item['input'])
        cfg, cfg_ref = load(regular(base, str(Path('sampling')/inp.parent.name/'batches'/inp.stem/'config.json'))); contract(cfg)
        own_ids, own_hashes, own_tids = [], [], []
        for row in rows:
            qid, qhash, ordinal = row['qid'], row['question_hash'], row['candidate_ordinal']
            if row['split'] != 'train' or type(ordinal) is not int or qid in ids or qid in old_ids or ordinal in candidates:
                raise ValueError('Delta duplicate/imported question or ordinal overlap')
            ids.add(qid); hashes.add(qhash); candidates.add(ordinal); own_ids.append(qid); own_hashes.append(qhash)
            domain = json.loads(qid)[0]; domains[domain] = domains.get(domain,0)+1
            if qhash in old_hashes:overlap_hashes.add(qhash)
            own = [RUN+'-outer000001-c%04d-g%d' % (ordinal,j) for j in range(5)]
            if tids.intersection(own) or old_tids.intersection(own):
                raise ValueError('Delta/imported authentic trajectory namespace overlap')
            tids.update(own); own_tids.extend(own)
        if len(rows) != item['questions']:
            raise ValueError('Delta actual input question count differs')
        item.update(input_ref=input_ref, config=cfg_ref, ordered_qids_sha256=digest(own_ids),
                    ordered_question_hashes_sha256=digest(own_hashes), expected_own_G5_IDs_sha256=digest(own_tids))
    audit_latest, audit_latest_ref = load(regular(base, 'audit/latest.json'))
    if (audit_latest.get('run_id') != RUN or audit_latest.get('seed') != SEED
            or audit_latest.get('model_revision') != REVISION or audit_latest.get('auditor_sha256') != AUDITOR):
        raise ValueError('Current independent audit namespace differs')
    audit_summary = {k:audit_latest.get(k) for k in ('status','epoch','verified_batches','questions','trajectories','failed_batches',
                    'sealed_pending_batches','strict_em_recomputed','states_shards_independently_audited')}
    accepted = [v for v in delta if v['audit']['status']=='EXISTING_INDEPENDENT_AUDIT_CURRENT_FSTAT_BOUND']
    pending = [v for v in delta if v not in accepted]
    if len(ids) != sum(x['questions'] for x in delta) or len(tids) != 5*len(ids):
        raise ValueError('Delta canonical identity counts differ')
    plan = dict(schema='PEER_NEWSEED_SEALED_METADATA_DELTA_V1', source_base=str(base), source_run_id=RUN,
        seed=SEED, model_revision=REVISION, sampling_attempt_id=ATTEMPT, source_config=root_ref,
        input_manifest=manifest_ref, imported_snapshot_complete=complete_ref, imported_snapshot_plan=previous_ref,
        imported_identity_membership=member_ref, actual_HPC3_imported_binding=imported_binding,
        selected_delta_batches=sorted(delta,key=lambda v:v['input']),
        selection_rule='all source-sealed canonical batches absent from the accepted244-batch snapshot; no EM/probe-score selection',
        current_sealed_batches=len(current), current_sealed_questions=sum(x['questions'] for x in current),
        current_audit_fstat_bound_batches=sum(x['audit']['status']=='EXISTING_INDEPENDENT_AUDIT_CURRENT_FSTAT_BOUND' for x in current),
        imported_batches=len(previous['selected_batches']), delta_batches=len(delta), delta_questions=len(ids), delta_trajectories=len(tids),
        audited_delta_batches=len(accepted), audited_delta_questions=sum(x['questions'] for x in accepted),
        sealed_pending_audit_batches=len(pending), sealed_pending_audit_questions=sum(x['questions'] for x in pending),
        domain_counts=domains, imported_overlap=dict(batch_paths=0,qids=0,trajectory_ids=0,question_hashes=len(overlap_hashes)),
        delta_unique_question_hashes=len(hashes), old_archive_payload_SHA_recomputed=False,
        prefix_full_EM_payload_opened=False, strict_EM_tokenizer_recomputed=False,
        existing_audit_proofs_inherited_only_when_current_all_file_fstat_binding_matches=True,
        payload_migration_started=False, GPU_started=False, source_mutated=False,
        current_audit_latest=audit_latest_ref, current_audit_summary=audit_summary,
        remaining_unsealed_batches=len(unsealed), all169028_complete=not unsealed)
    return dict(status='READONLY_SEALED_DELTA_PLAN_BOUND_NOT_EXPORTED', plan=plan, plan_sha256=digest(plan), epoch=time.time())
