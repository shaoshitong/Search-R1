"""Verify the actual R4 final evaluation without rerunning generation/scoring.

Run on HPC3 CPU. Stream full outputs, retain scalar audit evidence remotely,
and distinguish seven-set macro EM from the question-weighted overall EM.
This does not certify the remaining full-goal/checkpoint/cleanup requirements.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

BASE = Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4')
RUNTIME = '2000d46a530c64d78be7802bf11175a68b65effe'
COUNT = 51713


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for raw in iter(lambda: f.read(8*1024**2), b''):
            h.update(raw)
    return h.hexdigest()


def main():
    assert BASE.resolve() == BASE and BASE.stat().st_uid == os.getuid()
    assert read(BASE/'preparation/source-publication.json')['commit'] == RUNTIME
    target = BASE/'final-evaluation/complete.json'
    if not target.exists():
        print(json.dumps(dict(status='PENDING_NO_ACTUAL_FINAL51713_COMPLETE')))
        return
    proof = BASE/'preparation/final51713-integrity-and-stored-strict-metrics.json'
    if proof.exists():
        old = read(proof)
        assert old['status'] == 'FINAL51713_FRESH_SHA_IDENTITIES_AND_STORED_STRICT_METRICS_PASS'
        assert old['operator_sha256'] == sha(__file__)
        for row in old['verified_files']:
            p = Path(row['path']); s = p.stat()
            assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns) == (row['dev'],row['inode'],row['bytes'],row['mtime_ns'])
        print(json.dumps(dict(status='EXISTING_FINAL51713_VERIFICATION_STAT_SEALS_PASS',proof=str(proof),sha256=sha(proof))))
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['DYNRPO_PROGRESSIVE_PREFIX'] = 'cp50_prefix20_40_80_three_epoch_v1'
    os.environ['DYNRPO_PREFIX_TOKENS'] = '80'
    sys.path.insert(0,str(BASE/'repo'))
    from reproduction.prefix_selection200.eval import load_questions, CONTRACT
    from reproduction.prefix_selection200.interaction_protocol import CONTRACT as PROTOCOL
    c = read(BASE/'preparation/launch-config.json')
    joint_path = BASE/'outers/outer-000200/complete.json'
    j = read(joint_path); result = read(target)
    assert c['source_commit'] == j['runtime_source_commit'] == RUNTIME
    assert j['outer'] == 200 and j['run_id'] == result['run_id'] == c['run_id']
    assert result['actor_version'] == j['actor']['actor_version']
    assert result['actor_model_dir'] == j['actor']['hf_model_dir']
    assert result['evaluation_contract'] == CONTRACT and result['interaction_protocol'] == PROTOCOL
    assert result['actual_compute_dtype'] == 'torch.bfloat16'
    assert result['qa_count'] == COUNT and len(result['rank_receipts']) == len(result['output_shards']) == 4
    files = []

    def qualify(path, expected=None, output=False):
        p = Path(path)
        if output:
            assert p.parent == target.parent and p.resolve() == p
        s = p.stat(); digest = sha(p); after = p.stat()
        assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns) == (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
        if expected:
            assert digest == expected['sha256']
            if 'bytes' in expected: assert s.st_size == expected['bytes']
        files.append(dict(path=str(p),sha256=digest,bytes=s.st_size,dev=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns))
        return digest

    source_sha = qualify(c['final_questions'],dict(sha256=c['final_questions_sha256']))
    assert source_sha == result['question_source_sha256']
    qualify(target); qualify(joint_path); qualify(BASE/'preparation/launch-config.json')
    questions = load_questions(c['final_questions'],COUNT)
    canonical_counts = Counter(q['data_source'] for q in questions)
    assert len(questions) == len({q['qid'] for q in questions}) == COUNT and len(canonical_counts) == 7
    seen = set(); sums = defaultdict(lambda: [0,0.0,0]); rank_seconds = []
    for rank, expected in enumerate(result['rank_receipts']):
        path = target.parent/('rank-%d-complete.json'%rank)
        assert expected['path'] == str(path)
        qualify(path,expected,True); receipt = read(path)
        indices = set(range(rank,COUNT,4))
        assert receipt['rank'] == rank and receipt['qa_count'] == receipt['question_count'] == len(indices)
        for key in ('run_id','actor_version','actor_model_dir','question_source_sha256','evaluation_contract','interaction_protocol','actual_compute_dtype'):
            assert receipt[key] == result[key], key
        saved = {r['evaluation_ordinal']:r for r in receipt['rows']}
        assert len(saved) == len(receipt['rows']) == len(indices) and set(saved) == indices
        full = target.parent/('rank-%d-full.jsonl.gz'%rank)
        assert receipt['full_output'] == result['output_shards'][rank] and receipt['full_output']['path'] == str(full)
        qualify(full,receipt['full_output'],True)
        streamed = set()
        with gzip.open(full,'rt') as f:
            for raw in f:
                if not raw.strip(): continue
                row = json.loads(raw); ordinal = row['evaluation_ordinal']
                assert ordinal in saved and ordinal not in streamed and ordinal not in seen
                item = saved[ordinal]; canonical = questions[ordinal]
                assert all(row[key] == value for key,value in item.items())
                assert item['qid'] == canonical['qid'] and item['data_source'] == canonical['data_source']
                assert type(item['em']) is int and item['em'] in (0,1)
                assert math.isfinite(item['f1']) and 0 <= item['f1'] <= 1
                streamed.add(ordinal); seen.add(ordinal)
                acc = sums[item['data_source']]; acc[0] += item['em']; acc[1] += item['f1']; acc[2] += 1
        assert streamed == indices
        assert math.isfinite(receipt['elapsed_seconds']) and receipt['elapsed_seconds'] > 0
        rank_seconds.append(receipt['elapsed_seconds'])
    assert seen == set(range(COUNT)) and Counter({k:v[2] for k,v in sums.items()}) == canonical_counts
    per_source = {}
    for source,(correct,f1,count) in sums.items():
        measured = dict(qa_count=count,em=correct/count,f1=f1/count,strict_correct=correct)
        assert result['per_source'][source]['qa_count'] == count
        for key in ('em','f1'):
            assert math.isclose(result['per_source'][source][key],measured[key],rel_tol=0,abs_tol=1e-12)
        per_source[source] = measured
    weighted = dict(em=sum(v[0] for v in sums.values())/COUNT,f1=sum(v[1] for v in sums.values())/COUNT)
    assert set(result['per_source']) == set(per_source)
    for key,value in weighted.items(): assert math.isclose(result[key],value,rel_tol=0,abs_tol=1e-12)
    assert all(value == 0 for value in result['actual_worker_exit_codes'].values())
    report = dict(status='FINAL51713_FRESH_SHA_IDENTITIES_AND_STORED_STRICT_METRICS_PASS',checked_unix=time.time(),
        base=str(BASE),runtime_source_commit=RUNTIME,operator_sha256=sha(__file__),qa_count=COUNT,
        actor_version=result['actor_version'],question_source_sha256=source_sha,verified_files=files,
        per_source=per_source,question_weighted=weighted,
        seven_set_macro={key:sum(v[key] for v in per_source.values())/7 for key in ('em','f1')},
        rank_inference_seconds=rank_seconds,actual_worker_exit_codes=result['actual_worker_exit_codes'],
        phase_including_initialization=read(BASE/'logs/phase-timings.json').get('final51713'),
        model_evaluation_rerun=False,strict_scorer_changed=False,full200_goal_complete=False,
        scope='fresh full-file SHA, canonical identities and stored strict-score aggregation; remaining checkpoint and cleanup acceptance separate')
    with proof.open('x') as f:
        json.dump(report,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status=report['status'],path=str(proof),sha256=sha(proof),question_weighted=weighted,seven_set_macro=report['seven_set_macro'])))


if __name__ == '__main__':
    main()
