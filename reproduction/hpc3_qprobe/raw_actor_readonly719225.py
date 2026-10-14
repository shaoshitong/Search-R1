"""One fresh CPU process for a source-authenticated original formal1..200 outer.

Combines same-fd raw trace input and canonical-gold strict labels. No model,
generation, writer, actor, head, Slurm, or controller actions. Only small JSON.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

ROOT = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
HISTORY = (
    (1, 5, 'qprobe-teacher-warmup200-4gpu-jxu957-recovery718509-20261014',
     '5816a20f331e10aaff9bbb95c313611ead289fd7', '718650'),
    (6, 200, 'qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014',
     '1843babdd96b387dd10d91e3207c9ea9489b6941', '719225'),
)


def canonical(raw, run, row, raw_file):
    # These imports occur only after the shared reader binds this fresh process
    # to its original source. Never alternate original sources within a process.
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer
    from search_g1.mixed_runtime import strict_vectors
    from search_g1.full_state_restore import equal_tree
    from search_g1.qprobe_features import sha_file, sha_json, original_question_and_id
    from search_g1.qprobe_selection import PredictQuestion
    from reproduction.hpc3_qprobe.verify_goal import verify_selection
    from actor_trace15_readonly719225 import read_bound, need
    # real-rollout -> attempt -> attempts -> question-probe -> artifacts -> BASE
    base = Path(raw_file['path']).parents[4]
    step, aid = row['step'], row['attempt_id']
    directory = Path(raw_file['path']).parent
    q_raw, selection_file = read_bound(directory/'selection.json')
    a_raw, rollout_file = read_bound(directory/'rollout-audit.json')
    completed_path = base/'artifacts/question-probe/completed'/(aid+'.json')
    complete = row.get('complete_outer', True)
    terminal_path = completed_path if complete else base/'artifacts/question-probe/aborted'/(aid+'.json')
    e_raw, head_file = read_bound(terminal_path)
    q, a, event = map(json.loads, (q_raw, a_raw, e_raw))
    need(q['step'] == a['step'] == event['outer'] == step and
         q['attempt_id'] == a['attempt_id'] == event['attempt_id'] == aid,
         'canonical selection/rollout/head identity differs')
    ident = raw['identity']
    need(ident['policy_snapshot_id'] == q['policy_snapshot_id'] == a['policy_snapshot_id'],
         'canonical original policy snapshot differs')
    train = base/'data/train.parquet'
    train_sha = sha_file(train)
    need(train_sha == run['input_sha256']['train.parquet'], 'canonical train fullSHA differs')
    frame = pd.read_parquet(train, columns=['data_source','extra_info','prompt','reward_model'])
    need(sha_file(train) == train_sha, 'canonical train bytes changed during read')
    tokenizer_proofs = {}
    original = ROOT/'source'/run['source_commit']
    for name in ('search_g1/signals.py','search_r1/qa_protocol.py','verl/utils/reward_score/qa_em.py'):
        need(sha_file(original/name) == run['source_sha256'][name], 'original strict scorer bytes differ')
    for name in ('config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'):
        digest = sha_file(Path(run['initial_model'])/name)
        need(digest == run['initial_model_sha256'][name], 'actual bound tokenizer bytes differ')
        tokenizer_proofs[name] = digest
    tokenizer = AutoTokenizer.from_pretrained(run['initial_model'], local_files_only=True, trust_remote_code=False)
    vectors = strict_vectors(raw['parents'], raw['output'], tokenizer)
    need(len(vectors) == 512 and all(len(v) == 5 for v in vectors), 'actual canonical G5 budget differs')
    equal_tree(vectors, raw['strict_correctness'])
    equal_tree(vectors, [g['strict_correctness'] for g in a['groups']])
    need(raw['selected_ordinals'] == [g['ordinal'] for g in a['groups']] and
         ident['trajectory_ids'] == [tid for g in a['groups'] for tid in g['trajectory_ids']],
         'actual selected groups/ordered IDs differ')
    measured = verify_selection(q, a, run['question_probe']['seed'])
    if complete:
        equal_tree(measured, event['prequential'])
        need(event['prediction_version'] == q['head_version'] and event['head_update']['update_count'] == step and
             event['label_policy_snapshot_id'] == q['policy_snapshot_id'] and
             event['successful_actor_snapshot_id'] == run['run_id']+':'+str(step),
             'actual actor/head prequential transaction differs')
    else:
        need(not completed_path.exists() and event['head_updated'] is False and event['labels_observed'] is True and
             event['prediction_version'] == q['head_version'] and event['label_policy_snapshot_id'] == q['policy_snapshot_id'],
             'fully skipped actor attempt committed a head or lost real labels')
    def normal(v):
        if isinstance(v,np.ndarray): return normal(v.tolist())
        if isinstance(v,np.generic): return v.item()
        if isinstance(v,dict): return {k:normal(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)): return [normal(x) for x in v]
        return v
    nt = raw['parents'].non_tensor_batch
    wanted = {str(v) for v in nt['candidate_stable_id']}
    need(len(wanted) == 512, 'raw selected question IDs duplicate')
    wanted_pairs = {tuple(json.loads(v)) for v in wanted}
    matches = {v: [] for v in wanted}
    for src, extra, prompt, reward in frame.itertuples(index=False, name=None):
        index = extra.get('index') if isinstance(extra,dict) else None
        if index is not None and (str(src),str(index)) not in wanted_pairs:
            continue
        question,qid = original_question_and_id(dict(data_source=src, extra_info=extra, prompt=prompt))
        if qid in wanted:
            matches[qid].append((question,reward))
    bound = []
    for i, group in enumerate(a['groups']):
        selected = q['questions'][group['ordinal']]
        qid = str(nt['candidate_stable_id'][i])
        question = str(nt['candidate_original_question'][i])
        point = PredictQuestion(group['ordinal'], qid, question)
        need(qid == selected['qid'] == group['question_id'] and point.question_hash == selected['question_hash'] and
             point.content_hash == selected['content_sha256'] == group['content_sha256'],
             'canonical raw question/text/content identity differs')
        gold = normal(nt['reward_model'][i]['ground_truth']['target'])
        need(matches[qid] and any(t == question and normal(r['ground_truth']['target']) == gold for t,r in matches[qid]),
             'raw question/gold absent from canonical train')
        bound.append(dict(qid=qid,question_hash=point.question_hash,target=gold))
    counts = [sum(v) for v in vectors]
    return dict(checked_unix=time.time(), train_sha256=train_sha, canonical_train_rows=len(frame),
                actual_raw_decode_strict_terminal_em_equal=True, actual_512_qid_text_gold_equal=True,
                actual_selection_G5_E64_top448_prequential_head_equal=True, questions=512, trajectories=2560,
                actual_head_committed=complete, prequential=measured,
                strict_correct=sum(counts), K_distribution={str(k):counts.count(k) for k in range(6)},
                tokenizer_sha256=tokenizer_proofs, bound_parent_gold_digest=sha_json(bound),
                selection_file=selection_file, rollout_file=rollout_file, head_file=head_file)


def main(step, attempt_id):
    if type(step) is not int or not 1 <= step <= 200:
        raise ValueError('Only original successful formal1..200 is supported')
    first, last, name, commit, job = next(row for row in HISTORY if row[0] <= step <= row[1])
    path = Path(__file__).with_name('actor_trace15_readonly719225.py')
    spec = importlib.util.spec_from_file_location('actor_trace15_readonly719225', path)
    reader = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = reader
    spec.loader.exec_module(reader)
    result = reader.audit(base=ROOT.parent/name, source=ROOT/'source'/commit, step=step,
                          job_id=job, raw_sha=None, canonical_validator=canonical, attempt_id=attempt_id)
    result['history_partition'] = dict(first=first,last=last,base=str(ROOT.parent/name),source_commit=commit,job=job)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--attempt-id', required=True)
    args = p.parse_args()
    print(json.dumps(main(args.step,args.attempt_id)), flush=True)
