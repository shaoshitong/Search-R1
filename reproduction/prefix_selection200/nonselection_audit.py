"""Read-only CPU audit of actual rollout budgets and historical loss weights.

No model/tensor loading, new generation, evaluator calls, or training mutation.
"""
import collections
import datetime
import gzip
import json
from pathlib import Path
import statistics
import sys


def read(path):
    return json.loads(Path(path).read_text())


def collect(base, outer=50):
    base = Path(base)
    sys.path.insert(0, str(base/'repo'))
    from reproduction.prefix_selection200.actor import rank_plan
    joint = read(base/'outers'/('outer-%06d'%outer)/'complete.json')
    actor, rollout = joint['actor'], joint['rollout']
    records=[]
    for shard in rollout['shards']:
        path=Path(shard['path'])
        with (gzip.open if path.suffix=='.gz' else open)(path,'rt') as stream:
            records.extend(json.loads(line) for line in stream)
    records.sort(key=lambda row:(row['candidate_ordinal'],row['response_index']))
    assert len(records)==2560
    turns=[turn for row in records for turn in row['behavior_turns']]
    lengths=[sum(row['response_info_mask']) for row in records]
    early=[turn for turn in turns if turn['turn_index']<4
           and len(turn['actual_request_token_ids'])+1000>4096]
    variations, ratios, empty_pairs = [], [], 0
    for plan in rank_plan(1729,outer):
        for ids in plan:
            total=sum(lengths[i] for i in ids)
            variation=0
            for start in range(0,16,2):
                pair=ids[start:start+2];mass=sum(lengths[i] for i in pair)
                if total and mass:
                    old, new = 1/(8*mass), 1/total
                    variation+=sum(lengths[i]*abs(old-new) for i in pair)
                    ratios.extend(new/old for i in pair if lengths[i])
                else:
                    empty_pairs+=1
            variations.append(.5*variation)
    first_updates=[]
    for receipt in actor['rank_receipts']:
        values=[json.loads(line) for line in Path(receipt['ledger_path']).read_text().splitlines()]
        first=next(row for row in values if row['event']=='ACTUAL_ADAM_WRITE')
        first_updates.append({key:first[key] for key in ('rank','actual_learning_rate','grad_norm','metrics')})
    return dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        run_id=joint['run_id'],outer=outer,trajectories=len(records),turn_count=len(turns),
        policy_lengths=dict(mean=statistics.mean(lengths),min=min(lengths),max=max(lengths)),
        early_final_budget_triggered_turns=len(early),
        early_final_affected_trajectories=len({turn['trajectory_id'] for turn in early}),
        observation_clipped_turns=sum(bool(turn.get('observation_clipped')) for turn in turns),
        total_variation_of_policy_token_weights_new_vs_legacy_micro2=dict(
            mean=statistics.mean(variations),min=min(variations),max=max(variations)),
        token_weight_ratio_new_over_legacy=dict(min=min(ratios),max=max(ratios)),
        empty_legacy_pairs=empty_pairs,
        weighting_comparison_scope='same_actual_8rank_plan; legacy_equal_micro2_means_vs_current_local16_token_mean; not_a_gradient_or_EM_replay',
        actual_actor_binding=actor['binding'],actual_reward_proof=actor['reward'],
        first_update_per_rank=first_updates,
        training_modified=False,model_or_optimizer_loaded=False,gpu_forward_performed=False,
        all_trajectory_payloads_remain_remote=True)


def qa_shape(base, outer=50):
    """Read existing answer-presence and observation-wrapper counts; no scoring rerun."""
    from transformers import AutoTokenizer
    base=Path(base)
    config=read(base/'preparation/launch-config.json')
    tokenizer=AutoTokenizer.from_pretrained(config['official_model_dir'],local_files_only=True,
                                          trust_remote_code=False)
    joint=read(base/'outers'/('outer-%06d'%outer)/'complete.json')
    source=Path(joint['rollout']['own_em']['path']).parent.parent
    qa=read(source/'validation/complete.json')
    rows=[row for item in qa['rank_receipts'] for row in read(item['path'])['rows']]
    assert len(rows)==512 and sum(row['em'] for row in rows)/512==qa['em']
    def observations(turns):
        values=[]
        for turn in turns:
            ids=turn.get('observation_token_ids') or []
            text=tokenizer.decode(ids) if ids else ''
            if '<information>' in text:
                values.append(dict(tokens=len(ids),closing='</information>' in text))
        return dict(information_observations=len(values),
                    missing_closing=sum(not row['closing'] for row in values),
                    exactly500=sum(row['tokens']==500 for row in values))
    qa_turns, policy_lengths=[], []
    contingencies=collections.defaultdict(lambda:dict(questions=0,missing_answer=0,correct=0))
    missing_answer_finish=collections.Counter()
    for shard in qa['output_shards']:
        with gzip.open(shard['path'],'rt') as stream:
            for line in stream:
                row=json.loads(line)
                qa_turns.extend(row['behavior_turns'])
                policy_lengths.append(sum(row['response_info_mask']))
                broken=observations(row['behavior_turns'])['missing_closing']>0
                cell=contingencies['unclosed_information' if broken else 'closed_or_no_information']
                cell['questions']+=1
                cell['missing_answer']+=int(not row['answer_present'])
                cell['correct']+=row['em']
                if not row['answer_present']:
                    last=row['behavior_turns'][-1]
                    missing_answer_finish[str((last['ordinal'],last['finish_reason']))]+=1
    train_turns=[]
    for shard in joint['rollout']['shards']:
        with gzip.open(shard['path'],'rt') as stream:
            for line in stream:
                train_turns.extend(json.loads(line)['behavior_turns'])
    return dict(outer=outer,qa_count=512,em=qa['em'],f1=qa['f1'],
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        answer_present=sum(row['answer_present'] for row in rows),
        missing_answer=sum(not row['answer_present'] for row in rows),
        wrong_with_answer=sum(row['answer_present'] and not row['em'] for row in rows),
        mean_retrieval_calls=statistics.mean(row['actual_retrieval_calls'] for row in rows),
        mean_retained_policy_tokens=statistics.mean(policy_lengths),
        qa_observations=observations(qa_turns),train_observations=observations(train_turns),
        qa_observation_answer_contingency=dict(contingencies),
        missing_answer_final_turn_finish=dict(missing_answer_finish),
        model_weights_loaded=False,new_generation=False,new_scoring=False,training_modified=False,
        full_outputs_remain_remote=True)


def rejected_answer_audit(base, outer=50):
    """Classify existing strict-score rejections; never replace official scores."""
    import re
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    base=Path(base)
    sys.path.insert(0,str(base/'repo'))
    from search_g1.signals import decode_segments
    from search_r1.qa_protocol import parse_model_action, _ACTION, _RESERVED, _EOS
    from verl.utils.reward_score.qa_em import em_check
    config=read(base/'preparation/launch-config.json')
    tokenizer=AutoTokenizer.from_pretrained(config['official_model_dir'],local_files_only=True,
                                          trust_remote_code=False)
    canonical=pq.read_table(config['validation_questions']).to_pylist()
    assert len(canonical)==512
    joint=read(base/'outers'/('outer-%06d'%outer)/'complete.json')
    source=Path(joint['rollout']['own_em']['path']).parent.parent
    qa=read(source/'validation/complete.json')
    counts=collections.Counter()
    by_broken=collections.defaultdict(collections.Counter)
    rejection_reasons=collections.Counter()
    for shard in qa['output_shards']:
        with gzip.open(shard['path'],'rt') as stream:
            for line in stream:
                row=json.loads(line)
                counts['questions']+=1
                counts['official_correct']+=row['em']
                if row['answer_present']:
                    continue
                counts['strict_missing_or_rejected']+=1
                text,ranges=decode_segments(tokenizer,row['response_token_ids'],row['response_info_mask'])
                span=text[slice(*ranges[-1])] if ranges else ''
                matches=list(re.finditer(r'<answer>(.*?)</answer>',span,re.DOTALL))
                answers=[match.group(1).strip() for match in matches if match.group(1).strip()]
                gold=canonical[row['evaluation_ordinal']]['reward_model']['ground_truth']['target']
                has_broken=any('<information>' in tokenizer.decode(t.get('observation_token_ids')or[])
                    and '</information>' not in tokenizer.decode(t.get('observation_token_ids')or[])
                    for t in row['behavior_turns'])
                cell=by_broken['unclosed_information' if has_broken else 'closed_or_no_information']
                cell['strict_missing_or_rejected']+=1
                if not answers:
                    counts['no_nonempty_complete_answer_tag']+=1
                    cell['no_nonempty_complete_answer_tag']+=1
                    continue
                counts['has_nonempty_complete_answer_tag_but_rejected']+=1
                cell['has_nonempty_complete_answer_tag_but_rejected']+=1
                counts['last_tag_payload_matches_gold_but_rejected']+=int(em_check(answers[-1],gold))
                clean=_EOS.sub('',span.strip()).strip()
                actions=list(_ACTION.finditer(clean))
                if len(actions)!=1:
                    reason='not_exactly_one_action_in_final_policy_span'
                else:
                    action=actions[0]
                    payload=action.group(2).strip()
                    prefix=clean[:action.start()].strip()
                    if not payload:
                        reason='empty_payload'
                    elif _RESERVED.search(payload):
                        reason='protocol_tag_inside_answer_payload'
                    elif clean[action.end():].strip():
                        reason='nonwhitespace_after_action_close'
                    elif _RESERVED.search(prefix):
                        reason='invalid_protocol_tag_in_reasoning_prefix'
                        counts['rejected_prefix_has_information_close']+=int('</information>' in prefix)
                        if '</information>' in prefix:
                            remaining=clean[clean.rfind('</information>',0,action.start())+len('</information>'):]
                            parsed_after_info=parse_model_action(remaining)
                            if parsed_after_info is not None and parsed_after_info[0]=='answer':
                                counts['valid_answer_after_model_finished_information_block']+=1
                                counts['gold_matching_valid_answer_after_model_finished_information_block']+=int(em_check(parsed_after_info[1],gold))
                    else:
                        reason='other'
                rejection_reasons[reason]+=1
                # Diagnostic only: remove solely leading environment-closing repair tags.
                repaired,n=re.subn(r'^\s*</information>\s*','',span,count=1)
                parsed=parse_model_action(repaired) if n else None
                if parsed is not None and parsed[0]=='answer':
                    counts['rejected_only_by_leading_information_close_repair']+=1
                    correct=int(em_check(parsed[1],gold))
                    counts['gold_matching_answer_rejected_only_by_information_close_repair']+=correct
                    cell['rejected_only_by_leading_information_close_repair']+=1
                    cell['gold_matching_repair_rejection']+=correct
    assert counts['questions']==512 and counts['official_correct']/512==qa['em']
    return dict(outer=outer,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        counts=dict(counts),rejections_by_observation=dict(by_broken),
        strict_rejection_reason_counts=dict(rejection_reasons),official_em=qa['em'],
        diagnostic_not_replacement_scoring=True,official_scores_or_rules_modified=False,
        generated_text_exported=False,new_generation=False,model_weights_loaded=False,
        training_modified=False)


if __name__=='__main__':
    if len(sys.argv)>2 and sys.argv[2]=='qa':
        print(json.dumps(qa_shape(sys.argv[1],int(sys.argv[3]) if len(sys.argv)>3 else 50)))
    elif len(sys.argv)>2 and sys.argv[2]=='rejections':
        print(json.dumps(rejected_answer_audit(sys.argv[1],int(sys.argv[3]) if len(sys.argv)>3 else 50)))
    else:
        print(json.dumps(collect(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 50)))
