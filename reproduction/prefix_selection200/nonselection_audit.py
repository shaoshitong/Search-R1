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


if __name__=='__main__':
    print(json.dumps(collect(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 50)))
