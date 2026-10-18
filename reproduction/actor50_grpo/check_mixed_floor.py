"""CPU-only receipt integration fixtures for exact conditional refill semantics."""
import gzip
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from .schedule import BRANCH_SCHEDULE, outer_plan
from .mixed_floor import complete_pool, refill


def fixture(outer,mixed_ordinals,expected_mixed,expected_batches,*,corrupt=False):
    from reproduction.prefix_selection200 import rollout
    from reproduction.prefix_selection200.periodic_data import read,digest,immutable,iter_rows
    from reproduction.prefix_selection200.interaction_protocol import CONTRACT
    from reproduction.prefix_selection200.contract import PREFIX_TOKENS
    with tempfile.TemporaryDirectory(prefix='synthetic-mixed30-') as temp:
        base=Path(temp);root=base/'outers'/('outer-%06d'%outer);(root/'prefix').mkdir(parents=True)
        (root/'probe-score').mkdir();plan=outer_plan(outer)
        c=dict(base=str(base),run_id='fixture',actor_version='fixture-actor',actor_model_dir='/data/synthetic',
            world_size=4,prefix_policy_tokens=PREFIX_TOKENS,interaction_protocol=CONTRACT,
            grpo_from_actor50=True,actor50_branch_schedule=BRANCH_SCHEDULE,actor50_pool_consumed_candidates=230400,
            candidate_schedule=dict(effective_from_outer=6,legacy_candidates=2560,ordinary_candidates=5120,
                update_candidates=2560,probe_update_period=10,probe_update_at_final=False,selected_questions=512))
        config=base/'config.json';immutable(config,c)
        prefix=dict(run_id='fixture',outer=outer,actor_version=c['actor_version'],scan_sha256='a'*64)
        immutable(root/'prefix/complete.json',prefix)
        groups=[dict(qid='synthetic-q%d'%i,candidate_ordinal=i,q=1-i/3000,question_hash='%064x'%i,
            trajectory_ids=['synthetic-%d-%d'%(i,j) for j in range(5)]) for i in range(2560)]
        common=dict(prefix,prefix_scan_sha256=prefix['scan_sha256'],candidate_questions=2560,
            trajectories_per_question=5,exploration=0,outcome_labels_used=False)
        predicted_path=root/'probe-score/selection.json'
        immutable(predicted_path,dict(common,selected=groups[:plan['predicted_top_questions']],selected_questions=plan['predicted_top_questions']))
        if plan['probe_update']:immutable(root/'probe-score/pool-selection.json',dict(common,selected=groups,selected_questions=2560))
        index=[dict(trajectory_id=tid,prefix_sha256='%064x'%(g['candidate_ordinal']*5+j))
            for g in groups for j,tid in enumerate(g['trajectory_ids'])]
        index_path=root/'probe-score/index.json';immutable(index_path,index);prefixes={r['trajectory_id']:r['prefix_sha256'] for r in index}
        scores=root/'probe-score/candidate-scores.jsonl';scores.write_text(''.join(json.dumps(g)+'\n' for g in groups))
        immutable(root/'probe-score/manifest.json',dict(index_sha256=digest(index_path),candidate_scores_sha256=digest(scores),
            selection_sha256=digest(predicted_path),no_outcome_labels=True,prefix_scan_sha256=prefix['scan_sha256']))
        generated=[];original={}
        def native(config_path,actual_outer,phase,selection_path):
            cc=read(config_path);out=rollout.phase_output(cc,actual_outer,phase);out.mkdir()
            assert rollout.question_count(cc,phase,actual_outer)==len(read(selection_path)['selected'])
            selected=read(selection_path)['selected'];labels=[];shards=[]
            for rank in range(4):
                path=out/('rank-%d-full.jsonl.gz'%rank);rows=[]
                for g in selected[rank::4]:
                    assert g['qid'] not in generated;generated.append(g['qid'])
                    k=2 if g['candidate_ordinal'] in mixed_ordinals else (5 if g['candidate_ordinal']%2 else 0)
                    for j,tid in enumerate(g['trajectory_ids']):
                        row=dict(trajectory_id=tid,qid=g['qid'],candidate_ordinal=g['candidate_ordinal'],response_index=j,
                            actor_version=c['actor_version'],prefix_sha256=prefixes[tid],prefix_scan_sha256=prefix['scan_sha256'],
                            selection_sha256=digest(selection_path),strict_em=int(j<k),
                            initial_prompt_token_ids=[g['candidate_ordinal'],j],response_token_ids=[17,j,29],
                            response_info_mask=[1,0,1],response_log_probs=[-.1,-.2,-.3])
                        original[tid]=dict(row);rows.append(row);labels.append(row)
                with gzip.open(path,'wt') as f:
                    for row in rows:f.write(json.dumps(row)+'\n')
                shards.append(rollout.file_item(path,len(rows)))
            own=out/'own-em.jsonl'
            if corrupt:labels[0]=dict(labels[0],strict_em=.5)
            own.write_text(''.join(json.dumps(r)+'\n' for r in labels))
            receipt=dict(prefix,prefix_scan_sha256=prefix['scan_sha256'],questions=len(selected),trajectories=len(selected)*5,
                selection_sha256=digest(selection_path),shards=shards,own_em=rollout.file_item(own,len(labels)),
                native_generation_requests=len(labels),actual_sampled_tokens=len(labels)*3)
            immutable(out/'complete.json',receipt);return receipt
        before=digest(predicted_path)
        with patch.object(rollout,'launch_phase',native):
            if corrupt:
                try:complete_pool(config,outer)
                except ValueError:return dict(outer=outer,invalid_strict_EM_rejected=True)
                raise AssertionError('Fractional EM admitted')
            result=complete_pool(config,outer)
        stats=read(root/'mixed-floor-complete.json')
        assert stats['final_mixed']==expected_mixed and stats['extra_native_batches']==expected_batches,stats
        assert digest(predicted_path)==before and len(generated)==len(set(generated))==stats['completed_candidate_questions']
        selected=read(root/'probe-score/actual-em-selection.json')['selected'];wanted={t for g in selected for t in g['trajectory_ids']}
        rows=[r for shard in result['shards'] for r in iter_rows(shard['path'])]
        assert len(rows)==2560 and {r['trajectory_id'] for r in rows}==wanted
        for row in rows:
            old=original[row['trajectory_id']]
            for key in ('initial_prompt_token_ids','response_token_ids','response_info_mask','response_log_probs','strict_em','prefix_sha256'):
                assert row[key]==old[key]
        if not stats['triggered'] and not plan['probe_update'] and not plan['actual_em_rerank']:
            assert [g['qid'] for g in selected]==[g['qid'] for g in groups[:512]]
        return dict(outer=outer,final_mixed=expected_mixed,extra_batches=expected_batches,
            completed_questions=stats['completed_candidate_questions'],exhausted=stats['exhausted'],
            original_G5_tokens_masks_logprobs_and_EM_preserved=True)


def main():
    cases=[fixture(51,set(range(154)),154,0),
        fixture(51,set(range(153))|{512},154,1),
        fixture(51,set(range(100))|set(range(512,566)),154,1),
        fixture(51,set(range(100))|set(range(512,532)),120,16),
        fixture(151,set(range(100))|set(range(1024,1078)),154,1),
        fixture(150,set(range(100))|set(range(1024,1078)),154,0),
        fixture(60,set(range(120)),120,0),fixture(51,set(range(100)),0,0,corrupt=True)]
    print(json.dumps(dict(status='MIXED30_CPU_RECEIPT_FIXTURES_PASS',cases=cases,no_CUDA=True)))


if __name__=='__main__':main()
