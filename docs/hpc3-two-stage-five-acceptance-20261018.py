"""Independent scalar/real-ledger and authentic-G5 acceptance of R4's first five.

Run on HPC3 after complete66 and positive67. Tensor readback is the genuine
stopped CP61 CPU qualification; this audit does not rehash checkpoint payloads.
"""
import os,sys,json,hashlib,time,gzip,math
from pathlib import Path
from collections import defaultdict,Counter
B=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4')

def rd(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for raw in iter(lambda:f.read(1024**2),b''):h.update(raw)
    return h.hexdigest()
def rows(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:
        for line in f:
            if line.strip():yield json.loads(line)
def once(p,v):
    with p.open('x') as f:json.dump(v,f,sort_keys=True);f.flush();os.fsync(f.fileno())
def main():
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['DYNRPO_PROGRESSIVE_PREFIX']='cp50_prefix20_40_80_three_epoch_v1';os.environ['DYNRPO_PREFIX_TOKENS']='20'
    sys.path.insert(0,str(B/'repo'))
    from reproduction.prefix_selection200.i1_i6_checkpoint_cpu import actor_ledger
    from reproduction.prefix_selection200.two_stage_selection import specification
    from reproduction.prefix_selection200.progressive_resume import writers_gone
    from reproduction.prefix_selection200.adaptive_candidates import budget
    from reproduction.prefix_selection200.two_stage_resume import verify
    c=rd(B/'preparation/launch-config.json');verify(c)
    assert c['source_commit']=='2000d46a530c64d78be7802bf11175a68b65effe' and c['two_stage_selection']==specification(62)
    proof=B/'preparation/two-stage-five-actual-next-use.json'
    if proof.exists():
        assert rd(proof)['operator_script_sha256']==sha(__file__)
        print(json.dumps(dict(status='EXISTING_ACCEPTANCE',sha256=sha(proof))));return
    latest=rd(B/'latest-joint.json');stage=rd(B/'logs/pipeline-status.json')
    if latest['outer']<66 or stage['outer']<67:print(json.dumps(dict(status='WAITING_FOR_COMPLETE66_AND_POSITIVE67',complete_outer=latest['outer'],stage=stage)));return
    progress=[rd(B/'outers/outer-000067/prefix'/('rank-%d-progress.json'%rank)) for rank in range(4)]
    assert all(v['completed_trajectories']>0 for v in progress)
    source=Path(c['resume_joint']['source_base']);qualified=rd(c['resume_joint']['qualification_proof']);writers_gone(source,c)
    assert qualified['status']=='STOPPED_PROGRESSIVE_JOINT_FULL_SHA_CPU_READBACK_PASS' and qualified['outer']==61
    assert sha(c['resume_joint']['qualification_proof'])==c['resume_joint']['qualification_proof_sha256']
    assert qualified['actor_actual_adam_steps_per_rank']==[1220]*4 and qualified['probe']['actual_adam_steps_per_rank']==[550]*4
    for seal in qualified['verified_files']:
        s=Path(seal['path']).stat();assert (s.st_ino,s.st_size,s.st_mtime_ns,str(os.statvfs(seal['path']).f_fsid))==(seal['inode'],seal['bytes'],seal['mtime_ns'],str(seal['shared_filesystem_fsid']))
    imported=rd(B/'preparation/resume-joint-receipt.json');assert imported['completed_outer']==61 and imported['actor_updates_reexecuted']==imported['probe_updates_reexecuted']==0
    class Seals:
        def file(self,p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
    def qualified_rows(receipt):
        p=Path(receipt['path']);assert p.stat().st_size==receipt['bytes'] and sha(p)==receipt['sha256'];return list(rows(p))
    previous=rd(source/'outers/outer-000061/complete.json');checks=[]
    for n in range(62,67):
        root=B/'outers'/('outer-%06d'%n);j=rd(root/'complete.json');a=j['actor'];pf=rd(root/'prefix/complete.json');score=rd(root/'probe-score/manifest.json');inter_path=root/'probe-score/pool-selection.json';inter=rd(inter_path);final_path=root/'probe-score/actual-em-selection.json';final=rd(final_path);pool=rd(root/'full-pool/complete.json');selected=rd(root/'full/complete.json')
        assert j['run_id']==c['run_id'] and j['execution_id']==c['execution_id'] and j['outer']==n and j['runtime_source_commit']==c['source_commit']
        assert pf['questions']==5120 and pf['trajectories']==25600 and pool['questions']==1024 and pool['trajectories']==5120 and selected['questions']==512 and selected['trajectories']==2560
        assert inter['selected_questions']==1024 and inter['outcome_labels_used'] is False and final['outcome_labels_used'] is j['actor_selection_outcome_labels_used'] is True
        assert score['probe_before_checkpoint_sha256']==j['probe']['sha256']==previous['probe']['sha256'] and j['probe']['actual_adam_steps_per_rank']==[550]*4
        assert j['probe_metrics']['new_adam_steps_per_rank']==0 and j['grounding']['round']==50
        assert pf['actor_version']==pool['actor_version']==selected['actor_version']==previous['actor']['actor_version']
        assert pf['head_prefix_tokens']==20 and final['selection_scope']=='actor_only_probe_top1024_actual_EM'
        assert a['original_rollout_sha256']==sha(root/'full/complete.json')
        scores_path=root/'probe-score/candidate-scores.jsonl';assert sha(scores_path)==score['candidate_scores_sha256'];scores=list(rows(scores_path))
        expected=sorted(scores,key=lambda r:(-r['q'],r['candidate_ordinal']))[:1024]
        assert inter['selected']==expected and len({r['qid'] for r in expected})==1024
        assert sha(inter_path)==pool['selection_sha256']==final['original_pool_selection_sha256']
        own=qualified_rows(pool['own_em']);labels={r['trajectory_id']:r for r in own};assert len(labels)==5120
        expected_ids={tid for g in expected for tid in g['trajectory_ids']};assert set(labels)==expected_ids
        idx=rd(root/'probe-score/index.json');assert sha(root/'probe-score/index.json')==score['index_sha256'];prefixes={r['trajectory_id']:r for r in idx};assert len(prefixes)==25600
        for g in expected:
            for tid in g['trajectory_ids']:
                r=labels[tid];assert type(r['strict_em']) is int and r['strict_em'] in (0,1) and r['qid']==g['qid'] and r['prefix_sha256']==prefixes[tid]['prefix_sha256'] and r['prefix_scan_sha256']==pf['scan_sha256'] and r['selection_sha256']==sha(inter_path)
        actual=[]
        for g in expected:
            k=sum(labels[tid]['strict_em'] for tid in g['trajectory_ids']);actual.append(dict(g,actual_correct_count=k,actual_mixed=0<k<5))
        ranked=sorted(actual,key=lambda r:(not r['actual_mixed'],-r['q'],r['candidate_ordinal']))[:512]
        assert final['selected']==ranked and len({r['qid'] for r in ranked})==512
        mixed=sum(r['actual_mixed'] for r in actual);assert final['chosen_mixed_count']==min(mixed,512) and final['fallback_count']==512-min(mixed,512)
        assert selected['actual_selected_mixed_questions']==min(mixed,512) and selected['actual_selected_mixed_fraction']==min(mixed,512)/512 and selected['projected_generation_requests']==0
        originals={r['trajectory_id']:r for shard in pool['shards'] for r in qualified_rows(shard)}
        projected=[r for shard in selected['shards'] for r in qualified_rows(shard)];assert len(projected)==2560
        assert {r['trajectory_id'] for r in projected}=={tid for g in ranked for tid in g['trajectory_ids']}
        for row in projected:
            origin=originals[row['trajectory_id']]
            for key in ('qid','candidate_ordinal','response_index','initial_prompt_token_ids','response_token_ids','response_info_mask','response_log_probs','behavior_turns','sampling_journal','strict_em','prefix_sha256','prefix_scan_sha256','actor_version'):
                assert row[key]==origin[key],key
        rank_checks=[]
        for rank in range(4):
            ledger=actor_ledger(Seals(),a,n,rank);path=next(r['ledger_path'] for r in a['rank_receipts'] if r['rank']==rank);events=list(rows(path));restores=[e for e in events if e['event']=='ACTUAL_RESTORE'];assert len(restores)==1
            restore=restores[0];assert restore['restored_completed_outer']==n-1 and restore['actual_adam_before']==previous['actor']['actual_adam_steps_per_rank'][rank] and restore['restored_scheduler']==n-1
            assert restore['actual_learning_rates']==[1e-6] and restore['actual_model_master_dtypes']==['torch.float32'] and restore['actor_version']==previous['actor']['actor_version']
            monitors=[e for e in events if e['event']=='POLICY_MONITOR']
            assert len(monitors)==40 and Counter((m['opportunity'],m['stage']) for m in monitors)==Counter((i,stage) for i in range(20) for stage in ('before_minibatch','after_minibatch'))
            rank_checks.append(dict(rank=rank,ledger=ledger,restore=restore,actual_before_after_monitor_invocations=len(monitors)))
        assert a['scheduler_completed_outer']==n and [x-y for x,y in zip(a['actual_adam_steps_per_rank'],a['new_adam_steps_per_rank'])]==previous['actor']['actual_adam_steps_per_rank']
        bgt=rd(root/'candidate-budget.json');assert bgt==budget(c,n) and bgt['questions']==5120
        assert bgt['consumed_before']==rd(previous['pool_snapshot'])['consumed_candidates']
        state=rd(j['pool_snapshot']);assert sha(j['pool_snapshot'])==j['pool_sha256'] and state['next_outer']==n+1 and state['consumed_candidates']==bgt['consumed_after']
        checks.append(dict(outer=n,actor_actual_adam=a['actual_adam_steps_per_rank'],new_adam=a['new_adam_steps_per_rank'],probe_adam=[550]*4,candidates=5120,full_rollout_questions=1024,actual_mixed_in_1024=mixed,selected_mixed=min(mixed,512),selected_fraction=min(mixed,512)/512,rank_checks=rank_checks,complete_sha256=sha(root/'complete.json')))
        previous=j
    result=dict(status='CP61_TWO_STAGE_FIVE_COMPLETE_AND_ACTUAL_POSITIVE67_PASS',base=str(B),job=c['slurm_job'],runtime_commit=c['source_commit'],operator_script_sha256=sha(__file__),checked_unix=time.time(),imported_outer=61,checked_outers=list(range(62,67)),positive_next=67,checks=checks,source_stopped_cpu_proof_sha256=c['resume_joint']['qualification_proof_sha256'],tensor_payloads_rehashed=False,goal200_complete=False)
    once(proof,result);print(json.dumps(dict(status=result['status'],path=str(proof),sha256=sha(proof),mixed=[r['selected_mixed'] for r in checks],actor_adam=checks[-1]['actor_actual_adam'])))

if __name__=='__main__':main()
