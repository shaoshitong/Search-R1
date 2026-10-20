"""CPU-only genuine-event/own-EM/actual-journal fixture checks; no GPU/files banks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from . import g20_real_merge as m


def group(qid,offset):
    return dict(qid=qid,question_hash='hash-'+qid,trajectory_ids=['%s-event%d-r%d'%(qid,offset,j) for j in range(5)],
        labels=[0,1,0,1,0],response_indices=list(range(offset,offset+5)))

def conf(offset,base):
    return dict(role=m.ROLE,model_revision=m.REVISION,base=str(base),original_base='/unread-original',
        run_id='phase%d'%offset,sampling_attempt_id='attempt%d'%offset,original_run_id='phase0',original_sampling_attempt_id='attempt0',
        extra_response_offset=offset,extra_response_indices=list(range(offset,offset+5)),warmup_only=True,actor_updates_authorized=False,
        actor_optimizer_updates=0,capture_prefix_tokens=20,feature_layer=26,question_max_tokens=273,feature_dimension=3584,
        extra_shards=4,gpus_per_sampling_job=2,train_questions=4,train_trajectories=20,seed=1729,
        source_commit='a'*40,code_root='/unread-qualified-code',code_files_sha256={},final_plan='/MUST_NOT_OPEN_FINAL')

def fixture_seed(run_id,outer,tid,turn,chunk,seed,sampling_attempt_id=None):
    # Deterministic distinct per-event/per-question/per-slot synthetic native seeds.
    qid,event,response=tid.split('-');return int(qid[1:])*1000+int(event[5:])*10+int(response[1:])

def raw_rows(groups,c,prompts):
    result=[]
    for qid,g in groups.items():
        for j,(tid,label) in enumerate(zip(g['trajectory_ids'],g['labels'])):
            seed=fixture_seed(c['run_id'],1,tid,0,0,c['seed'],c.get('sampling_attempt_id'))
            result.append(dict(trajectory_id=tid,qid=qid,split='train',question_hash=g['question_hash'],response_index=g['response_indices'][j],
                run_id=c['run_id'],policy_snapshot_id='official7b-'+m.REVISION,initial_prompt_token_ids=prompts[qid]['initial_prompt_token_ids'],
                reward_model=prompts[qid]['reward_model'],strict_em=label,response_token_ids=[1,2],response_info_mask=[1,1],behavior_turns=[{}],
                sampling_journal=[dict(logical_turn_index=0,native_chunk_index=0,seed=seed,sampling=dict(seed=seed),sampling_attempt_id=c.get('sampling_attempt_id'),actor_version='official7b-'+m.REVISION,policy_snapshot_id='official7b-'+m.REVISION)],fixture_EM=label))
    return result


class G20MergeTests(unittest.TestCase):
    def test_exact_twenty_genuine_own_labels_and_no_selection(self):
        original={q:group(q,0) for q in ('q0','q1')};events={o:{q:group(q,o) for q in original} for o in (5,10,15)}
        joined=m.assemble(original,events,['q1','q0'])
        self.assertEqual([g['qid'] for g in joined],['q1','q0'])
        self.assertEqual(len({t for g in joined for t in g['trajectory_ids']}),40)
        self.assertEqual(joined[0]['labels'],[0,1,0,1,0]*4)
        self.assertEqual([e['response_indices'] for e in joined[0]['real_G5_events']],[list(range(o,o+5)) for o in (0,5,10,15)])
        events[10]['q0']['labels']=[1,0,1,0,1]
        self.assertEqual(m.assemble(original,events,['q1','q0'])[1]['labels'][10:15],[1,0,1,0,1])
        self.assertEqual(m.assemble(original,events,['q1','q0'])[1]['trajectory_ids'],joined[1]['trajectory_ids'])

    def test_missing_event_duplicate_ids_wrong_ownlabels_failclosed(self):
        original={q:group(q,0) for q in ('q0','q1')};events={o:{q:group(q,o) for q in original} for o in (5,10,15)}
        for mutate in ('missing','duplicate','bool_label','wrong_question'):
            changed=copy.deepcopy(events)
            if mutate=='missing':del changed[15]['q1']
            elif mutate=='duplicate':changed[10]['q0']['trajectory_ids'][0]=original['q0']['trajectory_ids'][0]
            elif mutate=='bool_label':changed[10]['q0']['labels'][0]=True
            else:changed[15]['q1']['question_hash']='foreign-question'
            with self.assertRaises((ValueError,KeyError)):m.assemble(original,changed,['q0','q1'])

    def test_all_four_actual_journal_seed_namespaces_and_labels(self):
        prompts={q:dict(initial_prompt_token_ids=[1,2,3],reward_model=dict(ground_truth=dict(target=['answer']))) for q in ('q0','q1')}
        ledger={}
        for offset in (0,5,10,15):
            groups={q:group(q,offset) for q in prompts};c=conf(offset,'/remote/event%d'%offset)
            result=m.audit_rows(iter(raw_rows(groups,c,prompts)),groups,c,prompts,ledger,event_offset=offset,score=lambda r:r['fixture_EM'],seed_function=fixture_seed)
            self.assertEqual(result['trajectories'],10);self.assertEqual(result['seed_chunks'],10)
        self.assertTrue(all(len(v)==20 for v in ledger.values()))
        # Distinct run/attempt strings alone do not prove actual seed separation.
        groups={q:group(q,15) for q in prompts};c=conf(15,'/remote/last');rows=raw_rows(groups,c,prompts)
        with self.assertRaisesRegex(ValueError,'overlap'):
            m.audit_rows(rows,groups,c,prompts,ledger,event_offset=15,score=lambda r:r['fixture_EM'],seed_function=fixture_seed)

    def test_real_em_recompute_prompt_seed_and_missing_full_row_are_enforced(self):
        prompts={'q0':dict(initial_prompt_token_ids=[1],reward_model={'ground_truth':{'target':['a']}})}
        groups={'q0':group('q0',5)};c=conf(5,'/remote/event5');rows=raw_rows(groups,c,prompts)
        for change in ('EM','prompt','seed','missing','duplicate'):
            own=copy.deepcopy(rows)
            if change=='EM':own[0]['fixture_EM']=1
            elif change=='prompt':own[0]['initial_prompt_token_ids']=[99]
            elif change=='seed':own[0]['sampling_journal'][0]['seed']+=1
            elif change=='missing':own=own[:-1]
            else:own.append(copy.deepcopy(own[0]))
            with self.assertRaises(ValueError):m.audit_rows(own,groups,c,prompts,{},event_offset=5,score=lambda r:r['fixture_EM'],seed_function=fixture_seed)

    def test_feature_events_remain_separate_and_require_authentic_identity(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);paths=[];wanted=set()
            for event in (0,5):
                d=root/str(event);d.mkdir();tids=group('q0',event)['trajectory_ids'];wanted.update(tids)
                index=d/'index.jsonl';index.write_text(''.join(json.dumps(dict(trajectory_id=t))+'\n' for t in tids))
                payload=d/'policy.bin';payload.write_bytes(b'CPU-fixture-no-real-tensor')
                m.atomic(d/'complete.json',dict(status='LOSSLESS_NATIVE_FEATURES_COMPLETE',feature_layer=26,prefix_tokens=20,dimension=3584,question_max_tokens=273,
                    files={'index.jsonl':m.ref(index),'policy.bin':m.ref(payload)}))
                paths.append(str(d/'complete.json'))
            refs=m.feature_membership(paths,wanted,fresh_all_payloads=True)
            self.assertEqual([r['path'] for r in refs],paths)
            self.assertFalse((root/'repacked').exists())
            with self.assertRaisesRegex(ValueError,'multiple'):m.feature_membership(paths+paths[:1],wanted,fresh_all_payloads=False)
            (root/'5/policy.bin').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'SHA'):m.feature_membership(paths,wanted,fresh_all_payloads=True)

    def test_completion_gate_precedes_bulk_and_rejects_wrong_namespace(self):
        with tempfile.TemporaryDirectory() as raw:
            phases={o:str(Path(raw)/str(o)) for o in (5,10,15)}
            # The original model/data/final paths deliberately do not exist.
            with self.assertRaisesRegex(ValueError,'twelve'):m.gate(phases,4)
            for offset,base in phases.items():
                b=Path(base);c=conf(offset,b);m.atomic(b/'config.json',c)
                for shard in range(4):
                    d=b/'sampling'/('shard-%02d'%shard)
                    m.atomic(d/'complete.json',dict(status='ALL_ASSIGNED_CANONICAL_G5_COMPLETE',shard=shard,run_id=c['run_id'],model_revision=m.REVISION,questions=1,trajectories=5))
                    m.atomic(d/'extra-g5-complete.json',dict(status='EXTRA_REAL_G5_SHARD_NATIVE_FULL_OWN_EM_SHA_VERIFIED',run_id=c['run_id'],sampling_attempt_id=c['sampling_attempt_id'],shard=shard,actor_updates=0,synthetic_EM=False,model_revision=m.REVISION,questions=1,trajectories=5,groups=[group('q%d'%shard,offset)]))
                    m.atomic(d/'extra-g5-adapter-controller.json',dict(run_id=c['run_id'],sampling_attempt_id=c['sampling_attempt_id'],source_commit=c['source_commit'],original_actor_fresh_all_file_SHA=True,response_indices=c['extra_response_indices']))
            got=m.gate(phases,4);self.assertEqual(set(got),{5,10,15})
            path=Path(phases[15])/'sampling/shard-03/complete.json';path.unlink()
            with self.assertRaisesRegex(ValueError,'twelve'):m.gate(phases,4)

if __name__=='__main__':unittest.main()
