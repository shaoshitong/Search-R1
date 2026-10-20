"""Small CPU fixtures for raw migration acceptance, not synthetic training data."""
import copy
import json
import importlib.util
import sys
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest
from . import peer_newseed_ingest as m
from .test_peer_newseed_receive import fixture as catalog_fixture


def group(qid,event=0,ordinal=0,qhash=None):
    return dict(qid=qid,question_hash=qhash or m.digest(qid),source_ordinal=ordinal,
        trajectory_ids=['%s-event%d-g%d'%(qid,event,j) for j in range(5)],labels=[0,1,0,1,0])


def exclusions():
    return {role:dict(role=role,identity_only=True,groups=[dict(qid='exclude-'+role,question_hash=m.digest(role))],
        source_ref=dict(path='/MUST_NOT_OPEN_FINAL_SOURCE',sha256='a'*64)) for role in m.SCOPES}


class CharTokenizer:
    def decode(self,ids): return ''.join(chr(i) for i in ids)


def trajectory(config):
    source=dict(qid='q',question_hash=m.digest('question'),candidate_ordinal=7,source_ordinal=7,split='train',
        initial_prompt_token_ids=[90,91],reward_model=dict(ground_truth=dict(target=['Paris'])),data_source='toy')
    tid='%s-outer000001-c0007-g0'%config['run_id'];actor='official7b-'+m.REVISION
    turns=[];response=[];mask=[];logs=[];segments=[];retained=[];positions=[];journal=[]
    for t,(text,observation) in enumerate([('<search>q</search>','<answer>fake</answer>'),('<answer>Paris</answer>','')]):
        request=source['initial_prompt_token_ids']+response;raw=list(map(ord,text));lp=[-.1]*len(raw)
        seed=m.audit.seed_for(config['run_id'],tid,t,0,config['seed'],config.get('sampling_attempt_id'))
        chunk=dict(actual_request_token_ids=request,raw_generated_token_ids=raw,selected_logprobs=lp,
            request_id=tid+'-t%d-chunk0'%t,logical_turn_index=t,native_chunk_index=0,seed=seed,
            sampling_attempt_id=config.get('sampling_attempt_id'),sampling=dict(seed=seed))
        turn=dict(actual_request_token_ids=request,raw_generated_token_ids=raw,selected_logprobs=lp,
            kept_token_count=len(raw),trajectory_id=tid,policy_snapshot_id=actor,turn_index=t,
            prefix_cut=dict(kept_token_ids=raw,kept_token_count=len(raw),extra_provenance=True),
            actual_sampling_chunks=[chunk],observation_token_ids=list(map(ord,observation)),correction_token_ids=[])
        take=min(len(raw),20-len(retained))
        if take:
            segments.append(dict(turn_index=t,policy_start=len(retained),policy_length=take,request_length=len(request),
                position_start=len(request),context_tokens=request+raw[:take]))
            retained.extend(raw[:take]);positions.extend(range(len(request),len(request)+take))
        turns.append(turn);journal.append(chunk);response.extend(raw+list(map(ord,observation)))
        mask.extend([1]*len(raw)+[0]*len(observation));logs.extend(lp+[0.]*len(observation))
    common=dict(run_id=config['run_id'],outer=1,trajectory_id=tid,qid='q',question_hash=source['question_hash'],
        candidate_ordinal=7,response_index=0,split='train',actor_version=actor,initial_prompt_token_ids=source['initial_prompt_token_ids'])
    prefix=dict(common,policy_token_ids=retained,policy_position_ids=positions,prefix_segments=segments,
        retained_policy_length=len(retained),terminal=False,terminal_reason=None,future_information_included=False)
    prefix['prefix_sha256']=m.digest(prefix)
    full=dict(common,prefix_sha256=prefix['prefix_sha256'],policy_snapshot_id=actor,reward_model=source['reward_model'],
        data_source='toy',strict_em=1,response_token_ids=response,response_info_mask=mask,response_log_probs=logs,
        behavior_turns=turns,sampling_journal=journal,terminal_reason='answer')
    return source,prefix,full


def strict_score(row):
    from reproduction.prefix20_online.rollout import strict_label
    # Load the exact QA module without verl.__init__ optional distributed deps.
    # Production strict_label still executes unchanged; no EM result is stubbed.
    path=Path(m.__file__).parents[3]/'verl/utils/reward_score/qa_em.py'
    spec=importlib.util.spec_from_file_location('cpu_actual_qa_em',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with patch.dict(sys.modules,{'verl.utils.reward_score.qa_em':module}):
        return strict_label(CharTokenizer(),row['response_token_ids'],row['response_info_mask'],row['reward_model']['ground_truth'])


def received_fixture(root):
    catalog,data=catalog_fixture();snapshot=catalog['snapshot_id'];base=root/snapshot;m.receiver.secure_dir(base)
    record=catalog['groups'][0];archive=base/'archives'/Path(record['archive']).name
    m.receiver.secure_dir(archive.parent);archive.write_bytes(b'fixture archive bytes, raw SHA gate only')
    record['archive_bytes']=archive.stat().st_size;record['archive_sha256']=m.receiver.sha(archive)
    catalog['source_snapshot']['groups'][0].update(archive_bytes=record['archive_bytes'],archive_sha256=record['archive_sha256'])
    raw=base/'raw'/record['files'][0]['path'];m.receiver.secure_dir(raw.parent);raw.write_bytes(data)
    original=m.canonical(catalog).decode();binding=dict(prefix=catalog['prefix'],catalog_commit='a'*40,
        catalog_bytes=original,catalog_sha256=m.digest(catalog))
    m.atomic(base/'catalog.json',catalog);m.atomic(base/'catalog-binding.json',binding)
    m.atomic(base/'accepted/group-00000.json',dict(status='TARGET_ARCHIVE_AND_ALL_MEMBER_FILES_FRESH_SHA_VERIFIED',
        catalog_sha256=binding['catalog_sha256'],archive_sha256=record['archive_sha256'],archive_bytes=record['archive_bytes'],files=1))
    m.atomic(base/'complete.json',dict(status=m.receiver.STATUS,snapshot_id=snapshot,target=str(base/'raw'),catalog_commit='a'*40,
        catalog_sha256=binding['catalog_sha256'],all_archive_SHA256=True,fresh_all_file_SHA256=True,feature_bank_created=False,
        actor_updates=0,probe_updates=0,groups=1,questions=1,trajectories=5,files=1))
    return base,raw


class IngestTests(unittest.TestCase):
    def test_real_received_allfile_SHA_gate_and_missing_target_do_not_prepare(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();base,payload=received_fixture(root)
            done,_,_=m.receiver_gate(base,fresh_files=True);self.assertEqual(done['questions'],1)
            payload.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'SHA or size'):m.receiver_gate(base,fresh_files=True)
            untouched=root/'future';result=m.prepare(root/'absent',untouched)
            self.assertFalse(result['prepared']);self.assertFalse(untouched.exists())

    def test_only_eval_identity_exclusions_preserve_existing_fit_and_ownG10(self):
        ex=exclusions();counts={k:1 for k in ex};old=[group('fit6528',0),group('new-train',0,1)]
        new=[group('fit6528',1),group('new-train',1,1)]
        accepted,report=m.qualify_membership(old,new,ex,counts)
        self.assertEqual([g['qid'] for g in accepted],['fit6528','new-train'])
        self.assertTrue(report['matched_fit6528_not_an_exclusion']);self.assertTrue(report['original_warmup_saw_all_169028_training_questions'])
        self.assertEqual(accepted[0]['labels'],[0,1,0,1,0]*2);self.assertEqual(len(set(accepted[0]['trajectory_ids'])),10)
        alias='hash-alias';old.append(group(alias,0,2,ex['selection2048']['groups'][0]['question_hash']))
        new.append(group(alias,1,2,ex['selection2048']['groups'][0]['question_hash']))
        # Poisoned labels of an excluded hash alias are deliberately not read.
        new[-1]['labels']=['MUST_NOT_INSPECT']
        _,report=m.qualify_membership(old,new,ex,counts);self.assertEqual(report['additional_excluded_by_question_hash'],1)

    def test_identity_labels_collision_and_missing_ownG5_failclosed(self):
        ex=exclusions();counts={k:1 for k in ex};old=[group('q')];new=[group('q',1)]
        for what in ('noncanonical','wrong_hash','missing','bool_label','duplicate_tid'):
            bad=copy.deepcopy(new)
            if what=='noncanonical':bad[0]['qid']='foreign'
            elif what=='wrong_hash':bad[0]['question_hash']='b'*64
            elif what=='missing':bad[0]['labels']=bad[0]['labels'][:4]
            elif what=='bool_label':bad[0]['labels'][0]=True
            else:bad[0]['trajectory_ids'][0]=old[0]['trajectory_ids'][0]
            with self.assertRaises((ValueError,KeyError)):m.qualify_membership(old,bad,ex,counts)
        bad=copy.deepcopy(ex);bad['frozen_final4096']['groups'][0]['labels']=[0]
        with self.assertRaisesRegex(ValueError,'only qid'):m.exclusion_sets(bad,counts)

    def test_tokenizer_strictEM_causal_tool_context_prefix20_and_true_seed_independence(self):
        ledger={};config=dict(run_id='new-real-run',seed=20261021,sampling_attempt_id='new-attempt')
        source,prefix,full=trajectory(config)
        result=m.audit_trajectory(prefix,full,source,config,1,strict_score,ledger,'new')
        self.assertEqual(result['multi_segment_prefix'],1);self.assertEqual(result['seed_chunks'],2)
        self.assertEqual(result['short_terminal_prefix'],0)
        old=dict(run_id='old-real-run',seed=1729)
        s,p,f=trajectory(old);p.update(feature_layer=26,feature_row_locator='unchanged-native-locator')
        m.audit_trajectory(p,f,s,old,1,strict_score,ledger,'old')
        self.assertEqual(len(ledger['q']),4)
        with self.assertRaisesRegex(ValueError,'seed overlap'):m.audit_trajectory(prefix,full,source,config,1,strict_score,ledger,'duplicate')

    def test_wrong_own_EM_future_context_positions_journal_and_seed_are_rejected(self):
        config=dict(run_id='new-real-run',seed=20261021,sampling_attempt_id='new-attempt')
        source,prefix,full=trajectory(config)
        for change in ('EM','context','position','seed','journal','future'):
            p,f=copy.deepcopy(prefix),copy.deepcopy(full)
            if change=='EM':f['reward_model']=copy.deepcopy(source['reward_model']);f['reward_model']['ground_truth']['target']=['London'];s=copy.deepcopy(source);s['reward_model']=f['reward_model']
            else:s=source
            if change=='context':f['behavior_turns'][1]['actual_request_token_ids']+=[77]
            elif change=='position':p['policy_position_ids'][-1]+=1;p['prefix_sha256']=m.digest({k:v for k,v in p.items() if k!='prefix_sha256'});f['prefix_sha256']=p['prefix_sha256']
            elif change=='seed':f['behavior_turns'][0]['actual_sampling_chunks'][0]['seed']+=1
            elif change=='journal':f['sampling_journal']=f['sampling_journal'][:1]
            elif change=='future':p['future_information_included']=True
            with self.assertRaises(ValueError):m.audit_trajectory(p,f,s,config,1,strict_score,{},'new')

    def test_protocol_sampling_contract_is_not_assumed_equal(self):
        old={key:dict(exact='contract') for key in m.PROTOCOL};new=copy.deepcopy(old)
        self.assertEqual(m.compare_protocol(old,new),old)
        new['sampling_contract']={'temperature':999}
        with self.assertRaisesRegex(ValueError,'protocols differ'):m.compare_protocol(old,new)

if __name__=='__main__':unittest.main()
