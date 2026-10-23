import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from reproduction.dynrpo7b.runtime import ProbeRuntime, restored_probe_metadata
from reproduction.dynrpo7b.reward_adapter import reuse_fitted
from reproduction.dynrpo7b.deploy import sha
from reproduction.dynrpo7b.collection_resume import begin_attempt


class RecoveryTransactions(unittest.TestCase):
    def test_state_snapshot_preserves_values_and_is_frozen_after_sync_save(self):
        import io
        from dataclasses import dataclass,asdict
        import torch
        from reproduction.dynrpo7b.collection_resume import snapshot_states
        @dataclass
        class State:
            index:int
            tokens:list
            native:dict
        native={'source':{'file':'a'*64},'sampling':{'temperature':1.0}}
        states=[State(i,[1,2,3],native) for i in range(2)]
        expected=[asdict(s) for s in states]
        payload=snapshot_states(states)
        self.assertEqual(payload,expected)
        self.assertIs(payload[0]['native'],payload[1]['native'])
        stream=io.BytesIO();torch.save(payload,stream)
        states[0].tokens.append(99);native['sampling']['temperature']=2.0
        stream.seek(0);loaded=torch.load(stream,weights_only=False)
        self.assertEqual(loaded,expected)
        self.assertEqual([asdict(State(**row)) for row in loaded],expected)

    def test_pinned_collection_origin_checks_actual_writer_logs(self):
        import hashlib
        from reproduction.dynrpo7b.collection_resume import find_resume
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'new';origin=root/'failed'
            (base/'preparation').mkdir(parents=True)
            (origin/'preparation/dynrpo7b-collection').mkdir(parents=True)
            (origin/'logs').mkdir()
            pointer=origin/'preparation/dynrpo7b-collection/outer-0004.json'
            value=dict(run_id='run',outer=4,prior_writes=60,stage='collected')
            pointer.write_text(json.dumps(value))
            pin=dict(source_base=str(origin),outer=4,
                     pointer_sha256=hashlib.sha256(pointer.read_bytes()).hexdigest())
            (base/'preparation/collection-resume-origin.json').write_text(json.dumps(pin))
            self.assertEqual(find_resume(base,{'run_id':'run'},4,60),value)
            self.assertIsNone(find_resume(base,{'run_id':'run'},5,80))
            (origin/'logs/optimizer-rank-0.jsonl').write_text(json.dumps(dict(kind='OPTIMIZER_WRITE',write_count=61))+'\n')
            with self.assertRaisesRegex(ValueError,'later real Adam'):
                find_resume(base,{'run_id':'run'},4,60)
            pointer.write_text('{}')
            with self.assertRaisesRegex(ValueError,'origin changed'):
                find_resume(base,{'run_id':'run'},4,60)

    def test_published_probe_overrides_old_wrapper_raw_ema_adam(self):
        old = dict(completed_outer=10, actor_snapshot_id='run:10', reference={'path':'old'},
                   head_version='old', probe_pending=True, fit_groups=['G5'],fit_output_dir='/old/fit')
        new = dict(path='epoch3',sha256='tensor-sha')
        phase = dict(completed_actor_outer=10,probe_completed_epoch=3,probe_pending=False,probe_ref=new)
        loaded=[]
        def rpc(operation,payload):
            if operation=='load_state_dict':
                loaded.append(payload['reference'])
                return [dict(completed_update_outer=10)]*4
            return [{}]*4
        with tempfile.TemporaryDirectory() as tmp,patch('torch.load',return_value=old):
            runtime=ProbeRuntime({'output_dir':tmp},tmp,NS(dynrpo7b_probe_operation=rpc))
            runtime.restore('old-wrapper','run:10',phase=phase)
            self.assertEqual(loaded,[new]);self.assertIsNone(runtime.pending)
            self.assertEqual(runtime.version,'seed1-E9-raw-online-0010')
            self.assertTrue(runtime.fit_after_checkpoint(None,'actor10')['already_published'])
        partial=dict(phase,probe_completed_epoch=1,probe_pending=True,probe_ref={'path':'epoch1'})
        state=restored_probe_metadata(old,partial)
        self.assertEqual(state['reference']['path'],'epoch1');self.assertTrue(state['probe_pending'])
        wrapper=dict(new,reference=new)
        self.assertEqual(restored_probe_metadata(old,dict(phase,probe_ref=wrapper))['reference'],new)

    def test_pending_reward_manifest_reused_with_original_version_and_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/'old';new=Path(tmp)/'new';cp=old/'actor50';cp.mkdir(parents=True)
            (cp/'complete.json').write_text(json.dumps(dict(run_id='run',step=50,files_sha256={'config.json':'cfg'})))
            archive=old/'artifacts/round_50-attempt-original'
            hashes={}
            for category,names in [('heads',['commit_probe.pkl','boundary_probe.pkl','calibration-report.json']),('dumps',['labels.json'])]:
                (archive/category).mkdir(parents=True)
                for name in names:
                    p=archive/category/name;p.write_text(name);hashes[name]=sha(p)
            value=dict(run_id='run',round=50,encoder=str(cp),version='original-version',encoder_sha256={'config.json':'cfg'},
                head_sha256={k:hashes[k] for k in ('commit_probe.pkl','boundary_probe.pkl')},
                calibration_report_sha256=hashes['calibration-report.json'],dump_sha256={'labels.json':hashes['labels.json']})
            (archive/'version-manifest.json').write_text(json.dumps(value))
            run=dict(run_id='run',full_state_restore={'source_base':str(old)})
            self.assertEqual(reuse_fitted(new,run,50,cp),value)
            self.assertEqual(reuse_fitted(new,run,50,cp),value)
            (archive/'heads/commit_probe.pkl').write_text('corrupt')
            with self.assertRaises(ValueError):reuse_fitted(new,run,50,cp)

    def test_production_collection_resumes_after_committed_128(self):
        try:
            from reproduction.dynrpo7b.collection import collect
            from reproduction.dynrpo7b.collection_resume import CollectionJournal,load_reference
            from search_g1.qprobe_runtime import QProbeCandidateStream
            from reproduction.common8.recoverable_attempt import AttemptProgress
            from verl import DataProto
            import numpy as np
            import torch
        except ModuleNotFoundError as error:
            self.skipTest('Run this production-entry CPU fixture in the existing remote runtime: '+str(error))
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,SEARCH_G1_BASE=tmp,SLURM_JOB_ID='new-job'):
            base=Path(tmp);(base/'logs').mkdir();run=dict(run_id='run',source_sha256={'x':'a'*64})
            (base/'logs/run-manifest.json').write_text(json.dumps(run))
            parents=DataProto.from_dict(dict(input_ids=torch.ones((5120,2),dtype=torch.long),attention_mask=torch.ones((5120,2),dtype=torch.long),position_ids=torch.ones((5120,2),dtype=torch.long)))
            loader=NS(dataset=list(range(5120)),sampler=list(range(5120)))
            stream=QProbeCandidateStream(loader)
            pool_calls=[];generation=[];restores=[]
            def pool(stream,count):
                pool_calls.append(count);stream._source.consumed=count
                return None,[NS(qid=str(i)) for i in range(count)]
            stream.materialize=lambda *a:parents
            runtime=NS(output_dir=base/'probe',version='probe0',_probe_ref={'path':'probe-memory','sha256':'test'},
                begin_outer=lambda *a:None,initialise=lambda:None,
                rpc=lambda *a:None,score_groups=lambda groups,**kw:[1-i/10000 for i in range(5120)])
            workers=NS(capture_mixed_state=lambda *a,**k:[],restore_dynrpo7b_collection_rng=lambda r:restores.append(r))
            progress=NS(pending=None,attempt_index=0,cumulative_writes=0)
            trainer=NS(qprobe_runtime=runtime,global_steps=1,inline_attempt_progress=progress,actor_rollout_wg=workers,tokenizer=None)
            attempt=dict(logical_update=1,attempt_index=1,attempt_id='attempt')
            identity=dict(run_id='run',source_sha256={'x':'a'*64},slurm_job_id='old-job',logical_update=1,
                attempt_id='attempt',attempt_index=1,policy_snapshot_id='snapshot',parent_group_ids=[str(i) for i in range(5120)],
                trajectory_ids=[str(i) for i in range(25600)])
            class Collector:
                def __init__(self,manager,identity,states,**kw):
                    self.identity=identity;self.states=states;self.meta={};self.sampling={};self.phase_events=[];self.raw_generated_policy_tokens=0
                def phase(self,states,identity,prefix_only):
                    ordinals=[s.index//5 for s in states[::5]]
                    generation.append((prefix_only,ordinals))
                    for s in states:
                        s.status='paused' if prefix_only else 'complete';s.current_raw=[42] if prefix_only else [];s.response=[42];s.info_mask=[1]
                    self.phase_events.append(dict(phase='prefix' if prefix_only else 'continuation'))
                def output(self,states,identity):
                    return DataProto.from_dict(dict(responses=torch.ones((len(states),1))),non_tensors={'ordinal':np.array([s.index//5 for s in states])})
                def release(self):return []
            def vectors(parent,output,tokenizer):
                return [[0,1,0,1,0] if int(i)>=1024 else [0]*5 for i in output.non_tensor_batch['ordinal'][::5]]
            commit=CollectionJournal.commit;crashed=[]
            def crash_after_commit(self,stage,*a,**kw):
                result=commit(self,stage,*a,**kw)
                if stage=='backstop_batch' and not crashed:
                    crashed.append(True);raise RuntimeError('injected after durable128')
                return result
            manager=NS(config=NS(max_start_length=2048))
            with patch('reproduction.dynrpo7b.collection.PrefixCollector',Collector), \
                 patch('reproduction.common8.behavior_runtime.prepare_identity',return_value=identity), \
                 patch('reproduction.common8.context_budget.protocol',return_value={'total_tokens':4096}), \
                 patch('search_g1.qprobe_runtime._collect_pool',pool), \
                 patch('search_g1.qprobe_runtime._contracts',return_value={}), \
                 patch('search_g1.qprobe_runtime._complete_generation'), \
                 patch('search_g1.mixed_runtime.strict_vectors',vectors), \
                 patch.object(CollectionJournal,'commit',crash_after_commit):
                with self.assertRaisesRegex(RuntimeError,'injected'):collect(trainer,manager,stream,attempt)
                before=len(generation)
                resumed=begin_attempt(trainer,progress)
                batch,output,identity2,audit=collect(trainer,manager,stream,resumed)
            self.assertEqual(pool_calls,[5120])
            self.assertEqual([len(q) for prefix,q in generation],[5120,1024,128,128])
            self.assertEqual(generation[before][1],list(range(1152,1280)))
            self.assertEqual(audit['selected_mixed'],154)
            self.assertEqual(audit['backstop']['additional_questions'],256)
            self.assertEqual(len(restores),1)
            self.assertEqual(identity2['slurm_job_id'],'new-job')
            self.assertEqual(identity2['generation_producers'][0]['slurm_job_id'],'old-job')


if __name__=='__main__':unittest.main()
