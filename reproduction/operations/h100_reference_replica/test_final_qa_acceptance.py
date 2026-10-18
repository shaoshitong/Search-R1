"""Tiny synthetic gzip/metric fixtures, not an actual final QA acceptance."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .final_qa_acceptance import FIELDS,read_shard,aggregates,matching_aggregate,inspect_final,scoring_publication
from .final_qa_acceptance_control import remote_inner


def score(q,p,response,mask):
    return dict(em=int(response[0]==q['gold']),f1=float(response[0]==q['gold']),
        answer_present=True,answer_sha256=hashlib.sha256(str(response[0]).encode()).hexdigest())


class FinalQAAcceptanceTests(unittest.TestCase):
    def fixture(self,base,mutate=None):
        questions=[dict(qid='q%d'%i,data_source='nq'if i<4 else'hotpotqa',gold=i//4,
            prompt=[123,i])for i in range(8)]
        full=[]
        for ordinal in (4,0):
            q=questions[ordinal];response=[q['gold']]
            row=dict(qid=q['qid'],data_source=q['data_source'],evaluation_ordinal=ordinal,
                actual_retrieval_calls=0,valid_actions=1,turns=1,response_token_count=1,
                terminal_reason='answer',initial_prompt_token_ids=q['prompt'],
                response_token_ids=response,response_info_mask=[1],behavior_turns=[{}],
                **score(q,q['prompt'],response,[1]))
            full.append(row)
        if mutate:mutate(full,questions)
        path=base/'rank-0-full.jsonl.gz';payload=gzip.compress(('\n'.join(json.dumps(row)for row in full)+'\n').encode(),mtime=0);path.write_bytes(payload)
        producer=dict(path=str(path),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
        worker=dict(rows=sorted([{key:row[key]for key in FIELDS}for row in full],key=lambda row:row['evaluation_ordinal']))
        return producer,worker,questions

    def test_real_gzip_pass_hashes_and_rescores_arrival_order(self):
        with tempfile.TemporaryDirectory()as tmp:
            p,w,q=self.fixture(Path(tmp));seen,proof=read_shard(p,w,q,0,lambda q:q['prompt'],score)
            self.assertEqual(set(seen),{0,4});self.assertEqual(proof['sha256'],p['sha256'])
            self.assertEqual(proof['compressed_file_read_passes'],1)
            self.assertTrue(proof['actual_original_full_sha_and_cpu_rescore'])

    def test_self_consistent_wrong_question_and_prompt_are_rejected(self):
        for field,value in [('qid','another-q'),('initial_prompt_token_ids',[999])]:
            with tempfile.TemporaryDirectory()as tmp:
                p,w,q=self.fixture(Path(tmp),lambda rows,questions:rows[0].update({field:value}))
                with self.assertRaises(ValueError):read_shard(p,w,q,0,lambda q:q['prompt'],score)

    def test_stored_em_is_recomputed_from_canonical_gold(self):
        with tempfile.TemporaryDirectory()as tmp:
            p,w,q=self.fixture(Path(tmp));q[4]['gold']=42
            with self.assertRaisesRegex(ValueError,'strict EM'):read_shard(p,w,q,0,lambda q:q['prompt'],score)

    def test_full_compressed_hash_and_coverage_are_required(self):
        with tempfile.TemporaryDirectory()as tmp:
            p,w,q=self.fixture(Path(tmp));p['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'full SHA'):read_shard(p,w,q,0,lambda q:q['prompt'],score)
            p,w,q=self.fixture(Path(tmp),lambda rows,questions:rows.pop())
            with self.assertRaisesRegex(ValueError,'identities'):read_shard(p,w,q,0,lambda q:q['prompt'],score)

    def test_every_source_aggregate_must_match(self):
        rows={0:dict(qid='a',data_source='nq',em=1,f1=1.),1:dict(qid='b',data_source='hotpotqa',em=0,f1=.5)}
        result=aggregates(rows);matching_aggregate(result,result)
        recorded=copy.deepcopy(result);recorded['per_source']['hotpotqa']['f1']=.6
        with self.assertRaisesRegex(ValueError,'Per-source'):matching_aggregate(result,recorded)

    def test_absent_final_output_is_pending_without_intent_or_model_import(self):
        with tempfile.TemporaryDirectory()as tmp:
            result=inspect_final(tmp)
            self.assertEqual(result['status'],'PENDING_FINAL_OUTPUT_NOT_COMMITTED_NO_AUDIT')
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_imported_scorer_must_match_actual_frozen_publication(self):
        with tempfile.TemporaryDirectory()as tmp:
            b=Path(tmp);code=b/'eval.py';code.write_text('original scorer')
            expected=hashlib.sha256(code.read_bytes()).hexdigest();proof=b/'compat.json'
            proof.write_text(json.dumps(dict(runtime_source_commit='runtime',target_publication=dict(files_sha256={'eval.py':expected}))))
            c=dict(source_commit='runtime',source_compatibility=str(proof),source_compatibility_sha256=hashlib.sha256(proof.read_bytes()).hexdigest())
            result=scoring_publication(b,c,{'eval.py':code});self.assertEqual(result['actual_imported_sources']['eval.py'],expected)
            code.write_text('another implementation')
            with self.assertRaisesRegex(ValueError,'imported scorer'):scoring_publication(b,c,{'eval.py':code})

    def test_control_pending_never_stages_or_starts_worker(self):
        with tempfile.TemporaryDirectory()as tmp:
            arms={'B':dict(base=tmp,run_id='b')}
            with patch('subprocess.Popen')as launch,patch('builtins.print'):
                exec(remote_inner(arms,'start'),{})
            launch.assert_not_called();self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_controller_stages_only_cpu_reader_outside_frozen_repo(self):
        class CapturedLaunch(Exception):pass
        with tempfile.TemporaryDirectory()as tmp:
            b=Path(tmp);prep=b/'preparation';prep.mkdir();(b/'repo').mkdir();(b/'repo/marker').write_text('frozen')
            for name in ('logs/pipeline-complete.json','latest-joint.json','final-evaluation/complete.json','preparation/final-evaluation-config.json'):
                path=b/name;path.parent.mkdir(exist_ok=True);path.write_text('{}')
            raw=json.dumps(dict(run_id='b')).encode();(prep/'launch-config.json').write_bytes(raw)
            arms={'B':dict(base=tmp,run_id='b',commit='runtime',config_sha256=hashlib.sha256(raw).hexdigest())};captured={}
            def launch(command,**kw):captured.update(command=command,**kw);raise CapturedLaunch()
            with patch('subprocess.Popen',side_effect=launch),self.assertRaises(CapturedLaunch):exec(remote_inner(arms,'start'),{})
            self.assertEqual(captured['env']['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(captured['env']['PYTHONPATH'],str(b/'repo'))
            self.assertEqual(list((b/'repo').iterdir()),[b/'repo/marker'])
            worker=Path(captured['command'][1]);self.assertTrue(worker.is_relative_to(prep/'final-qa-tools'))
            self.assertEqual(hashlib.sha256(worker.read_bytes()).hexdigest(),worker.parent.name)
            compile(worker.read_text(),str(worker),'exec')
            claim=json.loads((prep/'final-qa-cpu-control-intent.json').read_text());self.assertFalse(claim['frozen_repo_changed'])


if __name__=='__main__':unittest.main()
