"""Label-blind CPU metadata tests; no hidden tensors/model/GPU loading."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from . import peer_newseed_replay as p
from .test_peer_newseed_ingest import trajectory


def group(qid):
    return dict(qid=qid,question_hash=p.digest(qid),trajectory_ids=[qid+'-t%d'%j for j in range(10)],
        labels=['MUST_NOT_READ']*10)


class ReplayInputTests(unittest.TestCase):
    def test_hash_random_question_selection_is_label_blind_and_excludes_controlled_and_eval(self):
        source=[group('q%d'%i) for i in range(12)]
        controlled=[{k:source[0][k] for k in ('qid','question_hash')}]
        evaluations=[[{k:source[1][k] for k in ('qid','question_hash')}]]
        chosen,summary=p.choose(source,controlled,evaluations,budget=8)
        self.assertFalse(summary['labels_consulted']);self.assertEqual(summary['eligible_questions'],10)
        self.assertFalse({'q0','q1'}&{g['qid'] for g in chosen});self.assertTrue(all(set(g)=={'qid','question_hash','trajectory_ids'} for g in chosen))
        shuffled=list(reversed(copy.deepcopy(source)))
        for g in shuffled:g['labels']=object()
        self.assertEqual(p.choose(shuffled,controlled,evaluations,budget=8)[0],chosen)
        alias=group('alias');alias['question_hash']=source[0]['question_hash']
        self.assertNotIn('alias',{g['qid'] for g in p.choose(source+[alias],controlled,evaluations,budget=8)[0]})

    def test_insufficient_questions_or_fake_G10_never_falls_back(self):
        source=[group('q%d'%i) for i in range(3)]
        with self.assertRaisesRegex(ValueError,'available=3 required=4'):p.choose(source,[],[],budget=4)
        source[0]['trajectory_ids']=source[0]['trajectory_ids'][:5]
        with self.assertRaisesRegex(ValueError,'Ten complete'):p.choose(source,[],[],budget=3)

    def test_projection_uses_only_prefix_and_exact_saved_causal_positions(self):
        config=dict(run_id='cpu-prefix-fixture',seed=1729)
        _,prefix,_=trajectory(config)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();path=root/'prefix.jsonl.gz';declarations=[]
            with gzip.open(path,'wt') as f:
                for j in range(10):
                    row=copy.deepcopy(prefix);row['trajectory_id']='actual-t%d'%j
                    row['prefix_sha256']=p.digest({k:v for k,v in row.items() if k!='prefix_sha256'})
                    f.write(json.dumps(row)+'\n');declarations.append(dict(qid='q',trajectory_id=row['trajectory_id']))
            item=p.ref(path);wanted={g['trajectory_id']:g['qid'] for g in declarations}
            own=list(p.projected_rows([item],wanted));self.assertEqual(len(own),10)
            selection=dict(role='peer_newseed_train_G10',model_revision=p.r.REVISION,capture_layers=list(p.r.LAYERS),
                real_trajectories_per_question=10,label_blind_selection=True,trajectories=declarations,prefix_files=[item],questions=1,
                full='/DO_NOT_OPEN_FULL',own_em='/DO_NOT_OPEN_OWN_EM')
            normalized=p.load_selection(selection);self.assertEqual(len(normalized),10)
            self.assertEqual(normalized[0]['policy_position_ids'],prefix['policy_position_ids'])
            self.assertEqual(len(normalized[0]['segments']),2)
            path.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'SHA or size'):list(p.projected_rows([item],wanted))

    def test_selected_prefix_cannot_smuggle_EM_or_future_tokens(self):
        config=dict(run_id='cpu-prefix-fixture',seed=1729);_,prefix,_=trajectory(config)
        for key in ('strict_em','labels','reward_model'):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();path=root/'prefix.jsonl.gz';row=copy.deepcopy(prefix);row[key]=1
                row['prefix_sha256']=p.digest({k:v for k,v in row.items() if k!='prefix_sha256'})
                with gzip.open(path,'wt') as f:f.write(json.dumps(row)+'\n')
                with self.assertRaisesRegex(ValueError,'prefix-only'):list(p.projected_rows([p.ref(path)],{row['trajectory_id']:'q'}))
        self.assertGreater(p.dense_upper_bytes(8192),500_000_000_000)
        self.assertLess(p.dense_upper_bytes(8192),600*1024**3)

if __name__=='__main__':unittest.main()
