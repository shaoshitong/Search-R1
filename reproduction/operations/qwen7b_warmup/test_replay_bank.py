"""Offline synthetic metadata tests; no real bank/GPU/SSH/submission claims."""
import copy
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import tarfile
import sys
import unittest
from unittest.mock import patch
from .replay_bank_prepare import partition_questions,render_sbatch,checked
from .replay_bank_capture import load_fixture
from .replay_probe_layers import canonical,fingerprints
from .test_replay_probe_layers import row as prefix_row


def group(q):
    return dict(qid='q'+str(q),question_hash=('%064x'%q),trajectory_ids=['q%d-r%d'%(q,j) for j in range(5)])


class ReplayBankTests(unittest.TestCase):
    def test_real_matched_config_SHA_only_refs_and_optional_strict_bytes(self):
        path='/data/user/test/immutable-plan.json'
        actual=dict(path=path,sha256='a'*64,bytes=321)
        with patch('reproduction.operations.qwen7b_warmup.replay_bank_prepare.fingerprints',return_value=actual):
            self.assertEqual(checked(dict(path=path,sha256='a'*64)),Path(path))
            self.assertEqual(checked(actual),Path(path))
            with self.assertRaisesRegex(ValueError,'SHA/size'):
                checked(dict(path=path,sha256='b'*64))
            with self.assertRaisesRegex(ValueError,'SHA/size'):
                checked(dict(path=path,sha256='a'*64,bytes=320))

    def test_full_declared9088_scope_four_disjoint_complete_G5s(self):
        class NoEM(dict):
            def __getitem__(self,key):
                if key in ('labels','strict_em','predicted_mixed'):raise AssertionError('Label-aware selection forbidden')
                return super().__getitem__(key)
        train=[NoEM(group(q)) for q in range(6528)]
        diag=[NoEM(group(q)) for q in range(6528,7040)]
        held=[NoEM(group(q)) for q in range(7040,9088)]
        qids=[g['qid'] for g in train]
        plan=dict(cohort_qids=qids,batches=[qids[i:i+51] for i in range(0,6528,51)])
        parts,proof=partition_questions(plan,train,diag,held)
        self.assertEqual(proof['questions'],9088);self.assertEqual(proof['trajectories'],45440)
        self.assertEqual([len(p) for p in parts],[2272]*4)
        for p in parts:
            self.assertEqual(sum(g['role']=='fit' for g in p),1632)
            self.assertEqual(sum(g['role']=='diagnostic512' for g in p),128)
            self.assertEqual(sum(g['role']=='selection2048' for g in p),512)
        self.assertFalse(proof['labels_consulted_for_selection'])
        again,second=partition_questions(plan,train,diag,held)
        self.assertEqual(parts,again);self.assertEqual(proof,second)
        bad=copy.deepcopy(held);bad[0]['question_hash']=train[0]['question_hash']
        with self.assertRaisesRegex(ValueError,'overlap'):partition_questions(plan,train,diag,bad)
        plan['batches'][0][0]=qids[-1]
        with self.assertRaisesRegex(ValueError,'6528 plan'):partition_questions(plan,train,diag,held)

    def test_four_scripts_normal_independent_lock_and_safe_tokens(self):
        for shard in range(4):
            base='/data/user/test/isolated/shard-%02d'%shard
            script=render_sbatch(base,'/data/user/test/immutable-code','a'*64,shard)
            subprocess.run(['bash','-n'],input=script,text=True,check=True)
            self.assertIn('#SBATCH --partition=acd_u',script);self.assertIn('#SBATCH --qos=normal',script)
            self.assertIn('#SBATCH --gres=gpu:1',script);self.assertIn('#SBATCH --cpus-per-task=8',script)
            self.assertIn('#SBATCH --mem=64G',script);self.assertIn('#SBATCH --time=01:00:00',script)
            self.assertNotIn('--dependency',script);self.assertNotIn('@@',script)
            self.assertIn('flock -n 9',script)
            exports='\n'.join(line for line in script.splitlines() if line.startswith('export '))
            subprocess.run(['bash','-euc',exports+'\ntest "$QWEN7B_CODE_ROOT" = /data/user/test/immutable-code'],check=True)
        self.assertIn('#SBATCH --time=02:00:00',render_sbatch('/data/a','/data/b','a'*64,0,2))
        with self.assertRaises(ValueError):render_sbatch('/data/a','/data/b','a'*64,0,3)

    def test_fixture_loader_uses_sealed_own_G5_and_rejects_wrong_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'prefix.jsonl.gz';rows=[]
            for i in range(5):
                row=prefix_row();row.pop('prefix_sha256');row['trajectory_id']='q0-r%d'%i
                row['prefix_sha256']=__import__('hashlib').sha256(canonical(row)).hexdigest();rows.append(row)
            with gzip.open(path,'wt') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
            selection=dict(questions=1,trajectories=[dict(qid='q0',trajectory_id=row['trajectory_id']) for row in rows],prefix_files=[fingerprints(path)])
            selected,proof=load_fixture(selection,synthetic_fixture=True)
            self.assertEqual(len(selected),5);self.assertEqual(len(selected[0]['segments']),2)
            self.assertEqual(proof,[fingerprints(path)])
            bad=copy.deepcopy(selection);bad['trajectories'].pop()
            with self.assertRaisesRegex(ValueError,'undeclared'):load_fixture(bad,synthetic_fixture=True)
            selection['prefix_files'][0]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'SHA'):load_fixture(selection,synthetic_fixture=True)

    def test_source_bundle_and_every_quoted_control_command_under16KB(self):
        from .deploy_replay_bank import self_test
        root=Path(__file__).resolve().parents[3]
        result=self_test(root)
        self.assertFalse(result['remote_actions_executed']);self.assertFalse(result['jobs_submitted'])
        self.assertTrue(all(n<16000 for n in result['quoted_command_bytes'].values()))
        self.assertGreater(result['closure']['files'],8)

    def test_complete_code_bundle_cpu_imports_resolve_inside_isolated_root(self):
        from .deploy_replay_bank import bundle_from_sources
        repo=Path(__file__).resolve().parents[3]
        paths=[str(p.relative_to(repo)) for p in (repo/'reproduction').rglob('*')
               if p.is_file() and '__pycache__' not in p.parts]
        raw,_=bundle_from_sources(paths,lambda name:(repo/name).read_bytes(),'a'*40)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
                for member in archive.getmembers():
                    p=root/member.name;p.parent.mkdir(parents=True,exist_ok=True)
                    p.write_bytes(archive.extractfile(member).read())
            code="""import importlib,pathlib
root=pathlib.Path.cwd()
for name in ['replay_bank_prepare','replay_bank_capture','replay_probe_layers','replay_feature_reader','packed_features','native_capture']:
 m=importlib.import_module('reproduction.operations.qwen7b_warmup.'+name)
 assert str(pathlib.Path(m.__file__).resolve()).startswith(str(root)+'/'),name
"""
            env=dict(os.environ,PYTHONPATH=str(root),QWEN7B_QUESTION_MAX_TOKENS='273',SEARCH_G1_B_PREFIX_TOKENS='20',
                     CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([sys.executable,'-c',code],cwd=root,env=env,check=True,timeout=30)


if __name__=='__main__':unittest.main()
