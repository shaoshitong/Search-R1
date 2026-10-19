import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from reproduction.grounding_eval.asset_fetch import run,validate_job

class AssetFetchTests(unittest.TestCase):
    def job(self,p):
        files={('model-%02d.safetensors'%i):{'size':9,'sha256':hashlib.sha256(b'modeldata').hexdigest()} for i in range(12)}
        return {'repo':'Sutoonq/TMP_MODEL','revision':'a'*40,'prefix':'authorized/checkpoint','output':str(p),'files':files,'route':'CPU fixture','source_receipt_sha256':'b'*64,'concurrency':3,'resolve':['huggingface.co:443:3.167.112.25']}
    def test_restart_resume_and_fresh_verification(self):
        with tempfile.TemporaryDirectory()as t:
            j=self.job(Path(t));attempts={};secret='PRIVATE_TOKEN_SENTINEL'
            def curl(args,**kw):
                self.assertNotIn(secret,' '.join(args));self.assertIn(secret,kw['input'].decode())
                p=Path(args[args.index('--output')+1]);attempts[p.name]=attempts.get(p.name,0)+1
                if attempts[p.name]==1:p.write_bytes(b'mode');return SimpleNamespace(returncode=7)
                self.assertEqual(p.read_bytes(),b'mode');p.write_bytes(b'modeldata');return SimpleNamespace(returncode=0)
            with patch('reproduction.grounding_eval.asset_fetch.subprocess.run',side_effect=curl),patch('reproduction.grounding_eval.asset_fetch.time.sleep'):
                r=run(j,secret)
            self.assertEqual(r['status'],'MIGRATION_ALL_SELECTED_ASSETS_VERIFIED');self.assertEqual(r['verified_bytes'],108)
            self.assertTrue(all(v==2 for v in attempts.values()))
            for p in Path(t).glob('*.json'):self.assertNotIn(secret,p.read_text())
            with patch('reproduction.grounding_eval.asset_fetch.subprocess.run',side_effect=AssertionError('Must reuse verified final')):
                self.assertEqual(run(j,secret)['verified_bytes'],108)
    def test_changed_final_is_rejected_and_preserved(self):
        with tempfile.TemporaryDirectory()as t:
            j=self.job(Path(t));first=Path(t)/next(iter(j['files']));first.write_bytes(b'badmodel!')
            # Avoid real processes for unrelated files after the intentional failure.
            def curl(args,**kw):Path(args[args.index('--output')+1]).write_bytes(b'modeldata');return SimpleNamespace(returncode=0)
            with patch('reproduction.grounding_eval.asset_fetch.subprocess.run',side_effect=curl):
                with self.assertRaises(RuntimeError):run(j,'TOKEN_SENTINEL')
            self.assertEqual(first.read_bytes(),b'badmodel!');self.assertFalse((Path(t)/'complete.json').exists())
    def test_rejects_unplanned_and_escape(self):
        j=self.job(Path('/tmp/fixture'));j['prefix']='../../escape'
        with self.assertRaises(AssertionError):validate_job(j)
        j=self.job(Path('/tmp/fixture'));j['files']['training_state.pt']=j['files'].pop(next(iter(j['files'])))
        with self.assertRaises(AssertionError):validate_job(j)

if __name__=='__main__':unittest.main()
