import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from . import mixed_direct_fast_handoff as h
class Tests(unittest.TestCase):
 def fixture(self,root):
  main=root/'main';base=root/'fast';(main/'checkpoints').mkdir(parents=True);src=main/'checkpoints/.rolling-accepted-cursor-000513-tiny.pt';src.write_bytes(b'opaque-fixture');cursor=513;hashes={k:'a'*64 for k in h.KEYS};cp=dict(path=str(src),bytes=src.stat().st_size,sha256='b'*64,cursor=cursor,native_steps=cursor,head_steps=cursor,config_sha256=h.SOURCE_CONFIG['sha256'],allocation_job_id=h.SOURCE_JOB,full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed=True,accepted_hashes=hashes,sampler_state_sha256=hashes['sampler'],data_cursor=dict(cursor=cursor,sampler_state_sha256=hashes['sampler']))
  values={'config.json':{},'latest.json':cp,'progress.json':dict(cursor=cursor,native_steps=cursor,head_steps=cursor),'update-intent.json':dict(cursor=cursor,phase='accepted_update_complete'),'interrupted.json':dict(status='INTERRUPTED_AFTER_COMPLETE_ACCEPTED_UPDATE',cursor=cursor,native_steps=cursor,head_steps=cursor,checkpoint=cp,pending_evaluation=False),'mixed-checkpoint-transaction.json':dict(phase='published',new_checkpoint=cp)}
  for name,v in values.items():(main/name).write_text(json.dumps(v))
  d=main/'evaluations/epoch-00';d.mkdir(parents=True);(d/'complete.json').write_text(json.dumps(dict(cursor=0,config_sha256=h.SOURCE_CONFIG['sha256'])))
  return main,base,src,cp
 def test_hardlink_retains_source_without_payload_read(self):
  with tempfile.TemporaryDirectory()as tmp:
   main,base,src,cp=self.fixture(Path(tmp));original=Path.read_bytes
   def guarded(p):
    if p.suffix=='.pt':raise AssertionError('Checkpoint payload read')
    return original(p)
   with patch.object(h,'source_metadata',return_value=({},{})),patch.object(h,'command',return_value=''),patch.object(Path,'read_bytes',guarded):
    result=h.pin_terminal_source(base,main,terminal='746083|COMPLETED|0:0|00:10');target=Path(result['pinned_checkpoint']['path']);self.assertEqual(src.stat().st_ino,target.stat().st_ino);self.assertEqual(result['checkpoint'],cp);self.assertTrue(result['no_unresolved_tail']);src.unlink();self.assertEqual(h.pin_terminal_source(base,main),result)
 def test_unsaved_accepted_tail_rejected(self):
  with tempfile.TemporaryDirectory()as tmp:
   main,base,_,_=self.fixture(Path(tmp));p=json.loads((main/'progress.json').read_text());p['cursor']+=1;(main/'progress.json').write_text(json.dumps(p))
   with self.assertRaises(AssertionError):h.final_state_metadata(main)
 def test_wrong_main_terminal_rejected(self):
  with tempfile.TemporaryDirectory()as tmp:
   main,base,_,_=self.fixture(Path(tmp))
   with patch.object(h,'source_metadata',return_value=({},{})),self.assertRaises(AssertionError):h.pin_terminal_source(base,main,terminal='746083|FAILED|1:0|00:10')
if __name__=='__main__':unittest.main()
