"""Two bounded snapshot receipt / frozen-state inference fixtures, no real IO."""
import copy,gc,gzip,json,tempfile,unittest
from pathlib import Path
import torch
from . import mixed_group_checkpoint as ck
from . import evaluate_mixed_direct_checkpoint as e
from .test_joint_mixed_direct_pretrain import setup,snap,DEVICE,BIND,PROV
from .joint_mixed_direct_pretrain import direct_update


class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2)

 def test_pinned_cp128_changes_only_path_and_preserves_original_config_base(self):
  names=('raw','ema','native_adam','head_adam','raw_buffers','ema_buffers','rng','sampler','official_source');cp=dict(path=str(e.BASE/'inputs/cursor-000128.pt'),bytes=12345,sha256='a'*64,cursor=128,native_steps=128,head_steps=128,config_sha256=e.SOURCE_CONFIG_SHA,matched_binding_sha256='b'*64,accepted_hashes={n:'c'*64 for n in names},full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed=True);cfg=dict(path=str(e.BASE/'source-config.json'),bytes=9615,sha256=e.SOURCE_CONFIG_SHA);v=dict(mode='read_only_checkpoint_snapshot',source_config=cfg,source_main_config=dict(cfg,path=str(e.source.MAIN/'config.json')),checkpoint=cp,source_checkpoint_original=dict(cp,path=str(e.source.MAIN/'checkpoints/.rolling-accepted-cursor-000128.pt')))
  self.assertTrue(e.validate_receipts(v));wrong=copy.deepcopy(v);wrong['checkpoint']['native_steps']=127
  with self.assertRaises(ValueError):e.validate_receipts(wrong)
  wrong=copy.deepcopy(v);wrong['source_config']['sha256']='E4-trained-source'
  with self.assertRaises(ValueError):e.validate_receipts(wrong)
  wrong=copy.deepcopy(v);wrong['checkpoint']['sha256']='d'*64
  with self.assertRaisesRegex(ValueError,'changes only pathname'):e.validate_receipts(wrong)

 def test_raw_EMA_scores_own_G5_only_and_preserves_two_Adam_RNG_buffers_sampler(self):
  model,ema,bo,ho,sampler,reader,lookup=setup();_,batch=direct_update(model,ema,bo,ho,reader,lookup,sampler,DEVICE);state=snap(model,ema,bo,ho,sampler,1,batch['source_qids']);ck.restore(state,model,ema,bo,ho,DEVICE,sampler,BIND);before=e.full_live_hashes(model,ema,bo,ho,DEVICE,sampler,PROV);groups=[dict(qid=g['qid'],trajectory_ids=g['trajectory_ids'][5:10],labels=g['labels'][5:10])for g in lookup.values()]
  with tempfile.TemporaryDirectory(dir='/tmp')as tmp:
   root=Path(tmp);value=e.score_pair(model,ema,bo,ho,DEVICE,sampler,PROV,dict(synthetic_ownG5=(groups,reader)),root,root/'progress.json',state['accepted_hashes']);self.assertEqual(before,value['full_components_after']);self.assertEqual(value['native_Adam_clock'],1);self.assertEqual(value['group_head_Adam_clock'],1);self.assertTrue(value['all_saved_training_components_unchanged'])
   for label in('raw','EMA'):
    metric=value['metrics'][label]['synthetic_ownG5'];self.assertFalse(metric['product_formula_used']);self.assertFalse(metric['trajectory_probability_or_accuracy_produced'])
    with gzip.open(metric['predictions']['path'],'rt')as f:rows=[json.loads(line)for line in f]
    self.assertEqual([r['trajectory_ids']for r in rows],[g['trajectory_ids']for g in groups]);self.assertEqual([r['strict_em']for r in rows],[g['labels']for g in groups]);self.assertTrue(all(set(r)=={'qid','trajectory_ids','strict_em','true_mixed','predicted_mixed'}for r in rows));self.assertTrue(all(0<=r['predicted_mixed']<=1 for r in rows))
  self.assertTrue(model.training);self.assertFalse(ema.training);self.assertTrue(all(t.grad is None for t in reader.last));del state;gc.collect()


if __name__=='__main__':unittest.main()
