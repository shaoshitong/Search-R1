"""Real entry parser accepts2018 without GPU, actor, or training side effects."""
import unittest
from unittest.mock import patch
from . import online_packed_window_topk_seeded_v2 as r
class WindowCLI(unittest.TestCase):
 def test_exact_entry_accepts2018_and_binds_explicit_seeds(self):
  argv=['--action','train','--base','/tmp/unused-window-cli','--fixture-base','/tmp/unused-fixture','--cache-window','2018','--layer','15','--prefix','80','--epochs','10','--lr','0.000005','--ema-decay','0.9995','--margin','1.0','--init-seed','19137','--train-plan-seed','2026102302','--start-mode','fresh']
  with patch.object(r,'run',side_effect=AssertionError('No GPU run')),patch.object(r,'prepare',side_effect=AssertionError('No prepare')):
   a=r.build_parser().parse_args(argv)
  self.assertEqual((a.cache_window,a.init_seed,a.train_plan_seed,a.layer,a.prefix,a.epochs),(2018,19137,2026102302,15,80,10))
 def test_current64_remains_allowed_and_history_count_is_not_total_capacity(self):
  argv=['--action','train','--base','/tmp/unused','--cache-window','0','--init-seed','19137','--train-plan-seed','2026102302'];self.assertEqual(r.build_parser().parse_args(argv).cache_window,0)
  argv[5]='1954'
  with self.assertRaises(SystemExit):r.build_parser().parse_args(argv)
if __name__=='__main__':unittest.main()
