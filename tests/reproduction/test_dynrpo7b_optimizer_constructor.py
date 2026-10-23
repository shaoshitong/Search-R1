import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from search_g1.full_state_restore import optimizer_construction_config


class DynRPOOptimizerConstructor(unittest.TestCase):
    def test_actual_constructor_accepts_native_outer_and_preserves_resolved_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);(base/'logs').mkdir();receipt=base/'boundary.json';receipt.write_text('{}')
            cp=(base/'actor').resolve()
            for outer in (3,10,50,190,200):
                with self.subTest(outer=outer):
                    binding=dict(kind='dynrpo7b_native4',source_receipt=str(receipt),source_receipt_sha256='a'*64,completed_outer=outer)
                    (base/'logs/run-manifest.json').write_text(json.dumps(dict(full_state_restore=binding)))
                    config=dict(completed_updates=outer,lr=1e-6,lr_warmup_steps=57)
                    with patch.dict(os.environ,SEARCH_G1_BASE=str(base)),patch('search_g1.full_state_restore.source',return_value=(receipt,dict(step=outer),cp,None,None)),patch('search_g1.full_state_restore.digest',return_value='a'*64):
                        construction=optimizer_construction_config(config,str(cp))
                        self.assertEqual(config['completed_updates'],outer)
                        self.assertEqual(construction,dict(config,completed_updates=0))
                        with self.assertRaises(AssertionError):
                            optimizer_construction_config(dict(config,completed_updates=outer-1),str(cp))
                        with self.assertRaises(AssertionError):
                            optimizer_construction_config(config,str(base/'another-actor'))


if __name__=='__main__':unittest.main()
