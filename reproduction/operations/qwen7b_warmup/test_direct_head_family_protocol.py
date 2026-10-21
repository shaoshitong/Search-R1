import unittest
from . import direct_head_family_protocol as f

class FamilyTests(unittest.TestCase):
 def test_complete_family_and_fixed_ties(self):
  def m(ba,n):return dict(Mixed_BA=ba,Top25percent=dict(Mixed_count=n))
  baseline=dict(NQ=m(.5,100),half0=m(.5,50),half1=m(.5,50))
  rows=[dict(recipe=r,epoch=e,model=v,NQ=m(.6,101),half0=m(.5,50),half1=m(.5,50))for r in f.RECIPES for e in f.EPOCHS for v in f.MODES]
  with self.assertRaises(ValueError):f.select(rows[:-1],baseline)
  selected=f.select(rows,baseline);self.assertEqual((selected['recipe'],selected['epoch'],selected['model']),('Count',0,'EMA'))
  rows[1]['half0']=m(.49,51)
  selected=f.select(rows,baseline);self.assertEqual((selected['recipe'],selected['epoch'],selected['model']),('binary',0,'EMA'))
  for r in rows:r['NQ']=m(.5,101)
  self.assertIsNone(f.select(rows,baseline))

if __name__=='__main__':unittest.main()
