import copy,unittest,tempfile
from pathlib import Path
from .prepare_hpc3_tail9344_l26 import physical_binding,checked,fp
class PhysicalSourceTests(unittest.TestCase):
 def setUp(self):
  self.row=dict(qid='q',trajectory_id='t',prefix_sha256='a'*64,policy_token_ids=[1,2]);self.seal=dict(path='/data/HPC3/complete.json',bytes=5,sha256='b'*64);self.source=dict(seal=self.seal,prefix=dict(path='/data/HPC3/prefix.gz',bytes=8,sha256='c'*64));self.selected=dict(qid='q',seal=self.seal,event='HPC3_newseed20261021',execution='hpc3-real');self.meta=dict(execution_run_id='hpc3-real',execution_host_cluster='HPC3')
 def test_bound_physical_row_and_digest(self):
  x=physical_binding(self.row,self.source,self.selected['event'],self.selected,self.meta);self.assertEqual(x['physical_prefix'],self.source['prefix']);self.assertEqual(len(x['canonical_prefix_row_sha256']),64)
  altered=copy.deepcopy(self.row);altered['policy_token_ids']=[1,3];self.assertNotEqual(physical_binding(altered,self.source,self.selected['event'],self.selected,self.meta)['canonical_prefix_row_sha256'],x['canonical_prefix_row_sha256'])
 def test_same_protocol_tid_different_execution_rejected(self):
  other=copy.deepcopy(self.source);other['seal']['path']='/data/donor/complete.json'
  with self.assertRaises(ValueError):physical_binding(self.row,other,self.selected['event'],self.selected,self.meta)
 def test_no_labels_and_wrong_host_rejected(self):
  row=dict(self.row,strict_em=1)
  with self.assertRaises(ValueError):physical_binding(row,self.source,self.selected['event'],self.selected,self.meta)
  with self.assertRaises(ValueError):physical_binding(self.row,self.source,self.selected['event'],self.selected,dict(self.meta,execution_host_cluster='donor'))
class MetadataReferenceTests(unittest.TestCase):
 def test_authenticated_metadata_extensions_keep_strict_core_SHA(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'prefix.gz';p.write_bytes(b'code-only-fixture');ref=dict(fp(p),count=640)
   self.assertEqual(checked(ref),p)
   with self.assertRaises(AssertionError):checked(dict(ref,sha256='0'*64))
if __name__=='__main__':unittest.main()
