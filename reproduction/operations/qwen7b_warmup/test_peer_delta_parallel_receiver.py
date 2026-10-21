import hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from . import peer_newseed_receive_parallel as p
class Response:
 def __init__(self,data,start,end,total,bad=False):self.data=data[start:end+1];self.status_code=200 if bad else 206;self.headers={'Content-Range':'bytes %d-%d/%d'%(start,end,total)}
 def raise_for_status(self):pass
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def iter_content(self,n):yield self.data
class Session:
 def __init__(self,data,bad=False):self.data=data;self.bad=bad
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def get(self,url,headers,**kwargs):start,end=map(int,headers['Range'][6:].split('-'));return Response(self.data,start,end,len(self.data),self.bad)
class Tests(unittest.TestCase):
 def test_exact_ranges_sha_assembly_and_accepted_reuse(self):
  data=bytes(range(251))*101;record=dict(group='group-00000',archive='/synthetic/snapshot-group-00000.tar',archive_bytes=len(data),archive_sha256=hashlib.sha256(data).hexdigest(),hf_archive_path='prefix/archives/snapshot-group-00000.tar')
  with tempfile.TemporaryDirectory()as tmp:
   b=Path(tmp).resolve();(b/'archives').mkdir();directory=p.paths(b,record)[0].with_name(p.paths(b,record)[0].name+'.ranges8');directory.mkdir();(directory/'range-00.part').write_bytes(data[:2])
   with patch.object(p,'_new_session',return_value=Session(data)):
    out=p.download(None,b,record,'a'*40,retries=1,workers=4);self.assertEqual(out.read_bytes(),data)
    self.assertEqual(p.download(None,b,record,'a'*40,retries=1,workers=4),out)
 def test_wrong_http_status_never_publishes_archive(self):
  data=b'synthetic'*100;record=dict(group='group-00000',archive='/synthetic/snapshot-group-00000.tar',archive_bytes=len(data),archive_sha256=hashlib.sha256(data).hexdigest(),hf_archive_path='prefix/archives/snapshot-group-00000.tar')
  with tempfile.TemporaryDirectory()as tmp:
   b=Path(tmp).resolve();(b/'archives').mkdir()
   with patch.object(p,'_new_session',return_value=Session(data,True)):
    with self.assertRaisesRegex(ValueError,'range not honored'):p.download(None,b,record,'a'*40,retries=1,workers=4)
   self.assertFalse(p.paths(b,record)[0].exists())
if __name__=='__main__':unittest.main()
