import hashlib,json,os,tempfile,unittest
from pathlib import Path
from .retained_corpus_cache_binding import apply,identity

class RetainedCacheBindingTest(unittest.TestCase):
    def fixture(self,root):
        r=root/'retained-assets';r.mkdir();c=r/'shaoshitong/search-g1-prefix20-online1000-20261016-ca1c60f/retrieval-cache';c.mkdir(parents=True)
        a=r/'shaoshitong/corpus.tar';a.write_bytes(b'proven archive');oldmtime=a.stat().st_mtime_ns-10000000
        (c/'wiki_dump.jsonl').write_bytes(b'proven extracted contents');dh=hashlib.sha256((c/'wiki_dump.jsonl').read_bytes()).hexdigest()
        old=dict(source='/mnt/data/shaoshitong/corpus.tar',source_size=a.stat().st_size,source_mtime_ns=oldmtime,content_bytes=(c/'wiki_dump.jsonl').stat().st_size,content_sha256=dh,member='one-member')
        (c/'corpus.json').write_text(json.dumps(old));(c/'arrow').mkdir();(c/'arrow/dataset_info.json').write_text('{"real":"cached metadata"}')
        rows=[dict(kind='HARDLINK_TO_EXISTING_IMMUTABLE_FULL_SHA_BACKUP',target=str(a),identity=identity(a),proven_sha256=hashlib.sha256(a.read_bytes()).hexdigest())]
        for p in c.rglob('*'):
            if p.is_file():rows.append(dict(kind='NEW_COPY_SOURCE_TARGET_FULL_SHA_PASS',target=str(p),bytes=p.stat().st_size,source_identity=identity(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),destination_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        proof=r/'cache2-RESULT.json';proof.write_text(json.dumps(dict(status='FIXED_INPUT_RETENTION_COMPLETE',live_training_changed=False,files=rows)))
        return r,c,a,old,hashlib.sha256(proof.read_bytes()).hexdigest()
    def test_private_receipt_rebind_preserves_original_and_is_idempotent(self):
        with tempfile.TemporaryDirectory()as tmp:
            r,c,a,old,digest=self.fixture(Path(tmp));before=identity(a);result=apply(r,digest);target=Path(result['cache'])
            self.assertEqual(json.loads((c/'corpus.json').read_text()),old)
            self.assertEqual(identity(a),before)
            self.assertEqual(json.loads((target/'corpus.json').read_text())['source_mtime_ns'],a.stat().st_mtime_ns)
            self.assertNotEqual((target/'corpus.json').stat().st_ino,(c/'corpus.json').stat().st_ino)
            self.assertEqual((target/'wiki_dump.jsonl').read_bytes(),(c/'wiki_dump.jsonl').read_bytes())
            self.assertEqual(apply(r,digest),result)
    def test_reject_archive_change_even_when_size_stays_equal(self):
        with tempfile.TemporaryDirectory()as tmp:
            r,c,a,old,digest=self.fixture(Path(tmp));s=a.stat();os.utime(a,ns=(s.st_atime_ns,s.st_mtime_ns+1))
            with self.assertRaisesRegex(ValueError,'archive inode'):apply(r,digest)
    def test_reject_unverified_retention_manifest(self):
        with tempfile.TemporaryDirectory()as tmp:
            r,c,a,old,digest=self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError,'retention result changed'):apply(r,'0'*64)
if __name__=='__main__':unittest.main()
