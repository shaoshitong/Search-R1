import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from reproduction.prefix20_joint4.canonical_assets import unpack_selected


class CanonicalAssetsTests(unittest.TestCase):
    def test_complete_archive_sha_but_only_selected_canonical_inputs_materialized(self):
        data={'canonical/train.parquet':b'constructed canonical bytes',
              'canonical/eligible-question-pool.json':b'constructed pool bytes',
              'epochs/old-rollout.pt':b'old replay must not be extracted'}
        manifest={'files':[dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in data.items()]}
        selected={name:(len(raw),hashlib.sha256(raw).hexdigest()) for name,raw in data.items() if name.startswith('canonical/')}
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w:gz') as archive:
            for name,raw in dict(data,**{'MANIFEST.json':json.dumps(manifest).encode()}).items():
                member=tarfile.TarInfo(name);member.size=len(raw);archive.addfile(member,io.BytesIO(raw))
        raw=stream.getvalue()
        with tempfile.TemporaryDirectory() as directory,patch('reproduction.prefix20_joint4.canonical_assets.SELECTED',selected):
            root=Path(directory);(root/'part000').write_bytes(raw)
            archive=dict(parts=[{'name':'part000'}],concatenated_sha256=hashlib.sha256(raw).hexdigest())
            proof=unpack_selected(root,archive,manifest)
            self.assertEqual(len(proof['files']),2);self.assertTrue(proof['full_compressed_archive_verified'])
            self.assertFalse(proof['historical_rollouts_extracted']);self.assertFalse((root/'selected/epochs').exists())
            for name,original in data.items():
                if name in selected:self.assertEqual((root/'selected'/name).read_bytes(),original)
            archive['concatenated_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'FULL_ARCHIVE_SHA'):unpack_selected(root,archive,manifest)


if __name__=='__main__':unittest.main()
