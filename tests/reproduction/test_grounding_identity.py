import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from reproduction.grounding_eval.core import canonical, digest, sha
from reproduction.grounding_eval.identity import (FINAL_DATA_SHA256, validate_normal_identity,
    _validate_occurrences, canonical_nq)

class NormalIdentityAcceptance(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.cp=self.root/'checkpoint';self.cp.mkdir()
        for n,v in {'model-00001.safetensors':b'model-A','config.json':b'{"model":"test"}',
                    'tokenizer_config.json':b'{}','tokenizer.json':b'{}',
                    'model.safetensors.index.json':b'{"weight_map":{"weight":"model-00001.safetensors"}}'}.items():
            (self.cp/n).write_bytes(v)
        self.files={p.name:sha(p)for p in self.cp.iterdir()}
        self.ids={i:'canonical-original-'+str(i)for i in range(3610)}
    def tearDown(self):self.tmp.cleanup()
    def write(self,path,obj):path.write_bytes(canonical(obj));return path
    def h100(self):
        b=dict(checkpoint_files=self.files,data_sha256=FINAL_DATA_SHA256,data_path='/real/final.parquet')
        p=self.write(self.root/'normal-binding.json',b)
        rows=[dict(data_source='nq',evaluation_ordinal=i,qid=self.ids[i],normal_binding_sha256=digest(b),
                   checkpoint_files_sha256=digest(self.files))for i in range(3)]
        return p,rows
    def validate(self,rows,path):
        with patch('reproduction.grounding_eval.identity.canonical_nq',return_value=self.ids):
            return validate_normal_identity(rows,self.cp,path)
    def test_h100_pilot_to_full_prefix_identity_stays_bound(self):
        p,rows=self.h100();a=self.validate(rows[:2],p);b=self.validate(rows,p)
        self.assertEqual(a['verified_checkpoint_files'],b['verified_checkpoint_files'])
        self.assertEqual(a['normal_binding_sha256'],b['normal_binding_sha256'])
        self.assertNotEqual(a['source_occurrences_sha256'],b['source_occurrences_sha256'])
    def test_same_sized_wrong_checkpoint_is_rejected(self):
        p,rows=self.h100();(self.cp/'model-00001.safetensors').write_bytes(b'model-B')
        with self.assertRaisesRegex(ValueError,'checkpoint bytes differ'):self.validate(rows,p)
    def test_wrong_original_qid_is_rejected(self):
        p,rows=self.h100();rows[1]['qid']='lookalike-other-question'
        with self.assertRaisesRegex(ValueError,'original final parquet'):self.validate(rows,p)
    def test_duplicate_ordinal_even_same_qid_is_rejected(self):
        p,rows=self.h100();rows[1]=dict(rows[0])
        with self.assertRaisesRegex(ValueError,'duplicated'):self.validate(rows,p)
    def test_missing_tokenizer_identity_is_rejected(self):
        self.files.pop('tokenizer.json');p,rows=self.h100()
        with self.assertRaisesRegex(ValueError,'omits HF'):self.validate(rows,p)
    def test_hpc3_map_and_source_checksum(self):
        fs={n:dict(size=(self.cp/n).stat().st_size,sha256=h)for n,h in self.files.items()}
        b=dict(checkpoint_files=fs,checkpoint_hash=digest(fs),data_sha256=FINAL_DATA_SHA256,
               job=dict(data_path='/actual/final.parquet'),normal_protocol=dict(greedy=True))
        p=self.write(self.root/'normal-binding.json',b)
        r=dict(data_source='nq',evaluation_ordinal=0,qid=self.ids[0],checkpoint_hash=digest(fs),normal_config_hash=digest(b))
        r['source_record_hash']=digest(r);r['source_shard_sha256']='sealed-later-batch-sha'
        self.assertEqual(self.validate([r],p)['schema'],'hpc3-legacy-manager-v1')
        r['qid']='other'
        with self.assertRaisesRegex(ValueError,'checksum'):self.validate([r],p)
    def test_a800_publication_complete_identity(self):
        self.write(self.cp/'complete.json',dict(files_sha256=self.files,run_id='actual'))
        fs=dict(self.files,**{'complete.json':sha(self.cp/'complete.json')})
        normal=dict(greedy=True)
        b=dict(checkpoint=str(self.cp),checkpoint_identity_sha256=digest(fs),dataset_sha256=FINAL_DATA_SHA256,
               dataset_path='/actual/final.parquet',normal_config=normal,normal_config_sha256=digest(normal))
        self.write(self.root/'binding.json',b);p=self.write(self.root/'normal-config.json',normal)
        r=dict(data_source='nq',evaluation_ordinal=0,qid=self.ids[0],checkpoint_id=digest(fs),normal_config_sha256=digest(normal))
        r['source_record_hash']=digest(r)
        self.assertEqual(self.validate([r],p)['schema'],'a800-common8-v2')
    def test_unbound_normal_input_is_rejected(self):
        p=self.write(self.root/'normal-config.json',dict(greedy=True))
        with self.assertRaisesRegex(ValueError,'No supported'):self.validate([],p)
    def test_actual_data_hash_must_match_before_reading(self):
        p=self.root/'wrong.parquet';p.write_bytes(b'other dataset')
        with self.assertRaisesRegex(ValueError,'fixed common final'):canonical_nq(p)

if __name__=='__main__':unittest.main()
