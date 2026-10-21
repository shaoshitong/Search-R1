"""Three tiny stdlib fixtures for the one explicit metadata-chain exception."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from . import peer_newseed_target_attestation as v


def fixture(root):
    root=Path(root);base=root/'source';base.mkdir();conf=dict(run_id=v.RUN,seed=v.SEED,model_revision=v.REVISION,sampling_attempt_id='independent-seed20261021-g5-20261020',actor_optimizer_updates=0)
    def write(rel,value):
        p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(json.dumps(value,sort_keys=True,separators=(',',':')).encode());return dict(path=rel,bytes=p.stat().st_size,sha256=v._sha(p))
    rows=[dict(qid='toy%d'%i,question_hash=hashlib.sha256(str(i).encode()).hexdigest(),candidate_ordinal=i)for i in range(2)];batch=dict(rows=rows);write('source/config.json',conf);write('source/'+v.INPUT,batch)
    stem='sampling/shard-01/batches/batch-000300/outers/outer-000001/'
    items={}
    for role,rel in [('prefix',stem+'prefix/rank-0-prefix.jsonl.gz'),('full',stem+'full/rank-0-full.jsonl.gz'),('own_em',stem+'full/rank-0-own-em.jsonl')]:
        p=base/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(('synthetic-'+role).encode());items[role]=dict(path=v.SOURCE_ROOTS[0]+'/'+rel,bytes=p.stat().st_size,sha256=v._sha(p))
    prefix=dict(actor_version='official7b-'+v.REVISION,prefix=items['prefix']);full=dict(prefix_scan_sha256='1'*64,selection_sha256='2'*64,full=items['full'],own_em=items['own_em'])
    scan=dict(ranks=[prefix],questions=2,trajectories=10,head_prefix_tokens=20,model_revision=v.REVISION);sr=write('source/'+stem+'prefix/scan.json',scan);write('source/'+stem+'prefix/complete.json',dict(scan,scan_sha256=sr['sha256']));write('source/'+stem+'prefix/rank-0-complete.json',prefix);write('source/'+stem+'full/rank-0-complete.json',full)
    groups=[dict(q,trajectory_ids=[v.RUN+'-outer000001-c%04d-g%d'%(q['candidate_ordinal'],j)for j in range(5)])for q in rows];selection=dict(run_id=v.RUN,outer=1,actor_version=prefix['actor_version'],prefix_scan_sha256=sr['sha256'],kind='ALL_G5_NO_SELECTION',selected=groups);ss=write('source/'+stem+'probe-score/selection.json',selection)
    flags=dict(original_receipt_chain_equal_current=False,original_continuation_scan_sha256='1'*64,original_continuation_selection_sha256='2'*64,fresh_current_scan_sha256=sr['sha256'],fresh_current_selection_sha256=ss['sha256'],same_prefix_rank_receipt_and_payload=True,same_all_G5_IDs=True,full_rows_original_chain_and_current_prefix_causal_tokens_exact=True,producer_metadata_modified=False)
    tids=sorted(t for g in groups for t in g['trajectory_ids']);d=hashlib.sha256(json.dumps(tids,sort_keys=True,separators=(',',':')).encode()).hexdigest();independent={k:i['sha256']for k,i in items.items()}
    proof=dict(status='VERIFIED',binding=dict(auditor_sha256=v.AUDITOR,cache_version=1,run_id=v.RUN,seed=v.SEED,input_sha256=v._sha(base/v.INPUT),root_config_sha256=v._sha(base/'config.json')),proof=dict(questions=2,trajectories=10,labels=10,actual_trajectory_ids_sha256=d,independent_sha256=independent,metadata_reseal_attestation=flags));pref=write('provenance/audit/shard-01/batch-000300-proof.json',proof)
    att=dict(status='ONE_METADATA_RESEALED_BATCH_FRESH_SHA_AND_ORIGINAL_CHAIN_CAUSAL_RECONCILIATION_PASSED',input=v.INPUT,auditor_sha256=v.AUDITOR,pinned_original_auditor_sha256=v.ORIGINAL,proof=dict(pref,path='/original/source/proof.json'),metadata_reseal_attestation=flags,independent_sha256=independent);ar=write('provenance/audit/shard-01/batch-000300-attestation.json',att)
    write('cohort.json',dict(metadata_resealed_audits={v.INPUT:dict(auditor_sha256=v.AUDITOR,original_receipt_chain_equal_current=False,proof=pref,attestation=ar)}))
    return (root,v.INPUT,base,conf,batch,prefix,full,sr['sha256'],ss['sha256'])


class TargetAttestationTests(unittest.TestCase):
    def test_explicit_old_chain_and_own_payload_SHA_no_EM_claim(self):
        with tempfile.TemporaryDirectory()as d:
            args=fixture(d);r=v.validate(*args);self.assertEqual(r['old_scan_sha'],'1'*64);self.assertFalse(r['descriptor']['original_receipt_chain_equal_current']);self.assertFalse(r['descriptor']['target_tokenizer_strict_EM_verified'])
    def test_wrong_member_missing_packaged_proof_and_payload_corruption_reject(self):
        with tempfile.TemporaryDirectory()as d:
            args=fixture(d);bad=list(args);bad[1]='inputs/shard-01/batch-000301.json'
            with self.assertRaises(ValueError):v.validate(*bad)
            p=Path(d)/'source/sampling/shard-01/batches/batch-000300/outers/outer-000001/full/rank-0-full.jsonl.gz';p.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'payload SHA'):v.validate(*args)
        with tempfile.TemporaryDirectory()as d:
            args=fixture(d);p=Path(d)/'cohort.json';x=json.loads(p.read_bytes());x['metadata_resealed_audits'][v.INPUT]['proof']['path']='../outside.json';p.write_text(json.dumps(x))
            with self.assertRaisesRegex(ValueError,'path'):v.validate(*args)
    def test_no_forged_byte_chain_or_ownID_substitution(self):
        with tempfile.TemporaryDirectory()as d:
            args=fixture(d);bad=list(args);bad[6]=dict(args[6],prefix_scan_sha256='3'*64)
            with self.assertRaises(ValueError):v.validate(*bad)
            bad=list(args);bad[4]=copy.deepcopy(args[4]);bad[4]['rows'][0]['qid']='foreign'
            with self.assertRaisesRegex(ValueError,'input'):v.validate(*bad)

if __name__=='__main__':unittest.main()
