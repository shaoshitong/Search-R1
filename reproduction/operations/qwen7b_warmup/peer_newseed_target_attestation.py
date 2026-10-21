"""Verify the one packaged metadata-reseal exception before target strict EM.

This validates small source attestations and their exact received payload SHA,
current own-G5 selection and old continuation-chain identity. It does not assert
that old/current metadata bytes are equal, and does not replace per-trajectory
causal/seed/EM reconstruction by the ingester.
"""
import hashlib,json,stat
from pathlib import Path,PurePosixPath
RUN='70188b8e50fd4b45bc420ce58014f46e';SEED=20261021
REVISION='a09a35458c702b33eeacc393d103063234e8bc28'
AUDITOR='e01c2b14c7e78f10141f03fd22385d08c65017379f7b515f75b705f82fd9ceae'
ORIGINAL='3def055adebfdb38e9326b9db385338c44800160c1b0a64e1ee78d8c3b68b906'
INPUT='inputs/shard-01/batch-000300.json'
SOURCE_ROOTS=('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020','/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')


def _regular(root,relative):
    root=Path(root);relative=PurePosixPath(str(relative))
    if relative.is_absolute()or'..'in relative.parts or str(relative)in('','.'):raise ValueError('Unsafe packaged attestation path')
    p=root
    for part in relative.parts:
        p=p/part
        if p.is_symlink():raise ValueError('Packaged symlink forbidden')
    if not stat.S_ISREG(p.stat().st_mode):raise ValueError('Regular packaged artifact required')
    return p


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _read(root,ref,exact):
    if ref.get('path')!=exact or type(ref.get('bytes'))is not int or not 0<ref['bytes']<65536:raise ValueError('Exact bounded packaged proof path/size required')
    p=_regular(root,exact)
    if p.stat().st_size!=ref['bytes']or _sha(p)!=ref['sha256']:raise ValueError('Packaged proof SHA/bytes differ')
    return json.loads(p.read_bytes())


def validate(prefix_group_root,input_relative,source_base,conf,batch,prefix_receipt,full_receipt,scan_sha,selection_sha):
    root=Path(prefix_group_root);base=Path(source_base)
    if input_relative!=INPUT or base!=root/'source':raise ValueError('Only explicitly packaged one-batch exception is allowed')
    if any(conf.get(k)!=v for k,v in dict(run_id=RUN,seed=SEED,model_revision=REVISION,sampling_attempt_id='independent-seed20261021-g5-20261020',actor_optimizer_updates=0).items()):raise ValueError('Received source model/seed/run differs')
    cohort=json.loads(_regular(root,'cohort.json').read_bytes());mapping=cohort.get('metadata_resealed_audits',{}).get(INPUT)
    if not isinstance(mapping,dict)or mapping.get('auditor_sha256')!=AUDITOR or mapping.get('original_receipt_chain_equal_current')is not False:raise ValueError('Explicit published metadata-reseal mapping missing')
    proof=_read(root,mapping['proof'],'provenance/audit/shard-01/batch-000300-proof.json')
    att=_read(root,mapping['attestation'],'provenance/audit/shard-01/batch-000300-attestation.json')
    binding=proof.get('binding',{});detail=proof.get('proof',{});flags=detail.get('metadata_reseal_attestation',{})
    if proof.get('status')!='VERIFIED'or binding.get('auditor_sha256')!=AUDITOR or binding.get('run_id')!=RUN or binding.get('seed')!=SEED or binding.get('cache_version')!=1:raise ValueError('Packaged source proof identity differs')
    inp=_regular(base,INPUT);stored=json.loads(inp.read_bytes())
    if stored!=batch or binding.get('input_sha256')!=_sha(inp)or binding.get('root_config_sha256')!=_sha(_regular(base,'config.json')):raise ValueError('Packaged input/root config SHA differs')
    if(att.get('status')!='ONE_METADATA_RESEALED_BATCH_FRESH_SHA_AND_ORIGINAL_CHAIN_CAUSAL_RECONCILIATION_PASSED'or att.get('input')!=INPUT or att.get('auditor_sha256')!=AUDITOR or att.get('pinned_original_auditor_sha256')!=ORIGINAL or any(att.get('proof',{}).get(k)!=mapping['proof'][k]for k in('bytes','sha256'))or att.get('metadata_reseal_attestation')!=flags):raise ValueError('Packaged independent attestation differs from exact proof')
    if(flags.get('original_receipt_chain_equal_current')is not False or any(flags.get(k)is not True for k in('same_prefix_rank_receipt_and_payload','same_all_G5_IDs','full_rows_original_chain_and_current_prefix_causal_tokens_exact'))or flags.get('producer_metadata_modified')is not False):raise ValueError('Attestation did not retain own identity and causal payload')
    stem='sampling/shard-01/batches/batch-000300/outers/outer-000001/'
    scanpath=_regular(base,stem+'prefix/scan.json');selpath=_regular(base,stem+'probe-score/selection.json')
    scan=json.loads(scanpath.read_bytes());selection=json.loads(selpath.read_bytes());prefixseal=json.loads(_regular(base,stem+'prefix/complete.json').read_bytes())
    if _sha(scanpath)!=scan_sha or _sha(selpath)!=selection_sha or prefixseal.get('scan_sha256')!=scan_sha or scan.get('ranks')!=[prefix_receipt]or any(prefixseal.get(k)!=v for k,v in scan.items()):raise ValueError('Received current prefix scan/seal is inconsistent')
    if json.loads(_regular(base,stem+'prefix/rank-0-complete.json').read_bytes())!=prefix_receipt or json.loads(_regular(base,stem+'full/rank-0-complete.json').read_bytes())!=full_receipt:raise ValueError('Received actual rank receipt differs')
    oldscan=full_receipt.get('prefix_scan_sha256');oldselection=full_receipt.get('selection_sha256')
    want=dict(original_continuation_scan_sha256=oldscan,original_continuation_selection_sha256=oldselection,fresh_current_scan_sha256=scan_sha,fresh_current_selection_sha256=selection_sha)
    if any(flags.get(k)!=v for k,v in want.items())or(oldscan==scan_sha and oldselection==selection_sha):raise ValueError('Explicit old/fresh metadata chain differs from attestation')
    if selection.get('kind')!='ALL_G5_NO_SELECTION'or selection.get('run_id')!=RUN or selection.get('outer')!=1 or selection.get('prefix_scan_sha256')!=scan_sha or selection.get('actor_version')!=prefix_receipt.get('actor_version'):raise ValueError('Actual ALL_G5 selection binding differs')
    expected={r['qid']:r for r in batch['rows']};selected={};tids=[]
    for g in selection['selected']:
        if g['qid']not in expected or g['qid']in selected:raise ValueError('Actual selection repeats/changes qid')
        q=expected[g['qid']];own=[RUN+'-outer000001-c%04d-g%d'%(q['candidate_ordinal'],j)for j in range(5)]
        if(g.get('question_hash')!=q['question_hash']or g.get('candidate_ordinal')!=q['candidate_ordinal']or len(g.get('trajectory_ids',[]))!=5 or set(g['trajectory_ids'])!=set(own)):raise ValueError('Actual selection lacks exact own G5')
        selected[g['qid']]=own;tids+=own
    if set(selected)!=set(expected)or len(tids)!=len(set(tids))or detail.get('questions')!=len(expected)or detail.get('trajectories')!=5*len(expected)or detail.get('labels')!=5*len(expected):raise ValueError('Actual selection/proof totals differ')
    canonical=json.dumps(sorted(tids),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest()!=detail.get('actual_trajectory_ids_sha256'):raise ValueError('Packaged own G5 identity digest differs')
    for role,item,rel in [('prefix',prefix_receipt['prefix'],stem+'prefix/rank-0-prefix.jsonl.gz'),('full',full_receipt['full'],stem+'full/rank-0-full.jsonl.gz'),('own_em',full_receipt['own_em'],stem+'full/rank-0-own-em.jsonl')]:
        path=_regular(base,rel)
        if not any(item.get('path')==source+'/'+rel for source in SOURCE_ROOTS)or item.get('bytes')!=path.stat().st_size or item.get('sha256')!=_sha(path)or detail.get('independent_sha256',{}).get(role)!=item['sha256']or att.get('independent_sha256',{}).get(role)!=item['sha256']:raise ValueError('Actual received payload SHA differs from proof/receipt')
    return dict(old_scan_sha=oldscan,old_selection_sha=oldselection,descriptor=dict(input=INPUT,auditor_sha256=AUDITOR,proof=mapping['proof'],attestation=mapping['attestation'],original_receipt_chain_equal_current=False,actual_current_G5_selection_verified=True,actual_received_payload_SHA_verified=True,target_tokenizer_strict_EM_verified=False,per_trajectory_causal_seed_and_EM_acceptance_still_required=True))
