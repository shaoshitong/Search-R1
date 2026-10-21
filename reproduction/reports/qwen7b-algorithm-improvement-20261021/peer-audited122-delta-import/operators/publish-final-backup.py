"""Publish only code, metadata references and small aggregate final evidence."""
import hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
C=Path(__file__).resolve().parents[3];R=C/'work/qwen7b-probe-warmup-20261019';O=Path(__file__).resolve().parent
D=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-audited122-delta-import'
D.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(Path(p).read_bytes())
audit=read(O/'final-audit-run.json');s=audit['summary']
assert audit['exit_code']==0 and s['status']=='AUDITED122_DELTA_RECEIVER_AND_ELIGIBLE_RAW_G10_INDEPENDENTLY_ACCEPTED'
files=[]
for name in ('source-publication-local.json','source-stage.json','source-start.json','source-status.json','hf-stage.json','hf-start.json','hf-first-archive.json','hf-status.json','receive-stage.json','receive-preflight.json','receive-start.json','receive-status.json','ingest-stage.json','ingest-tests.json','ingest-prepare.json','ingest-start.json','ingest-status.json','final-audit-stage.json','final-audit-run.json','audit-source-publication-local.json'):
 p=O/name
 if p.exists():q=D/name;q.write_bytes(p.read_bytes());files.append(str(q.relative_to(R)))
for name in ('hpc3-storage-probe.py','publish-final-backup.py'):
 p=O/name;q=D/'operators'/name;q.parent.mkdir(exist_ok=True);q.write_bytes(p.read_bytes());files.append(str(q.relative_to(R)))
metadata=dict(status='BOUNDED_AUDITED122_RAW_TRANSFER_AND_STRICT_EM_G10_ACCEPTED',received_questions=s['received_questions'],received_newseed5_trajectories=s['received_newseed5_trajectories'],eligible_questions=s['eligible_training_questions'],eligible_authentic_G10_trajectories=s['eligible_authentic_G10_trajectories'],domains=s['eligible_domains'],groups=s['groups'],source_events=s['source_events'],receiver_complete=s['receiver_complete'],ingest_complete=s['ingest_complete'],independent_join_audit=audit['audit'],target_independent_full_SHA_audit=dict(path=str(Path(s['ingest_complete']['path']).parents[1]/'peer-audited122-target-independent-v1/complete.json'),bytes=5937,sha256='2cfe0417cdecac5d53d9c565c81cbb21d0fdf0e9ec894c67a2009d6482d9b28c'),runtime_commit='e7cedec11904271d748f75fe60df3c33b85e1d45',operator_revision='4cbe640c83e98632fc8c7357a3d659bb2f85f656',join_audit_commit='88d96c4b9b053682243a9dd27d724fba9abcf63c',all_full_predictions_rollouts_tensors_and_weights_remote_only=True,existing_dataset_or_training_changed=False,new_GPU_requested=False,epoch=time.time())
p=D/'FINAL.json';p.write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n');files.append(str(p.relative_to(R)))
text=f'''# Audited122 delta: transfer and strict-EM ingestion accepted

The exact published subset of122batches was imported independently: **15,616questions /78,080genuine new-seed trajectories**, NQ12,300 +HotpotQA3,316. It did not rediscover or export other newly sealed batches. The source169,028question generation remains outside this bounded completion claim.

- Source snapshot: `audited122-delta-20261021-673b8cb6`;8archives/722,391,040bytes/1,734members/719,345,030member bytes.
- HF pinned catalog: `30efd6dd3abaa6cc4cfe5587c2e57eae06ba3000`, catalog SHA`3a24be0718992cbb3d2f2f69e0c417a7c1f67198d69c343511d0147da34efe54`.
- H100→management bounded RAM→private HF→HKUST2→HKUST3→HPC3, with source/HF/per-hop archive checks and fresh shared-storage member SHA. Independent final readback covered all8archives +1,734members in41.01seconds and proved zero old244 overlap by batch/qid/question-hash/authentic trajectory IDs.
- Receiver complete: `{s['receiver_complete']['path']}`; SHA`{s['receiver_complete']['sha256']}`.

The unchanged production ingester then recomputed own strict EM and verified complete causal contexts, question/token identity, and real old5+new5 seed namespaces. Its final accepted training bank has **{s['eligible_training_questions']:,}questions /{s['eligible_authentic_G10_trajectories']:,}authentic trajectories**. Diagnostic512/selection2048/final4096 identity exclusions remained enforced: `{json.dumps(s['excluded_by_qid'],sort_keys=True)}`; additional hash exclusions `{s['additional_excluded_by_question_hash']}`. No synthetic repacking, duplicate-seed substitution, or validation/final scoring was performed.

- Groups: `{s['groups']['path']}`, {s['groups']['bytes']}bytes, SHA`{s['groups']['sha256']}`.
- Source events: `{s['source_events']['path']}`, SHA`{s['source_events']['sha256']}`.
- Ingest complete: `{s['ingest_complete']['path']}`, SHA`{s['ingest_complete']['sha256']}`.
- Independent own-ID/ordered-EM/exclusion/histogram join: `{audit['audit']['path']}`, SHA`{audit['audit']['sha256']}`.

Actual verification progressed 11,040→31,840→53,760→74,560→99,840 trajectories before the terminal result. The entire source subset/old244 identity boundary and all payload hash checks were independently accepted. Initial real fsync probes confirmed HKUST2 `/data` ext4, HKUST3 `/backup` ext4, and HPC3 `/data` `/LK02` dpc shared storage; probes were removed. HF used the previously owned HKUST2 loopback17898 exit after actual private archive Range verification. Mac carried only code/configuration/small aggregate receipts; credentials stayed on stdin/process RAM.

Original warmup had seen the canonical169,028training questions. This is newly imported genuine sampling data, **not an unseen-question claim**. No feature tensor, actor/probe update, GPU request, or modification to active training datasets was made. New raw data remains isolated for future explicitly selected use.

Immutable runtime `e7cedec11904271d748f75fe60df3c33b85e1d45`; operator revision `4cbe640c83e98632fc8c7357a3d659bb2f85f656`; independent join source `88d96c4b9b053682243a9dd27d724fba9abcf63c`. Code and all small receipts are backed up on `codex/peer-audited122-delta-transfer-20261021`.
'''
p=D/'RESULT.md';p.write_text(text);files.append(str(p.relative_to(R)))
branch='codex/peer-audited122-delta-transfer-20261021';parent=subprocess.check_output(['git','rev-parse','refs/heads/'+branch],cwd=R,text=True).strip();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();index=Path(tempfile.mktemp(prefix='delta-final-report-index-',dir='/tmp'));env=dict(os.environ,GIT_INDEX_FILE=str(index))
def git(*args,**kw):return subprocess.check_output(['git',*args],cwd=R,env=env,**kw).decode().strip()
try:
 git('read-tree',parent);git('add','--',*files);tree=git('write-tree');commit=git('commit-tree',tree,'-p',parent,input=b'Archive completed audited122 raw transfer and independent strict-EM G10 acceptance\n');git('update-ref','refs/heads/'+branch,commit,parent);subprocess.run(['git','push','origin',commit+':refs/heads/'+branch],cwd=R,check=True)
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=R,text=True).split()[0];assert remote==commit
 inventory={}
 for name in files:
  raw=(R/name).read_bytes();assert raw==subprocess.check_output(['git','show',commit+':'+name],cwd=R);inventory[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==head
 receipt=dict(status='ALL_FINAL_SMALL_EVIDENCE_PUSHED_AND_BLOB_SHA_VERIFIED',commit=commit,remote_sha=remote,branch=branch,files=inventory,shared_main_HEAD_unchanged=True,epoch=time.time());(O/'final-publication.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
finally:index.unlink(missing_ok=True)
