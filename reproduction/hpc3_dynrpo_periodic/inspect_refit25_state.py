"""HPC3 R16 only: read SHA-checked refit25 metadata without reading weight/Adam tensors.

Run on the remote host with CUDA_VISIBLE_DEVICES empty. Temporary hardlinks are
removed in finally. This backs up the exact inspection code used for epoch20.
"""
from pathlib import Path
import json,time,hashlib,os,uuid,resource
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
base=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-periodic25-thread5-20261016-r16')
phase=base/'artifacts/question-probe/periodic-refits/25'
holder=base/'diagnostics/phase-readback'
holder.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for v in iter(lambda:f.read(4*1024**2),b''):h.update(v)
 return h.hexdigest()
out=None
for attempt in range(3):
 pointer=json.loads((phase/'latest-state.json').read_text())
 snapshot=holder/('reader-'+uuid.uuid4().hex+'.pt')
 try:
  os.link(phase/'latest-state.pt',snapshot)
  actual_sha=sha(snapshot)
  if actual_sha!=pointer['sha256']:continue
  saved=torch.load(snapshot,map_location='cpu',weights_only=False,mmap=True)
  p=saved['phase']
  assert p['epoch']==pointer['epoch'] and p['batch']==pointer['batch']
  assert saved['outer']==25 and saved['pre_receipt_sha256']=='ef05cc61cdd37c7fe6852fdddb93e60a26c3c1c399ae704c3dee082637acf09d'
  assert saved['head']['update_count']==400+20*p['epoch']+p['batch']==pointer['head_adam_steps']
  assert len(saved['indices'])==len(saved['labels'])==5120
  ranks=[{k:r[k] for k in ('rank','parameter_sha256','local_parameter_count','actual_optimizer_writes','completed_outer','scheduler_sha256','adam_steps')} for r in saved['frozen_before']]
  assert len(ranks)==4 and all(r['actual_optimizer_writes']==500 and r['completed_outer']==25 and r['adam_steps']==[500] for r in ranks)
  out=dict(checked_unix=time.time(),snapshot_sha256=actual_sha,source_pointer=pointer,epoch=p['epoch'],batch=p['batch'],head_adam_steps=saved['head']['update_count'],start_adam_steps=p['start_steps'],saved_questions=len(saved['indices']),saved_labels=len(saved['labels']),completed_minibatches=len(p['minibatches']),summaries=p['summaries'],frozen_student_before=ranks,feature_hashes_present=saved['feature_hashes'] is not None,feature_hash_container_type=type(saved['feature_hashes']).__name__,cuda_initialized=torch.cuda.is_initialized(),max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scope='Read-only mmap metadata from an SHA-checked temporary hardlink; no model/Adam tensor data accessed, no inference, temporary link removed')
  del saved
  break
 finally:
  snapshot.unlink(missing_ok=True)
if out is None:out=dict(checked_unix=time.time(),status='RETRY_AFTER_ATOMIC_WRITER_PROGRESS',reader_retry_count=3)
print(json.dumps(out))

