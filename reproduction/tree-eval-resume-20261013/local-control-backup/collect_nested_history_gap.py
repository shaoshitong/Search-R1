"""Read fixed historical scalar/rank/dev/native JSON evidence; never touch training or PT."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

SOURCE = '778b2b43dad90fc0c1c5d95d77c2326286011b73'
REMOTE = r'''
import hashlib,json,os,re,subprocess,time
from pathlib import Path
cfg=__CONFIG__
source='778b2b43dad90fc0c1c5d95d77c2326286011b73'
run=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:'+source)
unit='sst-tree-nested-branch-fresh200-778b2b43.service'
os.nice(15)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(path):
 if not path.is_relative_to(run) or path.is_symlink():raise ValueError('Fixed own regular JSON required')
 with path.open('rb') as f:
  before=os.fstat(f.fileno())
  if before.st_size>2*1024*1024:raise ValueError('Small JSON limit exceeded')
  raw=f.read();after=os.fstat(f.fileno())
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Input changed during read')
 return json.loads(raw),dict(path=str(path),sha256=digest(raw),bytes=len(raw),mtime_ns=before.st_mtime_ns)
def native(outer):
 path=run/'output/native-manifest-history'/('source-'+source+'-outer-%06d.receipt.json'%outer)
 receipt,rmeta=read(path);mp=Path(receipt['history_manifest_path']);manifest,mmeta=read(mp)
 if receipt.get('source_commit')!=source or receipt.get('completed_outer')!=outer:raise ValueError('Wrong native source/outer')
 identity=manifest['identity']
 if identity.get('code_commit')!=source or identity.get('completed_step')!=outer:raise ValueError('Wrong manifest identity')
 if mmeta['sha256']!=receipt.get('history_manifest_sha256') or mmeta['sha256']!=receipt.get('source_manifest_sha256'):raise ValueError('Archive differs from original writer manifest SHA')
 return dict(observed_epoch=time.time(),source_commit=source,tensor_contents_read=False,independent_cold_reload_performed=False,outer=outer,receipt_path=str(path),receipt_sha256=rmeta['sha256'],manifest_sha256=mmeta['sha256'],recorded_original_manifest_sha256=receipt['source_manifest_sha256'],archive_matches_original_manifest=True,manifest_status=manifest.get('status'),identity=identity,receipt=receipt)
if cfg['native_outer'] is not None:
 print(json.dumps(dict(mode='native',native=native(cfg['native_outer'])),allow_nan=False));raise SystemExit
started=time.time();first,last=cfg['first'],cfg['last']
state=subprocess.check_output(['systemctl','show',unit,'--property=MainPID,InvocationID,ActiveState,SubState,Result,ExecMainStatus'],text=True)
state_map=dict(line.split('=',1) for line in state.splitlines() if '=' in line)
if state_map.get('MainPID')!='503255' or state_map.get('InvocationID')!='a313bda919834d24b4c5d491698d39ce':raise ValueError('Original training instance identity differs')
raw=(run/'resolved.yaml').read_bytes();configmeta=dict(path=str(run/'resolved.yaml'),sha256=digest(raw),bytes=len(raw))
if configmeta['sha256']!='355edf5232dc963deb526281f7e1e61b98eaaa73b0bffc5e7855ef3d262799b3':raise ValueError('Resolved configuration differs')
metrics=[];lines=[];hasher=hashlib.sha256();offset=0;stage=run/'stage.log'
with stage.open('rb') as stream:
 before=os.fstat(stream.fileno())
 for raw in stream:
  hasher.update(raw);start=offset;offset+=len(raw)
  if not raw.endswith(b'\n'):continue
  line=raw.decode('utf8','replace')
  if 'timing_s/step:' not in line:continue
  found={k:float(v) for k,v in re.findall(r'([a-zA-Z0-9_./-]+):(-?[0-9]+(?:[.][0-9]+)?(?:[eE][+-]?[0-9]+)?)',line)}
  outer=found.get('step')
  if outer is None or outer!=int(outer) or not first<=int(outer)<=last:continue
  if found.get('training/actual_outer_updates')!=outer:raise ValueError('Historical full metric credit differs')
  metrics.append(found);lines.append(dict(outer=int(outer),path=str(stage),byte_start=start,byte_end=offset,line_sha256=digest(raw)))
 after=os.fstat(stream.fileno())
if [int(m['step']) for m in metrics]!=list(range(first,last+1)):raise ValueError('Historical full metric range missing or duplicated')
rank_fields=('outer_step','attempt_id','rank','successful_optimizer_calls','skipped_optimizer_calls','skipped_empty_minibatches','opportunities_processed','opportunities_skipped','scheduler_step','learning_rate','finite_grad_norms','nonzero_grad_norms','local_shard_sample_delta')
ranks={};rankfiles={};natives=[];rollouts={};rolloutfiles={}
for outer in range(first,last+1):
 ranks[str(outer)]=[];rankfiles[str(outer)]=[]
 for rank in range(8):
  path=run/'output/tree-rank-audit'/('step-%06d-attempt-%06d-rank-%02d.json'%(outer,outer,rank));value,meta=read(path)
  if value.get('outer_step')!=outer or value.get('attempt_id')!=outer or value.get('rank')!=rank:raise ValueError('Direct rank identity differs')
  ranks[str(outer)].append({k:value[k] for k in rank_fields if k in value});rankfiles[str(outer)].append(meta)
 path=run/'output/training-rollout-cost'/('attempt-%06d.json'%outer);value,meta=read(path)
 if value.get('candidate_outer')!=outer or value.get('code_commit')!=source or value.get('status')!='TRAINING_ROLLOUT_COST_COMPLETE':raise ValueError('Controller source/outer differs')
 rollouts[str(outer)]=value;rolloutfiles[str(outer)]=meta;natives.append(native(outer))
dev=[];devfiles=[]
for outer in cfg['dev']:
 path=run/'output/validation-cost'/('validation-506917-%d.json'%(outer//5-1));value,meta=read(path)
 if value.get('status')!='VALIDATION_COST_COMPLETE' or value.get('actual_completed_outer')!=outer or value.get('evaluated_rows')!=512:raise ValueError('Historical development identity differs')
 dev.append(dict(path=path.name,outer=outer,phase=value.get('phase'),status=value['status'],rows=value['evaluated_rows'],metrics=value['metrics'],ledger_sha256=value['ledger_sha256']));devfiles.append(meta)
result=dict(summary_schema='tree_nested_historical_gap_scalar_readback_v1',observed_start_epoch=started,observed_epoch=time.time(),source_commit=source,config_sha256=configmeta['sha256'],run=str(run),unit=unit,training_systemd=state,requested_outers=list(range(first,last+1)),iteration_metrics=metrics,metric_line_sources=lines,stage_log_snapshot=dict(path=str(stage),bytes_read=offset,sha256=hasher.hexdigest(),bytes_before=before.st_size,bytes_after=after.st_size,mtime_ns_before=before.st_mtime_ns,mtime_ns_after=after.st_mtime_ns,scope='Actual bytes streamed during this read; append-only current log may grow after observation'),resolved_config=configmeta,direct_rank_summaries_by_outer=ranks,direct_rank_source_files_by_outer=rankfiles,development=dev,development_source_files=devfiles,rollout_cost_by_outer=rollouts,rollout_cost_source_files_by_outer=rolloutfiles,native_metadata_readbacks=natives,read_only=True,remote_files_written=False,training_signals_sent=False,tensor_contents_read=False,raw_rollout_transferred=False,full_evaluation_output_transferred=False,exact_executor_resume_claimed=False,PT_current_availability_unknown=True)
encoded=json.dumps(dict(mode='gap',gap=result),allow_nan=False)
if len(encoded.encode())>800000:raise ValueError('Scalar gap summary exceeds compact budget')
print(encoded)
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--ssh-helper',type=Path,default=Path(__file__).with_name('ssh_retained18.py'))
    p.add_argument('--native-outer',type=int)
    p.add_argument('--first',type=int,default=107)
    p.add_argument('--last',type=int,default=119)
    args=p.parse_args()
    if args.native_outer is not None and not 107<=args.native_outer<=120:raise ValueError('Only native metadata107..120 permitted')
    if (args.first,args.last)!=(107,119):raise ValueError('Only fixed historical107..119 gap permitted')
    spec=importlib.util.spec_from_file_location('gap_route',args.ssh_helper)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    cfg=dict(native_outer=args.native_outer,first=args.first,last=args.last,dev=[110,115])
    code=REMOTE.replace('__CONFIG__',repr(cfg))
    response=helper.run('env CUDA_VISIBLE_DEVICES= python3 -B -c '+shlex.quote(code),timeout=60)
    if response.returncode:raise RuntimeError('Read-only gap unavailable; current unknown: '+response.stderr[-1600:])
    payload=json.loads(response.stdout)
    if payload['mode']=='native':items=[(f'NATIVE_HISTORY_{args.native_outer}_METADATA_READBACK.json',payload['native'])]
    else:
        gap=payload['gap'];gap['collector_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();gap['remote_program_sha256']=hashlib.sha256(code.encode()).hexdigest();gap['ssh_helper_sha256']=hashlib.sha256(args.ssh_helper.read_bytes()).hexdigest()
        items=[('HISTORICAL_GAP107_119_SCALAR_READBACK.json',gap)]+[(f'NATIVE_HISTORY_{x["outer"]}_METADATA_READBACK.json',x) for x in gap['native_metadata_readbacks']]
    args.base.mkdir(parents=True,exist_ok=True)
    for name,value in items:
        raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();dest=args.base/name
        if dest.exists():
            old=json.loads(dest.read_bytes())
            if name.startswith('NATIVE_HISTORY_') and all(old.get(k)==value.get(k) for k in ['source_commit','outer','receipt_sha256','manifest_sha256','identity','receipt']):continue
            raise ValueError('Immutable local evidence exists with different content: '+name)
        with dest.open('xb') as stream:stream.write(raw)
        if dest.read_bytes()!=raw:raise ValueError('Local byte readback differs')
        print(json.dumps(dict(path=str(dest.resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),observed_epoch=value.get('observed_epoch'))),flush=True)


if __name__=='__main__':main()
