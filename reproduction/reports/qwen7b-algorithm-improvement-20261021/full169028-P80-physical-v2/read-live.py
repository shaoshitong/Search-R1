"""Read-only actual process/lock/progress metadata of owned CPU preparation."""
import hashlib,json,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full169028-policy80-physical-ownG10-20261023-v2')
def read(p):return json.loads(Path(p).read_bytes())
out=dict(time=time.time(),base=str(B),final4096_read=False,Slurm_changed=False,files={})
for name in('prepare-process.json','controller.json','progress.json','failure.json','complete.json','fixture/accepted.json'):
 if(B/name).is_file():out['files'][name]=read(B/name)
ref=out['files'].get('prepare-process.json');live=False
if ref:
 p=Path('/proc/%d'%ref['pid'])
 if p.exists():live=(p/'stat').read_text().rsplit(')',1)[1].split()[19]==ref['start_ticks']and b'prepare_full169028_policy80_physical_v2.py'in(p/'cmdline').read_bytes()
pcontract=B/'physical-source-contract.json'
if pcontract.is_file():
 c=read(pcontract);raw=pcontract.read_bytes();out['physical_source_contract']=dict(proof=dict(path=str(pcontract),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),**{k:c[k]for k in('status','training_questions','training_trajectories','explicit_chosen_physical_questions','excluded_same_protocol_tid_wrong_physical_rows','excluded_wrong_execution_label_disagreements','duplicate_equal_payload_rows','all_selected_own_EM_matches_canonical','actual_physical_execution_and_seal_source_exact','whole_G5_not_mixed','selection_used_EM','canonical_labels_changed')});out['physical_source_contract']['selected_source_files']=len(c['selected_sources'])
out['actual_owned_process_live']=live
if live:
 out['process_resources']={line.split(':',1)[0]:line.split(':',1)[1].strip()for line in(p/'status').read_text().splitlines()if line.startswith(('VmRSS:','VmPeak:','Threads:'))}
if'fixture/accepted.json'in out['files']:
 a=out['files']['fixture/accepted.json'];f=B/'fixture/fixture-complete.json';raw=f.read_bytes();assert a['fixture']==dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
if(B/'prepare.log').exists():
 with(B/'prepare.log').open('rb')as f:f.seek(max(0,(B/'prepare.log').stat().st_size-1800));out['log_tail']=f.read().decode(errors='replace')
print(json.dumps(out))
