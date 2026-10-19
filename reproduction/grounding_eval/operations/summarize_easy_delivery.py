"""Export a small final receipt only after normal and both CF acceptances."""
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

parser=argparse.ArgumentParser(__doc__);parser.add_argument('--audit-dir',required=True);args=parser.parse_args()
audit=Path(args.audit_dir);upload=json.loads((audit/'easy-hf-verified-status.json').read_text())
assert upload['state']=='HF_FILES_VERIFIED' and len(upload['receipts'])==13
observations=[]; maximum_attempts={}; observed_curl_codes=[]
for path in sorted(audit.glob('easy-a800-asset-observation-*.json')):
    raw=path.read_bytes(); observation=json.loads(raw); status=observation.get('status',{})
    if not isinstance(status,dict) or 'files' not in status:
        continue
    attempts={name:record['attempt'] for name,record in status['files'].items() if 'attempt' in record}
    for name,attempt in attempts.items():
        maximum_attempts[name]=max(maximum_attempts.get(name,0),attempt)
    observed_curl_codes.extend(record['curl_exit'] for record in status['files'].values() if 'curl_exit' in record)
    observations.append(dict(snapshot=path.name,snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        at=observation['at'],observed_bytes=status['observed_bytes'],attempts=attempts,
        partial_file_sizes={record['name']:record['size'] for record in observation.get('files',[]) if record['name'].endswith('.part')}))
retry_event_path=audit/'easy-a800-retry-events.json'
retry_events=json.loads(retry_event_path.read_text()) if retry_event_path.exists() else None
if retry_events:
    observed_curl_codes.extend(record['curl_exit'] for event in retry_events['events']
        for record in event['files'].values() if 'curl_exit' in record)
minimum_previous=sum(max(0,value-1) for value in maximum_attempts.values())
upload['_download_recovery_audit']=dict(maximum_observed_attempts=maximum_attempts,
    minimum_nonzero_previous_attempts=minimum_previous,
    observed_curl_exit_codes=observed_curl_codes,
    prior_exit_code_null_reason='Legacy worker overwrites per-attempt exit codes; uncaptured earlier codes cannot be reconstructed from final status.' if len(observed_curl_codes)<minimum_previous else None,
    per_connection_max_time_seconds=600,recovery_kept_existing_writer_and_sha_named_partials=True,
    attribution='Bounded600s per-connection HTTP timeout followed by retained-part continuation; not an inference failure.',
    observations=observations,retry_events=retry_events,
    retry_event_audit_sha256=hashlib.sha256(retry_event_path.read_bytes()).hexdigest() if retry_events else None)
remote=r'''
import pathlib,json,sys,hashlib,time
upload=json.load(sys.stdin);base=pathlib.Path('/data/shaoshitong/grounding-eval-20261019')
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb')as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()
control=base/'easy-selection-dual-cf-control';completion=read(control/'complete.json');assert completion['state']=='EASY_DUAL_CF_COMPLETE'
normal=base/'easy-selection-normal3610';nr=read(normal/'normal-independent-verification.json');assert nr['state']=='VERIFIED'and nr['count']==3610
content=nr['normal_content_order_sha256'];assert content=='23c0a8307f38a7a8d1cbd9e71fb115b7bd091421a3d3b10952fb71c8210b551c'
normal_sha=sha(normal/'normal.jsonl');assert normal_sha==nr['normal_jsonl_sha256']
asset=base/'assets/easy-selection-iter200';ar=read(asset/'asset-migration-complete.json');assert ar['revision']==upload['final_revision']and ar['status']=='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED'
assert {r['name']:(r['bytes'],r['sha256'])for r in upload['receipts']}=={n:(v['size'],v['sha256'])for n,v in ar['files'].items()}
reports={}
for version,name in [('v4','easy-selection-cf-full-v4'),('v5','easy-selection-cf-strict-v5')]:
 root=base/name;manifest=read(root/'manifest.json');summary=read(root/'verified-summary.json');accept=read(root/'independent-acceptance-20261019.json')
 assert manifest['source']['sha256']==normal_sha and accept['normal_sha256']==normal_sha and accept['normal_content_order_sha256']==content
 assert accept['state']=='INDEPENDENT_CF_FULL_READ_VERIFIED'and summary['status']=='complete'and summary['counts']['normal_questions']==3610
 assert accept['cf_attempts']['errors']==0 and accept['cf_attempts']['failed_or_interrupted_attempts']==0
 batches=[]
 for line in(root/'attempts.jsonl').open():
  event=json.loads(line)
  if event['status']=='INFERENCE_STARTED':batches.append(len(event['attempt_ids']))
 reports[version]={'root':str(root),'counts':summary['counts'],'TC':summary['TC'],'D_Inv':summary['D_Inv'],'TC_denominator':summary['TC_denominator'],'D_Inv_denominator':summary['D_Inv_denominator'],'coverage':summary['coverage'],'cf_tokens':summary['token_usage']['counterfactual'],'cf_attempt_tokens':accept['cf_attempts'],'actual_inference_batch_sizes':batches,'manifest_path':str(root/'manifest.json'),'manifest_file_sha256':sha(root/'manifest.json'),'source_normal_sha256':normal_sha,'acceptance_path':str(root/'independent-acceptance-20261019.json'),'acceptance_sha256':sha(root/'independent-acceptance-20261019.json'),'normal_content_order_sha256':content}
 if version=='v4':
  replay=read(root/'official-changed-rule-replay.json');method=replay['methods']['Easy Selection'];reports[version]['changed_rule_only']=method['changed_rule_only'];reports[version]['official_parser_plus_changed_rule']=method['official_parser_plus_changed_rule'];reports[version]['rule_replay_valid_cf']=method['valid_cf_records'];reports[version]['rule_replay_path']=str(root/'official-changed-rule-replay.json');reports[version]['rule_replay_sha256']=sha(root/'official-changed-rule-replay.json');reports[version]['scope']='Same frozen-v4 interventions and prompts as other five diagnostic rows; official changed-answer scoring replay only, not a full official reproduction; known replacement limitations retained.'
 else:reports[version]['scope']='Corrected conservative typed strict-v5; finite eligible-type conditional coverage, not all searched-correct questions or full official reproduction.'
v4=read(base/'easy-selection-cf-full-v4/verified-summary.json');v5=read(base/'easy-selection-cf-strict-v5/verified-summary.json');assert v4['counts']['normal_correct']==v5['counts']['normal_correct']
normal_tokens=v4['token_usage']['normal'];assert all(v['measured_questions']==3610 for v in normal_tokens.values())
for key,value in normal_tokens.items():assert nr['token_totals'][key]==value['total']
exits={'pilot96':read(base/'easy-selection-normal-pilot96/control/exit.json')['exit_code'],'normal3610':read(normal/'control/exit.json')['exit_code']}
for version in('v4','v5'):exits[version+'-CF']=read(control/(version+'-run-exit.json'))['exit_code']
assert all(v==0 for v in exits.values())
metrics=[read(p)for p in sorted((normal/'batch-metrics').glob('*.json'))]
report={'state':'EASY_NORMAL_AND_DUAL_CF_INDEPENDENTLY_VERIFIED','at':time.time(),'method':'Easy Selection','checkpoint':'iter200','normal':{'root':str(normal),'questions':3610,'correct':v4['counts']['normal_correct'],'EM':v4['counts']['normal_correct']/3610,'token_usage':normal_tokens,'actual_batch_sizes':nr['actual_batch_sizes'],'per_batch_execution_metrics':metrics,'independent_verification_path':str(normal/'normal-independent-verification.json'),'independent_verification_sha256':sha(normal/'normal-independent-verification.json'),'source_sha256':normal_sha,'normal_content_order_sha256':content},'v4_diagnostic':reports['v4'],'v5_corrected_strict':reports['v5'],'assets':{'repo':ar['repo'],'revision':ar['revision'],'prefix':ar['prefix'],'verified_files':len(ar['files']),'verified_bytes':ar['verified_bytes'],'all_selected_file_source_and_target_sha_verified':True,'file_inventory':ar['files'],'receipt_path':str(asset/'asset-migration-complete.json'),'receipt_sha256':sha(asset/'asset-migration-complete.json'),'source_preserved':True,'model_bytes_through_Mac':0},'exit_codes':exits,'all_raw_outputs_remain_remote':True,'normal_recomputed_for_CF':False,'two_CF_protocols_kept_separate':True,'normal_content_order_sha256':content}
report['assets']['download_recovery_audit']=upload['_download_recovery_audit']
out=base/'easy-selection-final-delivery.json'
if out.exists():raise FileExistsError('Do not overwrite final delivery receipt')
with out.open('x')as f:json.dump(report,f,indent=2);f.flush();import os;os.fsync(f.fileno())
print(json.dumps(report))
'''
compile(remote,'easy-final-delivery','exec')
spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
command=ctl.route('baai-a800','python3 -B -c '+shlex.quote(remote));command[1:1]=['-o','ServerAliveInterval=5','-o','ServerAliveCountMax=6']
result=subprocess.run(command,input=json.dumps(upload),capture_output=True,text=True,timeout=45)
if result.returncode:raise RuntimeError('Final delivery not accepted: '+result.stderr[-1200:])
report=json.loads(result.stdout);(audit/'easy-final-delivery.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
