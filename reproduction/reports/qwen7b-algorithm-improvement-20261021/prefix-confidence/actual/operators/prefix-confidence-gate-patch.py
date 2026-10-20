from pathlib import Path
p=Path('reproduction/operations/qwen7b_warmup/prefix_confidence_pipeline.py');s=p.read_text();s=s.replace("BASE=d.BASE;FIT=BASE/'fit'","BASE=d.BASE;FIT=BASE/'fit';FIT_CONFIG='fit-config-v2.json'")
s=s.replace("BASE/'fit-config.json'","BASE/FIT_CONFIG")
start=s.index("   if gate['passed']:\n");end=s.index("   assert d.stamp",start);block=s[start:end];body=block.split('\n',1)[1];body='\n'.join(line[3:] if line.startswith('   ') else line for line in body.splitlines());body=body.replace("qid not in qids","qid not in train_qids").replace("state=states['prefix_full8'];",'').replace("cprefs['prefix_full8']","checkpoint")
helper="\ndef external_after_independent_gate(config,freezeref,state,checkpoint,train_qids):\n import numpy as np\n import torch\n from . import prefix_confidence_calibration as c,platt_g10_calibration as p\n"+body+"\n return external\n"
s=s[:start]+s[end:];index=s.index('\ndef run(config_ref):');s=s[:index]+helper+s[index:]
s=s.replace("status='PREFIX_CONFIDENCE_V2_SINGLE_INFORMATION_TEST_COMPLETE',config=config_ref", "status=('PREFIX_CONFIDENCE_V2_INTERNAL_COMPLETE_AWAITING_INDEPENDENT_AUDIT' if gate['passed'] else 'PREFIX_CONFIDENCE_V2_SINGLE_INFORMATION_TEST_COMPLETE'),config=config_ref")
s=s.replace("external_payload_opened=gate['passed'],fit_loss_progress_observations", "external_payload_opened=False,fit_loss_progress_observations")
s=s.replace("once(FIT/'complete.json',final);return final", "once(FIT/'internal-complete.json',final)\n   if not gate['passed']:once(FIT/'complete.json',final)\n   return final")
idx=s.index("if __name__=='__main__':")
finish='''\ndef finish_external(config_ref,audit_ref):
 import torch
 from . import prefix_confidence_calibration as c
 config=read(checked(config_ref));internal_ref=ref(FIT/'internal-complete.json');internal=read(internal_ref['path']);audit=read(checked(audit_ref))
 assert audit['status']=='PREFIX_CONFIDENCE_INTERNAL_FULLSTATE_AND_METRICS_INDEPENDENTLY_PASSED' and audit['config']==config_ref and audit['internal_complete']==internal_ref and audit['full_aux_checkpoints']==internal['full_aux_checkpoints'] and audit['external_opening_permitted'] is True and internal['internal_gate']['passed']
 with (FIT/'calibration.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (FIT/'complete.json').exists():return read(FIT/'complete.json')
  if (FIT/'external-open-intent.json').exists():raise ValueError('Prior external opening preserved; no implicit rescore')
  cp=internal['full_aux_checkpoints']['prefix_full8'];checked(cp);state=torch.load(cp['path'],map_location='cpu',weights_only=False);c.restore(state);truth=read(checked(config['groups']));truth=truth['groups'] if isinstance(truth,dict) else truth
  external=external_after_independent_gate(config,internal['frozen_candidate'],state,cp,[g['qid'] for g in truth]);final=dict(internal,status='PREFIX_CONFIDENCE_V2_SINGLE_INFORMATION_TEST_COMPLETE',external=external,external_payload_opened=True,independent_internal_audit=audit_ref,external_completed_epoch=time.time());once(FIT/'complete.json',final);return final
'''
s=s[:idx]+finish+s[idx:];s=s.replace("choices=('prepare','run')","choices=('prepare','run','external')");s=s.replace("p.add_argument('--config-sha256');a=p.parse_args()", "p.add_argument('--config-sha256');p.add_argument('--internal-audit-sha256');a=p.parse_args()")
s=s.replace("r=ref(BASE/FIT_CONFIG);assert r['sha256']==a.config_sha256;v=run(r)","r=ref(BASE/FIT_CONFIG);assert r['sha256']==a.config_sha256\n   if a.action=='run':v=run(r)\n   else:\n    audit=ref(FIT/'independent-internal-audit.json');assert audit['sha256']==a.internal_audit_sha256;v=finish_external(r,audit)")
p.write_text(s)
p=Path('reproduction/operations/qwen7b_warmup/deploy_prefix_confidence_fit.py');s=p.read_text().replace("ROOT=d.BASE/'fit-deployment'","ROOT=d.BASE/'fit-deployment-v2'").replace("experiment/'fit-config.json'","experiment/'fit-config-v2.json'");p.write_text(s)
