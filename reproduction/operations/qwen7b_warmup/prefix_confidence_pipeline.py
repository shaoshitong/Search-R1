"""One preregistered CPU information test with matched2 internal control and full8.

No probe/actor/Adam loading; only a new scalar optimizer. External native LP and
sealed E4 predictions remain unopened unless full8 passes the frozen jointgate.
"""
import argparse,fcntl,gzip,hashlib,json,math,os,threading,time
from pathlib import Path
from . import prefix_confidence_data as d
BASE=d.BASE;FIT=BASE/'fit'
def read(p):return d.read(p)
def ref(p):return d.ref(p)
def checked(v):return d.checked(v)
def atomic(p,v):return d.atomic(p,v)
def once(p,v):return d.once(p,v)
def gate_decision(raw,control,full):
 key='Top512';rba=raw['balanced_accuracy'];cba=control['balanced_accuracy'];fba=full['balanced_accuracy'];rt=raw[key]['Mixed'];ct=control[key]['Mixed'];ft=full[key]['Mixed']
 return dict(raw_strict_BA=fba>rba,raw_strict_Top512=ft>rt,control_BA_nondecrease=fba>=cba,control_Top512_nondecrease=ft>=ct,control_at_least_one_strict=fba>cba or ft>ct,passed=fba>rba and ft>rt and fba>=cba and ft>=ct and (fba>cba or ft>ct))
def prepare(code,commit,protocol_commit):
 code=Path(code);extract=ref(BASE/'data/complete.json');accepted=read(extract['path']);assert accepted['status']==d.STATUS and accepted['questions']==8192 and accepted['trajectories']==81920 and accepted['availability_qualified'] is True and accepted['all_own_ID_EM_order_native_LP_and_P20_exact'] is True;checked(accepted['features']);c=read(checked(accepted['config']));protocol=ref(code/'reproduction/reports/qwen7b-algorithm-improvement-20261021/prefix-confidence-PROTOCOL.md');assert protocol_commit=='a1ec8bbe44a24deb9ada77b59f3edf57cdc653e2'
 config=dict(role='PREREGISTERED_PREFIX_CONFIDENCE_INFORMATION_TEST_V2',source_commit=commit,code_root=str(code),protocol=protocol,protocol_git_commit=protocol_commit,extract_complete=extract,features=accepted['features'],groups=accepted['groups'],split=accepted['split'],E4_predictions=accepted['E4_predictions'],source_checkpoint=accepted['source_checkpoint'],source_EMA_sha256=accepted['source_EMA_sha256'],external_predictions=c['external_predictions'],external_validation_complete=c['external_validation_complete'],external_groups=c['external_groups'],external_fixed_G5_plan=c['external_fixed_G5_plan'],external_source_audit=c['external_source_audit'],modes=['matched_platt2','prefix_full8'],objective='trajectory_BCE1+uniform_all252_threeevent_NLL1',regularizer=.01,CPU=2,RSS_limit_bytes=3*1024**3,whole_fit_stage_deadline_seconds=1560,internal_G5_indices=[5,6,7,8,9],external_candidate='prefix_full8_only',external_gate='strict BA and Top512 improve raw E4; neither decrease vs matched2 and at least one strict',external_payload_opened=False,final_read=False)
 FIT.mkdir(exist_ok=True);once(BASE/'fit-config.json',config);return dict(status='PREFIX_CONFIDENCE_V2_REAL_DATA_AND_QID_SPLIT_ACCEPTED_FIT_NOT_STARTED',config=ref(BASE/'fit-config.json'),extract_complete=extract)
def load_features(reference,groups):
 import numpy as np
 rows=[];ordered=[]
 with gzip.open(checked(reference),'rt') as f:
  for i,line in enumerate(f):
   x=json.loads(line)
   if i>=len(groups) or x['qid']!=groups[i]['qid'] or x['trajectory_ids']!=groups[i]['trajectory_ids']:raise ValueError('Confidence feature ordered authentic IDs differ')
   if set(x)!={'qid','trajectory_ids','features'}:raise ValueError('Only prescribed features and own identities allowed')
   rows.append(x['features']);ordered.append(x['qid'])
 x=np.asarray(rows,dtype=np.float64)
 if len(rows)!=len(groups) or x.shape!=(len(groups),10,6) or not np.isfinite(x).all():raise ValueError('All authentic G10 six features required')
 return x

def extract_external(config,freeze):
 """Only called after full8 CP/internalgate freeze; same original fixed G5."""
 frozen=read(checked(freeze));assert frozen['internal_gate']['passed'] and frozen['mode']=='prefix_full8' and frozen['external_payload_opened'] is False;checked(frozen['checkpoint']);source=read(checked(config['external_validation_complete']));fixed=read(checked(config['external_fixed_G5_plan']));assert fixed['summary']['labels_consulted'] is False and fixed['summary']['role']=='validation';groups=fixed['groups'];assert len(groups)==2048
 wanted={t:row['qid'] for row in groups for t in row['trajectory_ids']};assert len(wanted)==10240 and all(len(r['trajectory_ids'])==5 for r in groups);audit=read(checked(config['external_source_audit']));found={};proofs=[]
 for event,batches in audit['events'].items():
  for batch in batches:
   prefix=batch['phase_payloads']['prefix'];full=batch['phase_payloads']['full'];checked(prefix);p={}
   with gzip.open(prefix['path'],'rt') as f:
    for line in f:
     row=json.loads(line);tid=row['trajectory_id']
     if tid in wanted:
      assert tid not in p and row['qid']==wanted[tid];p[tid]=row
   if not p:continue
   checked(full);seen=set()
   with gzip.open(full['path'],'rt') as f:
    for line in f:
     row=json.loads(line);tid=row['trajectory_id']
     if tid not in p:continue
     assert tid not in seen and tid not in found and row['qid']==wanted[tid];seen.add(tid);value=d.projection(p[tid],row);found[tid]=value['features']
   assert seen==set(p);proofs.append(dict(prefix=prefix,full=full));atomic(FIT/'external-progress.json',dict(stage='FROZEN_AFTER_GATE_FIXED_G5_NATIVE_PREFIX_CONFIDENCE',trajectories=len(found),expected=10240,fit_or_threshold_updates=0,time=time.time()))
 assert set(found)==set(wanted);output=FIT/'external-prefix-features.jsonl.gz'
 with gzip.open(output,'wb') as f:
  for g in groups:f.write(d.a.canonical(dict(qid=g['qid'],trajectory_ids=g['trajectory_ids'],features=[found[t] for t in g['trajectory_ids']]))+b'\n')
 return groups,found,dict(features=ref(output),source_batches=proofs,trajectories=10240,freeze=freeze,all_selected_native_LP_and_P20_exact=True)

def run(config_ref):
 import numpy as np
 import torch
 from . import prefix_confidence_calibration as c,platt_g10_calibration as p
 assert not torch.cuda.is_initialized();config=read(checked(config_ref));assert Path(config['code_root'])==Path(__file__).resolve().parents[3];checked(config['protocol']);accepted=read(checked(config['extract_complete']));assert accepted['features']==config['features'] and accepted['availability_qualified'];extractcfg=read(checked(accepted['config']));checked(extractcfg['source_protocol']);FIT.mkdir(exist_ok=True)
 with (FIT/'calibration.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (FIT/'complete.json').exists():return read(FIT/'complete.json')
  if (FIT/'fit-started.json').exists():raise ValueError('Prior fit attempt preserved; no implicit refit/extra optimization budget')
  torch.set_num_threads(2);torch.set_num_interop_threads(1);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);start=time.monotonic();deadline=start+1560;stop=threading.Event()
  def guard():
   while not stop.wait(.25):
    rss=int(next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')).split()[1])*1024
    if rss>3*1024**3 or time.monotonic()>deadline+5:atomic(FIT/'failure.json',dict(type='WHOLE_CPU_FIT_RESOURCE_LIMIT',RSS_bytes=rss,elapsed_seconds=time.monotonic()-start));os._exit(74)
  threading.Thread(target=guard,daemon=True).start()
  try:
   truth=read(checked(config['groups']));truth=truth['groups'] if isinstance(truth,dict) else truth;qids=[g['qid'] for g in truth];split=read(checked(config['split']));mask=c.question_split(qids);assert [q for q,m in zip(qids,mask) if m]==split['held_qids'] and [q for q,m in zip(qids,mask) if not m]==split['fit_qids'];assert len(truth)==8192 and mask.sum()==2048
   y=np.asarray([g['labels'] for g in truth],dtype=np.float64);features=load_features(config['features'],truth);c.validate_features(features);scored=p.read_scored(config['E4_predictions'],truth);z=scored['logits'];prob=scored['probabilities'];hold_q=[q for q,m in zip(qids,mask) if m];fit_q=[q for q,m in zip(qids,mask) if not m];y5=y[mask,5:];z5=z[mask,5:];x5=features[mask,5:];p5=prob[mask,5:]
   cpstat=d.stamp(config['source_checkpoint']['path']);once(FIT/'fit-started.json',dict(config=config_ref,extract=config['extract_complete'],split=config['split'],models=['E4_only'],modes=config['modes'],start_epoch=time.time(),external_payload_opened=False));states={};reports={};cprefs={};internals={};readbacks={};counter=[]
   for mode,use in [('matched_platt2',False),('prefix_full8',True)]:
    bind=dict(availability_qualified=True,availability_receipt=config['extract_complete'],calibration_mode=mode,config=config_ref,protocol=config['protocol'],features=config['features'],groups=config['groups'],E4_predictions=config['E4_predictions'],split=config['split'],source_checkpoint=config['source_checkpoint'],source_EMA_sha256=config['source_EMA_sha256'])
    def progress(calls,loss):
     row=dict(mode=mode,closure_calls=calls,loss=loss,elapsed_seconds=time.monotonic()-start,external_payload_opened=False,time=time.time());atomic(FIT/'progress.json',row);counter.append(row)
    def checkpoint(state):
     path=FIT/('aux-'+mode+('-complete.pt' if state['phase']==c.PHASE else '-partial.pt'));tmp=path.with_suffix('.pt.partial');torch.save(state,tmp);tmp.replace(path);saved=torch.load(path,map_location='cpu',weights_only=False);assert c.same_state(state,saved)
     if state['phase']==c.PHASE:
      restored=c.restore(saved);assert c.same_state(restored['optimizer'].state_dict(),state['optimizer']) and c.same_state(c.capture_rng(),state['rng']);assert torch.equal(restored['raw'].detach(),state['raw'])
      reference=c.calibrated(z5[:8],x5[:8] if use else None,state['raw'],state['normalizer'],use_prefix=use);again=c.calibrated(z5[:8],x5[:8] if use else None,restored['raw'],restored['normalizer'],use_prefix=use);assert torch.equal(reference,again)
      readbacks[mode]=dict(checkpoint=ref(path),full_raw_optimizer_RNG_normalizer_sources_phase_reload=True,optimizer_iterations=state['fit']['optimizer_iterations'],scalar_mode=mode,source_probe_not_loaded=True,time=time.time());atomic(FIT/('checkpoint-'+mode+'-reload-audit.json'),readbacks[mode])
     return ref(path)
    state,report=c.fit(z[~mask],y[~mask],features[~mask] if use else None,fit_q,bind,progress=progress,checkpoint_callback=checkpoint,use_prefix=use,deadline_monotonic=deadline);states[mode]=state;reports[mode]=report;cprefs[mode]=report['full_checkpoint'];assert readbacks[mode]['checkpoint']==cprefs[mode]
    internals[mode]=c.summary(z5,y5,x5 if use else None,hold_q,state);q=c.score(z5,x5 if use else None,state);pred=FIT/('internal-'+mode+'.jsonl.gz')
    candidate_z=c.calibrated(z5,x5 if use else None,state['raw'],state['normalizer'],use_prefix=use);candidate_p=torch.sigmoid(candidate_z).detach().numpy()
    with gzip.open(pred,'wb') as out:
     for i,own in enumerate(np.flatnonzero(mask)):out.write(d.a.canonical(dict(qid=qids[own],trajectory_ids=truth[own]['trajectory_ids'][5:],strict_em=truth[own]['labels'][5:],p_j=candidate_p[i].tolist(),q=float(q[i])))+b'\n')
    reports[mode]['internal_predictions']=ref(pred);atomic(FIT/('internal-'+mode+'.json'),dict(mode=mode,checkpoint=cprefs[mode],metrics=internals[mode],predictions=ref(pred),phase='CP_SAVED_AND_RELOADED_BEFORE_THIS_INTERNAL_ASSESSMENT',external_payload_opened=False))
   raw=p.raw_probability_summary(p5,y5,hold_q);identity_p=1/(1+np.exp(-z5));clipped=p.raw_probability_summary(identity_p,y5,hold_q);gate=gate_decision(raw,internals['matched_platt2'],internals['prefix_full8']);clipinfo=dict(original_raw=raw,clipped_identity=clipped,probability_clip=1e-6,threshold_flips=int(((1-p5.prod(1)-(1-p5).prod(1)>=.5)!=(1-identity_p.prod(1)-(1-identity_p).prod(1)>=.5)).sum()))
   freeze=dict(mode='prefix_full8',checkpoint=cprefs['prefix_full8'],control_checkpoint=cprefs['matched_platt2'],config=config_ref,protocol=config['protocol'],internal_gate=gate,internal_raw_E4=raw,internal_matched2=internals['matched_platt2'],internal_full8=internals['prefix_full8'],external_payload_opened=False,original_probe_file_stamp=cpstat,source_probe_optimizer_reset=False,time=time.time());once(FIT/'frozen-candidate-and-gate.json',freeze);freezeref=ref(FIT/'frozen-candidate-and-gate.json');external=None
   if gate['passed']:
    atomic(FIT/'external-open-intent.json',dict(frozen_candidate=freezeref,checkpoint=cprefs['prefix_full8'],open_once=True,time=time.time()));groups,featuremap,qualification=extract_external(config,freezeref);by={g['qid']:g for g in read(checked(config['external_groups']))['validation']};expected={g['qid']:g for g in groups};seen=set();output=[]
    with gzip.open(checked(config['external_predictions']),'rt') as f:
     for line in f:
      row=json.loads(line);qid=row['qid'];assert qid in expected and qid not in seen and qid not in qids;seen.add(qid);g=expected[qid];truthmap=dict(zip(by[qid]['trajectory_ids'],by[qid]['labels']));assert row['trajectory_ids']==g['trajectory_ids'] and row['strict_em']==[truthmap[t] for t in g['trajectory_ids']];output.append(row)
    assert len(output)==2048 and seen==set(expected);eq=[g['qid'] for g in output];ep=np.asarray([g['p_j'] for g in output]);ez=c.probability_logits(ep);ey=np.asarray([g['strict_em'] for g in output]);ex=np.asarray([[featuremap[t] for t in g['trajectory_ids']] for g in output]);state=states['prefix_full8'];measured=c.summary(ez,ey,ex,eq,state);original=p.raw_probability_summary(ep,ey,eq);clipped_external=p.raw_probability_summary(1/(1+np.exp(-ez)),ey,eq);scores=c.score(ez,ex,state);newz=c.calibrated(ez,ex,state['raw'],state['normalizer'],use_prefix=True);newp=torch.sigmoid(newz).detach().numpy();pred=FIT/'external-frozen-full8-predictions.jsonl.gz'
    with gzip.open(pred,'wb') as f:
     for i,g in enumerate(output):f.write(d.a.canonical(dict(qid=g['qid'],trajectory_ids=g['trajectory_ids'],strict_em=g['strict_em'],original_p_j=g['p_j'],prefix_features=ex[i].tolist(),p_j=newp[i].tolist(),q=float(scores[i])))+b'\n')
    external=dict(metrics=measured,uncalibrated_E4=original,clipped_identity_E4=clipped_external,predictions=ref(pred),qualification=qualification,external_source_predictions=config['external_predictions'],frozen_checkpoint=cprefs['prefix_full8'],fixed_G5_plan=config['external_fixed_G5_plan'],fixed_E2_BA_frontier=.5625049686633887,fixed_E4_Top512_frontier=196,frontier_joint_pass=measured['balanced_accuracy']>.5625049686633887 and measured['Top512']['Mixed']>196,source_payload_opened_once=True,refit_or_threshold_tuning=False)
   assert d.stamp(config['source_checkpoint']['path'])==cpstat
   final=dict(status='PREFIX_CONFIDENCE_V2_SINGLE_INFORMATION_TEST_COMPLETE',config=config_ref,extract_complete=config['extract_complete'],full_aux_checkpoints=cprefs,checkpoint_reload_audits=readbacks,fit_reports=reports,internal_raw_and_clipping=clipinfo,internal_metrics=internals,internal_gate=gate,frozen_candidate=freezeref,external=external,external_payload_opened=gate['passed'],fit_loss_progress_observations=counter,source_probe_file_stat_unchanged=True,source_probe_model_or_Adam_loaded=False,source_probe_updates=0,actor_updates=0,GPU_used=False,final_read=False,elapsed_seconds=time.monotonic()-start,time=time.time());once(FIT/'complete.json',final);return final
  finally:stop.set()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','run'),required=True);p.add_argument('--code-root');p.add_argument('--commit');p.add_argument('--protocol-commit');p.add_argument('--config-sha256');a=p.parse_args()
 try:
  if a.action=='prepare':v=prepare(a.code_root,a.commit,a.protocol_commit)
  else:
   r=ref(BASE/'fit-config.json');assert r['sha256']==a.config_sha256;v=run(r)
  print(json.dumps(dict(status=v['status'],internal_gate=v.get('internal_gate'),external_payload_opened=v.get('external_payload_opened'),elapsed_seconds=v.get('elapsed_seconds')),sort_keys=True),flush=True)
 except BaseException as e:atomic(FIT/'failure.json',dict(type=type(e).__name__,message=str(e)[:1500],time=time.time()));raise
