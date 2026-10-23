"""Fit both actual G1 heads and publish one immutable encoder/readout version."""
import argparse,hashlib,json,os,subprocess,time,uuid
from pathlib import Path
import joblib,numpy as np
from search_g1.resource import evaluator_gpu_lock
from reproduction.common8.verify_data import digest
from reproduction.a800.recovery_binding import finite_fitted_values
from reproduction.common8.formal_contract import PROBE_ROUNDS, assert_formal_run

ap=argparse.ArgumentParser();ap.add_argument('--step',type=int,required=True);ap.add_argument('--model',required=True);args=ap.parse_args()
base=Path(os.environ.get('SEARCH_G1_BASE','/data/search-g1'));run=json.loads((base/'logs/run-manifest.json').read_text())
assert run['profile']=='common8' and run.get('completed_updates',0)==0
assert_formal_run(run)
assert args.step in PROBE_ROUNDS, 'Terminal200 does not refit an unused reward evaluator'
for name,sha in run['input_sha256'].items(): assert digest(base/'data'/name)==sha
assert run['probe_visible_device'] is not None, 'Probe placement has not passed resource admission'
if args.step:
 complete=json.loads((Path(args.model)/'complete.json').read_text());assert complete['run_id']==run['run_id'] and complete['step']==args.step
encoder=Path(args.model)
index=json.loads((encoder/'model.safetensors.index.json').read_text())
names=set(index['weight_map'].values())|{'model.safetensors.index.json','config.json','tokenizer.json','tokenizer_config.json'}
encoder_hashes={name:digest(encoder/name) for name in sorted(names)}
expected=complete['files_sha256'] if args.step else run['initial_model_sha256']
assert all(encoder_hashes[name]==value for name,value in expected.items() if name in encoder_hashes), 'Encoder changed before calibration'
round_dir=base/'artifacts'/('round_'+str(args.step)+'-attempt-'+uuid.uuid4().hex);round_dir.mkdir(parents=True,exist_ok=False)
from search_g1.service_ports import retriever_url
dump=round_dir/'dumps';heads=round_dir/'heads';env=dict(os.environ,RETRIEVER_URL=retriever_url(run),NO_PROXY='127.0.0.1,localhost')
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:env.pop(k,None)
python=str(base/'env/bin/python');repo=base/'repo'
with evaluator_gpu_lock():
 with (round_dir/'calibration.log').open('w') as log:
  subprocess.run([python,str(repo/'ts_probe_dump.py'),'--model',args.model,'--data',str(base/'data/calibration-300.parquet'),'--tag','nq-hotpot-common','--cap','300','--G','5','--layers','21,24,27','--gpu_mem','0.25','--rollout_max_new','500','--correction_feedback','--out',str(dump)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run([python,str(repo/'ts_probe_train.py'),'--dumps',str(dump),'--layers','21,24,27','--commit-position','Xa','--commit-label','flip','--save',str(heads),'--require-all-cv-folds'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 report=json.loads((heads/'calibration-report.json').read_text());proof=json.loads((dump/'calibration-label-proof.json').read_text())
 assert proof['question_count']==300 and report['unique_question_count']==300
 assert set(proof['question_ids'])==set(run['calibration_question_ids'])
 assert report['require_all_cv_folds'] is True and report['configured_cv_folds']==2
 assert proof['calibration_protocol']['rollout_max_new']==500
 assert proof['calibration_protocol']['correction_feedback'] is True
 assert proof['evidence_deletion']=='typed_retrieval_ranges_only_preserve_model_and_correction'
 assert np.isfinite(report['commit_cv_auc']) and np.isfinite(report['boundary_cv_auc'])
 assert min(report['commit_class_counts'])>0 and min(report['boundary_class_counts'])>0
 for name in ['commit_probe.pkl','boundary_probe.pkl']:
  obj=joblib.load(heads/name);assert list(obj['clf'].classes_)==[0,1] and np.isfinite(obj['auc'])
  finite_fitted_values(vars(obj['clf']))
 for L in (21,24,27):
  for file in dump.glob('*.npz'):
   with np.load(file,allow_pickle=False) as z:
    for key in ('Xa_'+str(L),'Xp_question_'+str(L)):assert np.isfinite(z[key]).all()
 assert all(digest(encoder/name)==sha for name,sha in encoder_hashes.items()), 'Encoder changed during calibration'
 version=run['run_id']+':'+str(args.step)+':'+uuid.uuid4().hex
 payload={'run_id':run['run_id'],'round':args.step,'version':version,'encoder':str(Path(args.model).resolve()),'commit':str(heads/'commit_probe.pkl'),'boundary':str(heads/'boundary_probe.pkl'),'encoder_sha256':encoder_hashes,'head_sha256':{n:hashlib.sha256((heads/n).read_bytes()).hexdigest() for n in ['commit_probe.pkl','boundary_probe.pkl']},'calibration_report':report,'label_proof':proof,'published_unix':time.time()}
 payload['calibration_report_sha256']=digest(heads/'calibration-report.json')
 payload['dump_sha256']={p.name:digest(p) for p in sorted(dump.iterdir()) if p.is_file()}
 with (round_dir/'version-manifest.json').open('x') as f:json.dump(payload,f,indent=2)
 live=base/'artifacts/live';live.mkdir(exist_ok=True);tmp=live/'manifest.json.tmp';tmp.write_text(json.dumps(payload,indent=2));os.replace(tmp,live/'manifest.json')
 print('SEARCH_G1_BOTH_PROBES_FITTED='+version,flush=True)
