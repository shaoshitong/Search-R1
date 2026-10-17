"""Actual reward-head refit at50/100/150; separate from the selection probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refit(config_path, outer, model):
    import joblib
    import numpy as np
    from reproduction.a800.recovery_binding import finite_fitted_values
    c=read(config_path);base=Path(c['base']);model=Path(model)
    if outer not in (50,100,150):raise ValueError('Only actually used reward refit rounds')
    actor=read(base/'actor/latest.json');joint=read(base/'latest-joint.json')
    if (actor['completed_outer']!=outer or joint['outer']!=outer or actor!=joint['actor']
            or Path(actor['hf_model_dir'])!=model or actor['run_id']!=c['run_id']):
        raise ValueError('Refit requires the actual completed own actor milestone')
    data=Path(c['grounding_calibration_questions'])
    if sha(data)!=c['grounding_calibration_sha256']:raise ValueError('Calibration split changed')
    completed=base/'grounding'/('round-%03d-complete.json'%outer)
    if completed.exists():
        live=read(completed);manifest=read(live['manifest'])
        if (live['run_id']!=c['run_id'] or live['round']!=outer
                or live['actor_version']!=actor['actor_version']
                or sha(live['manifest'])!=live['manifest_sha256']
                or Path(manifest['encoder'])!=model):
            raise ValueError('Existing reward fit belongs to a different actual milestone')
        for name,key in (('commit_probe.pkl','commit'),('boundary_probe.pkl','boundary')):
            if sha(manifest[key])!=manifest['head_sha256'][name]:
                raise ValueError('Existing fitted reward head changed')
        temporary=base/'grounding/live.json.tmp';temporary.write_text(json.dumps(live,indent=2)+'\n')
        os.replace(temporary,base/'grounding/live.json');print(json.dumps(live));return live
    # Failed fitting attempts remain evidence/cost; they never block a later
    # explicit recovery or masquerade as a completed fit.
    target=base/'grounding'/('round-%03d-attempt-%s'%(outer,uuid.uuid4().hex))
    target.mkdir(parents=True,exist_ok=False)
    intent=dict(run_id=c['run_id'],round=outer,actor_version=actor['actor_version'],
        actor_receipt_sha256=sha(base/'actor/latest.json'),model=str(model),
        calibration=str(data),calibration_sha256=sha(data),created_unix=time.time())
    (target/'intent.json').write_text(json.dumps(intent,indent=2)+'\n')
    env=dict(os.environ,RETRIEVER_URL=c['search_url'],NO_PROXY='127.0.0.1,localhost')
    for k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(k,None)
    repo=base/'repo';dump=target/'dumps';heads=target/'heads'
    with (target/'calibration.log').open('x')as stream:
        subprocess.run([sys.executable,str(repo/'ts_probe_dump.py'),'--model',str(model),
            '--data',str(data),'--tag','nq-hotpot-common','--cap','300','--G','5',
            '--layers','21,24,27','--gpu_mem','0.25','--rollout_max_new','500',
            '--correction_feedback','--out',str(dump)],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,str(repo/'ts_probe_train.py'),'--dumps',str(dump),
            '--layers','21,24,27','--commit-position','Xa','--commit-label','flip',
            '--save',str(heads),'--require-all-cv-folds'],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    report=read(heads/'calibration-report.json');proof=read(dump/'calibration-label-proof.json')
    if proof['question_count']!=300 or report['unique_question_count']!=300:
        raise ValueError('Reward grounding must refit the canonical300 questions')
    if not report['require_all_cv_folds']or report['configured_cv_folds']!=2:
        raise ValueError('Original reward fitting CV contract changed')
    initial=read(c['grounding_manifest'])
    if set(proof['question_ids'])!=set(initial['label_proof']['question_ids']):
        raise ValueError('Reward calibration question identities changed')
    for name in ('commit_probe.pkl','boundary_probe.pkl'):
        head=joblib.load(heads/name)
        if list(head['clf'].classes_)!=[0,1]or not np.isfinite(head['auc']):
            raise ValueError('Reward readout was not actually fitted with both classes')
        finite_fitted_values(vars(head['clf']))
    files={name:info['sha256']for name,info in actor['hf_files'].items()}
    if any(sha(model/name)!=digest for name,digest in files.items()):
        raise ValueError('Frozen calibration encoder differs from actual actor export')
    version=c['run_id']+':'+str(outer)+':'+uuid.uuid4().hex
    manifest=dict(run_id=c['run_id'],round=outer,version=version,encoder=str(model),
        commit=str(heads/'commit_probe.pkl'),boundary=str(heads/'boundary_probe.pkl'),
        encoder_sha256=files,head_sha256={name:sha(heads/name)for name in ('commit_probe.pkl','boundary_probe.pkl')},
        calibration_report=report,label_proof=proof,calibration_report_sha256=sha(heads/'calibration-report.json'),
        calibration_data_sha256=sha(data),published_unix=time.time(),actor_version=actor['actor_version'])
    path=target/'version-manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    live=dict(run_id=c['run_id'],round=outer,version=version,manifest=str(path),manifest_sha256=sha(path),
        actor_version=actor['actor_version'],actual_both_heads_fitted=True)
    complete_tmp=completed.with_name(completed.name+'.tmp')
    complete_tmp.write_text(json.dumps(live,indent=2)+'\n');os.replace(complete_tmp,completed)
    temporary=base/'grounding/live.json.tmp';temporary.write_text(json.dumps(live,indent=2)+'\n')
    os.replace(temporary,base/'grounding/live.json')
    print(json.dumps(live));return live


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--outer',required=True,type=int)
    p.add_argument('--model',required=True);a=p.parse_args();refit(a.config,a.outer,a.model)
