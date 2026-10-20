"""Immutable prepare/normal-one-GPU submit/status for fixed validation EMA3.

Code and small metadata only cross the control connection. Checkpoint bytes,
features and predictions remain remote. Explicit submit flag, durable intent,
held job identity before release, and no previous-job success dependency.
"""
import argparse
import base64
import importlib.util
import inspect
import json
from pathlib import Path
import shlex
import subprocess
from . import deploy_replay_downstream_comparison as transport
from .deploy_confirmation_materialization import make_bundle

PARENT=transport.PARENT
BASE=PARENT+'/frozen-validation-ema3-20261021'
VALIDATION=PARENT+'/algorithm-improvement-20261021/validation-g10-20261021/materialized/bank/complete.json'
BALANCED=PARENT+'/balanced-rank1-from-epoch1-prefix20-20261020'
EVENT=PARENT+'/algorithm-improvement-20261021/matched-loss128-v2/event'
PACKAGE='reproduction/operations/qwen7b_warmup/'
SEEDS=tuple(PACKAGE+name for name in ('evaluate_frozen_validation.py','model.py','packed_features.py',
    'probe_ablation_train.py','probe_ablation_loss.py','runtime.sh'))
MANIFEST='_frozen_validation_source_manifest.json'
JOB_NAME='q7b-fixed-validation-ema3'
CHUNK_BYTES=4096
BRANCH=transport.BRANCH
digest=transport.digest


def render_sbatch(base,code_root,config,config_sha256,job_name=JOB_NAME):
    for p in (base,code_root,config):
        if not Path(p).is_absolute() or any(c.isspace() for c in str(p)):
            raise ValueError('Absolute no-whitespace remote paths required')
    if len(config_sha256)!=64 or any(c not in '0123456789abcdef' for c in config_sha256):
        raise ValueError('Actual SHA-bound config required')
    if not job_name or any(not(c.isalnum() or c in '-_') for c in job_name):
        raise ValueError('Safe job name required')
    script='#!/usr/bin/env bash\n#SBATCH --job-name=@@FV_JOB@@\n#SBATCH --partition=acd_u\n#SBATCH --account=mou176\n#SBATCH --qos=normal\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=8\n#SBATCH --mem=64G\n#SBATCH --gres=gpu:1\n#SBATCH --time=00:30:00\n#SBATCH --output=@@FV_BASE_RAW@@/logs/evaluation-%j.log\nset -euo pipefail\nFV_BASE=@@FV_BASE_Q@@\nFV_CODE_ROOT=@@FV_CODE_Q@@\nFV_CONFIG=@@FV_CONFIG_Q@@\nexec 9>"$FV_BASE/evaluation-run.lock"\nflock -n 9\nexport QWEN7B_CODE_ROOT="$FV_CODE_ROOT"\nexport OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8\nexport PYTHONFAULTHANDLER=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\nexport FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8\nsrun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.evaluate_frozen_validation --config "$FV_CONFIG" --config-sha256 @@FV_CONFIG_SHA@@\n'
    for key,value in {'@@FV_JOB@@':job_name,'@@FV_BASE_RAW@@':str(base),
        '@@FV_BASE_Q@@':shlex.quote(str(base)),'@@FV_CODE_Q@@':shlex.quote(str(code_root)),
        '@@FV_CONFIG_Q@@':shlex.quote(str(config)),'@@FV_CONFIG_SHA@@':config_sha256}.items():
        script=script.replace(key,value)
    if '@@FV_' in script:raise ValueError('Unrendered unique placeholder')
    return script


def bundle_from(repo,commit,published=True):
    if published:
        paths=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=repo,text=True).splitlines()
        cache={}
        def load(name):
            if name not in cache:cache[name]=subprocess.check_output(['git','show',commit+':'+name],cwd=repo)
            return cache[name]
    else:
        paths=[str(p.relative_to(repo)) for directory in ('reproduction','search_g1','search_r1','verl')
               for p in (repo/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        load=lambda name:(repo/name).read_bytes()
    return make_bundle(paths,load,commit,seeds=SEEDS,resource_files=(),manifest_name=MANIFEST)


def control_source(values,block,extra=None):
    own=dict(values);own.update(extra or {})
    source='inputs='+repr(own)+'\n'+COMMON+'\n'
    if block==SUBMIT:source+=inspect.getsource(render_sbatch)+'\n'
    return source+block


def self_test(repo):
    bundle,report=bundle_from(repo,'a'*40,False)
    values=dict(base=BASE,parent=PARENT,validation=VALIDATION,balanced=BALANCED,event=EVENT,
        commit='a'*40,job_name=JOB_NAME,bundle_sha256=digest(bundle),bundle_bytes=len(bundle),
        chunks=(len(bundle)+CHUNK_BYTES-1)//CHUNK_BYTES,
        source_manifest_sha256=report['source_manifest_sha256'],
        renderer_sha256=digest(inspect.getsource(render_sbatch).encode()))
    sizes={}
    for name in ('CHECK','INIT','CHUNK','EXTRACT','PREPARE','SUBMIT','STATUS'):
        extra=dict(chunk_index=0,chunk_data=base64.b64encode(b'x'*CHUNK_BYTES).decode(),chunk_sha256='b'*64) if name=='CHUNK' else None
        source=control_source(values,globals()[name],extra);compile(source,'fixed-validation-'+name,'exec')
        sizes[name]=len(('python3 -c '+shlex.quote(source)).encode())
        if sizes[name]>=16000:raise AssertionError('Remote command >=16KB: '+name)
    script=render_sbatch(BASE,BASE+'/code-0123456789ab',BASE+'/inputs/config.json','a'*64)
    subprocess.run(['bash','-n'],input=script,text=True,check=True)
    setup='\n'.join(line for line in script.splitlines() if line.startswith(('FV_','export ')))
    subprocess.run(['bash','-euc',setup+'\ntest "$QWEN7B_CODE_ROOT" = '+shlex.quote(BASE+'/code-0123456789ab')+'\ntest "$FV_CONFIG" = '+shlex.quote(BASE+'/inputs/config.json')],check=True)
    if 'afterok' in script or '--dependency' in script:raise AssertionError('Unexpected dependency')
    result=dict(status='LOCAL_FIXED_VALIDATION_EVAL_DEPLOYMENT_SELFTEST_PASS',closure=report,
        bundle_bytes=len(bundle),remote_command_bytes=sizes,actual_bash_export_execution=True,
        remote_actions_executed=False,jobs_submitted=False)
    print(json.dumps(result));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path)
    p.add_argument('--action',choices=('prepare','submit','status'),default='prepare')
    p.add_argument('--remote-base',default=BASE);p.add_argument('--prepared-commit')
    p.add_argument('--submit-authorized',action='store_true');p.add_argument('--self-test',action='store_true')
    a=p.parse_args();repo=Path(__file__).resolve().parents[3]
    if a.self_test:self_test(repo);return
    if a.control_root is None:p.error('--control-root is required')
    if a.action=='submit' and not a.submit_authorized:p.error('Explicit --submit-authorized required')
    base=Path(a.remote_base)
    if base.parent!=Path(PARENT) or any(not(c.isalnum() or c in '-_.') for c in base.name):
        raise ValueError('Use a new isolated direct child of experiment root')
    commit=subprocess.check_output(['git','rev-parse',a.prepared_commit or 'HEAD'],cwd=repo,text=True).strip()
    if a.action=='prepare':
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
        published=subprocess.check_output(['git','ls-remote','origin',BRANCH],cwd=repo,text=True).split()[0]
        if head!=published:raise ValueError('Publish latest source before stage')
        subprocess.run(['git','merge-base','--is-ancestor',commit,head],cwd=repo,check=True)
        for f in (__file__,Path(__file__).with_name('deploy_confirmation_materialization.py'),Path(transport.__file__)):
            rel=str(Path(f).relative_to(repo))
            if Path(f).read_bytes()!=subprocess.check_output(['git','show',head+':'+rel],cwd=repo):
                raise ValueError('Unpublished helper: '+rel)
        bundle,report=bundle_from(repo,commit)
    else:bundle,report=b'',{}
    chunks=[bundle[i:i+CHUNK_BYTES] for i in range(0,len(bundle),CHUNK_BYTES)]
    values=dict(base=str(base),parent=PARENT,validation=VALIDATION,balanced=BALANCED,event=EVENT,
        commit=commit,job_name=JOB_NAME,bundle_sha256=digest(bundle),bundle_bytes=len(bundle),chunks=len(chunks),
        source_manifest_sha256=report.get('source_manifest_sha256'),renderer_sha256=digest(inspect.getsource(render_sbatch).encode()))
    adapter=a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py'
    spec=importlib.util.spec_from_file_location('frozen_eval_private',adapter);control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    def call(block,extra=None):
        source=control_source(values,block,extra);command='python3 -c '+shlex.quote(source)
        if len(command.encode())>=16000:raise ValueError('SSH code command >=16KB')
        r=control.run(command,account='mou176',source_address=address,timeout=55)
        rows=[json.loads(line) for line in r.stdout.decode(errors='replace').splitlines() if line.strip().startswith('{')]
        if r.returncode:
            print(r.stderr.decode(errors='replace')[-2500:]);raise SystemExit(r.returncode)
        if not rows:raise RuntimeError('No actual structured response; reconcile before retry')
        return rows[-1]
    if a.action=='status':print(json.dumps(call(STATUS)));return
    if a.action=='submit':print(json.dumps(call(SUBMIT)));return
    print(json.dumps(call(CHECK)));print(json.dumps(call(INIT)))
    for i,raw in enumerate(chunks):
        result=call(CHUNK,dict(chunk_index=i,chunk_data=base64.b64encode(raw).decode(),chunk_sha256=digest(raw)))
        if i%16==0 or i+1==len(chunks):print(json.dumps(dict(status='CODE_UPLOAD_PROGRESS',verified=i+1,chunks=len(chunks),last=result)))
    print(json.dumps(call(EXTRACT)));print(json.dumps(call(PREPARE)))


COMMON=transport.COMMON.replace('_replay_comparison_source_manifest.json',MANIFEST)
# Transport-only functions (metadata/code bytes), no prior GPU entrypoint.
COMMON=COMMON[:COMMON.index('def checked_qualification():')]
COMMON+="\ndef validation():\n p=Path(inputs['validation']);v=read(p)\n assert v['status']=='VALIDATION_ACTUAL_TWO_FRESH_G5_G10_SHA_OWN_EM_VERIFIED'\n assert v['questions']==2048 and v['trajectories']==20480 and v['role']=='validation'\n assert v['untouched_final'] is False and v['fitting_on_validation_allowed'] is False and v['tuning_allowed'] is True\n assert v['fixed_G5_was_chosen_before_own_EM'] is True and v['synthetic_EM'] is False\n assert v['model_revision']=='a09a35458c702b33eeacc393d103063234e8bc28'\n for d in v['outputs'].values():assert fp(d['path'])==d\n assert fp(v['plan_freeze']['path'])==v['plan_freeze']\n return fp(p),v\n"
CHECK="f,v=validation();print(json.dumps(dict(status='ACCEPTED_FIXED_G5_VALIDATION_2048_READY',validation=f,questions=2048,trajectories=10240,submitted=False)))"
INIT="b.mkdir(parents=True,exist_ok=True);guard=lock();unsubmitted()\nfor name in ('logs','inputs'):(b/name).mkdir(exist_ok=True)\nexpected={k:inputs[k] for k in ('commit','bundle_sha256','bundle_bytes','chunks','source_manifest_sha256','validation','balanced','event','job_name','renderer_sha256')}\np=b/'upload-controller.json'\nif p.exists():assert read(p)==expected,'Conflicting isolated deployment/version'\nelse:atomic(p,expected)\nuploads.mkdir(exist_ok=True)\nprint(json.dumps(dict(status='FROZEN_EVAL_CODE_UPLOAD_BOUND',base=str(b),commit=commit,chunks=inputs['chunks'])))"
CHUNK=transport.CHUNK
EXTRACT=transport.EXTRACT.replace('_replay_comparison_source_manifest.json',MANIFEST)
PREPARE="guard=lock();unsubmitted();publication=verify_code();validation_ref,v=validation();descriptors=[]\nfor name,epoch,steps,root,file in [('E2',2,11782,inputs['balanced'],'probe/epoch-0002.json'),('E4',4,14418,inputs['balanced'],'probe/epoch-0004.json'),('event128',None,10592,inputs['event'],'latest.json')]:\n root=Path(root);p=root/file;r=read(p);c=read(root/'config.json')\n assert r['steps']==steps and Path(r['path']).is_file() and Path(r['path']).stat().st_size==r['bytes']>0\n if name=='event128':assert r['cursor']==128 and c['update_budget']==128 and c['recipe']=={'id':'event','loss_kwargs':{'trajectory_weight':1.0,'three_event_weight':1.0}}\n else:\n  assert r['epoch']==epoch and r['cursor']==0 and r['completed_epoch'] is True and r['independent_cpu_reload'] is True\n  assert c['prefix_tokens']==20 and c['question_max_tokens']==273 and c['native_head_layer']==27 and c['fit_ema']==.99\n if name=='E2':assert r['sha256']=='6a970815ff3aec26eedf4d6a65589bbebac3c4ccbdf8eefdf859a83e2a11bd2d'\n sealed=b/'inputs'/('source-'+name+'-sealed-receipt.json')\n if sealed.exists():assert sealed.read_bytes()==p.read_bytes(),'Sealed source receipt changed'\n else:\n  with sealed.open('xb') as stream:stream.write(p.read_bytes());stream.flush();os.fsync(stream.fileno())\n descriptor={k:r[k] for k in ('path','bytes','sha256','steps')}\n descriptor.update(name=name,epoch=epoch,source_receipt=fp(sealed),original_source_receipt=fp(p),source_config=fp(root/'config.json'),model_dir=c['model_dir'])\n descriptors.append(descriptor)\nassert len(set(d['model_dir'] for d in descriptors))==1\nmodel=Path(descriptors[0]['model_dir']);modelconfig=fp(model/'config.json');mc=read(model/'config.json')\nassert mc['num_hidden_layers']==28 and mc['hidden_size']==3584\nmanifest=read(code/'_frozen_validation_source_manifest.json')\nconfig=dict(base=str(b),source_commit=commit,validation_complete=validation_ref,checkpoints=descriptors,\n model_dir=str(model),model_config=modelconfig,micro_groups=8,evaluation_role='viewed_recipe_selection_validation_not_untouched_final',\n probe_optimizer_updates_authorized=False,actor_optimizer_updates_authorized=False,\n code_hashes={name:item['sha256'] for name,item in manifest['files'].items() if name.endswith('.py')})\np=b/'inputs/config.json'\nif p.exists():assert read(p)==config,'Immutable evaluator source/data binding changed'\nelse:atomic(p,config)\nprepared=dict(status='FIXED_VALIDATION_EMA3_GPU_EVALUATION_PREPARED',commit=commit,code_root=str(code),config=fp(p),\n questions=2048,trajectories=10240,job_name=inputs['job_name'],renderer_sha256=inputs['renderer_sha256'],\n checkpoint_full_SHA_deferred_to_allocated_GPU=True,login_model_or_tensor_loaded=False)\np=b/'preparation.json'\nif p.exists():assert read(p)==prepared\nelse:atomic(p,prepared)\nprint(json.dumps(prepared))"
SUBMIT="guard=lock();unsubmitted();verify_code();p=read(b/'preparation.json')\nassert p['status']=='FIXED_VALIDATION_EMA3_GPU_EVALUATION_PREPARED' and p['commit']==commit\nassert p['job_name']==inputs['job_name'] and p['renderer_sha256']==inputs['renderer_sha256']\nassert fp(p['config']['path'])==p['config']\nassert not command(['squeue','-h','-u','mou176','-n',JOB_NAME,'-o','%i']),'Same-name active job exists'\nassert not (b/'evaluation-output').exists(),'Preserve previous GPU output; do not resubmit'\nwith (b/'evaluation-run.lock').open('a') as h:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)\nscript=render_sbatch(str(b),str(code),p['config']['path'],p['config']['sha256'],JOB_NAME)\ns=b/'evaluation.sbatch'\nif s.exists():assert s.read_text()==script\nelse:s.write_text(script)\ncommand(['bash','-n',str(s)])\nintent=dict(status='FROZEN_EVAL_SUBMIT_INTENT',commit=commit,script=fp(s),job_name=JOB_NAME,\n requested_resources=dict(partition='acd_u',qos='normal',gpus=1,cpus=8,memory='64G',time='00:30:00'),dependency=None,epoch_time=time.time())\natomic(b/'submission-intent.json',intent)\ntry:r=subprocess.run(['sbatch','--hold','--parsable',str(s)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)\nexcept subprocess.TimeoutExpired:\n intent.update(status='SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPEAT');atomic(b/'submission-intent.json',intent);raise\nif r.returncode:\n intent.update(status='SBATCH_REJECTED',stderr=r.stderr[-2000:]);atomic(b/'submission-intent.json',intent);raise RuntimeError(r.stderr[-2000:])\njob=r.stdout.strip().split(';')[0]\nif not job.isdigit():\n intent.update(status='SUBMIT_RESPONSE_UNKNOWN_DO_NOT_REPEAT');atomic(b/'submission-intent.json',intent);raise RuntimeError('Unknown submit response')\nwith (b/'job.id').open('x') as h:h.write(job+'\\n');h.flush();os.fsync(h.fileno())\nresult=dict(intent,status='FROZEN_EVAL_JOB_HELD',job_id=job,code_root=str(code),config=p['config']);atomic(b/'submission.json',result)\nstate=command(['squeue','-h','-j',job,'-o','%T|%R']);assert state.startswith('PENDING|') and 'JobHeldUser' in state,state\ncommand(['scontrol','release',job]);result.update(status='FROZEN_EVAL_JOB_RELEASED',queue_observation=command(['squeue','-h','-j',job,'-o','%i|%u|%j|%T|%P|%R']),epoch_time=time.time());atomic(b/'submission.json',result)\nprint(json.dumps(result))"
STATUS="result=dict(base=str(b),observed_epoch=time.time(),exists=b.exists())\nfor name in ('source-publication.json','preparation.json','submission-intent.json','submission.json'):\n p=b/name\n if p.exists():\n  v=read(p);result[name]={k:v[k] for k in ('status','commit','job_id','code_root','questions','trajectories','requested_resources') if k in v}\nif (b/'job.id').exists():\n job=(b/'job.id').read_text().strip();assert job.isdigit();result['job_id']=job\n result['squeue']=command(['squeue','-h','-j',job,'-o','%i|%u|%j|%T|%P|%R'])\n result['sacct']=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,JobName,State,ExitCode,Elapsed'])\n result['start_estimate']=command(['squeue','--start','-h','-j',job,'-o','%i|%S|%R'])\nfor name in ('progress.json','complete.json'):\n p=b/'evaluation-output'/name\n if p.exists():\n  v=read(p)\n  result[name]=v if name!='complete.json' else dict(status=v['status'],questions=v['questions'],trajectories=v['trajectories'],metrics={k:r['metrics'] for k,r in v['checkpoints'].items()})\nif (b/'failure.json').exists():result['failure']=read(b/'failure.json')\nprint(json.dumps(result))"

if __name__=='__main__':main()
