"""Independent label-blind own-G5 class-weighted E1 continuation; Root alone submits GPU."""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/';BRANCH='codex/mixed-direct-e1-uniform-g5-weighted-20261022';A='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021';MAIN=A+'/joint-mixed-direct-official-169028-b128-v1';BASE=A+'/mixed-direct-e1-uniform-g5-weighted-20261022-v1';JOB='q7b-e1-uniform-g5'
SOURCE_CONFIG=dict(path=MAIN+'/config.json',bytes=19415,sha256='e5f0027dfcff4b6db442b38e9b2452e86cc2c9adcabd97a194971dc4d8f8d837');SOURCE_PUBLICATION=dict(path=MAIN+'/source-publication.json',bytes=8208,sha256='1492aefafa3a5bce1b5fe9246ee90c97b24b093892c41b1d79377cf98bdf098a');SOURCE_E1=dict(path=MAIN+'/evaluations/epoch-01/complete.json',bytes=27129,sha256='3d832d00b2d40eaf5ffad4568c1785c84270b0a81e033c57e6c7030a93cd0de8')
WORKER=P+'joint_mixed_direct_labelblind.py';ORIGINS={'reproduction/operations/qwen7b_warmup/mixed_group_unique_checkpoint.py': '4ffae871859ab21c9a625b1f5b54ef6679676f07', 'reproduction/operations/qwen7b_warmup/unique_fixed_train_evaluation.py': 'e7bfaef501dac0fbfb7258adb688be79656742a4', 'reproduction/operations/qwen7b_warmup/unique_selected_feature_reader.py': '333033b1d134d8167287bd72446a22b01f0b0317', 'reproduction/operations/qwen7b_warmup/labelblind_question_g5_sampler.py': '5dbc3683fe195241caa3057903332484575f5670', 'reproduction/operations/qwen7b_warmup/direct_update_labelblind.py': '5dbc3683fe195241caa3057903332484575f5670', 'reproduction/operations/qwen7b_warmup/joint_mixed_direct_labelblind.py': '5dbc3683fe195241caa3057903332484575f5670'};NEW=list(ORIGINS);OUT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-e1-uniform-g5-weighted'
SOURCE_E1_PUBLICATION=dict(SOURCE_PUBLICATION)
SOURCE_PUBLICATION=dict(path=A+'/mixed-direct-e1-lr-headonly-20261022-v1/source-publication.json',bytes=8762,sha256='c1826a6eabb497178feeffd4701c586b30ae61b5ff4450d46db87f86a300ab52')
COMMON='''import pathlib,json,hashlib,os,fcntl,subprocess,time,re,base64,stat
b=pathlib.Path(v['base']);b.mkdir(exist_ok=True);code=b/('code-'+v['archive_sha256'][:12]);code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);assert p.stat().st_size<4*1024**2 and p.suffix!='.pt';raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(ref):assert proof(ref['path'])==ref;return read(ref['path'])
def once(p,x):
 raw=(json.dumps(x,sort_keys=True,separators=(',',':'))+chr(10)).encode()
 if p.exists():assert p.read_bytes()==raw
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);assert r.returncode==0,r.stderr[-1200:];return r.stdout.strip()
def active():return command(['squeue','-h','-u','mou176','-n',v['job'],'-o','%i|%T|%R'])
def publication():
 pub=read(b/'source-publication.json');assert pub['files']==v['manifest']['files']and pub['file_origins']==v['manifest']['origins']and pub['code_root']==str(code)
 for name,item in pub['files'].items():assert proof(code/name)==dict(path=str(code/name),**item)
 return pub
'''
PART='''raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha'];p=code/(v['upload']+'-%04d.part'%v['part'])
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(accepted=True,part=v['part'])))
'''
INSTALL='''old=checked(v['source_publication']);assert len(old['files'])==30
for name,item in v['manifest']['files'].items():
 if name in v['new']:
  x=v['parts'][name];raw=b''.join((code/(x['upload']+'-%04d.part'%i)).read_bytes()for i in range(x['count']))
 else:
  assert old['files'][name]==item and old['file_origins'][name]==v['manifest']['origins'][name];raw=(pathlib.Path(old['code_root'])/name).read_bytes()
 assert len(raw)==item['bytes']and hashlib.sha256(raw).hexdigest()==item['sha256'];p=code/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');script='import pathlib;root=pathlib.Path('+repr(str(code))+');names='+repr(list(v['manifest']['files']))+';[compile((root/n).read_bytes(),n,"exec")for n in names if n.endswith(".py")]';r=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-c',script],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=35);assert r.returncode==0,r.stderr[-1000:]
once(b/'source-publication.json',dict(archive_sha256=v['archive_sha256'],code_root=str(code),files=v['manifest']['files'],file_origins=v['manifest']['origins'],source_publication=v['source_publication'],original30_sources_unchanged=True,controller=v['controller']));publication();print(json.dumps(dict(status='E1_LABELBLIND_G5_CLASSWEIGHT_SOURCE_STAGED',files=len(v['manifest']['files']),publication=proof(b/'source-publication.json'),GPU_submitted=False)))
'''
PREPARE='''publication();assert not active()and not(b/'submission-intent.json').exists();terminal=command(['sacct','-X','-n','-P','-j','746266','--format=JobIDRaw,State,ExitCode']).splitlines();assert any(r.split('|')[:3]==['746266','COMPLETED','0:0']for r in terminal)
c=checked(v['source_config']);ep=checked(v['source_e1']);cp=ep['checkpoint'];oldpub=checked(v['source_e1_publication']);assert c['code_hashes']=={n:i['sha256']for n,i in oldpub['files'].items()};assert cp['cursor']==cp['native_steps']==cp['head_steps']==2577 and cp['bytes']==3781073882 and cp['sha256']=='c0a1c7dc3663d4216966a3440c2103348bcc52db3cd04c2f5c59a784a2792917'and cp['pinned']and cp['full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed']and cp['config_sha256']==v['source_config']['sha256'];path=pathlib.Path(cp['path']);s=path.stat();assert path==pathlib.Path(v['main'])/'checkpoints/cursor-002577.pt'and not path.is_symlink()and stat.S_ISREG(s.st_mode)and s.st_size==cp['bytes']and s.st_uid==os.getuid()
copy=b/'source-config.json';raw=pathlib.Path(v['source_config']['path']).read_bytes()
if copy.exists():assert copy.read_bytes()==raw
else:copy.write_bytes(raw)
c['base']=str(b);c['mode']='e1_labelblind_g5_each_epoch';c['architecture']['micro_questions']=128;c['batch_questions']=128;c['architecture']['batch_questions']=128;c['sampler_policy']='labelblind_question_once_G5_epoch_global_classweighted_BCE_v1';c['unique_sampling_seed']=2026102204;c['phase3_head_lr']=5e-7;c['epoch_unique_questions']=169028;c.pop('epoch_class_counts',None);c['loss_class_weighting']='N/(2*actual_epoch_class_count), weighted_BCE_sum/actual_batch_n';c['G5_sampling_label_blind']=True;c['tail_batch_questions']=68;c['views_per_question']=1;c['epochs']=3;c['epoch_end_cursors']=[3898,5219];c['update_budget']=5219;c['save_every']=128;c['lr']=5e-7;c['ema_decay']=.99;c['code_hashes']={n:i['sha256']for n,i in v['manifest']['files'].items()};c['source_e1']=dict(source_config=proof(copy),source_main_config=v['source_config'],source_checkpoint=cp,source_evaluation_complete=v['source_e1'],reader_integrity=proof(pathlib.Path(v['main'])/'reader-integrity.json'));c['phases']=[dict(name='phase2_joint_low_lr',start_cursor=2577,end_cursor=3898,native_lr=5e-7,head_lr=5e-7,native_frozen=False),dict(name='phase3_head_only',start_cursor=3898,end_cursor=5219,native_lr=5e-7,head_lr=5e-7,native_frozen=True)];once(b/'config.json',c);(b/'logs').mkdir(exist_ok=True);entry=b/'training.sbatch';text=v['entry'].replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code))
if entry.exists():assert entry.read_text()==text
else:entry.write_text(text)
subprocess.check_call(['bash','-n',str(entry)]);ready=dict(status='E1_LABELBLIND_G5_CLASSWEIGHT_READY_NOT_SUBMITTED',config=proof(b/'config.json'),entry=proof(entry),publication=proof(b/'source-publication.json'),source_checkpoint=cp,source_checkpoint_stat=dict(inode=s.st_ino,uid=s.st_uid,bytes=s.st_size),source_config=proof(copy),source_e1=v['source_e1'],reader_integrity=c['source_e1']['reader_integrity'],remaining_updates=2642,checkpoint_payload_read_on_CPU=False,GPU_fresh_full_CP_SHA_required=True,source_model_two_Adam_RNG_preserved=True,source_sampler_archived_new_unique_policy=True,GPU_submitted=False);once(b/'preparation.json',ready);print(json.dumps(ready))
'''
ENTRY='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-e1-uniform-g5
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --signal=USR1@900
#SBATCH --output=@@BASE@@/logs/train-%j.log
#SBATCH --chdir=@@BASE@@
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.joint_mixed_direct_labelblind --base @@BASE@@ --action train
'''
SUBMIT='''publication();guard=(b/'submit-control.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not active()and not(b/'submission-intent.json').exists(),'Actual same-name job/unknown intent: reconcile, never submit twice';ready=read(b/'preparation.json')
for key in('config','entry','publication','source_config','source_e1','reader_integrity'):assert proof(ready[key]['path'])==ready[key]
lock=(b/'training.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close();assert not any((b/n).exists()for n in('latest.json','progress.json','failure.json','complete.json'));once(b/'submission-intent.json',dict(ready=ready,controller=v['controller'],epoch=time.time()));r=subprocess.run(['sbatch','--hold','--parsable','--account=mou176',str(b/'training.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
if r.returncode or not re.fullmatch(r'[0-9]+(?:;[^ ]+)?',r.stdout.strip()):raise RuntimeError('Unknown/rejected submit; preserve intent and inspect same name: '+r.stdout[-200:]+r.stderr[-1000:])
job=r.stdout.strip().split(';')[0];identity=command(['scontrol','show','job','-o',job])
for item in('JobName='+v['job'],'Account=mou176','QOS=normal','Partition=acd_u','NumCPUs=8','MinMemoryNode=128G','Dependency=(null)','Command='+str(b/'training.sbatch'),'WorkDir='+str(b)):assert item in identity,item
assert 'TimeLimit=1-00:00:00'in identity or'TimeLimit=24:00:00'in identity;assert'gres/gpu=1'in identity or'TresPerNode=gres/gpu:1'in identity;once(b/'submission-identity.json',dict(job_id=job,job=identity,config=ready['config']));command(['scontrol','release',job]);result=dict(status='E1_LABELBLIND_G5_CLASSWEIGHT_JOB_RELEASED_NOT_PROGRESS_VERIFIED',job_id=job,config=ready['config'],identity=proof(b/'submission-identity.json'));once(b/'submission-result.json',result);print(json.dumps(result))
'''
STATUS='''result=dict(epoch=time.time(),active_same_name_jobs=active())
for name in('preparation.json','submission-result.json','controller.json','progress.json','latest.json','failure.json','complete.json'):
 if(b/name).exists():result[name[:-5]]=read(b/name)
print(json.dumps(result))
'''
def git(c,n):return subprocess.check_output(['git','show',c+':'+n],cwd=R)
def bundle():
 assert all(len(origin)==40 for origin in ORIGINS.values()),'Published per-file unique worker/updater/sampler/checkpoint origins required';old=P+'deploy_mixed_direct_e1_lr_headonly.py';ns=dict(__name__='accepted_e1_runtime_bundle',__file__=str(R/old));exec(compile(git('8cf67d463421c889e8c89b01ee335c08bb1ce4de',old),old,'exec'),ns);contents,previous,oldsha=ns['bundle']();assert len(previous['files'])==30;origins=dict(previous['origins'])
 for name in NEW:contents[name]=git(ORIGINS[name],name);origins[name]=ORIGINS[name]
 manifest=dict(files={n:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())for n,raw in contents.items()},origins=origins);buf=io.BytesIO()
 with gzip.GzipFile(fileobj=buf,mode='wb',mtime=0,filename='')as gz:
  with tarfile.open(fileobj=gz,mode='w')as tar:
   for name,raw in sorted(contents.items()):t=tarfile.TarInfo(name);t.size=len(raw);t.mode=0o444;t.mtime=0;tar.addfile(t,io.BytesIO(raw))
 return contents,manifest,hashlib.sha256(buf.getvalue()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','prepare','submit','status'),required=True);p.add_argument('--submit-authorized',action='store_true');a=p.parse_args()
 if a.action=='submit'and not a.submit_authorized:p.error('Root single-submit authorization required')
 contents,manifest,sha=bundle();rel=str(Path(__file__).resolve().relative_to(R));commit=subprocess.check_output(['git','log','-1','--format=%H','refs/heads/'+BRANCH,'--',rel],cwd=R).decode().strip();assert git(commit,rel)==Path(__file__).read_bytes();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R).decode().split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',commit,remote],cwd=R);v=dict(base=BASE,main=MAIN,job=JOB,manifest=manifest,archive_sha256=sha,new=NEW,source_config=SOURCE_CONFIG,source_publication=SOURCE_PUBLICATION,source_e1_publication=SOURCE_E1_PUBLICATION,source_e1=SOURCE_E1,controller=dict(commit=commit,path=rel,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 spec=importlib.util.spec_from_file_location('private_e1_phase_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 def call(body,**kw):
  text='v='+repr(dict(v,**kw))+'\n'+COMMON+body;compile(text,'e1_phase_controller','exec');packed=base64.b64encode(zlib.compress(text.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(cmd.encode())<16000;r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError(r.stdout[-2000:]+r.stderr[-2000:])
  return json.loads(r.stdout.splitlines()[-1])
 if a.action=='stage':
  prior=call("print(json.dumps(dict(exists=(b/'source-publication.json').exists())))")
  if prior['exists']:result=call("publication();print(json.dumps(dict(status='EXISTING_E1_PHASE_SOURCE_VERIFIED',publication=proof(b/'source-publication.json'))))")
  else:
   parts={}
   for name in NEW:
    raw=contents[name];upload='new-'+hashlib.sha256(name.encode()).hexdigest()[:8];chunks=[raw[i:i+2048]for i in range(0,len(raw),2048)];parts[name]=dict(upload=upload,count=len(chunks))
    for i,data in enumerate(chunks):call(PART,upload=upload,part=i,data=base64.b64encode(data).decode(),sha=hashlib.sha256(data).hexdigest())
   result=call(INSTALL,parts=parts)
 else:result=call(dict(prepare=PREPARE,submit=SUBMIT,status=STATUS)[a.action],entry=ENTRY)
 OUT.mkdir(parents=True,exist_ok=True);(OUT/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
for body in(COMMON,PART,INSTALL,PREPARE,SUBMIT,STATUS):compile(body,'e1_controller_static','exec')
if __name__=='__main__':main()
