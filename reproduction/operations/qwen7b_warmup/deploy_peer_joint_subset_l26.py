"""Published accepted-subset adapter; at most two normal2H100 pair writers."""
import argparse,base64,gzip,hashlib,importlib.util,inspect,io,json,shlex,subprocess,tarfile
from pathlib import Path
from . import deploy_peer_audited100_l26 as old
from . import deploy_peer_newseed_replay as legacy
from . import deploy_peer_batched_replay as batch
P=legacy.PACKAGE

def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--branch',required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--action',choices=('stage','prepare-start','finalize','submit','status'),required=True);p.add_argument('--pair',type=int);p.add_argument('--submit-authorized',action='store_true');a=p.parse_args();repo=Path(__file__).resolve().parents[3];name=str(a.config.resolve().relative_to(repo));raw=subprocess.check_output(['git','show',a.commit+':'+name],cwd=repo);assert raw==a.config.read_bytes();c=json.loads(raw);TARGET=Path(c['target']);INGEST=Path(c['raw_complete']['path']).parent
 if a.action=='submit'and(a.pair not in(0,1)or not a.submit_authorized):p.error('Explicit owned pair and submit authorization required')
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+a.branch],cwd=repo,text=True).split()[0]
 if remote!=a.commit and subprocess.run(['git','merge-base','--is-ancestor',a.commit,remote],cwd=repo).returncode:raise ValueError('Pinned source not published')
 tag=c['tag'].lower();assert all(v in 'abcdefghijklmnopqrstuvwxyz0123456789-'for v in tag)
 render_source=inspect.getsource(old.render_sbatch).replace('q7b-delta100-L26-','q7b-'+tag+'-L26-').replace('peer_audited100_l26_replay','peer_joint_subset_l26')
 files=tuple(dict.fromkeys(old.FILES+(P+'peer_joint_subset_l26.py',P+'deploy_peer_joint_subset_l26.py',name)));config_sha=hashlib.sha256(raw).hexdigest()
 inputs=dict(base=str(TARGET),target=str(TARGET),ingest=str(INGEST),receiver=str(legacy.RECEIVER),identity=str(TARGET/'identity-plan.json'),golden=str(batch.GOLDEN),commit=a.commit,pair=a.pair,config_name=name,config_sha256=config_sha,bundle_sha256='b'*64,bundle_bytes=1,chunks=1,files={})
 copy=old.COPY_CONFIG.replace('AUDITED100','JOINT_SUBSET')
 start=old.START.replace('peer_audited100_l26_replay','peer_joint_subset_l26')
 finalize=r'''
guard=lock();pub,root=publication();c=read(b/'cohort-config.json');done=read(b/'preparation.json');assert done['source_commit']==pub['commit']and done['questions']==c['questions']and done['persist_layers']==[26]
qualified={read(v['path'])['shard']:v for v in done['selections']};variants={}
for pair in range(2):
 members=[2*pair,2*pair+1]
 if not all(i in qualified for i in members):continue
 own=[qualified[i]for i in members]
 for item in own:assert fp(item['path'])==item and read(item['path'])['questions']>0
 base=b/('pair-%02d'%pair);(base/'logs').mkdir(parents=True,exist_ok=True);p=base/'capture.sbatch';text=render_sbatch(pair,str(base),str(root),fp(b/'preparation.json')['sha256'],inputs['config_sha256'])
 if p.exists():assert p.read_text()==text
 else:p.write_text(text)
 command(['bash','-n',str(p)]);variants[str(pair)]=dict(base=str(base),job_name='q7b-'+c['tag'].lower()+'-L26-%02d'%pair,selections=own,shards=members,sbatch=fp(p))
value=dict(status='ACCEPTED_SUBSET_NORMAL_TWO2GPU_SCRIPTS_READY',variants=variants,blocked_shards=done['blocked_shards'],preparation=fp(b/'preparation.json'),no_dependency=True);atomic(b/'submission-plan.json',value);print(json.dumps(value))
'''
 blocks=dict(init=legacy.INIT,chunk=legacy.CHUNK,extract=legacy.EXTRACT,copy=copy,start=start,finalize=finalize,submit=old.SUBMIT.replace('AUDITED100_AFTER366','JOINT_SUBSET'),status=old.STATUS.replace('AUDITED100_AFTER366','JOINT_SUBSET'))
 def source(action,extra=None):
  own=dict(inputs);own.update(extra or {})
  if action not in('init','extract'):own.pop('files',None)
  definitions='Path=pathlib.Path\nTARGET=Path('+repr(str(TARGET))+')\n'+render_source if action=='finalize'else''
  s='inputs='+repr(own)+'\n'+legacy.COMMON+'\n'+definitions+'\n'+blocks[action];compile(s,action,'exec');assert len(('python3 -c '+shlex.quote(s)).encode())<16000;return s
 spec=importlib.util.spec_from_file_location('joint_subset_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(action,extra=None):
  r=ctl.run('python3 -c '+shlex.quote(source(action,extra)),account='mou176',source_address=ip,timeout=55);rows=[json.loads(s)for s in r.stdout.decode(errors='replace').splitlines()if s.startswith('{')]
  if r.returncode or not rows:raise RuntimeError(r.stderr.decode(errors='replace')[-1800:]+r.stdout.decode(errors='replace')[-1800:])
  return rows[-1]
 if a.action=='stage':
  out=io.BytesIO();bound={}
  with gzip.GzipFile(fileobj=out,mode='wb',mtime=0)as gz:
   with tarfile.open(fileobj=gz,mode='w')as tar:
    for f in files:
     v=subprocess.check_output(['git','show',a.commit+':'+f],cwd=repo);assert v==(repo/f).read_bytes();bound[f]=dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest());item=tarfile.TarInfo(f);item.size=len(v);item.mode=0o444;tar.addfile(item,io.BytesIO(v))
  blob=out.getvalue();inputs.update(files=bound,bundle_sha256=hashlib.sha256(blob).hexdigest(),bundle_bytes=len(blob),chunks=(len(blob)+4095)//4096);call('init')
  for i,startat in enumerate(range(0,len(blob),4096)):
   part=blob[startat:startat+4096];call('chunk',dict(chunk_index=i,chunk_data=base64.b64encode(part).decode(),chunk_sha256=hashlib.sha256(part).hexdigest()))
  call('extract');value=call('copy')
 else:value=call('start'if a.action=='prepare-start'else a.action)
 d=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('joint-feature-'+tag);d.mkdir(parents=True,exist_ok=True);(d/(a.action+('-'+str(a.pair)if a.pair is not None else'')+'.json')).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');print(json.dumps(value),flush=True)
if __name__=='__main__':main()
