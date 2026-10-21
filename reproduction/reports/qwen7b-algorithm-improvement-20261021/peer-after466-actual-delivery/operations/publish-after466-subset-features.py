"""Local code-only immutable configuration publication after actual raw acceptance."""
import argparse,json,os,pathlib,subprocess,tempfile,hashlib,sys
C=pathlib.Path(__file__).resolve().parents[3]
R=C/'work/qwen7b-probe-warmup-20261019'
P='reproduction/operations/qwen7b_warmup/'
Q='reproduction/reports/qwen7b-algorithm-improvement-20261021/'
A='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/'
p=argparse.ArgumentParser();p.add_argument('number',choices=('243','262'));p.add_argument('--refs',type=pathlib.Path,required=True);a=p.parse_args();n=a.number
snap={'243':'audited243-after466-20261021-45227948','262':'audited262-after709-20261021-d3c45767'}[n]
v=json.loads(a.refs.read_text())['records'][snap];assert isinstance(v['complete'],dict)
expected={'243':31010,'262':33536}[n];assert(v['questions'],v['trajectories'])==(expected,expected*10)
audit=C/'outputs/qwen7b-warmup-20261019/peer-frozen-delta-transfer'/snap/'final-audit-run.json';av=json.loads(audit.read_text());assert av['exit_code']==0
cfgname=Q+'joint124194-feature-subsets/audited'+n+'-config.json'
cfg=dict(status='FROZEN_ACCEPTED_TRAIN_SUBSET_L26_CONFIG',tag='audited'+n,target=A+'peer-joint124194-audited'+n+'-L26-v1',questions=expected,trajectories=expected*10,raw_complete=v['complete'],raw_groups=v['outputs']['groups'],source_events=v['outputs']['source-events'],independent_raw_acceptance=av['audit'],prefix_tokens=20,question_max_tokens=273,batch_size=32,max_context_tokens=16384,compute_layers=[20,23,26],persist_layers=[26],minimum_free_bytes=850*1024**3,shared_free_is_not_user_quota=True,user_quota='unknown; actual shared write probe passed',future_active_training_checkpoint_reserve_bytes=400*1024**3,canonical_train_text_repeats_retained=True,former_internal5120_no_longer_unseen_if_fitted=True,true_evaluation_whole_question_exclusions_retained=True,actor_updates=0,probe_updates=0,new_rollouts_generated=0,expected_domains={'243':{'hotpotqa':28706,'nq':2304},'262':{'hotpotqa':629,'nq':32907}}[n])
out=R/cfgname;data=(json.dumps(cfg,sort_keys=True,indent=2)+'\n').encode()
if out.exists():assert out.read_bytes()==data
else:out.write_bytes(data)
os.chdir(R);sys.path.insert(0,str(R))
from reproduction.operations.qwen7b_warmup.deploy_peer_audited100_l26 import FILES
extra=tuple(P+x for x in ('peer_joint_subset_l26.py','deploy_peer_joint_subset_l26.py','peer_newseed_source_audit_dispatch.py','peer_newseed_audit_resealed_metadata.py','peer_newseed_target_attestation.py'))
files=tuple(dict.fromkeys(FILES+extra+(cfgname,)))
branch='codex/peer-joint-subset-L26-20261021';parent=subprocess.check_output(['git','rev-parse',branch],text=True).strip()
fd,index=tempfile.mkstemp(prefix='joint-subset-index-');os.close(fd);os.unlink(index);env=dict(os.environ,GIT_INDEX_FILE=index)
try:
 subprocess.run(['git','read-tree',parent],env=env,check=True);subprocess.run(['git','add','--',*files],env=env,check=True)
 tree=subprocess.check_output(['git','write-tree'],env=env,text=True).strip();commit=subprocess.check_output(['git','commit-tree',tree,'-p',parent],input='Publish accepted '+n+' G10 subset L26 feature configuration\n',text=True).strip();subprocess.run(['git','update-ref','refs/heads/'+branch,commit,parent],check=True);subprocess.run(['git','push','origin',branch],check=True)
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],text=True).split()[0];assert remote==commit
finally:
 if os.path.exists(index):os.unlink(index)
rec=dict(commit=commit,branch=branch,config=cfgname,config_sha256=hashlib.sha256(data).hexdigest(),remote_sha=remote,files=list(files),questions=expected)
(C/'outputs/qwen7b-warmup-20261019/reports'/('joint-audited'+n+'-publication.json')).write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec))
