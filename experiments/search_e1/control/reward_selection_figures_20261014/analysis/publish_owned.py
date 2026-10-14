from pathlib import Path
import json,subprocess,tempfile,os,hashlib,datetime,re
D=Path(__file__).resolve().parent;out=Path('/Users/shaoshitong/Downloads/WWW_2026_PAPER/drawio_figure/reward-selection-comparison');repo=D.parent.parent/'work/search-e1-publish';branch='codex/reward-selection-figures-20261014';base=['git','-C',str(repo)]
def run(args,**kw):
 p=subprocess.run(base+args,capture_output=True,check=True,**kw);return p.stdout
files={}
for p in D.glob('*.py'):
 files['analysis/'+p.name]=p.read_bytes()
files['helpers/C_stream.py']=(D.parent/'g1-reward-correlation-20261010/reconstruct_train_raw_c_preflight_v2.py').read_bytes();files['helpers/replay_core.py']=(D.parent/'g1-reward-correlation-20261010/historical_replay/replay_core.py').read_bytes();files['helpers/stats.py']=(D.parent/'g1-diversity-20261014/aggregate_remote.py').read_text().split('MODE=MODE_VALUE')[0].encode()
files['analysis/fixed512-curve-data.json']=(D/'fixed512-curve-data.json').read_bytes();files['analysis/native-reward-means.json']=(D/'native-reward-means.json').read_bytes()
for name in ['hard-conditioned-soft-data-20261015.json','hard-conditioned-soft-data-validation-20261015.json','hard-conditioned-verification.json','hard-conditioned-source-backup.json','all-correct-soft-binning-20261015.json','all-correct-soft-binning-validation-20261015.json','all-correct-source-backup.json']:
 files['analysis/'+name]=(D/name).read_bytes()
for name in ['selection-storyboard-source-backup.json','selection-storyboard-verification.json']:
 files['analysis/'+name]=(D/name).read_bytes()
storyboard_backup=json.loads((D/'selection-storyboard-source-backup.json').read_text())
files['figures/templates/user-three-chart-storyboard.drawio']=Path(storyboard_backup['path']).read_bytes()
backup=json.load((D/'hard-conditioned-source-backup.json').open())
files['figures/templates/user-edited-before-hard-conditioning.drawio']=Path(backup['path']).read_bytes()
for p in out.glob('*.drawio'):files['figures/'+p.name]=p.read_bytes()
for p in out.glob('*.canvas.tsx'):files['figures/'+p.name]=p.read_bytes()
files['README.md']='''# Reward selection comparison figures, 2026-10-14

Native editable draw.io diagrams, no bitmap-wrapped diagrams. Labels: Soft Reward; Soft Reward -> Hard Reward; Easy Selection+Soft Reward. Fixed512 EM is from immutable native evaluation identities at outer50/100/150/200. Native training reward means are from actual update ledgers. The two reward definitions have different scales.

The latest hard-conditioned-soft-reward-amplification.drawio is one compact three-chart storyboard based on the user's edited first page. It preserves iter20 histogram bins/counts, excludes all-wrong groups, and uses the original twelve fixed512 EM points. Equal-width charts share a frame and native curved arrows. Native robot, magnifier, question-pool and filter icons explain the progression. The actual selection prefers mixed strict-EM groups (Hard EM std>0), with all-correct fallback when needed; it does not rank questions by Soft std or remove all all-correct groups. Nonzero Soft preferences among equally correct answers are not by themselves proof of incorrect gradients. The user's incoming full drawing is retained as a template backup. build_selection_storyboard.py includes hash guards against overwriting subsequent edits.

Diversity is the sample standard deviation (correction1) within each original question's five responses. A=(r-group_mean)/(group_std+1e-6), not token weighted. Strict EM component distributions are reconstructed from source-bound token/scorer or immutable selection receipts, and are not the actual soft-policy advantages. Historical Soft Reward reconstruction is available at1/20/40/60/80/100/120/140. Its per-row soft rewards use frozen historical encoders and readout heads, not guaranteed bitwise online recovery.

Independent Easy Selection reward replay was blocked before GPU model loading when the original experiment's final evaluation3395 acquired all8 GPUs. No trainer, source manifest or original probe service was changed. The isolated replay script is a candidate, not validated GPU inference; aggregate values from failed attempts must not be published as data. Raw tokens, model weights and per-row replay outputs stay remote. Local artifacts only contain code, diagrams and compact aggregates.

The source root/work paths in audit tools locate existing deployments. These scripts do not launch or restart any original training/evaluation job. Do not interpret code backup as execution authorization.
'''.encode()
for name,raw in files.items():assert not re.search(rb'hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----',raw),name
target=branch if run(['ls-remote','origin','refs/heads/'+branch]).strip() else 'search-e1'
run(['fetch','origin',target]);parent=run(['rev-parse','FETCH_HEAD']).decode().strip();prefix='experiments/search_e1/control/reward_selection_figures_20261014/'
with tempfile.TemporaryDirectory(prefix='reward-figure-index-') as t:
 env=dict(os.environ,GIT_INDEX_FILE=str(Path(t)/'index'));run(['read-tree',parent],env=env)
 for name,raw in files.items():
  blob=run(['hash-object','-w','--stdin'],input=raw).decode().strip();run(['update-index','--add','--cacheinfo','100644',blob,prefix+name],env=env)
 tree=run(['write-tree'],env=env).decode().strip();commit=run(['commit-tree',tree,'-p',parent],input=b'Archive editable reward selection figures and evidence-bound audit tools\n').decode().strip()
run(['push','origin',commit+':refs/heads/'+branch]);remote=run(['ls-remote','origin','refs/heads/'+branch]).decode().split()[0];assert remote==commit
run(['fetch','origin',branch]);back=run(['rev-parse','FETCH_HEAD']).decode().strip();assert back==commit
for name,raw in files.items():assert run(['show',back+':'+prefix+name])==raw,name
v={'session':'01a101e0-90d3-7731-bae9-58c8f1fdf7e2','source_commit':commit,'remote_sha':remote,'branch':branch,'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_checkout_unchanged':True,'files':[{'path':prefix+n,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)} for n,b in files.items()]};(D/'publication-verified.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v[k] for k in ['source_commit','remote_sha','branch','verified_utc']}))
