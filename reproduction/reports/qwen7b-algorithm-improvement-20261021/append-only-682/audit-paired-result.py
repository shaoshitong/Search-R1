"""Fresh metadata/SHA acceptance of a finished fixed-protocol CPU bootstrap."""
import argparse,importlib.util,json,pathlib,shlex,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1];OUT=pathlib.Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--cursor',type=int,choices=(512,768,938),required=True);a=p.parse_args()
source='''import pathlib,json,hashlib,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');root=A/'peer-expanded-append-stability-1a0f5b4';work=root/('cursor-%06d-v2'%CURSOR);evalbase=A/'peer-expanded-append-only-train-v2/evaluations'/('cursor-%06d'%CURSOR)
def read(p):return json.loads(pathlib.Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def ref(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024**2),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
manifest=read(work/'manifest.json');result=read(work/'result.json');exit=read(work/'worker-exit.json');audit=read(evalbase/'independent-audit.json')
assert exit['returncode']==0 and result['status']=='DESCRIPTIVE_PAIRED_QUESTION_STABILITY_COMPLETE'
assert result['source_manifest']==manifest and result['source_manifest_sha256']==ref(work/'manifest.json')['sha256']
assert manifest['role']=='already_viewed_recipe_selection_validation' and manifest['bootstrap_replicates']==2000 and manifest['questions']==2048
assert manifest['independent_identity_audit']==ref(evalbase/'independent-audit.json')
assert audit['cursor']==CURSOR and audit['status']=='APPEND_MILESTONE_OWN_EM_PLAN_METRICS_CP_SHA_INDEPENDENTLY_ACCEPTED'
assert manifest['predictions']['candidate']==audit['evaluations']['selection2048FreshG5HF']['predictions']
for v in manifest['predictions'].values():assert ref(v['path'])==v
assert result['seed']==9137 and result['threshold']==.5 and result['selection_fraction']==.25
assert not any(result[k] for k in ('parameters_fitted','threshold_selected','candidate_selected','external_final_read','generalization_claim'))
strata=result['strata'];assert strata['labels_consulted'] is False and strata['scores_consulted'] is False and strata['strata_sha256']==digest(strata['strata'])
flat=[q for group in strata['strata'] for q in group];assert len(flat)==len(set(flat))==2048 and all(len(group)==512 for group in strata['strata'])
assert set(result['scopes'])=={'all_questions','hash_stratum_0','hash_stratum_1','hash_stratum_2','hash_stratum_3'}
for scope in result['scopes'].values():assert scope['bootstrap_replicates_requested']==scope['bootstrap_replicates_used']==2000 and scope['single_class_replicates_skipped']==0
point=result['scopes']['all_questions']['models']['candidate'];accepted=audit['evaluations']['selection2048FreshG5HF']['metrics']
assert point['mixed_balanced_accuracy']==accepted['pool']['mixed_balanced_accuracy'] and point['top_quarter_mixed_count']==accepted['selected512']['mixed_questions']
assert result['scopes']['all_questions']['models']['E2']['mixed_balanced_accuracy']==0.5625049686633887 and result['scopes']['all_questions']['models']['E4']['top_quarter_mixed_count']==196
summary=dict(status='FIXED_PROTOCOL_PAIRED_CPU_BOOTSTRAP_FINAL_SHA_AND_BINDINGS_ACCEPTED',cursor=CURSOR,bootstrap_result=ref(work/'result.json'),manifest=ref(work/'manifest.json'),worker_exit=exit,independent_milestone_audit=ref(evalbase/'independent-audit.json'),source=ref(root/'paired_question_stability.py'),strata_sha256=strata['strata_sha256'],scopes=result['scopes'],all_5_scopes_exact2000_replicates=True,no_local_probabilities_or_qids=True,external_final_read=False,threshold_selected=False,parameters_fitted=False,time=time.time())
p=work/'independent-result-audit.json';assert not p.exists();p.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\\n');summary['audit_ref']=ref(p);print(json.dumps(summary))'''
spec=importlib.util.spec_from_file_location('paired_result_audit_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
command='python3 -c '+shlex.quote('CURSOR='+str(a.cursor)+'\n'+source)
if len(command.encode())>=16000:raise ValueError('Control command exceeds16KB')
r=ctl.run(command,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=45)
if r.returncode:raise RuntimeError(r.stderr[-2000:])
v=json.loads(r.stdout);(OUT/('actual-paired-%d-independent-audit.json'%a.cursor)).write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(dict(status=v['status'],cursor=a.cursor,audit=v['audit_ref'],result=v['bootstrap_result'])))
