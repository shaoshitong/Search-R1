"""Accept actual startup receipts/first eight transactions, no tensor reload."""
import base64,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
BODY='''import pathlib,json,hashlib,subprocess,time,statistics
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-124194-v1')
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
v={n:read(b/(n+'.json'))for n in('progress','first-update-replay','first-cp-independent-audit','durable-next-update-replay','latest')}
p=v['progress'];one=v['first-update-replay'];audit=v['first-cp-independent-audit'];two=v['durable-next-update-replay'];cp=v['latest']
assert p['cursor']>=8 and p['steps']==14418+p['cursor']and not(b/'failure.json').exists()
assert one['status']=='PASSED'and one['expected']==one['actual']and one['block_changed']and one['readout_changed']and one['source_steps']==14418 and one['accepted_steps']==14419
assert audit['status']=='NATIVE_JOINT_CP1_INDEPENDENT_CPU_FULL_SIX_STATE_AND_SAMPLER_RELOAD_PASSED'and audit['allocation_job_id']=='745501'and audit['host']=='ACD1-28'and audit['CUDA_initialized']is False and audit['next_sampler_batch_identity_exact']is True and audit['components']==one['actual']
assert audit['checkpoint']['sha256']=='8e5c69e8b995d86e47964f89a586f4fe92cde728683ff5f6d90d9d2dbc72db5d'and pathlib.Path(audit['checkpoint']['path']).stat().st_size==audit['checkpoint']['bytes']
assert two['status']=='PASSED'and two['cursor']==2 and two['steps']==14420 and two['expected']==two['actual']
rows=[json.loads(line)for line in(b/'curve.jsonl').read_bytes().splitlines(keepends=True)if line.endswith(bytes([10]))];first=[r for r in rows if 3<=r['cursor']<=8];assert[r['cursor']for r in first]==list(range(3,9))
for r in first:
 g=r['grouping'];assert r['steps']==14418+r['cursor']and r['sampler_accepted_updates']==r['cursor']and r['actor_updates']==0 and r['recomputed_logits_bit_identical']is True
 assert g['questions']==32 and g['views']==128 and g['mixed_views']==g['nonmixed_views']==64 and g['per_question_total_weight_exactly_one']is True and g['source_question_pool_size']==124194 and g['labels_recomputed_from_own_subset']is True and g['synthetic_labels']is False
assert cp['cursor']>=7 and cp['full_CPU_six_state_reload_passed']and cp['sampler_full_state_and_cursor_reload_passed']and cp['saved_all_four_source_RNG_states']and cp['allocation_job_id']=='745501';assert pathlib.Path(cp['path']).stat().st_size==cp['bytes']
e0ref=proof(b/'evaluations/epoch-00/complete.json');e0=read(e0ref['path']);assert e0['cursor']==e0['epoch']==0 and e0['checkpoint']['steps']==14418 and e0['training_components_unchanged']and e0['native_buffers_and_RNG_restored']and e0['Count5120_scored']is False and e0['final4096_scored']is False
main=[r for r in subprocess.check_output(['sacct','-n','-P','-j','745501','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'],universal_newlines=True).splitlines()if r.split('|')[0]=='745501'];assert len(main)==1 and main[0].split('|')[1]=='RUNNING'
out=dict(status='ACTUAL124194_NATIVE_JOINT_MAIN_STARTUP_QUALIFIED_EIGHT_ACCEPTED_UPDATES_AND_CONTINUING',epoch=time.time(),main_job=main[0],current_progress={k:p[k]for k in('cursor','steps','budget','seen_questions','loss','seconds','time')},receipt_refs={n:proof(b/(n+'.json'))for n in v},first_update=one,independent_CP1=audit,durable_next_update=two,first_eight_transactions=dict(first=1,second=2,following=[dict(cursor=r['cursor'],steps=r['steps'],seconds=r['seconds'],Mixed=r['grouping']['mixed_views'],nonMixed=r['grouping']['nonmixed_views'],repeated_subset_draws=r['grouping']['repeated_subset_draws'])for r in first]),latest_full_checkpoint=cp,epoch0_receipt=e0ref,epoch0_selection2048={name:e0['metrics'][name]['fixed_selection2048']for name in('raw','EMA')},epoch0_fixed512={name:e0['metrics'][name]['diagnostic512']for name in('raw','EMA')},no_login_checkpoint_tensor_load=True,CP_full_SHA_and_component_reload_inherited_from_owned_allocation_receipts=True,new_model_external_improvement_claim=False,whole_four_epoch_goal_complete=False)
print(json.dumps(out))
'''
spec=importlib.util.spec_from_file_location('native_startup_receipts_only',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();packed=base64.b64encode(zlib.compress(BODY.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');r=ctl.run(cmd,account='mou176',source_address=ip,timeout=30)
if r.returncode:raise RuntimeError(r.stdout[-3000:]+r.stderr[-1000:])
v=json.loads(r.stdout.splitlines()[-1]);Path(__file__).with_name('actual-startup-qualified.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:v[k]for k in('status','main_job','current_progress','latest_full_checkpoint','epoch0_receipt')}))
