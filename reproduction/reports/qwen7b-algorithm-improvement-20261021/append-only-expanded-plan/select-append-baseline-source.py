"""Apply the precommitted failed-ranking fallback after actual terminal audits."""
import hashlib,json,pathlib,subprocess,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');B=A/'peer-expanded-append-only-plan-v1';T=A/'peer-newseed8192-g10-train-v1'
def read(p):return json.loads(pathlib.Path(p).read_text())
def fp(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(r):
 actual=fp(r['path'])
 if any(actual[k]!=v for k,v in r.items() if k in actual):raise ValueError('Bound file proof differs')
 return read(r['path'])
records={}
for recipe in ('inter1','bothrank1'):
 p=A/('peer8192-'+recipe+'-matched256-v1')/'final-independent-audit.json';audit=read(p)
 assert audit['status']=='MATCHED_RANK256_CP_OWN_EM_FIXED_HF_METRICS_INDEPENDENTLY_ACCEPTED'
 assert audit['final_Adam_steps']==10720 and audit['same_actual_grouping_all256'] and '|COMPLETED|0:0|' in audit['sacct']
 metrics=audit['evaluations']['selection2048FreshG5HF']['metrics'];ba=metrics['pool']['mixed_balanced_accuracy'];top=metrics['selected512']['mixed_questions']
 assert not(ba>0.5625049686633887 and top>196),'Joint winner must be frozen/confirmed before enlargement'
 records[recipe]=dict(audit=fp(p),BA=ba,Top512=top,joint_pass=False)
bank=dict(path=str(A/'peer-expanded-L26-replay-v1/bank/complete.json'),bytes=2280,sha256='12f5e89581628e1e6d08f2d1b76829445e9f69237cecf77db057ff8b377e340d')
bank_audit=dict(path=str(A/'peer-expanded-L26-replay-v1/bank/independent-input-audit.json'),bytes=2346,sha256='866291abff96cf1302ce78065582b06fca784f94f14b0456750f214a640f4691')
metadata=checked(bank);checked(bank_audit);assert metadata['questions']==30016 and metadata['trajectories']==300160
source=read(T/'complete.json');config=read(T/'config.json');receipt=source['checkpoint']
assert source['cursor']==256 and source['steps']==10720 and source['optimizer_reset'] is False and source['actor_updates']==0
assert config['recipe']['loss_kwargs']==dict(trajectory_weight=1.,three_event_weight=1.)
assert source['config_sha256']==fp(T/'config.json')['sha256'] and source['plan_sha256']=='0b2212b46e2046d097f7c1a90490406ba67babf81002f15b129a61efe4dba012'
assert receipt['cursor']==256 and receipt['steps']==10720 and receipt['sha256']=='36d9bcc3be71ba94eaca3a1e1b292c0ed9a65297fb96a746ea5295432f09c000'
actual=fp(receipt['path']);assert all(actual[k]==receipt[k] for k in ('path','bytes','sha256'))
state=subprocess.check_output(['sacct','-n','-X','-j','743367','-o','JobIDRaw,State,ExitCode','-P'],universal_newlines=True,timeout=15)
assert '743367|COMPLETED|0:0' in state
done=dict(status='PRECOMMITTED_UNCHANGED_PEER_CP256_APPEND_ONLY_BASELINE_SELECTED_AFTER_RANK_FINALS_AND_BANK_ACCEPTANCE',source_job=743367,source_Slurm=state.strip(),source_checkpoint=receipt,source_complete=fp(T/'complete.json'),source_config=fp(T/'config.json'),source_plan=config['plan'],new_bank=bank,new_bank_independent_audit=bank_audit,append_plan=read(B/'complete.json')['plan'],ranking_finals=records,fair_BA_frontier=0.5625049686633887,fair_Top512_frontier=196,loss_kwargs=dict(trajectory_weight=1.,three_event_weight=1.),source_cursor=256,source_Adam=10720,new_updates=682,cumulative_target_cursor=938,cumulative_target_Adam=11402,source_world_size=1,current_world_size=1,original_E1_archive_world_size=4,optimizer_reset=False,block_reset=False,readout_reset=False,model_or_optimizer_tensor_loaded=False,full_state_restore_and_replay_still_required=True,new_training_submitted=False,final_pool_read=False,time=time.time())
p=B/'source-selection.json'
if p.exists():
 old=read(p)
 for k in ('source_checkpoint','new_bank','new_bank_independent_audit','append_plan','ranking_finals','loss_kwargs'):
  assert old[k]==done[k],'Preserve a different prior source selection'
 done=old
else:
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(done,sort_keys=True)+'\n');temp.replace(p)
print(json.dumps(dict(receipt=fp(p),**done),sort_keys=True))
