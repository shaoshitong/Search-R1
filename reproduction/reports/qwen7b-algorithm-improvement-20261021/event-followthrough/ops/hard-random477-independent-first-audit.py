"""CPU-only independent SHA/binding/real-update acceptance; no tensor deserialization."""
import hashlib,json,pathlib,time,math
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
T=B/'algorithm-improvement-20261021/hard-vs-random-E4-477-v1'
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for raw in iter(lambda:f.read(8*1024**2),b''):h.update(raw)
 return h.hexdigest()
def fp(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def check(v):
 assert fp(v['path'])=={k:v[k] for k in ('path','bytes','sha256')};return pathlib.Path(v['path'])
prep=read(T/'preparation.json');check(prep['selection_complete']);check(prep['validation_complete']);check(prep['cpu_tests'])
assert read(T/'cpu-result.json')['status']=='E4_HARD_RANDOM_CPU_PREPARATION_COMPLETE'
out=dict(status='E4_HARD_RANDOM_ACTUAL_FIRST_UPDATE_AND_FULL_CHECKPOINT_SHA_INDEPENDENT_PASS',observed_epoch=time.time(),
    preparation=fp(T/'preparation.json'),variants={},torch_imported=False,model_or_optimizer_deserialized=False,
    raw_predictions_returned=False,untouched_final_read=False)
sources=[]
for name in ('targeted','random_control'):
 b=T/name;item=prep['variants'][name];c=read(check(item['config']));check(item['sbatch'])
 assert c['variant']==name and c['global_groups']==48 and c['update_budget']==477 and c['source_checkpoint']['steps']==14418
 assert c['lr']==5e-6 and c['ema_decay']==.99 and c['recipe']['loss_kwargs']==dict(trajectory_weight=1.,three_event_weight=1.,inter_rank_weight=1.)
 assert c['canonical_source_rng_rank']==0 and c['source_world_size']==4 and c['current_world_size']==1 and c['four_rank_bitexact_resume_claim'] is False
 assert c['E1_reset'] is False and c['source_raw_ema_adam_RNG_preserved'] is True
 sources.append(c['source_checkpoint'])
 for key in ('bank','features','plan','selection_complete','candidate_plan','scoring_complete','feature_manifest_refs'):check(c[key])
 code=pathlib.Path(read(T/'source-publication.json')['code_root'])
 for rel,wanted in c['code_hashes'].items():assert sha(code/rel)==wanted
 controller=read(b/'controller.json');assert controller['job_id']==(b/'job.id').read_text().strip()
 assert controller['config_sha256']==item['config']['sha256'] and controller['source_steps']==14418
 assert not (b/'failure.json').exists()
 proof=read(b/'next-update-replay-cursor000000.json')
 assert proof['passed'] and proof['next_update_replayed_twice'] and proof['source_steps']==14418 and proof['accepted_steps']==14419
 assert proof['four_rank_bitexact_resume_claim'] is False and proof['expected']==proof['actual']
 assert set(proof['actual'])=={'raw','ema','adam','raw_buffers','ema_buffers','rng'}
 for digest in proof['actual'].values():assert len(digest)==64 and all(ch in '0123456789abcdef' for ch in digest)
 cp1=b/'checkpoints/cursor-000001.pt';assert cp1.is_file() and cp1.stat().st_size>3*1024**3
 cp1_ref=fp(cp1)
 receipt=read(b/'latest.json');check(receipt)
 assert receipt['cursor']>=1 and receipt['steps']==14418+receipt['cursor'] and receipt['config_sha256']==item['config']['sha256']
 if receipt['cursor']==1:assert receipt['accepted_hashes']==proof['actual'] and receipt['sha256']==cp1_ref['sha256']
 assert receipt['matched_binding_sha256']==controller['matched_binding_sha256']
 progress=read(b/'progress.json');assert progress['cursor']>=2 and progress['steps']==14418+progress['cursor'] and progress['actor_updates']==0
 rows=[json.loads(s) for s in (b/'curve.jsonl').read_text().splitlines() if s.endswith('}')]
 assert len(rows)>=progress['cursor'] and [r['cursor'] for r in rows]==list(range(1,len(rows)+1))
 assert all(r['steps']==14418+r['cursor'] and r['global_questions']==48 and r['actor_updates']==0 and math.isfinite(r['loss']) for r in rows)
 assert rows[0]['steps']==14419
 out['variants'][name]=dict(job_id=controller['job_id'],config=item['config'],first_checkpoint=cp1_ref,
    latest_checkpoint={k:receipt[k] for k in ('path','bytes','sha256','cursor','steps')},
    first_update_actual_component_hashes=proof['actual'],double_next_update_replay_passed=True,
    accepted_complete_updates=len(rows),actual_adam_steps=14418+len(rows),all_complete_global_questions=48,
    first_loss=rows[0]['loss'],last_loss=rows[-1]['loss'],source_all_four_rank_RNG_preserved_by_GPU_acceptance=True,
    canonical_rank0_restored=True,four_rank_bitexact_resume_claim=False)
assert sources[0]==sources[1];check(sources[0]);out['source_E4_full_file_SHA_freshly_verified']=True
print(json.dumps(out))
