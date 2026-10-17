"""Root personal observer; no training mutations and no main-state writes."""
import argparse,datetime,json,hashlib,time
from pathlib import Path
from reproduction.h1008_qprobe.management_transport import remote

REMOTE = "\nimport sys,json,pathlib,hashlib,time\nb=pathlib.Path(BASE);sys.path.insert(0,str(b/'repo'))\nfrom reproduction.prefix_selection200.launch import observe\nv=observe(b);j=v['latest_joint']or{};n=j.get('outer',0)\ndef read(p):return json.loads(p.read_text())\ndef sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()\nfacts=[]\nfor outer in range(VERIFIED+1,n+1):\n root=b/'outers'/('outer-%06d'%outer);joint=read(root/'complete.json');a=joint['actor'];pr=[];ar=[]\n for rank in range(8):\n  path=pathlib.Path(a['rank_receipts'][rank]['ledger_path']);rows=[json.loads(s)for s in path.read_text().splitlines()];writes=[r for r in rows if r['event']=='ACTUAL_ADAM_WRITE'];restored=[r for r in rows if r['event']=='ACTUAL_RESTORE']\n  assert len(writes)==20 and [r['actual_adam_steps']for r in writes]==list(range(20*(outer-1)+1,20*outer+1)) and sha(path)==a['rank_receipts'][rank]['ledger_sha256']\n  ar.append(dict(rank=rank,writes=20,last_actual_adam=writes[-1]['actual_adam_steps'],restore=restored[0],sha256=sha(path),path=str(path)))\n prefix=read(root/'prefix/complete.json')\n assert len(prefix['ranks'])==8\n for row in prefix['ranks']:\n  assert row['independent_encoder_forward'] is False and row['actor_hidden_layer']==27 and row['actor_hidden_forward_calls']>0 and row['trajectories']==3200\n assert joint['probe']['actual_adam_steps_per_rank']==[100*(outer//20)]*8\n if outer<20:assert joint['probe']['completed_outer']==0 and joint['probe_metrics']['new_adam_steps_per_rank']==0\n pr=dict(actual_adam_per_rank=joint['probe']['actual_adam_steps_per_rank'],last_update_outer=joint['probe']['completed_outer'])\n facts.append(dict(outer=outer,actor=ar,probe=pr,phase_timings=joint['phase_timings'],probe_metrics=joint['probe_metrics'],native_capture=[{k:row[k] for k in ('rank','actor_hidden_forward_calls','actor_hidden_captured_rows','actor_hidden_forward_and_capture_seconds','independent_encoder_forward')} for row in prefix['ranks']],selected_mixed=joint['rollout']['actual_selected_mixed_fraction']))\nqa=b/'outers/outer-000005/validation/complete.json';qa_value=read(qa)if qa.exists()else None\nif qa_value:\n from reproduction.prefix_selection200.eval import CONTRACT as qa_contract\n assert qa_value['evaluation_contract']==qa_contract and qa_value['qa_count']==512 and qa_value['actual_worker_exit_codes']=={str(i):0 for i in range(8)}\n for ref in qa_value['rank_receipts']:\n  worker=read(pathlib.Path(ref['path']));assert worker['evaluation_contract']==qa_contract and worker['actual_compute_dtype']=='torch.bfloat16'\n qa_value={k:val for k,val in qa_value.items()if k not in('output_shards','rank_receipts')}\nstage=v.get('stage')or{};current=stage.get('outer',0);prefix=v.get('current_prefix')or[];continuation=v.get('current_continuation')or[]\nprint(json.dumps(dict(base=str(b),observed_unix=v['observed_unix'],native=v['native'],stage=stage,complete_outer=n,actor_adam_per_rank=j.get('actor',{}).get('actual_adam_steps_per_rank'),probe_adam_per_rank=j.get('probe',{}).get('actual_adam_steps_per_rank'),probe_metrics=j.get('probe_metrics'),facts=facts,qa5=qa_value,gpu_actual=v['gpu_actual'],prefix=prefix,continuation=continuation,cleanup=v['cleanup'],pipeline_complete=v['pipeline_complete'])))"

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--job',required=True);p.add_argument('--root',required=True);p.add_argument('--start',type=int,default=0);a=p.parse_args()
 root=Path(a.root);verified=a.start;facts={};last=None
 while True:
  tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
  try:
   code=REMOTE.replace('BASE',repr(a.base)).replace('VERIFIED',str(verified))
   value=json.loads(remote(code,root/('actorhidden-personal-five-'+tag+'-transport.json')))
   path=root/('actorhidden-personal-five-'+tag+'-actual.json');path.write_text(json.dumps(value,indent=2)+'\n')
   for fact in value['facts']:facts[fact['outer']]=fact;verified=max(verified,fact['outer'])
   native=(value.get('native')or{}).get('JobState','UNKNOWN');stage=value['stage'];n=value['complete_outer']
   next_started=stage.get('outer',0)>=a.start+6 and any((r or{}).get('completed_trajectories',0)>0 for r in value['prefix'])
   accepted=n>=a.start+5 and set(range(a.start+1,a.start+6))<=set(facts) and next_started
   compact=dict(job=a.job,native=native,stage=stage,complete_outer=n,actor_adam_per_rank=value['actor_adam_per_rank'],probe_adam_per_rank=value['probe_adam_per_rank'],new_verified_outers=sorted(facts),initial_five_accepted=accepted,last_original=str(path),observed_utc=tag,prefix=value['prefix'],qa5=value.get('qa5'))
   tmp=root/'actorhidden-period20-latest-compact.json.partial';tmp.write_text(json.dumps(compact,indent=2)+'\n');tmp.replace(root/'actorhidden-period20-latest-compact.json')
   key=(native,stage.get('stage'),n)
   if key!=last:print(json.dumps({k:v for k,v in compact.items() if k not in ('prefix','qa5')}),flush=True);last=key
   if accepted:
    proof=root/'actorhidden-period20-first-five-actual-facts.json';proof.write_text(json.dumps(dict(job=a.job,base=a.base,facts=facts,next_started_original=str(path),next_started=True),indent=2)+'\n');print('FIVE_NATIVE_COMPLETE_AND_NEXT_ACTUAL_PREFIX_STARTED',flush=True);return
   if native in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY') or (value.get('cleanup') and not value.get('pipeline_complete')):
    raise SystemExit(3)
  except (RuntimeError,ValueError,KeyError,AssertionError) as error:
   print(json.dumps(dict(observation_error=repr(error),training_modified=False,tag=tag)),flush=True)
  time.sleep(45)

if __name__=='__main__':main()
