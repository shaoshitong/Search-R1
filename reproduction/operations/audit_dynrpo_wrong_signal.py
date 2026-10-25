"""Read saved remote rollouts on CPU; emit only aggregate signal counts."""
from reproduction.dynrpo7b.local_control import remote
import json

SCRIPT = r'''
import pathlib,json,torch,gc,collections
from transformers import AutoTokenizer
from search_g1.signals import decode_segments,contexts
from verl.utils.reward_score.qa_em import em_check
from search_g1.prefix20_advantage import anchored_grpo
from search_g1.logical_micro_loss import logical_masked_mean
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo7b-formal-20261023-recoveryfix-r6')
torch.set_num_threads(2)
c=json.loads((b/'logs/resolved-training-config.json').read_text())['resolved_config']
t=AutoTokenizer.from_pretrained(c['actor_rollout_ref']['model']['path'],local_files_only=True)
rows=[]
for p in (b/'artifacts/question-probe/attempts').glob('*/rollout-audit.json'):
 a=json.loads(p.read_text())
 if a.get('step') not in range(91,100):continue
 files=list(p.parent.glob('real-rollout-*.pt'))
 if len(files)!=1:raise ValueError('Ambiguous committed rollout')
 d=torch.load(files[0],map_location='cpu',weights_only=False,mmap=True)
 o=d['output'];hist=collections.Counter();mismatches=0
 for i,labels in enumerate(d['strict_em']):
  if sum(labels):continue
  valid=0
  for j in range(5):
   x=o[i*5+j];plen=x.batch['prompts'].shape[-1]
   n=int(x.batch['attention_mask'][plen:].sum())
   resp,ranges=decode_segments(t,x.batch['responses'][:n],x.batch['info_mask'][plen:plen+n])
   ctx=contexts('',resp,ranges)
   valid+=int(ctx is not None)
   gold=d['parents'].non_tensor_batch['reward_model'][i]['ground_truth']['target']
   mismatches+=int(ctx is not None and em_check(ctx['answer'],gold))
  hist[valid]+=1
 row=dict(step=a['step'],selected_groups=a['selected_groups'],all_wrong=a['selected_all_wrong'],valid_answer_count_histogram=dict(hist),zero_advantage_groups=hist[0]+hist[5],format_only_groups=sum(v for k,v in hist.items() if 0<k<5),reward_em_disagreements=mismatches)
 assert sum(hist.values())==row['all_wrong']
 rows.append(row);print('ROW='+json.dumps(row),flush=True)
 del d,o;gc.collect()
rr=torch.full((5,1),.05);mask=torch.ones_like(rr)
aa,_=anchored_grpo(rr,mask,['q']*5,[0]*5)
rr1=torch.ones((5,1));ac,_=anchored_grpo(rr1,mask,['q']*5,[1]*5)
m=torch.zeros((4,100));m[:2,:10]=1;m[2:]=1
v=torch.zeros_like(m);v[:2]=1
math_check=dict(equal_wrong_advantages=aa.flatten().tolist(),equal_correct_anchored_advantages=ac.flatten().tolist(),logical2=float(logical_masked_mean(v,m,2)),logical4=float(logical_masked_mean(v,m,4)))
print('RESULT='+json.dumps(dict(math_check=math_check,rows=sorted(rows,key=lambda x:x['step']),gpu_forwards=0,training_modified=False,wrong_floor=.05,invalid_reward=-.1,wrong_ground_scale=0)),flush=True)
'''
if __name__=='__main__':
    base='/data/user/jxu957/experiments/dynrpo7b-formal-20261023-recoveryfix-r6'
    remote('import subprocess,os\nenv=dict(os.environ,CUDA_VISIBLE_DEVICES="",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="2")\nraise SystemExit(subprocess.call('+repr([base+'/env/bin/python','-B','-c',SCRIPT])+',cwd='+repr(base+'/repo')+',env=env))')
