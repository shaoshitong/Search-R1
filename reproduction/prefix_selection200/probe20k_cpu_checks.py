"""Small CPU fixtures for the exact20k EMA/mixup loss and complete-G5 plan."""
import json
import random
import torch
from .probe20k_fit import mix_batch,update_ema
from .probe20k_plan import plan_epoch
from .actor_hidden import WIDTH,DIM
from reproduction.prefix20_group_joint100.loss import joint_loss


def run():
    torch.set_num_threads(2)
    model=torch.nn.Linear(2,1,bias=False);ema=torch.nn.Linear(2,1,bias=False)
    with torch.no_grad():model.weight.fill_(3);ema.weight.fill_(1)
    update_ema(model,ema,.99)
    torch.testing.assert_close(ema.weight,torch.full_like(ema.weight,1.02))
    labels=torch.tensor([[1.,1.,1.,1.,1.],[0.,0.,0.,0.,0.],[1.,0.,1.,0.,1.]])
    batch=dict(features=torch.zeros((15,WIDTH,DIM),dtype=torch.bfloat16))
    batch['features'][5:10]=2;batch['features'][10:15]=4
    for key in ('attention_mask','question_mask','policy_mask'):
        batch[key]=torch.ones((15,WIDTH),dtype=torch.bool)
    batch['position_ids']=torch.arange(WIDTH).repeat(15,1)
    mixed,target,event,stats=mix_batch(batch,labels,[(0,1)],123,.2)
    lam=random.Random(123).betavariate(.2,.2)
    torch.testing.assert_close(target[0],torch.full((5,),lam))
    torch.testing.assert_close(target[0]+target[1],labels[0]+labels[1])
    torch.testing.assert_close(event,torch.tensor([0.,0.,1.]))
    torch.testing.assert_close(mixed['features'][0],torch.full((WIDTH,DIM),2*(1-lam),dtype=torch.bfloat16))
    torch.testing.assert_close(mixed['features'][10:],batch['features'][10:])
    assert stats['augmented_questions']==2
    z=torch.tensor([[.4,-.2,1.,-.8,.7],[-.1,.8,.3,.2,-.9],[.6,.5,.2,.1,-.3]],dtype=torch.float64,requires_grad=True)
    p=z.sigmoid();q=1-p.prod(1)-(1-p).prod(1)
    ordinary=joint_loss(z,labels.double())
    wanted=torch.nn.functional.binary_cross_entropy_with_logits(z,labels.double())+torch.nn.functional.binary_cross_entropy(q,torch.tensor([0.,0.,1.],dtype=torch.float64))
    torch.testing.assert_close(ordinary['loss'],wanted)
    augmented=joint_loss(z,target.double(),mixed_target=event.double())
    wanted_soft=torch.nn.functional.binary_cross_entropy_with_logits(z,target.double())+torch.nn.functional.binary_cross_entropy(q,event.double())
    torch.testing.assert_close(augmented['loss'],wanted_soft)
    augmented['loss'].backward();assert torch.isfinite(z.grad).all()and torch.count_nonzero(z.grad)>0
    from reproduction.prefix20_group_joint100.train_epoch import evaluate_predictions
    index={};predictions=[]
    for i in range(20480):
        for j in range(5):
            tid=str(i)+'-'+str(j);label=int(j==0)
            index[tid]=dict(qid=str(i),question_hash='question-'+str(i),strict_em=label)
            predictions.append((tid,2. if label else -2.,label))
    metrics,rows=evaluate_predictions(predictions,index,20480)
    assert metrics['count']==20480 and metrics['mixed_count']==20480 and len(rows)==20480
    fixtures=[]
    for name,keys in [('all_same',[0]*20480),('177_layouts',[i%177 for i in range(20480)]),('all_distinct',list(range(20480)))]:
        plan=plan_epoch(keys,20261017,1);again=plan_epoch(keys,20261017,2)
        assert plan!=again and len(plan)==400
        seen=[i for step in plan for i in step['question_indices']]
        assert sorted(seen)==list(range(20480))
        paired=0
        for step in plan:
            at=0
            for rank,size in enumerate(step['rank_questions']):
                local=step['question_indices'][at:at+size];at+=size
                for a,b in step['mix_pairs'][rank]:assert keys[local[a]]==keys[local[b]];paired+=2
        fixtures.append(dict(layout=name,questions=20480,updates_per_rank=400,paired_question_fraction=paired/20480))
    return dict(status='CPU_EMA_MIXUP_JOINT_G5_FIXTURES_PASS',ema_orientation=True,
        ownEM_soft_labels=True,separate_soft_mixed_event=True,default_joint_objective_unchanged=True,
        differentiable_group=True,complete_20480_question_evaluation=True,whole_G5_plans=fixtures,gpu_forward_performed=False)

if __name__=='__main__':print(json.dumps(run()))
