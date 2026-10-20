"""CPU-only loss landscape; no model/checkpoint load or training job mutation."""
import argparse,json,math
from pathlib import Path
import torch
from reproduction.prefix20_group_joint100.loss import joint_loss


def value(p,label,trajectory_weight):
    z=torch.full((1,5),math.log(p/(1-p)),dtype=torch.float64,requires_grad=True);y=torch.full_like(z,float(label));v=joint_loss(z,y);loss=trajectory_weight*v['trajectory_bce']+v['group_bce']
    return dict(p=p,trajectory_bce=float(v['trajectory_bce'].detach()),group_bce=float(v['group_bce'].detach()),total=float(loss.detach()),tied_logit_gradient=float(torch.autograd.grad(loss,z)[0].sum()))


def roots(label,weight):
    found=[];last=.001;sgn=value(last,label,weight)['tied_logit_gradient']
    for i in range(2,1000):
        p=i/1000;g=value(p,label,weight)['tied_logit_gradient']
        if g*sgn<0:
            a,b=last,p;ga=sgn
            for _ in range(45):
                m=(a+b)/2;gm=value(m,label,weight)['tied_logit_gradient']
                if ga*gm<=0:b=m
                else:a=m;ga=gm
            r=(a+b)/2;row=value(r,label,weight);z=torch.full((5,),math.log(r/(1-r)),dtype=torch.float64,requires_grad=True)
            def loss(v):
                d=joint_loss(v.unsqueeze(0),torch.full((1,5),float(label),dtype=torch.float64));return weight*d['trajectory_bce']+d['group_bce']
            eigen=torch.linalg.eigvalsh(torch.autograd.functional.hessian(loss,z));row.update(full_five_logit_hessian_eigenvalues=eigen.tolist(),full_gradient_max_abs=float(torch.autograd.grad(loss(z),z)[0].abs().max()),strict_local_minimum_in_independent_logits=bool((eigen>0).all()));found.append(row)
        last=p;sgn=g
    return found


def toy(weight):
    z=torch.nn.Parameter(torch.full((1,5),math.log(.2/.8),dtype=torch.float64));opt=torch.optim.Adam([z],lr=.02);curve=[]
    for step in range(1201):
        v=joint_loss(z,torch.ones_like(z));loss=weight*v['trajectory_bce']+v['group_bce']
        if step in [0,100,300,600,1200]:curve.append(dict(step=step,p_j=z.detach().sigmoid().tolist()[0],trajectory_bce=float(v['trajectory_bce'].detach()),group_bce=float(v['group_bce'].detach()),total=float(loss.detach())))
        if step<1200:opt.zero_grad();loss.backward();opt.step()
    return curve


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    result=dict(allcorrect_stationary=roots(1,1),allwrong_stationary=roots(0,1),allcorrect_fivefold_trajectory_stationary=roots(1,5),allwrong_fivefold_trajectory_stationary=roots(0,5),allcorrect_samples=[value(p,1,1) for p in [.1,.167,.2,.3,.4,.5,.8]],allcorrect_independent_logit_Adam_current=toy(1),allcorrect_independent_logit_Adam_candidate=toy(5),scope='Independent five-logit objective. Real shared-network trapping is not established. Candidate trajectory coefficient5 is only a CPU comparison; current GPU recipe remains unchanged.',gpu_training_changed=False)
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
