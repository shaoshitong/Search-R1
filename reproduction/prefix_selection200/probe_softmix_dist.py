"""Train the SAME mixed-event h used for selection, with0/.9/1 soft targets."""
import torch
import torch.nn.functional as F
from .probe_group_vector_dist import bernoulli_kl,dominant_hidden_mixup,evaluation_metrics as diversity_diagnostics


def soft_mixed_target(labels):
    k=labels.float().sum(-1)
    return torch.where((k==2)|(k==3),torch.ones_like(k),torch.where((k==1)|(k==4),torch.full_like(k,.9),torch.zeros_like(k)))


def mixed_event_logit(logits):
    z=logits.float();lp,ln=F.logsigmoid(z),F.logsigmoid(-z)
    bits=((torch.arange(32,device=z.device).unsqueeze(1)>>torch.arange(5,device=z.device).unsqueeze(0))&1).bool()
    logs=torch.where(bits.unsqueeze(0),lp.unsqueeze(1),ln.unsqueeze(1)).sum(-1)
    return torch.logsumexp(logs[:,1:31],dim=1)-torch.logsumexp(logs[:,[0,31]],dim=1)


def softmix_loss(logits,vector_target,mixed_target=None,dist_weight=1,mixed_weight=50):
    p=torch.sigmoid(logits.float());r=vector_target.float()
    pc,rc=p-p.mean(-1,keepdim=True),r-r.mean(-1,keepdim=True)
    vp,vr=pc.square().sum(-1),rc.square().sum(-1)
    rho=(pc*rc).sum(-1)/torch.sqrt((vp+1e-6)*(vr+1e-6))
    relation=((1-rho)*(vr>1e-12).float()).mean()
    target=soft_mixed_target(r)if mixed_target is None else mixed_target.float()
    event_logit=mixed_event_logit(logits)
    loss_kl=bernoulli_kl(event_logit,target).mean()
    total=dist_weight*relation+mixed_weight*loss_kl
    return dict(loss=total,total_loss=total,dist_loss=relation,soft_mixed_kl=loss_kl,
        predicted_mixed_probability=torch.sigmoid(event_logit),soft_mixed_target=target)


def mix_batch(batch,labels,pairs,seed,alpha):
    # Same continuous hidden coefficient also mixes the original G5 soft targets.
    # Never derive a new hard K from fractional mixed own-EM vectors.
    groups=soft_mixed_target(labels);targets=groups.clone()
    mixed,vectors,augmentation=dominant_hidden_mixup(batch,labels,pairs,seed,alpha,with_pair_lambdas=True)
    for (a,b),lam in zip(pairs,augmentation.pop('pair_lambdas')):
        targets[a]=lam*groups[a]+(1-lam)*groups[b]
        targets[b]=lam*groups[b]+(1-lam)*groups[a]
    return mixed,vectors,targets,augmentation


def evaluation_metrics(rows,dist_weight=1,mixed_weight=50):
    # Retain former D/trajectory diagnostics, explicitly excluded from optimization.
    result=diversity_diagnostics(rows,dist_weight=dist_weight,mean_weight=0,diversity_weight=0)
    z=torch.tensor([r['trajectory_logits']for r in rows],dtype=torch.float32)
    labels=torch.tensor([r['strict_em']for r in rows],dtype=torch.float32)
    values=softmix_loss(z,labels,dist_weight=dist_weight,mixed_weight=mixed_weight)
    for row,t in zip(rows,values['soft_mixed_target'].tolist()):
        row['soft_mixed_training_target']=t
        row['score_semantics']='mixed_probability_h_is_primary_prediction_and_training_target_function;D_is_diagnostic_only'
    result.update(**{k:float(values[k])for k in('dist_loss','soft_mixed_kl','total_loss')},
        soft_mixed_target_mean=float(values['soft_mixed_target'].mean()),
        soft_mixed_mae=float((values['predicted_mixed_probability']-values['soft_mixed_target']).abs().mean()),
        objective='G5_Pearson_DIST_plus_soft_mixed_event_KL',soft_mixed_KL_in_training_loss=True,
        soft_mixed_KL_weight=mixed_weight,diversity_KL_in_training_loss=False,
        group_mean_KL_in_training_loss=False,trajectory_BCE_in_training_loss=False,
        true_mixed_decision='0<K<5',primary_prediction='h=1-prod(p)-prod(1-p)',threshold=.5,
        soft_target_by_K=[0,.9,1,1,.9,0],legacy_diversity_KL_is_diagnostic_only=True)
    return result
