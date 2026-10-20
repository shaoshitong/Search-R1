"""Independent hidden-direct G5 count readout; no actor/model asset loading.

Legacy block27 and trajectory head keep their own exact optimizer. A new
attention/projector/count classifier has separate fresh Adam and EMA. Only
the actual retained policy-prefix tokens are pooled; question tokens supply
causal context through the unchanged legacy block, not a second data source.
"""
import copy
from fractions import Fraction
import torch
from torch import nn
import torch.nn.functional as F


class DirectCountHead(nn.Module):
    def __init__(self, dimension=3584, embedding_dimension=128):
        super().__init__()
        self.dimension, self.embedding_dimension = dimension, embedding_dimension
        self.attention = nn.Linear(dimension, 1, bias=False)
        self.projection = nn.Linear(dimension, embedding_dimension)
        self.classifier = nn.Sequential(nn.Linear(3*embedding_dimension, 128), nn.GELU(), nn.Linear(128, 6))

    def encode(self, prefix_hidden, prefix_mask):
        if (prefix_hidden.ndim != 3 or prefix_hidden.shape[-1] != self.dimension
                or prefix_mask.shape != prefix_hidden.shape[:2] or prefix_mask.dtype != torch.bool):
            raise ValueError('Real prefix hidden and matching boolean mask required')
        if not bool(torch.isfinite(prefix_hidden[prefix_mask]).all()):
            raise ValueError('Nonfinite real prefix hidden')
        # Padding is explicitly erased before attention, including short/zero
        # prefixes; no masked NaN or padding vector can become a model signal.
        clean = prefix_hidden.masked_fill(~prefix_mask[..., None], 0)
        scores = self.attention(clean).squeeze(-1)
        if scores.device.type == 'cuda':scores = scores.float()
        scores = scores.masked_fill(~prefix_mask, -1e30)
        mass = (scores-scores.max(-1, keepdim=True).values).exp()*prefix_mask
        weights = mass/mass.sum(-1, keepdim=True).clamp_min(1)
        pooled = (clean*weights[..., None]).sum(-2)
        embedding = F.gelu(self.projection(pooled))
        return embedding.float() if embedding.device.type == 'cuda' else embedding

    def count_logits(self, five_embeddings):
        if five_embeddings.ndim != 3 or five_embeddings.shape[1:] != (5, self.embedding_dimension):
            raise ValueError('Exactly five actual trajectory embeddings required')
        # Population variance, denominator5, and symmetric coordinatewise max.
        mean = five_embeddings.mean(1)
        variance = (five_embeddings-mean[:, None]).square().mean(1)
        maximum = five_embeddings.max(1).values
        return self.classifier(torch.cat((mean, variance, maximum), -1))

    def probabilities(self, five_embeddings):
        counts = self.count_logits(five_embeddings).softmax(-1)
        return dict(count_probabilities=counts, mixed_probability=counts[:, 1:5].sum(-1))


def native_hidden_forward(legacy, batch, head):
    """One unchanged block forward yields both old p logits and new embedding.

    This explicit adapter reproduces Probe.forward without modifying Probe or
    installing a hook on an active model. Student features stay detached.
    """
    width = legacy.qmax+legacy.prefix
    features = batch['features'][:, :width].detach().contiguous()
    mask = batch['attention_mask'][:, :width].contiguous()
    positions = batch['position_ids'][:, :width].contiguous()
    if mask.dtype != torch.bool or not bool(mask[:, 0].all()) or features.shape[1] != width:
        raise ValueError('Original actual question/prefix geometry required')
    kwargs = dict(attention_mask=mask, position_ids=positions,
        position_embeddings=legacy.rotary(features, positions), use_cache=False, output_attentions=False)
    if legacy.cache_position:kwargs['cache_position'] = torch.arange(width, device=features.device)
    hidden = legacy.block(features, **kwargs)
    if isinstance(hidden, tuple):hidden = hidden[0]
    trajectory_logits = legacy.output(hidden.masked_fill(~mask[..., None], 0).flatten(1)).squeeze(-1).float()
    embedding = head.encode(hidden[:, legacy.qmax:width], mask[:, legacy.qmax:width])
    return trajectory_logits, embedding


def count_loss(trajectory_logits, embeddings, source_groups, views, head):
    """Mean_q sum_views(1/4 * [mean5 trajectoryBCE + true countNLL])."""
    if trajectory_logits.shape != (len(source_groups), 10) or embeddings.shape[:2] != trajectory_logits.shape:
        raise ValueError('Every original question needs its own ten real outputs')
    by = {g['qid']:i for i,g in enumerate(source_groups)}
    if len(by) != len(source_groups):raise ValueError('Original questions repeat')
    indices, targets, weights, totals = [], [], [], {q:Fraction(0) for q in by}
    labels = []
    for view in views:
        i = by[view['qid']];g = source_groups[i]
        own = [g['trajectory_ids'].index(t) for t in view['trajectory_ids']]
        ys = [g['labels'][j] for j in own]
        if (len(own) != 5 or len(set(own)) != 5 or ys != view['labels']
                or any(type(y) is not int or y not in (0,1) for y in ys)):
            raise ValueError('View labels must remain bound to five actual own trajectories')
        weight = Fraction(view['weight_numerator'], view['weight_denominator'])
        if weight <= 0:raise ValueError('Positive question-view weight required')
        totals[view['qid']] += weight;weights.append(float(weight));targets.append(sum(ys));labels.append(ys)
        indices.append([i*10+j for j in own])
    if any(v != 1 for v in totals.values()):raise ValueError('Every original question must have exactly total weight1')
    device = embeddings.device;idx = torch.tensor(indices, dtype=torch.long, device=device)
    z = trajectory_logits.reshape(-1)[idx]
    em = embeddings.reshape(-1, embeddings.shape[-1])[idx]
    own_labels = torch.tensor(labels, dtype=z.dtype, device=device)
    target = torch.tensor(targets, dtype=torch.long, device=device)
    count_logits = head.count_logits(em)
    w = torch.tensor(weights, dtype=count_logits.dtype, device=device)/len(source_groups)
    trajectory = (F.binary_cross_entropy_with_logits(z, own_labels, reduction='none').mean(-1)*w).sum()
    count = (F.cross_entropy(count_logits, target, reduction='none')*w).sum()
    probability = count_logits.softmax(-1)
    return dict(loss=trajectory+count, trajectory_loss=trajectory, count_NLL=count,
        count_logits=count_logits, mixed_probability=probability[:,1:5].sum(-1),
        true_count=target, true_mixed=(target>0)&(target<5), view_weights=w*len(source_groups),
        original_questions=len(source_groups), views=len(views), labels_used_for_scoring=False)


def initialize_head(*, dimension=3584, embedding_dimension=128, seed=1729, device=torch.device('cpu')):
    """Fresh head initialization cannot consume/reset the inherited RNG clock."""
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        head=DirectCountHead(dimension,embedding_dimension).to(device)
    ema=copy.deepcopy(head).eval().requires_grad_(False)
    optimizer=torch.optim.Adam(head.parameters(),lr=5e-6,weight_decay=0)
    return head,ema,optimizer


def optimizer_steps(optimizer):
    values={int(s['step']) for s in optimizer.state.values() if s}
    if not values:return 0
    if len(values)!=1:raise ValueError('New head Adam counters differ')
    return values.pop()


def new_head_hashes(head,ema,optimizer):
    from . import probe_ablation_train as u
    return dict(raw=u.model_hash(head),ema=u.model_hash(ema),adam=u.optimizer_hash(optimizer))


def new_head_state(head,ema,optimizer):
    return dict(raw=head.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
        steps=optimizer_steps(optimizer),accepted_hashes=new_head_hashes(head,ema,optimizer),
        new_parameters_initialized_once=True,inherited_legacy_optimizer_reset=False)


def restore_new_head(saved,head,ema,optimizer,device):
    head.load_state_dict(saved['raw'],strict=True);ema.load_state_dict(saved['ema'],strict=True)
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
    for state in optimizer.state.values():
        for name,value in state.items():
            if torch.is_tensor(value) and name!='step':state[name]=value.to(device)
    if optimizer_steps(optimizer)!=saved['steps'] or new_head_hashes(head,ema,optimizer)!=saved['accepted_hashes']:
        raise ValueError('New head full optimizer/EMA restore differs')
