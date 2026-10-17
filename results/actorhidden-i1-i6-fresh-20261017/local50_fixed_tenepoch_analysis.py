"""CPU-only analysis of completed local50 predictions; return aggregates only."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect(base, helper_directory):
    import sys
    base, helper_directory = Path(base), Path(helper_directory)
    sys.path.insert(0, str(base/'repo'))
    audit = load_module(helper_directory/'probability_distribution_audit.py', 'existing_probability_audit')
    conflict = load_module(helper_directory/'local50_identical_input_analysis.py', 'local50_input_conflicts')
    epochs = []
    identities = {}
    for epoch in range(11):
        metrics_path = base/'train'/('epoch-%02d-metrics.json'%epoch)
        metrics = json.loads(metrics_path.read_text())
        assert metrics['epoch'] == epoch and metrics['actual_adam_per_rank'] == [200+epoch*100]*8
        item = dict(epoch=epoch, metrics=metrics, metrics_sha256=audit.sha(metrics_path), splits={})
        for split, count in (('train', 5120), ('test', 512)):
            path = base/'train'/('epoch-%02d-%s-predictions.jsonl'%(epoch, split))
            rows = audit.rows(path)
            assert len(rows) == count
            signature = sorted((str(r['qid']), tuple(r['trajectory_ids']), tuple(r['strict_em'])) for r in rows)
            if epoch == 0:
                identities[split] = signature
            else:
                assert signature == identities[split], 'Comparison must use identical trajectories and labels'
            summary = audit.summarize(rows)
            for k in (0, 5):
                values = [p for row in rows if sum(row['strict_em']) == k for p in audit.probabilities(row)]
                if values:
                    summary['by_correct_count'][str(k)]['fraction_p_between_012_022'] = audit.mean([.12 <= p <= .22 for p in values])
            ps = [p for row in rows for p in audit.probabilities(row)]
            ys = [y for row in rows for y in row['strict_em']]
            summary['high_confidence_wrong_fraction'] = audit.mean([(p >= .9 and y == 0) or (p <= .1 and y == 1) for p,y in zip(ps,ys)])
            item['splits'][split] = dict(summary, source_sha256=audit.sha(path))
        epochs.append(item)
    config = json.loads((base/'preparation/launch-config.json').read_text())
    data_base = Path(config['local_probe_fit50'].get('fixed_data_source_base', base))
    root = data_base/'outers/outer-000051'
    diagnostic = json.loads((root/'diagnostic-complete.json').read_text())
    scan = Path(diagnostic['scan_path'])
    conflicts = dict(train=conflict.inspect(root/'prefix/scan', root/'full/own-em.jsonl'),
                     test=conflict.inspect(scan, scan.parent.parent/'full/own-em.jsonl'))
    loss_path = base/'train/global-loss-updates.jsonl'
    updates = audit.rows(loss_path)
    assert len(updates) == 1000 and [r['actual_adam'] for r in updates] == list(range(201,1201))
    return dict(base=str(base), actor_updated=False, same_fixed_inputs_and_labels_verified=True,
                epochs=epochs, input_label_conflicts=conflicts, online_step_loss=updates,
                loss_updates_sha256=audit.sha(loss_path), raw_predictions_returned=False,
                tensor_loaded=False, model_forward_performed=False, training_modified=False,
                interpretation='One fixed-data LR1e-5 experiment; no causal LR/data/loss ablation performed')


def synthetic_loss_check(repo):
    """Confirm an analytic bad stationary point in the actual objective, CPU only."""
    import sys
    sys.path.insert(0,str(repo))
    import torch
    from reproduction.prefix20_group_joint100.loss import joint_loss
    p = .16693584131290795
    z = torch.full((1,5), math.log(p/(1-p)), dtype=torch.float64, requires_grad=True)
    y = torch.ones_like(z)
    def objective(value):
        return joint_loss(value.reshape(1,5), y)['loss']
    loss = objective(z)
    gradient = torch.autograd.grad(loss,z)[0]
    hessian = torch.autograd.functional.hessian(objective,z.reshape(5))
    losses = joint_loss(z,y)
    return dict(synthetic_not_model_measurement=True, probability=p, labels=[1]*5,
                true_mixed=0, predicted_mixed=float(losses['mixed_probability'].detach()),
                trajectory_bce=float(losses['trajectory_bce'].detach()),
                group_bce=float(losses['group_bce'].detach()), total_loss=float(loss.detach()),
                gradient_max_abs=float(gradient.abs().max()),
                logit_hessian_eigenvalues=torch.linalg.eigvalsh(hessian).tolist(),
                actual_frozen_loss_code_sha256=hashlib.sha256((Path(repo)/'reproduction/prefix20_group_joint100/loss.py').read_bytes()).hexdigest(),
                actual_model_trapped_here_not_established=True)
