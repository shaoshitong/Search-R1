"""Small CPU arithmetic/EMA/whole-G5 reduction checks; loads no LLM or dataset.

Run: python -m reproduction.question20_ema_warmup.selfcheck
"""

import copy
import datetime
import json
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F

from .checkpoint import capture_rng, make_checkpoint, restore_checkpoint, restore_rng
from .ema import ParameterEMA
from .loss import joint_loss, mixed_event_logits
from .metrics import GroupMetrics
from .model import causal_padding_mask
from .training import TrainCounters, build_epoch_plan, train_update, validate_groups


class TinyProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output = torch.nn.Linear(3, 1)
        self.register_buffer("integer_buffer", torch.tensor(7, dtype=torch.long))
        self.register_buffer("native_nonpersistent", torch.tensor([1.5]), persistent=False)

    def forward(self, batch):
        if batch is None:
            return sum(p.reshape(-1)[0] * 0 for p in self.parameters())
        return self.output(batch["features"]).reshape(-1)


class TinyReader:
    def __init__(self, groups):
        generator = torch.Generator().manual_seed(19)
        self.index = {}
        for group in groups:
            for tid, label in zip(group["trajectory_ids"], group["labels"]):
                self.index[tid] = (torch.randn(3, generator=generator), label)

    def get_batch(self, ids, device=None):
        return dict(features=torch.stack([self.index[tid][0] for tid in ids]).to(device),
                    labels=torch.tensor([self.index[tid][1] for tid in ids], device=device),
                    trajectory_ids=ids)


def _groups(n):
    return [dict(event_group_id="event-%d" % i, qid="q-%d" % i,
                 trajectory_ids=["e%d-r%d" % (i, j) for j in range(5)],
                 labels=[int(j < (i % 6)) for j in range(5)]) for i in range(n)]


def _new(seed=23):
    torch.manual_seed(seed)
    model = TinyProbe()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=0)
    ema = ParameterEMA(model)
    return model, optimizer, ema, TrainCounters()


def _assert_states(left, right):
    for name, value in left.state_dict().items():
        assert torch.allclose(value, right.state_dict()[name], atol=5e-7, rtol=1e-6), name


def _ddp_worker(rank, world, rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                            world_size=world, timeout=datetime.timedelta(seconds=45))
    try:
        # Step1 has unequal 3/2 groups with unequal micro counts. Step2 has one
        # group and one genuinely empty rank; no fabricated duplicate data.
        groups = _groups(6)
        reader = TinyReader(groups)
        model, optimizer, ema, counters = _new()
        model = DistributedDataParallel(model, broadcast_buffers=False)
        reference, ropt, rema, _ = _new()
        plan = build_epoch_plan(6, epoch=1, seed=29, world=world, global_group_pattern=(5,))
        rplan = build_epoch_plan(6, epoch=1, seed=29, world=1, global_group_pattern=(5,))
        for update_number, (step, rstep) in enumerate(zip(plan, rplan), start=1):
            train_update(model, optimizer, ema, counters, groups, step, rank=rank,
                         reader=reader, micro_groups=1, device="cpu", autocast_dtype=None)
            selected = [groups[i] for i in rstep.group_indices]
            ids = [tid for group in selected for tid in group["trajectory_ids"]]
            target = torch.tensor([group["labels"] for group in selected], dtype=torch.float32)
            ropt.zero_grad(set_to_none=True)
            direct_logits = reference(reader.get_batch(ids)).reshape(len(selected), 5)
            joint_loss(direct_logits, target)["loss"].backward()
            ropt.step()
            rema.update(reference, update_number)
            _assert_states(model.module, reference)
            _assert_states(ema.model, rema.model)
        assert counters.optimizer_steps == counters.ema_updates == 2
        assert counters.global_groups_seen == 6 and counters.global_trajectories_seen == 30
    finally:
        dist.destroy_process_group()


def main():
    torch.set_num_threads(1)
    z = torch.tensor([[-2., -.5, 0., 1., 2.], [1., 2., 3., 4., 5.]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[1., 0., 0., 0., 0.], [1., 1., 1., 1., 1.]], dtype=torch.float64)
    values = joint_loss(z, labels)
    p = z.sigmoid()
    direct_h = 1 - p.prod(dim=1) - (1 - p).prod(dim=1)
    direct = F.binary_cross_entropy_with_logits(z, labels) + F.binary_cross_entropy(direct_h, values["mixed_label"])
    assert torch.allclose(values["loss"], direct, atol=1e-12)
    grad = torch.autograd.grad(values["loss"], z, retain_graph=True)[0]
    assert torch.allclose(grad, torch.autograd.grad(direct, z)[0], atol=1e-12)
    assert values["mixed_label"].tolist() == [1., 0.]
    assert torch.allclose(mixed_event_logits(torch.zeros((1, 5))).sigmoid(), torch.tensor([30 / 32]))
    extreme = torch.tensor([[1000.] * 5, [-1000.] * 5, [1000., -1000., -1000., -1000., -1000.]], requires_grad=True)
    loss = joint_loss(extreme, torch.tensor([[0.] * 5, [1.] * 5, [1., 0., 0., 0., 0.]]))["loss"]
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(extreme.grad).all()
    assert extreme.grad[0].abs().sum() > 0 and extreme.grad[1].abs().sum() > 0
    mask = torch.tensor([[True, True, False, False, True]])
    additive = causal_padding_mask(mask, torch.float32)[0, 0]
    assert additive[4, 0] == 0 and additive[4, 4] == 0 and additive[4, 2] < -1e30
    assert additive[0, 4] < -1e30
    groups = _groups(103)
    assert validate_groups(groups) == dict(groups=103, trajectories=515)
    plan = build_epoch_plan(103, epoch=1, seed=31, world=4)
    assert [step.global_groups for step in plan] == [51, 51, 1]
    assert sorted(i for step in plan for i in step.group_indices) == list(range(103))
    model, optimizer, ema, counters = _new()
    initial = copy.deepcopy(model.state_dict())
    group = _groups(3)
    reader = TinyReader(group)
    step = build_epoch_plan(3, epoch=1, seed=1, global_group_pattern=(3,))[0]
    train_update(model, optimizer, ema, counters, group, step, rank=0, reader=reader,
                 micro_groups=1, device="cpu", autocast_dtype=None)
    for name, value in ema.model.state_dict().items():
        expected = initial[name] * .99 + model.state_dict()[name] * .01 if value.is_floating_point() else initial[name]
        assert torch.allclose(value, expected, atol=1e-7), name
    try:
        ema.update(model, 1)
        raise AssertionError("Repeated EMA step was incorrectly accepted")
    except ValueError:
        pass
    metrics = GroupMetrics()
    metrics.add(z.detach(), labels)
    result = metrics.result(weight_kind="ema0.99")
    assert result["groups"] == 2 and result["trajectories"] == 10
    assert result["mixed_group"]["positive"] == 1
    rng = capture_rng(rank=0, sampler_state=dict(epoch=1, next_step=1))
    saved = make_checkpoint(model, optimizer, ema, counters, binding=dict(run_id="tiny-new-run"),
                            epoch=1, next_update_in_epoch=1, epoch_plan_sha256="a" * 64, rank_rng=[rng])
    resumed, resumed_opt, resumed_ema, _ = _new(101)
    resumed_counters = restore_checkpoint(saved, resumed, resumed_opt, resumed_ema,
                                          expected_binding=dict(run_id="tiny-new-run"), device="cpu")
    _assert_states(model, resumed)
    _assert_states(ema.model, resumed_ema.model)
    assert torch.equal(model.native_nonpersistent, resumed.native_nonpersistent)
    assert torch.equal(ema.model.native_nonpersistent, resumed_ema.model.native_nonpersistent)
    assert resumed_counters.state_dict() == counters.state_dict()
    assert restore_rng(rng) == dict(epoch=1, next_step=1)
    import torch.multiprocessing as mp
    with tempfile.TemporaryDirectory(prefix="q20-ema-cpu-") as temp:
        mp.spawn(_ddp_worker, args=(2, str(Path(temp) / "gloo")), nprocs=2, join=True)
    print(json.dumps(dict(status="PASS", checks=["joint_math_and_gradient", "extreme_logits_finite_no_clamp",
          "original_causal_padding_mask", "exact_once_whole_G5_plan", "EMA_exact_initialization_and_single_update",
          "distinct_group_and_trajectory_metrics", "raw_EMA_Adam_RNG_checkpoint_roundtrip",
          "two_rank_unequal_micro_and_empty_final_rank_equal_global_mean"], gpu_tests=0,
          llm_loads=0, dataset_reads=0), sort_keys=True))


if __name__ == "__main__":
    main()
