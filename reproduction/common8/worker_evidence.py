"""Durable per-rank scalar evidence, independent of Ray console forwarding."""
import json
import math
import os
from pathlib import Path

from reproduction.common8.topology import monitor_trajectories_per_rank, ranks
from reproduction.common8.safety_calibration_evidence import objective_weight_definition
from search_g1.monitor_forward_reuse import summarize_monitor_execution


def append(kind, payload, *, collective=False, group=None):
    """A receipt failure is fatal on every rank before later model collectives."""
    if not collective:
        return _append_local(kind, payload)
    import torch
    import torch.distributed as dist
    failure = None
    try:
        _append_local(kind, payload)
    except Exception as error:
        failure = type(error).__name__+': '+str(error)
    device = ('cuda:'+str(torch.cuda.current_device()) if dist.is_initialized() and
              dist.get_backend(group)=='nccl' else 'cpu')
    accepted = torch.tensor(int(failure is None), dtype=torch.int32, device=device)
    if dist.is_initialized():
        dist.all_reduce(accepted, op=dist.ReduceOp.MIN, group=group)
    if not accepted.item():
        raise RuntimeError('Fatal durable worker receipt failure: '+kind+'; local='+str(failure))


def _append_local(kind, payload):
    base=Path(os.environ['SEARCH_G1_BASE'])
    run=json.loads((base/'logs/run-manifest.json').read_text())
    assert run['profile']=='common8'
    rank=payload['rank'];assert type(rank) is int and rank in ranks(run)
    row=dict(kind=kind,run_id=run['run_id'],**payload)
    path=base/'logs'/f'optimizer-rank-{rank}.jsonl'
    fd=os.open(path,os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o600)
    try:
        data=(json.dumps(row)+'\n').encode()
        written=os.write(fd,data)
        assert written==len(data)
        os.fsync(fd)
        parent_fd=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(parent_fd)
        finally:os.close(parent_fd)
    finally:os.close(fd)


def read_console_form(base, run_id):
    run=json.loads((Path(base)/'logs/run-manifest.json').read_text())
    if run.get('easy2560_selection') is True:
        from reproduction.common8.precision_contract import easy2560_native4
        from reproduction.common8.topology import formal_monitor_geometry
        assert easy2560_native4(run)
        geometry = formal_monitor_geometry(run)
    lines=[]
    for rank in ranks(run):
        path=Path(base)/'logs'/f'optimizer-rank-{rank}.jsonl'
        monitors=[];budgets=[];opportunities=[]
        for line in path.read_text().splitlines():
            row=json.loads(line)
            assert row.pop('run_id')==run_id and row['rank']==rank
            kind=row.pop('kind')
            if kind=='POLICY_MONITOR':
                assert isinstance(row['allowed'],bool) and row['optimizer_writes']>=0
                if run.get('easy2560_selection') is True:
                    assert row['objective_weight_definition'] == objective_weight_definition(run)
                    identity = row['monitor_identity']
                    assert len(identity['selection_indices']) == len(set(identity['selection_indices'])) == 8
                    assert len(identity['actor_token_lengths']) == 8
                    if row['kl_estimate_complete']:
                        assert math.isclose(row['effective_weight_mass'],
                            geometry['expected_complete_effective_weight_mass'], rel_tol=1e-12, abs_tol=1e-12)
                monitors.append(row)
                continue
            if kind=='OUTER_BUDGET':budgets.append(row)
            if kind=='OPPORTUNITY':
                assert run.get('policy_safety_mode')=='inline_first_five'
                opportunities.append(row)
                lines.append('SEARCH_G1_OPPORTUNITY='+json.dumps(row))
                continue
            assert kind in ('OPTIMIZER_WRITE','OUTER_BUDGET','ABORTED_ATTEMPT')
            if kind=='ABORTED_ATTEMPT':
                assert run.get('policy_safety_mode')=='inline_first_five'
                assert row['recoverable_guard'] is True and row['complete_outer'] is False
                assert row['scheduler_advances']==0 and 0<=row['optimizer_writes']<=20
                lines.append('SEARCH_G1_ABORTED_ATTEMPT='+json.dumps(row))
                continue
            lines.append('SEARCH_G1_'+kind+'='+json.dumps(row))
        if run.get('policy_safety_required',False):
            for budget in budgets:
                selected=[x for x in monitors if x['context']['logical_update']==budget['logical_update'] and
                          (run.get('policy_safety_mode')!='inline_first_five' or
                           x['context']['attempt_id']==budget['attempt_id'])]
                execution=summarize_monitor_execution(selected)
                for name in ('actual_monitor_forward_count','reused_monitor_forward_count'):
                    if name in budget: assert budget[name]==execution[name]
                inline=run.get('policy_safety_mode')=='inline_first_five'
                version=budget.get('recovery_protocol_version',1)
                if version==2:
                    selected_opportunities=[x for x in opportunities if x['context']['attempt_id']==budget['attempt_id']]
                    processed=budget['opportunities_processed']
                    assert [x['opportunity_index'] for x in selected_opportunities]==list(range(processed))
                    assert 1<=processed<=20 and 1<=budget['optimizer_writes']<=processed
                    assert sum(x['optimizer_write_succeeded'] for x in selected_opportunities)==budget['optimizer_writes']
                    assert budget['full_budget_complete']==(budget['optimizer_writes']==20)
                    if run.get('extra_policy_monitor_forwards',True):
                        assert len(selected)==sum(2 if x['guard_stage']=='after_minibatch' or x['status']=='written' else 1
                            for x in selected_opportunities)
                        assert all(x['monitor_identity']==selected[0]['monitor_identity'] for x in selected)
                    else:
                        from search_g1.actor_performance import lean_run
                        assert run.get('finite_thresholds_diagnostic_only') is True or lean_run(run)
                        assert selected==[] and budget.get('extra_policy_monitor_forwards') is False
                        assert budget['actual_monitor_forward_count']==budget['reused_monitor_forward_count']==0
                    continue
                assert len(selected)==40,'Missing pre/post policy monitors for completed outer'
                if inline:
                    selected_opportunities=[x for x in opportunities if x['context']['attempt_id']==budget['attempt_id']]
                    assert [x['opportunity_index'] for x in selected_opportunities]==list(range(20))
                    assert budget['opportunities_processed']==20 and 1<=budget['optimizer_writes']<=20
                    assert sum(x['optimizer_write_succeeded'] for x in selected_opportunities)==budget['optimizer_writes']
                start=budget['cumulative_writes']-budget['optimizer_writes']
                identity=selected[0]['monitor_identity'];calibration=selected[0]['calibration_sha256']
                count=monitor_trajectories_per_rank(run)
                assert len(identity['selection_indices'])==len(set(identity['selection_indices']))==count
                assert len(identity['actor_token_lengths'])==count and all(x>0 for x in identity['actor_token_lengths'])
                for index,row in enumerate(selected):
                    mini=index//2;post=index%2
                    assert row['context']['minibatch_index']==mini
                    assert row['stage']==('after_minibatch' if post else 'before_minibatch')
                    if inline:
                        opportunity=selected_opportunities[mini]
                        assert row['optimizer_writes']==opportunity['optimizer_writes_after' if post else 'optimizer_writes_before']
                        if opportunity['status']=='written':
                            assert row['allowed'] is True and row['kl_estimate_complete'] is True
                        else:
                            assert opportunity['status'] in ('skipped_guard','written_with_post_guard')
                            assert isinstance(row['allowed'],bool)
                    else:
                        assert row['optimizer_writes']==start+mini+post
                        assert row['allowed'] is True and row['kl_estimate_complete'] is True
                    assert row['calibration_sha256']==calibration and row['monitor_identity']==identity
                    assert row['objective_weight_definition'] == objective_weight_definition(run)
    return '\n'.join(lines)
