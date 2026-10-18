"""Preserve actor50's exact scientific optimizer binding, with explicit provenance."""
import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path

from reproduction.prefix_selection200 import actor

ACTOR_SOURCE_SHA256='7f4bbcdbd23648569afe8ee8d1b35066e46d8da72709e765cb6f5fc0e16943de'


@contextmanager
def binding_context(config):
    from search_g1 import training_state
    from verl.trainer.main_ppo_eco import EcoRewardManager
    from .reward_transport import chunked_call
    assert hashlib.sha256(Path(actor.__file__).read_bytes()).hexdigest()==ACTOR_SOURCE_SHA256
    retained=config['retained_actor50_binding'];original=actor._state_binding
    old_json=actor.atomic_json;old_save=training_state.atomic_save
    old_probe_call=EcoRewardManager._probe_call
    rpc_batch=config.get('grounding_rpc_batch',32)
    if rpc_batch!=32:raise ValueError('Bind the aligned32-item required-reward transport repair')
    def binding(current):
        actual=original(current)
        if (current['retained_actor50_binding']!=retained or set(actual)!=set(retained)
                or any(actual[k]!=retained[k] for k in actual if k!='source_commit')):
            raise ValueError('The actor objective, inputs, budget, safeguards or lineage changed')
        return dict(retained)
    def annotate(value):
        if not isinstance(value,dict) or value.get('run_id')!=config['run_id']:return value
        return dict(value,runtime_source_commit=config['source_commit'],
            science_source_commit=retained['source_commit'],
            actor50_branch_schedule=config['actor50_branch_schedule'],
            grounding_rpc_items=rpc_batch,required_reward_transport_only=True,
            source_warmup_sha256=config['probe_source_sha256'])
    actor._state_binding=binding
    actor.atomic_json=lambda path,value,**kw:old_json(path,annotate(value),**kw)
    training_state.atomic_save=lambda path,value:old_save(path,annotate(value))
    EcoRewardManager._probe_call=lambda owner,items:chunked_call(old_probe_call,owner,items,rpc_batch)
    try:yield
    finally:
        actor._state_binding=original;actor.atomic_json=old_json;training_state.atomic_save=old_save
        EcoRewardManager._probe_call=old_probe_call


def collect_oom_retry(output,config_path,*args,**kwargs):
    config=actor.read(config_path)
    with binding_context(config):return actor.collect_oom_retry(output,config_path,*args,**kwargs)


def archive_zero_write_attempt(base,config,*args,**kwargs):
    from reproduction.prefix_selection200 import guard_retry
    original=guard_retry._state_binding
    with binding_context(config):
        guard_retry._state_binding=actor._state_binding
        try:return guard_retry.archive_zero_write_attempt(base,config,*args,**kwargs)
        finally:guard_retry._state_binding=original


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--config',required=True)
    a,_=p.parse_known_args();config=actor.read(a.config)
    with binding_context(config):actor.main()


if __name__=='__main__':main()
