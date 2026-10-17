"""Explicit source-compatibility envelope around the byte-identical actor.

The mathematical actor module is unchanged. Only its legacy source binding and
new metadata envelopes distinguish immutable actor science from the new runtime
controller and the separately authorized candidate schedule delta.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

from . import actor
from .i1_i6_joint_resume import (read, verify_source_compatibility, scientific_binding,
                               source_envelope)


@contextmanager
def binding_context(config, *, actor_module=actor, state_module=None):
    proof=verify_source_compatibility(config);envelope=source_envelope(config,proof)
    if state_module is None:
        from search_g1 import training_state as state_module
    old_binding=actor_module._state_binding;old_json=actor_module.atomic_json;old_save=state_module.atomic_save
    def binding(current):
        if (current.get('source_compatibility_sha256')!=config['source_compatibility_sha256'] or
                current.get('source_compatibility')!=config['source_compatibility']):
            raise ValueError('Actor attempt changed its explicit scientific source compatibility proof')
        return scientific_binding(current,original_binding=old_binding,proof=proof)
    def annotate(value):
        if not isinstance(value,dict) or value.get('run_id')!=config['run_id']:return value
        # Shallow metadata extension: optimizer/model/RNG/scheduler objects are
        # passed through untouched, and old checkpoint blobs are never rewritten.
        return dict(value,**envelope)
    def atomic_json(path,value,**kwargs):return old_json(path,annotate(value),**kwargs)
    def atomic_save(path,value):return old_save(path,annotate(value))
    actor_module._state_binding=binding;actor_module.atomic_json=atomic_json;state_module.atomic_save=atomic_save
    try:yield envelope
    finally:
        actor_module._state_binding=old_binding;actor_module.atomic_json=old_json;state_module.atomic_save=old_save


@contextmanager
def control_binding_context(config):
    """Metadata-only parent helpers must interpret the same explicit binding."""
    from . import guard_retry
    proof=verify_source_compatibility(config)
    old_binding=actor._state_binding;old_guard_binding=guard_retry._state_binding
    def binding(current):return scientific_binding(current,original_binding=old_binding,proof=proof)
    actor._state_binding=binding;guard_retry._state_binding=binding
    try:yield
    finally:actor._state_binding=old_binding;guard_retry._state_binding=old_guard_binding


def collect_oom_retry(output,config_path,*args,**kwargs):
    config=read(config_path)
    with control_binding_context(config):return actor.collect_oom_retry(output,config_path,*args,**kwargs)


def archive_zero_write_attempt(base,config,*args,**kwargs):
    from .guard_retry import archive_zero_write_attempt as archive
    with control_binding_context(config):return archive(base,config,*args,**kwargs)


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--config',required=True)
    args,_=parser.parse_known_args();config=read(Path(args.config))
    with binding_context(config):actor.main()


if __name__=='__main__':main()
