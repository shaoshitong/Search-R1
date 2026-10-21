"""Explicit optional infrastructure flags; immutable run manifests choose them."""
import json
import os
from pathlib import Path

PROFILES = ('baseline', 'sparse-exact-v1', 'exact-cache-v1', 'fa3-exact-cache-v1')


def profile_flags(profile):
    if profile not in PROFILES:
        raise ValueError('Unknown Easy7B infrastructure profile')
    if profile == 'fa3-exact-cache-v1':
        return dict(selected_policy_logits=False,exact_retrieval_cache=True,flash_attention_3_actor=True)
    return dict(selected_policy_logits=profile == 'sparse-exact-v1',
                exact_retrieval_cache=profile in ('sparse-exact-v1', 'exact-cache-v1'))


def flags(run=None):
    if run is None:
        base = os.environ.get('SEARCH_G1_BASE')
        p = Path(base)/'logs/run-manifest.json' if base else None
        run = json.loads(p.read_text()) if p and p.is_file() else {}
    values = run.get('infra_optimizations', {})
    if not values:
        return dict(selected_policy_logits=False, exact_retrieval_cache=False)
    original = {'selected_policy_logits', 'exact_retrieval_cache'}
    if set(values) not in (original,original|{'flash_attention_3_actor'}):
        raise ValueError('Unknown optional Easy7B infrastructure flags')
    if any(type(v) is not bool for v in values.values()):
        raise ValueError('Infrastructure flags must be explicit booleans')
    if not any(values.values()):
        return dict(values)
    from reproduction.common8.model_identity import SEVEN, identity
    if identity(run) != SEVEN:
        raise ValueError('Optional infrastructure candidate is scoped to Easy7B')
    if values.get('flash_attention_3_actor'):
        from reproduction.common8.precision_contract import easy2560_native4
        if not easy2560_native4(run):raise ValueError('FA3 is scoped to the native4 Easy7B actor')
    return dict(values)
