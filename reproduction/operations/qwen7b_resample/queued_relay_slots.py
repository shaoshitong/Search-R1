"""Finite future node10 co-share slots; a declaration never proves GPU fit.

The downstream owns all eight primary leases. Each overlapping model slot
borrows only its declared per-slot leases and must pass its real GPU profile.
Existing owner registries and immutable stages are not edited.
"""
import copy

TP1_SLOTS = {'gpu%d-tp1' % index: dict(indices=[index], tp=1, seqs=64, tokens=8192, blocks=4096, reserve=1.0)
             for index in (1, 2, 3, 5)}
DEFAULT_SLOTS = ('tp4-0-3', 'tp4-4-7', 'gpu1-tp1', 'gpu2-tp1', 'gpu3-tp1', 'gpu5-tp1')


def specifications(original):
    value = copy.deepcopy(original)
    value.update(copy.deepcopy(TP1_SLOTS))
    assert set(value) == set(DEFAULT_SLOTS)
    return value


def install(owner):
    owner.SLOTS = specifications(owner.SLOTS)
    owner.DEFAULT_SLOTS = DEFAULT_SLOTS
