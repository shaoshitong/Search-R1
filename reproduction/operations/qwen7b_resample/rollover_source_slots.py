"""Finite next-generation source slots; declarations do not prove GPU fit.

Only new staged entry processes install this copied registry. Existing owner
and consumer sources, stages and running registries remain unchanged.
"""
import copy

TP1_SLOTS = {'gpu%d-tp1' % index: dict(indices=[index], tp=1, seqs=32, tokens=8192, blocks=4096, reserve=1.0)
             for index in (3, 4, 6, 7)}
GPU3_SLOT = TP1_SLOTS['gpu3-tp1']
DEFAULT_SLOTS = ('base-tp4', 'retrieval-tp4', 'gpu1-tp1', 'gpu2-tp1',
                 'gpu3-tp1', 'gpu4-tp1', 'gpu6-tp1', 'gpu7-tp1')


def specifications(original):
    value = copy.deepcopy(original)
    value.update(copy.deepcopy(TP1_SLOTS))
    assert set(value) == set(DEFAULT_SLOTS)
    return value


def install(owner):
    owner.SLOTS = specifications(owner.SLOTS)

