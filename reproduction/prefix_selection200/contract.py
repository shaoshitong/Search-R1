"""Shared science with an explicit four- or eight-physical-rank profile."""

import os
# An explicit process-wide profile is inherited by every native child. The
# original source arm keeps its four-rank default; the independent prepared
# environment opts into eight physical ranks before importing this package.
WORLD = int(os.environ.get('SEARCH_G1_PHYSICAL_WORLD_SIZE', '4'))
if WORLD not in (4, 8):
    raise ValueError('Only the original four-rank or prepared eight-rank profile is supported')
from .b_progressive import retained_prefix
PREFIX_TOKENS = retained_prefix()
QMAX = 273
DIM = 2048
WIDTH = QMAX + PREFIX_TOKENS
G = 5
PERIODIC_CANDIDATES = 2560
ORDINARY_CANDIDATES = 5120
# Historical/probe-update compatibility name; never use for an ordinary pool.
CANDIDATE_QUESTIONS = PERIODIC_CANDIDATES
SELECTED_QUESTIONS = 512
FINAL_OUTER = 200
PROBE_PERIOD = 10
PROBE_UPDATES = 50
PROBE_LR = 5e-6
SAMPLING_CONTRACT = 'chunked_native_action500_prefix5_v1'
HIDDEN_CAPTURE_CONTRACT = 'sample%d_keep%d_postblock27_native_residual_sum_v1' % (PREFIX_TOKENS+1,PREFIX_TOKENS)
