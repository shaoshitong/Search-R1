"""Shared scientific dimensions for the four-GPU question+prefix5 variant."""

import os

WORLD = 4
# A new immutable runtime opts in before importing any model/capture module.
# Legacy checkouts and workers retain the original five-token contract.
ACTOR50_BRANCH = os.environ.get('SEARCH_G1_GRPO_FROM50') == '1'
PREFIX_TOKENS = int(os.environ.get('SEARCH_G1_GRPO_PREFIX', '20')) if ACTOR50_BRANCH else 5
if ACTOR50_BRANCH and PREFIX_TOKENS not in (20, 40):
    raise ValueError('The actor50 branch admits only actual Prefix20 or Prefix40')
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
SAMPLING_CONTRACT = 'chunked_native_action500_prefix%d_v1' % PREFIX_TOKENS
HIDDEN_CAPTURE_CONTRACT = 'sample%d_keep%d_postblock27_native_residual_sum_v1' % (PREFIX_TOKENS+1, PREFIX_TOKENS)
