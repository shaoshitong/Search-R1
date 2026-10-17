"""Shared scientific dimensions for the four-GPU question+prefix5 variant."""

WORLD = 4
PREFIX_TOKENS = 5
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
HIDDEN_CAPTURE_CONTRACT = 'sample6_keep5_postblock27_native_residual_sum_v1'
