"""Dimensions for actual native question273+policy20 frozen-actor rollout."""
WORLD=4
PREFIX_TOKENS=20
QMAX=273
DIM=2048
WIDTH=QMAX+PREFIX_TOKENS
G=5
PERIODIC_CANDIDATES=512
SELECTED_QUESTIONS=512
SAMPLING_CONTRACT='chunked_native_action500_prefix20_v1'
HIDDEN_CAPTURE_CONTRACT='sample21_keep20_postblock27_native_residual_sum_v1'
