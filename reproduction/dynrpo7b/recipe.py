"""Confirmed algorithm constants, separate from infrastructure flags."""
from pathlib import Path

MODEL = '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/assets/qwen25-7b-instruct-a09a35458c70'
WARMUP = ('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/'
          'algorithm-improvement-20261021/full169028-L15P80-MSE-ownG5-physical-seed9137-plan2026102227-20261023-v3/checkpoint-epoch-09.pt')
WARMUP_SHA = '5f90fb17bee993353f47a7d687bdf2ea235dcd3d2eba287465567f4c286b13d7'
REWARD_SOURCE = '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021'
TOTAL_OUTERS, G, SELECTED, CANDIDATES, CONTINUATIONS = 200, 5, 512, 5120, 1024
UPDATE_CANDIDATES, UPDATE_EPOCHS = 2560, 3
MILESTONES = (5, 50, 100, 150, 200)


def probe_update(outer):
    return 0 < outer < TOTAL_OUTERS and outer % 10 == 0


def prefix_tokens(outer):
    return 80 if outer <= 100 else 160


def fit_prefix_tokens(completed_outer):
    return prefix_tokens(completed_outer + 1)


def probe_settings(base):
    return dict(probe_architecture='qwen7b_L15_packed_group_variance_v1',
                model_dir=MODEL, checkpoint_path=WARMUP, checkpoint_sha256=WARMUP_SHA,
                output_dir=str(Path(base) / 'artifacts/question-probe'),
                selection_state='raw', learning_rate=5e-6, ema_decay=.9995,
                fit_epochs=UPDATE_EPOCHS, fit_global_batch=128, fit_micro=8)
