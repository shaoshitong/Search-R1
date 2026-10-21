"""Fixed full-data comparison cadence; no models, data access or submission.

The existing124194 trial is immutable. This separate169028 trial uses the
same E4 optimizer initialization, native updater, loss, LR and real-G5 sampler.
Integer-epoch nodes and the four original trial update counts are both scored.
Final4096 remains outside this protocol and outside checkpoint selection.
"""
from pathlib import Path

QUESTIONS = 169028
QUESTIONS_PER_UPDATE = 32
STEPS_PER_EPOCH = (QUESTIONS + QUESTIONS_PER_UPDATE - 1) // QUESTIONS_PER_UPDATE
EPOCHS = 4
UPDATE_BUDGET = STEPS_PER_EPOCH * EPOCHS
MATCHED_UPDATES = (3882, 7764, 11646, 15528)
BASE = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-169028-v1')


def evaluation_node(cursor):
    if type(cursor) is not int or not 0 <= cursor <= UPDATE_BUDGET:
        raise ValueError('Actual accepted full-data cursor required')
    if cursor % STEPS_PER_EPOCH == 0:
        epoch = cursor // STEPS_PER_EPOCH
        return dict(kind='integer_epoch', epoch=epoch, cursor=cursor,
                    folder='epoch-%02d' % epoch)
    if cursor in MATCHED_UPDATES:
        return dict(kind='matched_124194_update_budget', epoch=None, cursor=cursor,
                    matched_reference_epoch=MATCHED_UPDATES.index(cursor) + 1,
                    folder='matched-update-%06d' % cursor)
    return None


def pinned(cursor):
    return cursor in (0, 1) or evaluation_node(cursor) is not None


def required_past_nodes(cursor):
    evaluation_node(cursor)  # Validate the actual resume cursor even off-node.
    nodes = set(range(0, UPDATE_BUDGET + 1, STEPS_PER_EPOCH)) | set(MATCHED_UPDATES)
    return [evaluation_node(v) for v in sorted(nodes) if v < cursor]


def validate_config(base, config):
    if (Path(base) != BASE or config['base'] != str(BASE)
            or config['mode'] != 'main' or config['fit_questions'] != QUESTIONS
            or config['fit_real_trajectories'] != 10 * QUESTIONS
            or config['steps_per_epoch'] != STEPS_PER_EPOCH
            or config['epochs'] != EPOCHS or config['update_budget'] != UPDATE_BUDGET
            or config['matched_evaluation_updates'] != list(MATCHED_UPDATES)
            or config['save_every'] != 128 or config['current_world_size'] != 1
            or config['lr'] != 5e-6 or config['ema_decay'] != .99
            or config['loss_kwargs'] != dict(trajectory_weight=1., binary_mixed_weight=1.)
            or config['source_checkpoint']['sha256'] != 'e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8'
            or config['source_checkpoint']['steps'] != 14418):
        raise ValueError('Independent full169028 same-E4/native/sampler comparison required')
