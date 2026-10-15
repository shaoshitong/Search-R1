"""Opt-in question selection; the existing actor/reward recipe stays unchanged."""
import json
from pathlib import Path

FIELDS = frozenset((
    "candidate_questions", "selected_questions", "responses_per_question",
    "exploration_questions", "seed", "cache_dir", "output_dir", "snapshot_path", "initial_head_checkpoint",
    "head_learning_rate", "head_batch_size", "head_epochs_per_outer", "bootstrap_head_epochs",
    "loss_mode", "mean_loss_weight", "std_loss_weight", "count_aux_loss_weight", "replay_window_outer",
    "replay_half_life_outer", "head_hidden_size",
    "bootstrap_max_groups_per_outer", "teacher_warmup_cache",
))


def arguments(run, base):
    """Bind all selection controls in the run manifest, never in global env."""
    settings = run.get("question_probe")
    if settings is None:
        return []
    if not isinstance(settings, dict) or set(settings) - FIELDS:
        raise ValueError("Unknown question_probe manifest fields")
    base = Path(base).resolve()
    values = dict(settings)
    bootstrap_epochs = values.get("bootstrap_head_epochs")
    if bootstrap_epochs is not None and (type(bootstrap_epochs) is not int or bootstrap_epochs <= 0):
        raise ValueError("Bootstrap head epochs must be a positive integer or null")
    loss_mode = values.get("loss_mode", "count6_moments_v1")
    if loss_mode not in ("count6_moments_v1", "sigma3_std_aux6_v1"):
        raise ValueError("Unknown question-probe loss mode")
    if loss_mode == "sigma3_std_aux6_v1":
        if not {"mean_loss_weight", "std_loss_weight", "count_aux_loss_weight"} <= set(values):
            raise ValueError("Sigma3 loss must explicitly bind all loss weights")
        if values["mean_loss_weight"] != 0:
            raise ValueError("Sigma3 loss requires mean_loss_weight=0")
    values.setdefault("candidate_questions", 5120)
    values.setdefault("selected_questions", 512)
    values.setdefault("responses_per_question", 5)
    values.setdefault("exploration_questions", 64)
    values.setdefault("output_dir", str(base / "artifacts/question-probe"))
    for key in ("candidate_questions", "selected_questions", "responses_per_question",
                "exploration_questions"):
        if type(values[key]) is not int:
            raise ValueError("Question budgets must be integers: " + key)
    if values["selected_questions"] != 512 or values["responses_per_question"] != 5:
        raise ValueError("Question probe must preserve the common8 512 x 5 budget")
    if values["candidate_questions"] < 512 or not 0 <= values["exploration_questions"] <= 512:
        raise ValueError("Invalid candidate or exploration question budget")
    if not values.get("cache_dir"):
        raise ValueError("Question feature cache must be prepared and explicitly bound")
    for key in ("cache_dir", "output_dir"):
        if not Path(values[key]).is_absolute():
            raise ValueError(key + " must name an explicit absolute remote path")
    return ["data.selection_mode=question_probe"] + [
        "question_probe." + key + "=" + json.dumps(value, allow_nan=False)
        for key, value in sorted(values.items())
    ]


def validate(config, run, base):
    expected = arguments(run, base)
    if not expected:
        if config.data.get("selection_mode", "legacy") == "question_probe":
            raise ValueError("Unbound question probe cannot be admitted")
        return
    if config.data.selection_mode != "question_probe" or config.data.get("mixed_first_selection", False):
        raise ValueError("Question-probe and post-rollout mixed-first modes are mutually exclusive")
    for arg in expected[1:]:
        key, raw = arg.split("=", 1)
        actual = config.question_probe[key.split(".", 1)[1]]
        if actual != json.loads(raw):
            raise ValueError("Resolved question probe differs from manifest: " + key)
    if (config.data.train_batch_size, config.actor_rollout_ref.rollout.n_agent) != (512, 5):
        raise ValueError("Question probe changed the actor rollout budget")
