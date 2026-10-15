"""Question-only, uniform-first selection before any real rollout is observed.

Only question identities and a six-class predicted count distribution enter
this module. Strict correctness is deliberately absent from its public DTO.
"""
from dataclasses import dataclass
import hashlib

import numpy as np

from search_g1.mixed_selection import question_identity


COUNT_MEANS = np.arange(6, dtype=np.float64) / 5
COUNT_VARIANCES = np.arange(6, dtype=np.float64) * (5 - np.arange(6)) / 20
COUNT_STDS = np.sqrt(COUNT_VARIANCES)


@dataclass(frozen=True)
class PredictQuestion:
    """No answer, reward, teacher trajectory, or correctness is available here."""
    ordinal: int
    qid: str
    question: str

    @property
    def question_hash(self):
        # Cache identity is exact, whereas pool deduplication is canonicalized.
        return hashlib.sha256(self.question.encode("utf-8")).hexdigest()

    @property
    def content_hash(self):
        return question_identity(self.question)


@dataclass(frozen=True)
class SelectedQuestion:
    ordinal: int
    qid: str
    selection_source: str
    expected_std: float


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return value


def validate_budget(candidate_questions=5120, selected_questions=512,
                    exploration_questions=64, responses_per_question=5):
    _integer(candidate_questions, "candidate_questions", minimum=1)
    _integer(selected_questions, "selected_questions", minimum=1)
    _integer(exploration_questions, "exploration_questions")
    if selected_questions > candidate_questions or exploration_questions > selected_questions:
        raise ValueError("Invalid question-probe selection budget")
    if type(responses_per_question) is not int or responses_per_question != 5:
        raise ValueError("The six-count probe requires exactly five responses")


def count_predictions(probabilities):
    """Derive consistent moments without an independence/binomial assumption."""
    if hasattr(probabilities, "detach"):
        probabilities = probabilities.detach().cpu().numpy()
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
        raise ValueError("Expected finite [questions, 6] count probabilities")
    if (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(1), 1, atol=1e-5, rtol=0):
        raise ValueError("Count probabilities must lie on the simplex")
    return dict(probabilities=values, mean=values @ COUNT_MEANS,
                expected_std=values @ COUNT_STDS,
                expected_variance=values @ COUNT_VARIANCES,
                mixed_probability=values[:, 1:5].sum(1))


class QuestionPool:
    """Unique question reservation that consumes no actor-generation quota."""
    def __init__(self, limit):
        self.limit = _integer(limit, "candidate_questions", minimum=1)
        self.questions = []
        self._ids, self._contents = set(), set()

    def reserve(self, qid, question):
        if len(self.questions) >= self.limit:
            return None
        key, text = str(qid), str(question)
        content = question_identity(text)
        if key in self._ids or content in self._contents:
            return None
        ordinal = len(self.questions)
        self.questions.append(PredictQuestion(ordinal, key, text))
        self._ids.add(key)
        self._contents.add(content)
        return ordinal


def select_questions(questions, probabilities, *, selected_questions=512,
                     exploration_questions=64, rng, tie_seed=0):
    """Draw E uniformly, then Top-(512-E) from the remaining unique pool.

    Ties use an identity hash independent of exploration RNG and source order.
    The returned rows are sorted by original ordinal to preserve group order.
    """
    questions = list(questions)
    validate_budget(len(questions), selected_questions, exploration_questions)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("A private numpy Generator is required")
    ordinals = [q.ordinal for q in questions]
    if sorted(ordinals) != list(range(len(questions))):
        raise ValueError("Question ordinals must form a complete original-order pool")
    if len({q.qid for q in questions}) != len(questions) or len({q.content_hash for q in questions}) != len(questions):
        raise ValueError("Duplicate candidate question identity or content")
    moments = count_predictions(probabilities)
    if len(moments["expected_std"]) != len(questions):
        raise ValueError("Prediction/question count differs")
    by_ordinal = {q.ordinal: (q, index) for index, q in enumerate(questions)}
    # One permutation means the E=64/128/256 exploration sets are nested when
    # initialized with the same RNG state. E=0 also consumes the same RNG draw.
    permutation = rng.permutation(len(questions))
    exploration_indices = permutation[:exploration_questions].tolist()
    exploration_ordinals = {questions[i].ordinal for i in exploration_indices}

    def rank_key(ordinal):
        q, index = by_ordinal[ordinal]
        tie = hashlib.sha256((str(tie_seed) + "\0" + q.qid + "\0" + q.question_hash).encode()).hexdigest()
        return (-float(moments["expected_std"][index]), tie)

    focus = sorted((x for x in ordinals if x not in exploration_ordinals), key=rank_key)
    focus = focus[:selected_questions - exploration_questions]
    chosen = exploration_ordinals.union(focus)
    return [SelectedQuestion(ordinal, by_ordinal[ordinal][0].qid,
              "random_exploration" if ordinal in exploration_ordinals else "predicted_diversity",
              float(moments["expected_std"][by_ordinal[ordinal][1]]))
            for ordinal in sorted(chosen)]


def prequential_metrics(probabilities, strict_vectors, sources):
    """Prediction-before-fit diagnostics; no unselected candidate oracle claims."""
    prediction = count_predictions(probabilities)
    labels = np.asarray(strict_vectors)
    if labels.shape != (len(prediction["mean"]), 5) or labels.dtype.kind not in "iu" or not np.isin(labels, (0, 1)).all():
        raise ValueError("Exactly five binary strict-EM labels are required per selected question")
    if len(sources) != len(labels):
        raise ValueError("Selection-source count differs")
    counts = labels.sum(1).astype(np.int64)
    means, stds = COUNT_MEANS[counts], COUNT_STDS[counts]
    mixed = ((counts > 0) & (counts < 5)).astype(np.float64)
    result = {}
    for name in ("all", "random_exploration", "predicted_diversity"):
        mask = np.ones(len(labels), dtype=bool) if name == "all" else np.asarray(sources) == name
        result[name] = {"questions": int(mask.sum())}
        if not mask.any():
            # E=0 has no random measurement, not a fabricated zero error.
            continue
        rows = np.flatnonzero(mask)
        result[name].update(
            count_nll=float(-np.log(np.maximum(prediction["probabilities"][rows, counts[rows]], 1e-12)).mean()),
            mean_mae=float(np.abs(prediction["mean"][mask] - means[mask]).mean()),
            std_mae=float(np.abs(prediction["expected_std"][mask] - stds[mask]).mean()),
            std_mse=float(np.square(prediction["expected_std"][mask] - stds[mask]).mean()),
            mixed_brier=float(np.square(prediction["mixed_probability"][mask] - mixed[mask]).mean()),
            actual_mixed_fraction=float(mixed[mask].mean()),
            actual_sample_std_mean=float(stds[mask].mean()),
            actual_em_mean=float(means[mask].mean()),
            all_wrong_fraction=float((counts[mask] == 0).mean()),
            all_correct_fraction=float((counts[mask] == 5).mean()))
    score_delta = prediction['expected_std'][:, None] - prediction['expected_std'][None, :]
    truth_delta = stds[:, None] - stds[None, :]
    comparable = truth_delta > 0
    result['ranking'] = dict(scope='selected_questions_prediction_before_rollout_before_fit',
        comparable_pairs=int(comparable.sum()),
        pairwise_concordance=(float(((score_delta[comparable] > 0).sum() +
                                    .5*(score_delta[comparable] == 0).sum()) / comparable.sum())
                              if comparable.any() else None))
    top = result['predicted_diversity']; random = result['random_exploration']
    if top['questions'] and random['questions']:
        t, r = top['actual_mixed_fraction'], random['actual_mixed_fraction']
        result['top_vs_random'] = dict(top_questions=top['questions'], random_questions=random['questions'],
            mixed_hit_rate_difference=t-r, mixed_hit_rate_ratio=t/r if r > 0 else None,
            actual_sample_std_mean_difference=top['actual_sample_std_mean']-random['actual_sample_std_mean'])
    return result
