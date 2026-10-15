"""Question-only, uniform-first selection before any real rollout is observed.

Question-only predictions enter selection. Count-distribution and token-binary
heads have separate selectors and diagnostics; neither invents the other
representation. Strict correctness is absent from both selection DTOs.
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
        raise ValueError("The question probe requires exactly five responses")


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
            mixed_brier=float(np.square(prediction["mixed_probability"][mask] - mixed[mask]).mean()),
            actual_mixed_fraction=float(mixed[mask].mean()),
            actual_sample_std_mean=float(stds[mask].mean()),
            actual_em_mean=float(means[mask].mean()),
            all_wrong_fraction=float((counts[mask] == 0).mean()),
            all_correct_fraction=float((counts[mask] == 5).mean()))
        # Reuse the immutable pre-rollout predictions and actual selected labels.
        # These diagnostics add no inference, sampling, filtering, or loss weight.
        actual_mixed = mixed[mask].astype(bool)
        predicted_mixed = prediction["mixed_probability"][mask] >= 0.5
        tp = int((actual_mixed & predicted_mixed).sum())
        tn = int((~actual_mixed & ~predicted_mixed).sum())
        fp = int((~actual_mixed & predicted_mixed).sum())
        fn = int((actual_mixed & ~predicted_mixed).sum())
        ratio = lambda numerator, denominator: numerator / denominator if denominator else 0.0
        recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
        present_recalls = ([recall] if tp + fn else []) + ([specificity] if tn + fp else [])
        p = prediction["probabilities"][rows]
        sigma_probabilities = np.stack((p[:, 0] + p[:, 5], p[:, 1] + p[:, 4],
                                        p[:, 2] + p[:, 3]), axis=1)
        actual_counts = counts[mask]
        result[name].update(
            binary_threshold=0.5, binary_accuracy=ratio(tp + tn, len(rows)),
            binary_balanced_accuracy=float(np.mean(present_recalls)),
            binary_confusion=dict(tp=tp, tn=tn, fp=fp, fn=fn),
            binary_precision=ratio(tp, tp + fp), binary_recall=recall,
            binary_specificity=specificity, binary_f1=ratio(2 * tp, 2 * tp + fp + fn),
            predicted_mixed_fraction=float(predicted_mixed.mean()),
            binary_majority_baseline=max(float(actual_mixed.mean()), float((~actual_mixed).mean())),
            count_accuracy=float((p.argmax(1) == actual_counts).mean()),
            sigma3_accuracy=float((sigma_probabilities.argmax(1) ==
                                    np.minimum(actual_counts, 5 - actual_counts)).mean()),
            classification_zero_division=0.0,
            prediction_scope="selected_questions_before_rollout_and_before_fit")
    return result


@dataclass(frozen=True)
class BinarySelectedQuestion:
    ordinal: int
    qid: str
    selection_source: str
    predicted_mixed_probability: float
    predicted_sample_std: float


def binary_predictions(mixed_probability, predicted_sample_std):
    """Validate the two direct outputs without synthesizing count probabilities."""
    def vector(value, name):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float64)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError("Expected a finite question vector: " + name)
        return value
    probability = vector(mixed_probability, "mixed_probability")
    std = vector(predicted_sample_std, "predicted_sample_std")
    if probability.shape != std.shape or (probability < 0).any() or (probability > 1).any():
        raise ValueError("Binary prediction probabilities or lengths are invalid")
    if (std < 0).any() or (std > COUNT_STDS.max() + 1e-7).any():
        raise ValueError("Predicted sample std is outside its G5 bound")
    return dict(mixed_probability=probability, predicted_sample_std=std)


def select_binary_questions(questions, mixed_probability, predicted_sample_std, *,
                            selected_questions=512, exploration_questions=0):
    """Top binary pMixed only; qid hash breaks ties independently of std/order."""
    questions = list(questions)
    validate_budget(len(questions), selected_questions, exploration_questions)
    if exploration_questions != 0:
        raise ValueError("Token binary acquisition requires exploration_questions=0")
    if sorted(q.ordinal for q in questions) != list(range(len(questions))):
        raise ValueError("Question ordinals must form a complete original-order pool")
    if len({q.qid for q in questions}) != len(questions) or len({q.content_hash for q in questions}) != len(questions):
        raise ValueError("Duplicate candidate question identity or content")
    prediction = binary_predictions(mixed_probability, predicted_sample_std)
    if len(prediction["mixed_probability"]) != len(questions):
        raise ValueError("Prediction/question count differs")
    order = sorted(range(len(questions)), key=lambda i: (
        -float(prediction["mixed_probability"][i]), hashlib.sha256(questions[i].qid.encode()).hexdigest()))
    return [BinarySelectedQuestion(questions[i].ordinal, questions[i].qid, "predicted_binary_mixed",
                float(prediction["mixed_probability"][i]), float(prediction["predicted_sample_std"][i]))
            for i in sorted(order[:selected_questions], key=lambda i: questions[i].ordinal)]


def binary_prequential_metrics(mixed_probability, predicted_sample_std, strict_vectors, sources):
    """Selected G5 labels measured against frozen direct binary/std predictions."""
    prediction = binary_predictions(mixed_probability, predicted_sample_std)
    labels = np.asarray(strict_vectors)
    if labels.shape != (len(prediction["mixed_probability"]), 5) or labels.dtype.kind not in "iu" or not np.isin(labels, (0, 1)).all():
        raise ValueError("Exactly five binary strict-EM labels are required per selected question")
    if len(sources) != len(labels) or any(source != "predicted_binary_mixed" for source in sources):
        raise ValueError("Token binary labels require the binary selection source")
    counts = labels.sum(1).astype(np.int64)
    truth = (counts > 0) & (counts < 5)
    stds = COUNT_STDS[counts]
    result = {}
    for name in ("all", "predicted_binary_mixed"):
        probability = prediction["mixed_probability"]
        positive = probability >= .5
        tp, tn, fp, fn = [int(value.sum()) for value in
                         (truth & positive, ~truth & ~positive, ~truth & positive, truth & ~positive)]
        std_mse = float(np.square(prediction["predicted_sample_std"] - stds).mean())
        clipped = np.clip(probability, 1e-12, 1 - 1e-12)
        result[name] = dict(questions=len(labels), binary_threshold=.5,
            binary_accuracy=float((positive == truth).mean()),
            binary_confusion=dict(tp=tp, tn=tn, fp=fp, fn=fn),
            binary_bce=float(-(truth * np.log(clipped) + ~truth * np.log1p(-clipped)).mean()),
            mixed_brier=float(np.square(probability - truth).mean()),
            std_mae=float(np.abs(prediction["predicted_sample_std"] - stds).mean()),
            std_mse_raw=std_mse, std_mse_normalized=std_mse / .3,
            std_loss_contribution=.3 * std_mse / .3,
            actual_mixed_fraction=float(truth.mean()), predicted_mixed_fraction=float(positive.mean()),
            actual_sample_std_mean=float(stds.mean()), actual_em_mean=float((counts / 5).mean()),
            all_wrong_fraction=float((counts == 0).mean()), all_correct_fraction=float((counts == 5).mean()),
            prediction_scope="selected_questions_before_rollout_and_before_fit")
    return result
