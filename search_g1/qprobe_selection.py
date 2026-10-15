"""Question-only, uniform-first selection before any real rollout is observed.

Only question identities and directly predicted binary mixed probabilities enter
this module. Strict correctness is deliberately absent from its public DTO.
"""
from dataclasses import dataclass
import hashlib

import numpy as np

from search_g1.mixed_selection import question_identity


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
    mixed_probability: float


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
        raise ValueError("The binary mixed probe requires exactly five responses")



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


def binary_predictions(prediction):
    """Validate direct binary/variance outputs; never invent count probabilities."""
    if not isinstance(prediction, dict) or 'mixed_probability' not in prediction or 'expected_variance' not in prediction:
        raise ValueError('Expected direct mixed probability and variance')
    if any(name in prediction for name in ('probabilities', 'logits', 'mean')):
        raise ValueError('Legacy six-count outputs are not binary probe predictions')
    result = {}
    for name in ('mixed_probability', 'expected_variance'):
        value = prediction[name]
        if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float64)
        maximum = 1. if name == 'mixed_probability' else .3
        if value.ndim != 1 or not np.isfinite(value).all() or (value < 0).any() or (value > maximum + 1e-7).any():
            raise ValueError('Invalid binary probe output: ' + name)
        result[name] = value.copy()
    if result['mixed_probability'].shape != result['expected_variance'].shape:
        raise ValueError('Binary probability/variance shape differs')
    return result


def select_questions(questions, prediction, *, selected_questions=512,
                     exploration_questions=64, rng, tie_seed=0):
    """Draw64 uniform first, then Top448 direct p_mix from the remaining pool.

    Equal p_mix uses seeded identity hashes, not candidate labels or variance.
    Return original ordinal order to preserve downstream G5 group alignment.
    """
    questions = list(questions)
    validate_budget(len(questions), selected_questions, exploration_questions)
    if not isinstance(rng, np.random.Generator):
        raise TypeError('A private numpy Generator is required')
    ordinals = [q.ordinal for q in questions]
    if sorted(ordinals) != list(range(len(questions))):
        raise ValueError('Question ordinals must form a complete original-order pool')
    if len({q.qid for q in questions}) != len(questions) or len({q.content_hash for q in questions}) != len(questions):
        raise ValueError('Duplicate candidate question identity or content')
    prediction = binary_predictions(prediction)
    scores = prediction['mixed_probability']
    if len(scores) != len(questions): raise ValueError('Prediction/question count differs')
    by_ordinal = {q.ordinal: (q,index) for index,q in enumerate(questions)}
    permutation = rng.permutation(len(questions))
    exploration = {questions[i].ordinal for i in permutation[:exploration_questions]}
    def rank_key(ordinal):
        q,index = by_ordinal[ordinal]
        tie = hashlib.sha256((str(tie_seed)+'\0'+q.qid+'\0'+q.question_hash).encode()).hexdigest()
        return (-float(scores[index]), tie)
    focus = sorted((x for x in ordinals if x not in exploration),key=rank_key)[:selected_questions-exploration_questions]
    return [SelectedQuestion(ordinal,by_ordinal[ordinal][0].qid,
        'random_exploration' if ordinal in exploration else 'predicted_mixed',
        float(scores[by_ordinal[ordinal][1]])) for ordinal in sorted(exploration.union(focus))]


def prequential_metrics(prediction, strict_vectors, sources):
    """Selected-question labels measured against actual pre-rollout p_mix."""
    prediction = binary_predictions(prediction)
    labels = np.asarray(strict_vectors)
    p = prediction['mixed_probability']; variance = prediction['expected_variance']
    if labels.shape != (len(p),5) or labels.dtype.kind not in 'iu' or not np.isin(labels,(0,1)).all():
        raise ValueError('Exactly five binary strict-EM labels are required per selected question')
    if len(sources) != len(labels): raise ValueError('Selection-source count differs')
    counts = labels.sum(1).astype(np.int64)
    mixed = ((counts >= 1) & (counts <= 4)).astype(np.float64)
    true_variance = counts * (5-counts)/20
    result = {}
    for name in ('all','random_exploration','predicted_mixed'):
        mask = np.ones(len(labels),dtype=bool) if name=='all' else np.asarray(sources)==name
        result[name] = {'questions':int(mask.sum())}
        if not mask.any(): continue
        prob = np.clip(p[mask],1e-12,1-1e-12); target=mixed[mask]
        selected = p[mask]>=.5
        tp=int(((target==1)&selected).sum()); fp=int(((target==0)&selected).sum()); fn=int(((target==1)&~selected).sum())
        precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.
        result[name].update(binary_bce=float(-(target*np.log(prob)+(1-target)*np.log1p(-prob)).mean()),
            mixed_brier=float(np.square(p[mask]-target).mean()),
            binary_accuracy=float((selected==target).mean()), binary_precision=precision,binary_recall=recall,
            binary_f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
            variance_mse=float(np.square(variance[mask]-true_variance[mask]).mean()),
            variance_normalized_mse=float(np.square((variance[mask]-true_variance[mask])/.3).mean()),
            actual_mixed_fraction=float(target.mean()), actual_sample_variance_mean=float(true_variance[mask].mean()),
            actual_em_mean=float((counts[mask]/5).mean()),all_wrong_fraction=float((counts[mask]==0).mean()),
            all_correct_fraction=float((counts[mask]==5).mean()))
    comparable = mixed[:,None] > mixed[None,:]
    delta=p[:,None]-p[None,:]
    result['ranking']=dict(scope='selected_questions_prediction_before_rollout_before_fit',score='mixed_probability',
        comparable_pairs=int(comparable.sum()),pairwise_concordance=float(((delta[comparable]>0).sum()+.5*(delta[comparable]==0).sum())/comparable.sum()) if comparable.any() else None)
    top,random=result['predicted_mixed'],result['random_exploration']
    if top['questions'] and random['questions']:
        t,r=top['actual_mixed_fraction'],random['actual_mixed_fraction']
        result['top_vs_random']=dict(top_questions=top['questions'],random_questions=random['questions'],mixed_hit_rate_difference=t-r,mixed_hit_rate_ratio=t/r if r>0 else None)
    return result
