"""CPU sampling-journal audit and optional HF parity for one common8 outer.

The caller owns the native journal, forwards, persistence and collective dispatch.
This reader neither authenticates physical execution nor grants health/admission.
It does not mutate either log-probability tensor or construct an optimizer update.
"""
import hashlib
import math
from collections.abc import Mapping

import torch

from reproduction.common8.policy_calibration_plan import canonical_bytes, document_sha256


LOGPROB_ATOL = 0.05
TRAJECTORY_COUNT = 512 * 5
_FLOAT_DTYPES = (torch.float32, torch.float64)
_MASK_DTYPES = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


class BehaviorProbabilityError(ValueError):
    """The complete outer must remain unaccepted; retain its actual journal."""


def _require(condition, message):
    if not condition:
        raise BehaviorProbabilityError('Behavior probability rejected: ' + message)


def _tensor(value, name, shape=None, *, mask=False):
    _require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
             value.ndim == 2, name + ' must be a two-dimensional CPU tensor')
    _require(value.shape[0] == TRAJECTORY_COUNT and value.shape[1] > 0,
             name + ' must cover all2560 actual trajectories')
    if shape is not None:
        _require(value.shape == shape, 'misaligned ' + name)
    _require(value.dtype in (_MASK_DTYPES if mask else _FLOAT_DTYPES),
             'invalid dtype for ' + name)
    if mask:
        _require(bool(((value == 0) | (value == 1)).all()), 'nonbinary ' + name)
    return value


def _tensor_sha256(value):
    value = value.detach().contiguous()
    metadata = canonical_bytes(dict(dtype=str(value.dtype), shape=list(value.shape)))
    return hashlib.sha256(metadata + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def verify_sampling(sampling):
    """Only unfiltered full-softmax training sampling can share the HF denominator."""
    _require(isinstance(sampling, Mapping), 'missing actual sampler settings')
    required = dict(temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
                    presence_penalty=0.0, frequency_penalty=0.0,
                    repetition_penalty=1.0)
    for name, expected in required.items():
        value = sampling.get(name)
        _require(type(value) in (int, float) and math.isfinite(value) and value == expected,
                 'actual sampler differs: ' + name)
    _require(type(sampling.get('n')) is int and sampling['n'] == 1 and
             sampling.get('do_sample') is True and sampling.get('ignore_eos') is False,
             'actual sampler is not the bound single stochastic completion')
    _require(type(sampling.get('logprobs')) is int and sampling['logprobs'] == 1,
             'actual selected log-probabilities were not requested')
    _require(type(sampling.get('max_tokens')) is int and sampling['max_tokens'] == 500,
             'actual generation token budget differs from original500')
    _require('min_tokens' in sampling and (sampling['min_tokens'] is None or
             (type(sampling['min_tokens']) is int and sampling['min_tokens'] == 0)),
             'minimum-token filter is not neutral')
    _require('logits_processors' in sampling and sampling['logits_processors'] in (None, []),
             'custom logits processors are missing or not neutral')
    _require('best_of' in sampling and (sampling['best_of'] is None or
             (type(sampling['best_of']) is int and sampling['best_of'] == 1)),
             'best-of selection is missing or not neutral')
    _require('logit_bias' in sampling and sampling['logit_bias'] in (None, {}),
             'logit bias is missing or not neutral')
    for name in ('allowed_token_ids', 'guided_decoding', 'truncate_prompt_tokens'):
        _require(name in sampling and sampling[name] is None,
                 'external filtering or prompt truncation is missing or not neutral: ' + name)
    _require(sampling.get('logprob_semantics') == 'selected_token_logprob_at_actual_sampling_call',
             'missing actual selected log-probability semantics')
    return dict(sampling)


def verify(hf_old_log_probs, sampling_log_probs, actor_mask, attention_mask, *,
           trace, actor_rows, audit, expected_run_id, expected_source_sha256,
           expected_slurm_job_id, expected_logical_update, expected_parent_group_ids,
           expected_trajectory_ids, expected_policy_snapshot_id, expected_sampling,
           require_hf_parity=True, expected_generation_producers=None):
    """Audit every actual token, with the original .05 HF gate by default.

    ``actor_rows`` are the current complete padded rows used by the actor.
    The trace audit is rerun against those rows, including after a valid reorder.
    HF values on observations/corrections/padding may be nonfinite: they never
    enter this comparison. Actual old values outside the actor mask must be zero,
    so copying the passed sampling tensor as frozen old is globally finite.
    The returned dictionary contains measurement evidence only; inputs are intact.
    """
    _require(type(require_hf_parity) is bool, 'invalid HF parity policy')
    _require(require_hf_parity or hf_old_log_probs is None,
             'journal-only audit must not present copied sampling values as HF output')
    hf = _tensor(hf_old_log_probs, 'HF old log-probabilities') if require_hf_parity else None
    actual = _tensor(sampling_log_probs, 'actual sampling log-probabilities',
                     hf.shape if hf is not None else None)
    authored = _tensor(actor_mask, 'actor mask', actual.shape, mask=True).bool()
    attention = _tensor(attention_mask, 'attention mask', actual.shape, mask=True).bool()
    _require(bool((authored <= attention).all()), 'actor mask includes padding')
    _require(bool(authored.any()), 'no included actor tokens')
    _require(bool(torch.isfinite(actual).all()), 'actual frozen old contains a nonfinite value')
    _require(bool((actual[~authored] == 0).all()),
             'observation/correction/padding probabilities must be zero outside actor mask')
    if require_hf_parity:
        _require(bool(torch.isfinite(hf[authored]).all()), 'nonfinite HF actor log-probability')
    _require(bool((actual[authored] <= 0).all()), 'positive actual log-probability')
    verify_sampling(expected_sampling)
    _require(isinstance(trace, Mapping), 'missing actual trace')
    verify_sampling(trace.get('sampling'))
    _require(canonical_bytes(trace['sampling']) == canonical_bytes(expected_sampling),
             'actual sampler evidence differs from its frozen binding')
    _require(isinstance(actor_rows, (list, tuple)) and len(actor_rows) == TRAJECTORY_COUNT,
             'missing complete actual actor rows')
    _require(isinstance(audit, Mapping), 'missing actual context audit')
    from reproduction.common8.behavior_trace import project_sampling_logprobs
    projected = project_sampling_logprobs(trace, expected_run_id=expected_run_id,
                          expected_source_sha256=expected_source_sha256,
                          expected_slurm_job_id=expected_slurm_job_id,
                          expected_logical_update=expected_logical_update,
                          expected_parent_group_ids=expected_parent_group_ids,
                          expected_trajectory_ids=expected_trajectory_ids,
                          expected_policy_snapshot_id=expected_policy_snapshot_id,
                          expected_sampling=expected_sampling, actor_rows=actor_rows,
                          expected_generation_producers=expected_generation_producers)
    rebuilt = projected['audit']
    _require(canonical_bytes(rebuilt) == canonical_bytes(audit),
             'supplied audit differs from the reconstructed actual context audit')
    _require(rebuilt.get('context_verified') is True,
             'actual included-token contexts were not verified')
    _require(rebuilt.get('trajectory_ids') == [row.get('trajectory_id') for row in actor_rows],
             'actual actor-row order differs from audit')
    counts = authored.sum(-1).tolist()
    _require(rebuilt.get('actor_token_counts') == counts and
             rebuilt.get('included_token_count') == sum(counts),
             'included actor-token counts differ from actual audit')
    for index, row in enumerate(actor_rows):
        _require(isinstance(row, Mapping), 'invalid actual actor row')
        width = row.get('prompt_width')
        _require(type(width) is int and width > 0, 'invalid actual prompt width')
        info, valid = row.get('info_mask'), row.get('attention_mask')
        _require(isinstance(info, list) and isinstance(valid, list) and
                 len(info) == len(valid) == width + actual.shape[1],
                 'actual padded response width differs')
        _require(info[width:] == actor_mask[index].tolist() and
                 valid[width:] == attention_mask[index].tolist(),
                 'comparison mask differs from actual actor/context row')
    _require(projected.get('trajectory_ids') == rebuilt['trajectory_ids'],
             'actual probability projection has different trajectory order')
    journal = torch.tensor(projected['sampling_log_probs'], dtype=actual.dtype)
    journal_mask = torch.tensor(projected['sampling_mask'], dtype=torch.bool)
    _require(journal.shape == actual.shape and torch.equal(journal_mask, authored),
             'actual probability projection has a different actor mask')
    # Reconstruct the selected Python values with their tensor storage dtype.
    # Equality also proves that observations and padding received zero, not an LP.
    _require(torch.equal(actual, journal),
             'sampling tensor differs from actual selected log-probabilities')
    maximum = None
    tensors = dict(sampling_log_probs=actual, actor_mask=actor_mask, attention_mask=attention_mask)
    if require_hf_parity:
        discrepancy = (hf[authored].double() - actual[authored].double()).abs()
        maximum = discrepancy.max().item()
        _require(math.isfinite(maximum) and maximum <= LOGPROB_ATOL,
                 'HF versus actual sampling maximum exceeds original0.05: ' + repr(maximum))
        tensors['hf_old_log_probs'] = hf
    return dict(schema_version=1, role='common8_behavior_probability_gate', accepted=True,
                run_id=expected_run_id, source_sha256=dict(expected_source_sha256),
                slurm_job_id=expected_slurm_job_id, logical_update=expected_logical_update,
                policy_snapshot_id=expected_policy_snapshot_id,
                sampling=verify_sampling(trace['sampling']), trajectory_count=TRAJECTORY_COUNT,
                sampling_distribution='full_softmax; actual neutral filters and penalties verified',
                trajectory_ids=rebuilt['trajectory_ids'], actor_token_counts=counts,
                included_actor_token_count=sum(counts), max_abs_discrepancy=maximum,
                tolerance=LOGPROB_ATOL if require_hf_parity else None,
                hf_recomputation_performed=require_hf_parity,
                hf_parity_accepted=True if require_hf_parity else None,
                trace_sha256=rebuilt['trace_sha256'],
                actor_rows_sha256=rebuilt['actor_rows_sha256'], audit_sha256=document_sha256(rebuilt),
                tensor_sha256={name: _tensor_sha256(value) for name, value in tensors.items()},
                hf_dtype=str(hf.dtype) if hf is not None else None, sampling_dtype=str(actual.dtype),
                actual_frozen_old_all_finite=True,
                scope='all retained actor tokens in all2560 actual trajectories; '
                      'observations, corrections and padding excluded; not the32 monitor sample',
                production_admission=False, healthy_checkpoint_accepted=False,
                physical_execution_authenticated=False)
