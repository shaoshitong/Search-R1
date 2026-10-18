"""Read-only Tree behavior-probability provenance; no loss or policy correction.

This module assembles only actual retained emission scalars. Tensor rows and
their row-local provenance must move together through DataProto permutations.
"""
import copy
import math
import struct
from types import SimpleNamespace

import torch

from .tree_raw_tokens import (FIELD as RAW_FIELD, PROTOCOL as RAW_PROTOCOL,
                             fingerprint, trajectory_projection, validate_native_emission)


PROTOCOL = 'tree_behavior_mu_centered_ppo_v1'
TENSOR = 'sampled_old_log_probs'
FIELD = 'tree_behavior_mu_binding'
META = 'tree_behavior_mu_protocol'
PROVENANCE_META = 'tree_behavior_mu_runtime_provenance'
PROVENANCE_KEYS = ('SEARCH_R1_BASELINE_COMMIT', 'SEARCH_R1_BASELINE_CONFIG_SHA256',
                   'SEARCH_R1_BASELINE_DATA_IDENTITY', 'SEARCH_R1_MODEL_IDENTITY',
                   'SEARCH_R1_BASELINE_RUN_ID', ('RBA_H100_SHARED_EXECUTION_ID' if __import__('os').environ.get('RBA_H100_SHARED_EXECUTION_ID') else 'SLURM_JOB_ID'))
PLAIN_DISTRIBUTION = dict(temperature=1., top_p=1., top_k=-1, min_p=0.,
                          presence_penalty=0., frequency_penalty=0.,
                          repetition_penalty=1., n=1, best_of=1, _real_n=1, logits_processors=None,
                          truncate_prompt_tokens=None, guided_decoding=None,
                          logit_bias=None, allowed_token_ids=None)


def require(value, message):
    if not value:
        raise ValueError('Tree behavior mu: ' + message)


def validate_options(options):
    """Explicit opt-in; an absent or strictly disabled mapping preserves old data."""
    if options is None:
        return None
    require(set(options) == {'enabled', 'protocol'} and type(options['enabled']) is bool,
            'Explicit enabled bool and protocol only are required')
    require(options['protocol'] == PROTOCOL, 'Unknown behavior correction protocol')
    return dict(options) if options['enabled'] else None


def runtime_provenance(environ):
    value = {key: environ.get(key) for key in PROVENANCE_KEYS}
    return validate_provenance(value)


def validate_provenance(value):
    require(isinstance(value, dict) and set(value) == set(PROVENANCE_KEYS) and
            all(isinstance(x, str) and x for x in value.values()),
            'Original six source/config/data/model/run/job values required')
    return copy.deepcopy(value)


def validate_distribution(value):
    require(isinstance(value, dict) and set(value) == set(PLAIN_DISTRIBUTION),
            'Actual native sampling distribution metadata missing')
    for key, expected in PLAIN_DISTRIBUTION.items():
        actual = value[key]
        if key in ('best_of','_real_n'):
            require(type(actual) is int and actual == 1,
                    'Actual '+key+' must be the recorded integer1')
        require((actual is None if expected is None else
                 type(actual) in (int, float) and type(actual) is not bool and actual == expected),
                'Unsupported actual sampling distribution: ' + key)
    return copy.deepcopy(value)


def scalar32(value):
    require(type(value) in (int, float) and math.isfinite(value) and value <= 0.,
            'Missing/nonfinite/positive selected emission probability')
    try:
        bits = struct.pack('!f', value)
    except (OverflowError, struct.error) as exc:
        raise ValueError('Tree behavior mu: scalar is not FP32') from exc
    require(struct.unpack('!f', bits)[0] == value,
            'Original selected scalar is not exactly representable in FP32')
    return bits.hex()


def _emission(action, context, expected_provenance):
    emission = action['native_emission']
    require(emission.get('protocol') == RAW_PROTOCOL and emission.get('schema_version') == 1,
            'Original raw emission protocol missing')
    identity = emission['sampling_identity']
    require(set(identity) == {'call_id', 'rank', 'request_id', 'physical_row', 'runtime_provenance'} and
            isinstance(identity['call_id'], str) and identity['call_id'] and
            isinstance(identity['request_id'], str) and identity['request_id'] and
            type(identity['rank']) is int and identity['rank'] >= 0 and
            type(identity['physical_row']) is int and identity['physical_row'] >= 0,
            'Actual sampling call/rank/request/row identity missing')
    require(validate_provenance(identity['runtime_provenance']) == expected_provenance and
            emission['sample_id'] == fingerprint(identity), 'Sampling runtime identity changed')
    require(emission['prompt_token_ids'] == context and emission['finished'] is True and
            emission['finish_reason'] in ('stop', 'length'), 'Native prefix/finish changed')
    settings = emission['sampling_params']
    required = set(PLAIN_DISTRIBUTION)
    required |= {'min_tokens', 'max_tokens', 'ignore_eos', 'stop', 'stop_token_ids',
                 'include_stop_str_in_output', 'logprobs', 'prompt_logprobs', 'seed'}
    decoder_fields={'skip_special_tokens','spaces_between_special_tokens'}
    require(set(settings) in (required,required|decoder_fields), 'Complete recorded native settings required')
    if decoder_fields<=set(settings):
        require(all(settings[key] is True for key in decoder_fields),
                'Unsupported native decoder settings')
    validate_distribution({key: settings[key] for key in PLAIN_DISTRIBUTION})
    require(settings['min_tokens'] == 0 and settings['max_tokens'] == 500 and
            type(settings['min_tokens']) is int and type(settings['max_tokens']) is int and
            settings['ignore_eos'] is False and ((settings['stop'] in (None, []) and
            settings['include_stop_str_in_output'] is False) or
            (settings['stop'] in (['</search>'],['</search>','</answer>']) and settings['include_stop_str_in_output'] is True)) and
            settings['stop_token_ids'] in (None, []) and
            settings['logprobs'] == 1 and
            type(settings['logprobs']) is int and
            settings['prompt_logprobs'] is None and settings['seed'] is None,
            'Unsupported stop/length/logprob/seed settings')
    ids, n, values = validate_native_emission(emission, 151643, 151645)
    for value in values:
        scalar32(value)
    require(type(action['cut']) is int and 0 < action['cut'] <= n and
            action['action_token_ids'] == ids[:action['cut']] and
            action['selected_logprobs'] == values[:action['cut']],
            'Retained action is not the original sampled token/scalar prefix')
    tok = action['tokenizer_identity']
    require(tok['pad_token_id'] == 151643 and tok['eos_token_id'] == 151645 and
            tok['policy_decode_skip_special_tokens'] is True,
            'Original tokenizer/PAD/EOS protocol changed')
    return emission


class BehaviorMuProofCache:
    """Driver-local, bounded to one outer; never serialized into training data."""
    def __init__(self, max_rows=4096):
        self.max_rows=max_rows;self.rows={};self.contexts={};self.hits=0;self.misses=0


def _row(projection, expected_provenance, validation_context=None, proof_cache=None):
    require(projection.get('protocol') == RAW_PROTOCOL and projection['response_cap'] == 4096,
            'Accepted raw/semantic protocol and fixed cap required')
    bound = {key: value for key, value in projection.items() if key != 'binding_sha256'}
    require(fingerprint(bound) == projection['binding_sha256'], 'Raw projection binding changed')
    # Fresh content SHA is computed above, never trust a caller-supplied digest.
    # It binds every ledger/identity field; runtime tokenizer checks remain below.
    key=(projection['binding_sha256'],fingerprint(expected_provenance),
         fingerprint([turn['action']['tokenizer_identity'] for turn in projection['raw_ledger']]))
    if proof_cache is not None and key in proof_cache.rows:
        proof_cache.hits+=1
        return proof_cache.rows[key]

    state = SimpleNamespace(prompt=projection['prompt_token_ids'],
                            response=projection['policy_response_token_ids'],
                            model_mask=projection['policy_model_mask'],
                            raw_ledger=projection['raw_ledger'])
    rebuilt = (trajectory_projection(state, 4096) if validation_context is None else
               trajectory_projection(state, 4096, _contexts=validation_context))
    require(rebuilt == projection,
            'Raw/semantic projection did not reconstruct')
    require(state.prompt and len(state.prompt) <= 2048 and
            len(state.prompt) + len(state.response) <= 4096,
            'Original prompt/rolling crop protocol unsupported')
    values, actions, steps = [], [], []
    offset = 0
    for turn in state.raw_ledger:
        context = turn['generation_context_token_ids']
        action = turn['action']
        emission = _emission(action, context, expected_provenance)
        # The original ledger already stores every ID/scalar. Bind that entire
        # prefix once per action, rather than duplicating O(T^2) causal prefixes.
        actions.append(dict(sample_id=emission['sample_id'], response_start=offset,
            retained_count=action['cut'], native_emission_sha256=fingerprint(emission),
            action_binding_sha256=action['action_binding_sha256'],
            causal_context_sha256=fingerprint(dict(token_ids=context,
                position_ids=turn['generation_context_positions'],
                attention_mask=turn['generation_context_attention'])),
            sampled_prefix_sha256=fingerprint(list(zip(action['action_token_ids'],
                [scalar32(value) for value in action['selected_logprobs']])))))
        values.extend(action['selected_logprobs'])
        steps.extend([turn['turn']] * len(action['action_token_ids']))
        obs = turn['observation_token_ids']
        require(len(obs) <= 500, 'Unsupported observation cap')
        values.extend([0.] * len(obs)); steps.extend([-1] * len(obs))
        offset = len(values)
    require(len(values) == len(state.response) and actions,
            'Effective row has missing behavior probabilities')
    result=(values, actions, steps)
    if proof_cache is not None:
        proof_cache.misses+=1
        if len(proof_cache.rows)>=proof_cache.max_rows:proof_cache.rows.clear()
        proof_cache.rows[key]=result
    return result


def compose_behavior_mu_rows(states, projections, *, response_width, expected_provenance):
    """Pure CPU assembly from reviewed ledgers; no resampling or actor fallback."""
    expected = validate_provenance(expected_provenance)
    require(len(states) == len(projections) > 0 and type(response_width) is int and response_width > 0,
            'Nonempty complete row collection required')
    tensor = torch.zeros((len(states), response_width), dtype=torch.float32)
    bindings = []
    validation_context = {}
    for row, (state, projection) in enumerate(zip(states, projections)):
        values, actions, _ = _row(projection, expected, validation_context=validation_context)
        require(projection['prompt_token_ids'] == state.prompt and
                projection['policy_response_token_ids'] == state.response and
                projection['policy_model_mask'] == state.model_mask and
                len(values) <= response_width, 'Compose row/projection/cap changed')
        require(type(state.source) is int and state.source >= 0 and
                all(isinstance(uid, str) and uid for uid in (state.tree, state.trajectory)),
                'Original source/tree/leaf identity missing')
        tensor[row, :len(values)] = torch.tensor(values, dtype=torch.float32)
        binding = dict(protocol=PROTOCOL, raw_projection_sha256=projection['binding_sha256'],
            runtime_provenance=copy.deepcopy(expected), tree_uid=state.tree, trajectory_uid=state.trajectory,
            source_index=state.source, response_width=response_width, action_receipts=actions,
            sampled_row_fp32_sha256=fingerprint([scalar32(x) for x in tensor[row].tolist()]))
        binding['binding_sha256'] = fingerprint(binding)
        bindings.append(binding)
    for context in validation_context.values():
        context.check_complete()
    return tensor, bindings


def validate_behavior_mu_batch(data, *, expected_provenance, required=True, proof_cache=None):
    """Validate complete row-local binding after final balancing/permutation.

    Caller owns matched collective refusal. This function never edits tensors,
    substitutes old/actor probabilities, or accepts numerical/optimizer gates.
    """
    expected = validate_provenance(expected_provenance)
    present = (TENSOR in data.batch, FIELD in data.non_tensor_batch, META in data.meta_info)
    if not any(present) and not required:
        return dict(status='TREE_BEHAVIOR_MU_DISABLED', rows=0)
    require(all(present) and data.meta_info[META] == PROTOCOL and
            validate_provenance(data.meta_info.get(PROVENANCE_META)) == expected,
            'Opted-in behavior tensor/binding/protocol/runtime provenance dropped')
    names = ('prompts', 'responses', 'input_ids', 'attention_mask', 'info_mask', 'position_ids', 'step_ids')
    require(all(key in data.batch for key in names) and RAW_FIELD in data.non_tensor_batch and
            all(key in data.non_tensor_batch for key in ('tree_uid', 'trajectory_uid')),
            'Complete raw input/projection/row identity fields required')
    rows, width = data.batch['responses'].shape
    require(rows > 0 and data.batch[TENSOR].shape == (rows, width) and
            data.batch[TENSOR].dtype == torch.float32 and
            len(data.non_tensor_batch[FIELD]) == len(data.non_tensor_batch[RAW_FIELD]) == rows,
            'Behavior tensor dtype/shape/provenance rows changed')
    require(all(data.batch[key].ndim == 2 and data.batch[key].shape[0] == rows and
                data.batch[key].dtype == torch.long for key in names), 'Original input tensor dtype/shape changed')
    for key in ('response_mask', 'loss_mask'):
        if key in data.batch:
            require(data.batch[key].shape == (rows, width), 'Added mask shape differs from response')
    # Stage complete tensors once. Row-at-a-time CUDA .cpu() forced a separate
    # device wait for every row/field before doing the same Python comparisons.
    cpu_rows = {key: data.batch[key].detach().cpu().tolist()
                for key in names + (TENSOR, 'response_mask', 'loss_mask') if key in data.batch}
    require(proof_cache is None or type(proof_cache) is BehaviorMuProofCache,
            'Invalid per-outer behavior proof cache')
    validation_context = {} if proof_cache is None else proof_cache.contexts
    for context in validation_context.values():context.check_runtime()
    seen, total = {}, 0
    for row in range(rows):
        projection = data.non_tensor_batch[RAW_FIELD][row]
        binding = data.non_tensor_batch[FIELD][row]
        bound = {key: value for key, value in binding.items() if key != 'binding_sha256'}
        require(fingerprint(bound) == binding['binding_sha256'] and binding['protocol'] == PROTOCOL and
                binding['runtime_provenance'] == expected and binding['response_width'] == width and
                binding['raw_projection_sha256'] == projection['binding_sha256'] and
                binding['tree_uid'] == data.non_tensor_batch['tree_uid'][row] and
                binding['trajectory_uid'] == data.non_tensor_batch['trajectory_uid'][row],
                'Row permutation lost behavior/source/leaf identity')
        values, actions, steps = _row(projection, expected, validation_context=validation_context, proof_cache=proof_cache)
        require(binding['action_receipts'] == actions, 'Sampled action/context/identity binding changed')
        prompt = cpu_rows['prompts'][row]
        response = cpu_rows['responses'][row]
        actual_mu = cpu_rows[TENSOR][row]
        n = len(projection['policy_response_token_ids'])
        prompt_count = len(projection['prompt_token_ids'])
        want_prompt = [151643] * (len(prompt)-prompt_count) + projection['prompt_token_ids']
        want_response = projection['policy_response_token_ids'] + [151643] * (width-n)
        prompt_attention = [0] * (len(prompt)-prompt_count) + [1] * prompt_count
        response_attention = [1] * n + [0] * (width-n)
        attention = prompt_attention + response_attention
        positions = []; index = 0
        for valid in attention:
            positions.append(index if valid else 0); index += valid
        info = attention[:len(prompt)] + projection['policy_model_mask'] + [0] * (width-n)
        want_steps = steps + [-1] * (width-n)
        require(n <= width and len(projection['prompt_token_ids']) <= len(prompt) and
                prompt == want_prompt and response == want_response and
                cpu_rows['input_ids'][row] == want_prompt+want_response and
                cpu_rows['attention_mask'][row] == attention and
                cpu_rows['info_mask'][row] == info and
                cpu_rows['position_ids'][row] == positions and
                cpu_rows['step_ids'][row] == want_steps,
                'Policy prefix/token/position/attention/loss-mask/turn input changed')
        if 'response_mask' in data.batch:
            response_mask = data.batch['response_mask']
            require(response_mask.shape == (rows, width) and
                    cpu_rows['response_mask'][row] == response_attention,
                    'Added response mask differs from explicit retained row length')
        if 'loss_mask' in data.batch:
            loss_mask = data.batch['loss_mask']
            require(loss_mask.shape == (rows, width) and
                    cpu_rows['loss_mask'][row] == info[len(prompt):],
                    'Added loss mask differs from original sampled actor-token mask')
        want_mu = values + [0.] * (width-n)
        actual_bits = [scalar32(x) for x in actual_mu]
        require(actual_bits == [scalar32(x) for x in want_mu] and
                fingerprint(actual_bits) == binding['sampled_row_fp32_sha256'],
                'Behavior scalar altered, missing, or present at observation/padding')
        for action in actions:
            key = action['sample_id']
            value = {k: v for k, v in action.items() if k != 'response_start'}
            require(key not in seen or seen[key] == value, 'Inherited sampled action changed between leaves')
            seen[key] = value; total += action['retained_count']
    for context in validation_context.values():
        context.check_complete()
    unique = sum(action['retained_count'] for action in seen.values())
    return dict(status='TREE_BEHAVIOR_MU_DATA_BINDING_VALIDATED', rows=rows,
                leaf_effective_token_occurrences=total, unique_sampled_occurrences=unique,
                inherited_leaf_occurrences=total-unique, unique_sampling_actions=len(seen), probability_accepted=False,
                training_accepted=False, validator_optimizer_writes=0)
