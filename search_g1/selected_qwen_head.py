"""Optional Qwen2 LM-head row selection; never enabled by importing this file.

Install on the raw HF actor *before* FSDP wrapping, then call the normal outer
FSDP ``forward`` with ``prediction_positions=<1-D tensor>``. Positions refer to
the decoder's sequence axis, i.e. the position immediately BEFORE each desired
target token. The decoder, its attention mask, position IDs, and checkpointing
are unchanged. Only the input rows of the existing LM head are selected.

Without ``prediction_positions`` the original forward executes unchanged.
With it, the returned HF object's logits are [batch, selected_positions, vocab]
instead of [batch, sequence, vocab]. No labels/cache or HF tail-only projection
may be combined with this explicit row selection. No FSDP/optimizer parameters
or checkpoint keys are added. This is deliberately not wired into live entry
points: callers must scatter selected log-probabilities back to their existing
response layout and keep the existing authored-token loss mask.
"""
import functools
import types


def packed_prediction_mask(indices, sequence_length, response_length, response_positions,
                           loss_mask=None):
    """Map the ORIGINAL padded target loss mask onto packed prediction rows."""
    import torch
    selected = response_positions.clone()
    if loss_mask is None:
        return selected
    positions = response_positions.nonzero(as_tuple=False).flatten()
    rows = torch.div(indices[positions], sequence_length, rounding_mode='floor')
    columns = indices[positions].remainder(sequence_length) - (sequence_length-response_length-1)
    selected[positions] = loss_mask[rows, columns].bool()
    # Empty policy batches retain a real (unused) projection graph. The existing
    # positive-mass policy guard decides their disposition; no synthetic gradient.
    return selected if bool(selected.any()) else response_positions.clone()


def install_selected_qwen_head(model):
    """Enable a per-call optional arg on one unwrapped Qwen2ForCausalLM.

    Uses PyTorch's normal differentiable index_select and an instance-local,
    temporary LM-head pre-hook. Keeping the existing HF forward is compatible
    with both older (<4.48) Qwen2 implementations and their differing output
    casting. Calls on this model must remain serialized, as in the actor worker.
    """
    import torch

    if getattr(model, '_selected_qwen_head_installed', False):
        return model
    if (getattr(getattr(model, 'config', None), 'model_type', None) != 'qwen2'
            or type(model).__name__ != 'Qwen2ForCausalLM'
            or not isinstance(getattr(model, 'lm_head', None), torch.nn.Linear)):
        raise TypeError('Selected projection requires an unwrapped Qwen2ForCausalLM')
    if int(getattr(model.config, 'pretraining_tp', 1)) != 1:
        raise ValueError('Split pretraining TP may bypass the existing LM head')
    original_forward = model.forward

    @functools.wraps(original_forward)
    def forward(instance, *args, prediction_positions=None, **kwargs):
        if prediction_positions is None:
            return original_forward(*args, **kwargs)
        if args:
            raise TypeError('Selected projection requires explicit keyword HF inputs')
        if kwargs.get('labels') is not None or kwargs.get('past_key_values') is not None:
            raise ValueError('Selected projection is for label-free full-context actor forwards')
        if kwargs.get('use_cache') not in (None, False):
            raise ValueError('Selected projection cannot change cached generation')
        # HF versions expose different suffix-selection argument names. Their
        # neutral value is zero, which preserves the complete decoder output.
        for name in ('num_logits_to_keep', 'logits_to_keep'):
            value = kwargs.get(name, 0)
            if type(value) is not int or value != 0:
                raise ValueError('Do not combine tail-only and explicit row projections')
        if (not torch.is_tensor(prediction_positions)
                or prediction_positions.dtype != torch.long
                or prediction_positions.ndim != 1 or prediction_positions.numel() == 0):
            raise ValueError('Prediction positions must be a nonempty 1-D int64 tensor')
        if bool((prediction_positions < 0).any()):
            raise ValueError('Prediction positions cannot be negative')
        if prediction_positions.numel() > 1 and bool((prediction_positions[1:] <= prediction_positions[:-1]).any()):
            raise ValueError('Prediction positions must be unique and ascending')
        if getattr(instance, '_selected_qwen_head_active', False):
            raise RuntimeError('Concurrent or nested selected-head forward on the same model')
        calls = []

        def select_rows(module, inputs):
            if len(inputs) != 1 or inputs[0].ndim != 3:
                raise ValueError('Unexpected Qwen LM-head input layout')
            hidden = inputs[0]
            indices = prediction_positions.to(device=hidden.device)
            if int(indices[-1]) >= hidden.shape[1]:
                raise ValueError('Prediction position exceeds the full decoder context')
            calls.append(1)
            return (hidden.index_select(1, indices),)

        instance._selected_qwen_head_active = True
        hook = instance.lm_head.register_forward_pre_hook(select_rows)
        try:
            kwargs['use_cache'] = False
            output = original_forward(**kwargs)
            if len(calls) != 1:
                raise RuntimeError('HF forward did not use the selected LM head exactly once')
            return output
        finally:
            hook.remove()
            instance._selected_qwen_head_active = False

    model.forward = types.MethodType(forward, model)
    model._selected_qwen_head_installed = True
    return model
