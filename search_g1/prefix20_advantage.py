"""Strict-EM gated virtual-zero baseline; no virtual policy training row."""
from collections import defaultdict
import numpy as np
import torch


def anchored_grpo(token_level_rewards, eos_mask, index, strict_em, epsilon=1e-6):
    if token_level_rewards.ndim != 2 or eos_mask.shape != token_level_rewards.shape:
        raise ValueError('PREFIX20_REWARD_MASK_SHAPE_MISMATCH')
    if len(index) != token_level_rewards.shape[0] or len(strict_em) != len(index):
        raise ValueError('PREFIX20_STRICT_EM_MUST_BE_ROW_ALIGNED_AFTER_BALANCING')
    # DataProto non-tensor wire columns use object arrays. Convert their actual
    # values to a numeric tensor without casting/truncating binary EM labels.
    values = strict_em.tolist() if isinstance(strict_em, np.ndarray) and strict_em.dtype == object else strict_em
    labels = torch.as_tensor(values, device=token_level_rewards.device)
    if labels.ndim != 1 or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError('PREFIX20_REQUIRES_BINARY_STRICT_EM_NOT_SOFT_REWARD')
    if not bool(torch.isfinite(token_level_rewards).all()):
        raise ValueError('PREFIX20_NONFINITE_GRPO_REWARD')
    groups = defaultdict(list)
    for row, qid in enumerate(index):
        groups[qid].append(row)
    if any(len(rows) != 5 for rows in groups.values()):
        raise ValueError('PREFIX20_GRPO_REQUIRES_COMPLETE_G5_GROUPS')
    with torch.no_grad():
        rewards = token_level_rewards.sum(-1)
        normalized = torch.empty_like(rewards)
        for rows in groups.values():
            positions = torch.tensor(rows, device=rewards.device, dtype=torch.long)
            real = rewards[positions]
            baseline = torch.cat((real.new_zeros(1), real)) if bool(labels[positions].all()) else real
            normalized[positions] = (real-baseline.mean()) / (baseline.std(correction=1)+epsilon)
        advantages = normalized.unsqueeze(-1) * eos_mask
    return advantages, advantages


def strict_em_from_batch(data, tokenizer):
    """Use actual model-authored terminal answer spans at the current row order."""
    from search_g1.signals import contexts, decode_segments
    from verl.utils.reward_score.qa_em import em_check
    labels = []
    for row in range(len(data)):
        item = data[row]
        prompt_width = item.batch['prompts'].shape[-1]
        count = int(item.batch['attention_mask'][prompt_width:].sum())
        ids = item.batch['responses'][:count]
        model_mask = item.batch['info_mask'][prompt_width:prompt_width+count]
        text, ranges = decode_segments(tokenizer, ids, model_mask)
        answer = contexts('', text, ranges)
        ground_truth = item.non_tensor_batch['reward_model']['ground_truth']
        labels.append(int(answer is not None and em_check(answer['answer'], ground_truth['target'])))
    # This column travels with the balanced row order through DataProto.chunk.
    return np.asarray(labels, dtype=object)
