"""Current-student all-token features, held only in remote CPU/GPU memory."""
import hashlib
import json
from pathlib import Path

import torch


class QuestionMetadata:
    """Load canonical question/token metadata, never the initial tensor bank."""
    def __init__(self, directory):
        directory = Path(directory)
        manifest = json.loads((directory/'manifest.json').read_text())
        raw = (directory/'index.jsonl').read_bytes()
        expected = manifest['files']['index.jsonl']
        if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError('PERIODIC_CANONICAL_QUESTION_METADATA_CHANGED')
        self.index = {row['qid']: row for row in map(json.loads, raw.splitlines())}
        self.ids = list(self.index)
        self.spec, self.spec_sha = manifest['spec'], manifest['spec_sha']
        self.feature_dim, self.manifest = manifest['feature_dim'], manifest
        if len(self.index) != 169028 or self.spec['global_max_tokens'] != 244:
            raise ValueError('PERIODIC_CANONICAL_METADATA_INCOMPLETE')


class EphemeralFeatures:
    def __init__(self, rows):
        self.rows = dict(rows)

    def get_batch(self, qids, *, device='cpu'):
        values = [self.rows[q] for q in qids]
        lengths = torch.tensor([len(v) for v in values], dtype=torch.long, device=device)
        features = torch.zeros(len(values), int(lengths.max()), values[0].shape[-1],
                               dtype=values[0].dtype, device=device)
        for i,value in enumerate(values):
            features[i,:len(value)] = value.to(device)
        return dict(features=features, lengths=lengths)

    def hashes(self):
        return {q:hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                for q,v in self.rows.items()}


def encode(trainer, questions, metadata, *, batch_size=16):
    """Use the live FSDP student, with normal collectives and detached outputs."""
    from verl import DataProto
    questions = list(questions)
    rows = {}
    for start in range(0,len(questions),batch_size):
        chunk = questions[start:start+batch_size]
        if len(chunk) % 4:
            raise ValueError('PERIODIC_FEATURE_BATCH_REQUIRES_FOUR_RANK_BALANCE')
        tokens = trainer.tokenizer([q.question for q in chunk],padding=False,
                                  truncation=False,add_special_tokens=True)['input_ids']
        ids = torch.full((len(chunk),244),trainer.tokenizer.pad_token_id,dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i,(q,token) in enumerate(zip(chunk,tokens)):
            row = metadata.index.get(q.qid)
            token_sha = hashlib.sha256(__import__('numpy').asarray(token,dtype='<i8').tobytes()).hexdigest()
            if (row is None or row['question_hash'] != q.question_hash or
                    row['length'] != len(token) or row['token_ids_sha256'] != token_sha or
                    not 0 < len(token) <= 244):
                raise ValueError('PERIODIC_QUESTION_OR_TOKEN_BINDING_CHANGED')
            ids[i,:len(token)] = torch.tensor(token)
            mask[i,:len(token)] = 1
        data = DataProto.from_dict(tensors=dict(input_ids=ids,attention_mask=mask))
        result = trainer.actor_rollout_wg.compute_question_probe_features(data)
        features = result.batch['question_probe_features']
        if features.shape != (len(chunk),244,2048) or features.requires_grad:
            raise ValueError('PERIODIC_LIVE_STUDENT_FEATURE_SHAPE_OR_GRADIENT')
        for i,q in enumerate(chunk):
            rows[q.qid] = features[i,:metadata.index[q.qid]['length']].detach().cpu().contiguous().clone()
    return EphemeralFeatures(rows)


def worker_features(worker, data):
    """Normal FSDP forward; no full-param scope, state_dict, or probe gradients."""
    from verl import DataProto
    from verl.utils.fsdp_utils import load_fsdp_param_and_grad, offload_fsdp_param_and_grad
    worker._assert_generation_phase_inactive()
    if not worker._is_actor:
        raise ValueError('PERIODIC_FEATURES_REQUIRE_CURRENT_ACTOR')
    module = worker.actor_module_fsdp
    backbone = module.module.model
    captured = []
    before = (getattr(worker.actor,'_optimizer_writes',0),getattr(worker.actor,'_complete_outers',0))
    device = torch.cuda.current_device()
    prior = module.training
    if worker._is_offload_param:
        load_fsdp_param_and_grad(module=module,device_id=device,load_grad=worker._is_offload_grad)
    data = data.to(torch.device('cuda',device))
    mask = data.batch['attention_mask']
    def hook(_module,_inputs,output):
        value = output[0] if isinstance(output,tuple) else output
        captured.append(value.detach().to(dtype=torch.bfloat16).cpu().clone())
    handle = backbone.layers[27].register_forward_hook(hook)
    try:
        module.eval()
        # Fork preserves both the training and sampling CUDA/CPU RNG boundary.
        with torch.random.fork_rng(devices=[device]), torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            output = module(input_ids=data.batch['input_ids'],attention_mask=mask,
                position_ids=(mask.cumsum(-1)-1).clamp_min(0),
                use_cache=False,output_hidden_states=False,return_dict=True)
            del output
        if len(captured) != 1 or not torch.isfinite(captured[0]).all():
            raise ValueError('PERIODIC_STUDENT_FEATURE_HOOK_FAILED')
        if before != (getattr(worker.actor,'_optimizer_writes',0),getattr(worker.actor,'_complete_outers',0)):
            raise ValueError('PERIODIC_ENCODER_UPDATED_STUDENT')
        return DataProto.from_dict(tensors={'question_probe_features':captured[0]})
    finally:
        handle.remove()
        module.train(prior)
        if worker._is_offload_param:
            offload_fsdp_param_and_grad(module=module,offload_grad=worker._is_offload_grad)
        torch.cuda.empty_cache()
