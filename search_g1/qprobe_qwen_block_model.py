"""One copied native Qwen2 decoder layer and a fresh full-token binary readout.

No Qwen attention, RoPE, normalization, residual, or MLP is reimplemented here.
The only new model operation is masked global padding/flattening and Linear2.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path

import torch
from torch import nn

from search_g1.qprobe_token_model import _integer_vector, token_binary_loss


def _tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class QwenBlockQProbeConfig:
    max_tokens: int = 244
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    intermediate_size: int = 11008
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    max_position_embeddings: int = 32768
    source_layer: int = 35
    num_hidden_layers: int = 36
    attention_implementation: str = 'sdpa'
    epochs: int = 10
    batch_size: int = 256
    micro_batch_size: int = 32
    learning_rate: float = 3e-4
    auxiliary_weight: float = .3
    target_statistic: str = 'sample_std'
    top_k: int = 512
    seed: int = 1729

    def __post_init__(self):
        for key in ('max_tokens','hidden_size','num_attention_heads','num_key_value_heads',
                    'intermediate_size','max_position_embeddings','num_hidden_layers','epochs',
                    'batch_size','micro_batch_size','top_k'):
            if type(getattr(self,key)) is not int or getattr(self,key) <= 0:
                raise ValueError('QWEN_BLOCK_INVALID_CONFIG: '+key)
        if (self.hidden_size%self.num_attention_heads or
                (self.hidden_size//self.num_attention_heads)%2 or
                self.num_attention_heads%self.num_key_value_heads or
                type(self.source_layer) is not int or not 0 <= self.source_layer < self.num_hidden_layers or
                self.max_tokens > self.max_position_embeddings or self.micro_batch_size > self.batch_size):
            raise ValueError('QWEN_BLOCK_INVALID_HEAD_OR_POSITION_CONFIG')
        if (self.target_statistic != 'sample_std' or self.auxiliary_weight != .3 or
                self.attention_implementation != 'sdpa' or type(self.seed) is not int or not 0 <= self.seed < 2**63):
            raise ValueError('QWEN_BLOCK_OBJECTIVE_OR_NATIVE_ATTENTION_CHANGED')
        for key in ('learning_rate','rms_norm_eps','rope_theta'):
            if not math.isfinite(getattr(self,key)) or getattr(self,key) <= 0:
                raise ValueError('QWEN_BLOCK_INVALID_CONFIG: '+key)

    @property
    def feature_dim(self):
        return self.hidden_size

    @property
    def statistic_bound(self):
        return math.sqrt(.3)

    @property
    def loss_identity(self):
        return 'BCEWithLogits(0<K<5)+0.3*((predicted_sample_std-sqrt(K*(5-K)/20))/sqrt(0.3))^2'

    def native_config(self):
        from transformers import Qwen2Config
        config = Qwen2Config(hidden_size=self.hidden_size, intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,rope_theta=self.rope_theta,hidden_act='silu',
            attention_dropout=0.,use_sliding_window=False,use_cache=False)
        config._attn_implementation = self.attention_implementation
        return config


def native_implementation_binding(config=None):
    import transformers
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
    config = config or QwenBlockQProbeConfig()
    objects = dict(decoder_layer=Qwen2DecoderLayer,rotary_embedding=Qwen2RotaryEmbedding,
                   causal_padding_mask=_prepare_4d_causal_attention_mask_for_sdpa)
    binding = dict(transformers_version=transformers.__version__,torch_version=str(torch.__version__),
        attention_implementation=config.attention_implementation,
        implementations={key:dict(qualified_name=value.__module__+'.'+value.__name__,
            source_sha256=hashlib.sha256(Path(inspect.getsourcefile(value)).read_bytes()).hexdigest())
            for key,value in objects.items()},
        decoder_forward_parameters=list(inspect.signature(Qwen2DecoderLayer.forward).parameters),
        rotary_forward_parameters=list(inspect.signature(Qwen2RotaryEmbedding.forward).parameters),
        native_config=config.native_config().to_dict())
    # HF config includes integer id2label keys; persisted JSON uses string keys.
    return json.loads(json.dumps(binding,sort_keys=True,allow_nan=False))


class QwenLastBlockQuestionProbe(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding
        self.config = config
        native_config = config.native_config()
        self.block = Qwen2DecoderLayer(native_config,layer_idx=config.source_layer)
        self.rotary = Qwen2RotaryEmbedding(config=native_config)
        self.output = nn.Linear(config.max_tokens*config.hidden_size,2)
        self._forward_parameters = inspect.signature(self.block.forward).parameters
        if 'position_embeddings' not in self._forward_parameters:
            raise ValueError('QWEN_BLOCK_REQUIRES_NATIVE_EXTERNAL_ROTARY_API')

    def copy_official_block(self, state):
        prefix = f'model.layers.{self.config.source_layer}.'
        if not isinstance(state,dict) or not state:
            raise ValueError('QWEN_BLOCK_MISSING_OFFICIAL_TENSORS')
        prefixed = any(key.startswith(prefix) for key in state)
        if prefixed and not all(key.startswith(prefix) for key in state):
            raise ValueError('QWEN_BLOCK_MIXED_OFFICIAL_PREFIX')
        values = {key[len(prefix):] if prefixed else key:value for key,value in state.items()}
        expected = self.block.state_dict()
        if set(values) != set(expected):
            raise ValueError('QWEN_BLOCK_OFFICIAL_PARAMETER_KEYS_CHANGED')
        proof, copied = {}, {}
        for key,value in values.items():
            if (not isinstance(value,torch.Tensor) or value.shape != expected[key].shape or
                    not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError('QWEN_BLOCK_OFFICIAL_PARAMETER_SHAPE_OR_FINITE_CHANGED: '+key)
            copied[key] = value.detach().to(device='cpu',dtype=torch.float32).clone()
            proof[key] = dict(source_key=prefix+key,shape=list(value.shape),
                source_dtype=str(value.dtype).removeprefix('torch.'),source_sha256=_tensor_sha(value),
                copied_fp32_sha256=_tensor_sha(copied[key]),exact_cast_equal=True)
        self.block.load_state_dict(copied,strict=True)
        for key,value in self.block.state_dict().items():
            if not torch.equal(value.detach().cpu(),copied[key]):
                raise ValueError('QWEN_BLOCK_OFFICIAL_COPY_READBACK_FAILED')
        self.block.requires_grad_(True)
        return dict(source_layer=self.config.source_layer,tensor_count=len(proof),tensors=proof,
                    all_copied_parameters_trainable=all(p.requires_grad for p in self.block.parameters()))

    def forward(self, features, lengths):
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
        if (not isinstance(features,torch.Tensor) or features.ndim != 3 or not len(features) or
                features.shape[2] != self.config.hidden_size or features.shape[1] > self.config.max_tokens):
            raise ValueError('QWEN_BLOCK_FEATURE_SHAPE_OR_GLOBAL_BMAX')
        lengths = _integer_vector(lengths,'lengths',len(features)).to(features.device)
        if bool(((lengths < 1)|(lengths > features.shape[1])|(lengths > self.config.max_tokens)).any()):
            raise ValueError('QWEN_BLOCK_INVALID_OR_OVERSIZE_TOKEN_LENGTH')
        active = int(lengths.max())
        features = features[:,:active].detach().to(dtype=self.output.weight.dtype)
        mask = torch.arange(active,device=features.device)[None,:] < lengths[:,None]
        features = features.masked_fill(~mask[:,:,None],0.)
        if not torch.isfinite(features).all():
            raise ValueError('QWEN_BLOCK_NONFINITE_REAL_TOKEN')
        position = torch.arange(active,device=features.device)
        position_ids = position[None,:]
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(mask,features.shape[:2],features,0)
        position_embeddings = self.rotary(features,position_ids)
        kwargs = dict(attention_mask=attention_mask,position_ids=position_ids,
                      position_embeddings=position_embeddings,use_cache=False,output_attentions=False)
        if 'cache_position' in self._forward_parameters:
            kwargs['cache_position'] = position
        hidden = self.block(features,**kwargs)
        if isinstance(hidden,tuple): hidden = hidden[0]
        if not isinstance(hidden,torch.Tensor) or hidden.shape != features.shape:
            raise ValueError('QWEN_BLOCK_NATIVE_OUTPUT_SHAPE_CHANGED')
        hidden = hidden.masked_fill(~mask[:,:,None],0.)
        hidden = torch.nn.functional.pad(hidden,(0,0,0,self.config.max_tokens-active))
        raw = self.output(hidden.flatten(start_dim=1))
        if not torch.isfinite(raw).all():
            raise ValueError('QWEN_BLOCK_NONFINITE_OUTPUT')
        return dict(raw_outputs=raw,binary_logit=raw[:,0],mixed_probability=raw[:,0].sigmoid(),
                    predicted_statistic=raw[:,1].sigmoid()*self.config.statistic_bound)


def formal_replay_weights(ages):
    """Current and previous7 complete formal outers only; no normalization."""
    ages = _integer_vector(ages,'formal_replay_ages')
    if bool((ages < 0).any()): raise ValueError('QWEN_BLOCK_FUTURE_FORMAL_REPLAY')
    return torch.where(ages < 8,2.*torch.pow(2.,-ages.float()),torch.zeros_like(ages,dtype=torch.float32))


def formal_replay_loss(prediction, counts, ages, config):
    """Exactly mean(w(age)*[BCE+.3 normalized std MSE]); teacher rows excluded upstream."""
    counts = _integer_vector(counts,'counts',len(prediction['binary_logit'])).to(prediction['binary_logit'].device)
    if bool(((counts < 0)|(counts > 5)).any()): raise ValueError('QWEN_BLOCK_K_OUTSIDE_G5')
    weights = formal_replay_weights(ages).to(counts.device)
    if len(weights) != len(counts): raise ValueError('QWEN_BLOCK_REPLAY_WEIGHT_LENGTH')
    truth = ((counts > 0)&(counts < 5)).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(prediction['binary_logit'],truth,reduction='none')
    target = (counts.float()*(5-counts.float())/20).sqrt()
    normalized = ((prediction['predicted_statistic']-target)/config.statistic_bound).square()
    loss = (weights*(bce+config.auxiliary_weight*normalized)).mean()
    if not torch.isfinite(loss): raise ValueError('QWEN_BLOCK_NONFINITE_FORMAL_REPLAY_LOSS')
    return loss


class QwenBlockWarmup:
    def __init__(self, config, block_state, device='cpu'):
        self.config = config if isinstance(config,QwenBlockQProbeConfig) else QwenBlockQProbeConfig(**config)
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(self.config.seed).get_state())
            self.model = QwenLastBlockQuestionProbe(self.config)
        self._initial_block_proof = self.model.copy_official_block(block_state)
        self.initial_readout_proof = dict(fresh=True,seed=self.config.seed,
            weight_sha256=_tensor_sha(self.model.output.weight),bias_sha256=_tensor_sha(self.model.output.bias))
        self.native_implementation = native_implementation_binding(self.config)
        self.model.to(device=self.device,dtype=torch.float32)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.config.learning_rate)
        self.rng = torch.Generator(device='cpu').manual_seed(self.config.seed+1)
        self.update_count,self.fitted = 0,False

    @property
    def initial_block_proof(self):
        return copy.deepcopy(self._initial_block_proof)

    def _adam_steps(self):
        values = [int(state['step'].item()) for state in self.optimizer.state.values() if 'step' in state]
        if values and len(set(values)) != 1: raise ValueError('QWEN_BLOCK_INCONSISTENT_ADAM_STEPS')
        return values[0] if values else 0

    def state_dict(self):
        return copy.deepcopy(dict(schema_version=1,role='qwen_last_block_question_head',config=asdict(self.config),
            model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),private_rng=self.rng.get_state(),
            update_count=self.update_count,fitted=self.fitted,loss_identity=self.config.loss_identity,
            initial_block_proof=self._initial_block_proof,initial_readout_proof=self.initial_readout_proof,
            native_implementation=self.native_implementation))

    def _batch(self, source, index, lengths, ids):
        from search_g1.qprobe_token_model import TokenBinaryWarmup
        return TokenBinaryWarmup._batch(self,source,index,lengths,ids)

    def _summary(self, source, lengths, counts, ids, epoch):
        prior_mode = self.model.training
        probability,statistic = [],[]
        sums = dict(total_loss=0.,binary_bce=0.,regression_mse_raw=0.,regression_mse_normalized=0.,regression_contribution=0.)
        try:
            self.model.eval()
            with torch.no_grad():
                for index in torch.arange(len(ids)).split(self.config.micro_batch_size):
                    features,sizes = self._batch(source,index,lengths,ids)
                    prediction = self.model(features,sizes)
                    loss,parts = token_binary_loss(prediction,counts[index],self.config)
                    for key,value in dict(total_loss=loss,**parts).items(): sums[key] += float(value)*len(index)
                    probability.append(prediction['mixed_probability'].detach().cpu())
                    statistic.append(prediction['predicted_statistic'].detach().cpu())
        finally:
            self.model.train(prior_mode)
        p,std_prediction = torch.cat(probability),torch.cat(statistic)
        truth,positive = (counts > 0)&(counts < 5),p >= .5
        tp,tn,fp,fn = [int(value.sum()) for value in (truth&positive,~truth&~positive,~truth&positive,truth&~positive)]
        std = (counts.float()*(5-counts.float())/20).sqrt()
        order = sorted(range(len(ids)),key=lambda i:(-float(p[i]),hashlib.sha256(ids[i].encode()).hexdigest()))
        chosen = torch.tensor(order[:min(self.config.top_k,len(ids))],dtype=torch.long)
        return dict(epoch=epoch,training_groups=len(ids),dataset_scope='all_training_groups_not_independent_holdout',
            **{key:value/len(ids) for key,value in sums.items()},binary_accuracy=(tp+tn)/len(ids),accuracy=(tp+tn)/len(ids),
            binary_confusion=dict(true_positive=tp,true_negative=tn,false_positive=fp,false_negative=fn),threshold=.5,
            threshold_tie_count=int((p == .5).sum()),true_mixed_fraction=float(truth.float().mean()),
            predicted_mixed_fraction=float(positive.float().mean()),sample_std_mae=float((std_prediction-std).abs().mean()),
            target_statistic='sample_std',regression_bound=self.config.statistic_bound,auxiliary_weight=.3,
            loss_identity=self.config.loss_identity,actual_head_adam_updates=self._adam_steps(),batch_size=self.config.batch_size,
            micro_batch_size=self.config.micro_batch_size,learning_rate=float(self.optimizer.param_groups[0]['lr']),
            top_k=dict(requested_k=self.config.top_k,selected_count=len(chosen),selection_metric='binary_p_mixed',
                tie_break='sha256(question_id)',mixed_precision=float(truth[chosen].float().mean()),
                true_sample_std_mean=float(std[chosen].mean()),
                selected_question_ids_sha256=hashlib.sha256(json.dumps([ids[i] for i in chosen.tolist()],
                    separators=(',',':'),ensure_ascii=False).encode()).hexdigest()))

    def fit(self, token_features, lengths, counts, ids, *, progress_callback=None):
        if self.fitted or self.update_count or self.optimizer.state:
            raise ValueError('QWEN_BLOCK_WARMUP_REQUIRES_FRESH_COPIED_BLOCK_READOUT_AND_ADAM')
        ids = list(ids)
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(qid,str) or not qid for qid in ids):
            raise ValueError('QWEN_BLOCK_REQUIRES_UNIQUE_QUESTION_IDS')
        lengths = _integer_vector(lengths,'lengths',len(ids)).cpu()
        counts = _integer_vector(counts,'counts',len(ids)).cpu()
        if bool(((lengths < 1)|(lengths > self.config.max_tokens)).any()) or bool(((counts < 0)|(counts > 5)).any()):
            raise ValueError('QWEN_BLOCK_INVALID_LENGTH_OR_G5_COUNT')
        if not hasattr(token_features,'get_batch') and len(token_features) != len(ids):
            raise ValueError('QWEN_BLOCK_FEATURE_COUNT_MISMATCH')
        previous = self.model.training
        epochs,minis = [],[]
        def publish(stage,row):
            if progress_callback is not None: progress_callback(dict(stage=stage,**copy.deepcopy(row)))
        try:
            self.model.train()
            epochs.append(self._summary(token_features,lengths,counts,ids,0)); publish('epoch_training_summary',epochs[-1])
            for epoch in range(1,self.config.epochs+1):
                order = torch.randperm(len(ids),generator=self.rng)
                for batch_index,index in enumerate(order.split(self.config.batch_size)):
                    self.optimizer.zero_grad(set_to_none=True)
                    scalars = dict(pre_step_loss=0.,binary_bce=0.,regression_mse_raw=0.,
                                   regression_mse_normalized=0.,regression_contribution=0.)
                    correct = 0
                    for micro in index.split(self.config.micro_batch_size):
                        features,sizes = self._batch(token_features,micro,lengths,ids)
                        prediction = self.model(features,sizes)
                        loss,parts = token_binary_loss(prediction,counts[micro],self.config)
                        fraction = len(micro)/len(index)
                        (loss*fraction).backward()  # True full-mini mean; exactly one Adam per256.
                        for key,value in dict(pre_step_loss=loss,**parts).items(): scalars[key] += float(value.detach())*fraction
                        truth = ((counts[micro] > 0)&(counts[micro] < 5)).to(self.device)
                        correct += int(((prediction['mixed_probability'].detach() >= .5) == truth).sum())
                    row = dict(epoch=epoch,minibatch_index=batch_index,batch_size=len(index),
                        micro_batch_size=self.config.micro_batch_size,head_update_ordinal=self.update_count+1,
                        head_adam_steps_before=self._adam_steps(),learning_rate=float(self.optimizer.param_groups[0]['lr']),
                        binary_accuracy=correct/len(index),accuracy=correct/len(index),**scalars)
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                        raise ValueError('QWEN_BLOCK_NONFINITE_GRADIENT')
                    self.optimizer.step(); self.update_count += 1
                    if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                        raise ValueError('QWEN_BLOCK_NONFINITE_PARAMETERS')
                    row['head_adam_steps_after'] = self._adam_steps(); minis.append(row); publish('head_optimizer_update',row)
                epochs.append(self._summary(token_features,lengths,counts,ids,epoch)); publish('epoch_training_summary',epochs[-1])
            self.fitted = True
            return dict(schema_version=1,role='qwen_last_block_question_head_training_warmup',config=asdict(self.config),
                training_groups=len(ids),fit_epochs=self.config.epochs,actual_head_adam_updates=self._adam_steps(),
                expected_head_adam_updates=self.config.epochs*math.ceil(len(ids)/self.config.batch_size),
                epoch_history=epochs,minibatch_history=minis,training_summary=epochs[-1],primary_report='binary_accuracy',
                dataset_scope='training_only_not_independent_holdout',initial_block_proof=self.initial_block_proof,
                native_implementation=self.native_implementation,actor_updated=False,reference_updated=False,formal_training_started=False)
        finally:
            # Failure leaves truthful partial parameters/Adam/RNG/counters; no automatic retry.
            self.model.train(previous)
