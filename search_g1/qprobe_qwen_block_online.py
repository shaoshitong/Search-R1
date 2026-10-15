"""Native Qwen copied-block head: adopted warmup, then formal-only replay.

Warmup teacher labels authenticate initialization and are discarded. Formal
fits retain current/previous7 complete outers and use mean(2*2**(-age)*loss).
Publication rollback belongs to the existing caller transaction.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
import hashlib

import torch

from search_g1.qprobe_model import _required_text, _sha_file
from search_g1.qprobe_token_online import OnlineTokenBinaryQProbe, _identity, _sha
from search_g1.qprobe_qwen_block_model import (
    QwenBlockQProbeConfig, QwenLastBlockQuestionProbe, QwenBlockWarmup,
    formal_replay_weights, native_implementation_binding,
)
from search_g1.qprobe_token_online import token_binary_group_loss


HEAD_KIND = 'qwen_native_lastblock_binary_v1'


@dataclass(frozen=True)
class QwenBlockOnlineConfig(QwenBlockQProbeConfig):
    head_kind: str = HEAD_KIND
    ranking: str = 'binary_mixed_probability'
    head_epochs_per_outer: int = 2
    replay_window_outer: int = 8

    def __post_init__(self):
        super().__post_init__()
        if (self.head_kind != HEAD_KIND or self.ranking != 'binary_mixed_probability' or
                self.head_epochs_per_outer != 2 or self.replay_window_outer != 8 or
                self.epochs != 10 or self.batch_size != 256):
            raise ValueError('QWEN_ONLINE_FORMAL_OR_BOOTSTRAP_RECIPE_CHANGED')

    def core_config(self):
        names = {field.name for field in fields(QwenBlockQProbeConfig)}
        return QwenBlockQProbeConfig(**{key:value for key,value in asdict(self).items() if key in names})

    @property
    def head_batch_size(self): return self.batch_size

    @property
    def head_learning_rate(self): return self.learning_rate

    @property
    def std_loss_weight(self): return self.auxiliary_weight

    @property
    def bootstrap_head_epochs(self): return self.epochs

    @classmethod
    def from_mapping(cls, values=None):
        values = dict(values or {})
        aliases = dict(head_batch_size='batch_size',head_learning_rate='learning_rate',
            std_loss_weight='auxiliary_weight',head_seed='seed',bootstrap_head_epochs='epochs',
            feature_dim='hidden_size',head_hidden_size='hidden_size')
        for source,target in aliases.items():
            if source in values:
                if target in values and source != 'head_seed' and values[target] != values[source]:
                    raise ValueError('QWEN_ONLINE_CONFLICTING_CONFIG: '+source)
                values[target] = values[source]
        allowed = {field.name for field in fields(cls)}
        return cls(**{key:value for key,value in values.items() if key in allowed})


class OnlineQwenBlockQProbe(OnlineTokenBinaryQProbe):
    """Reuse strict qid/provenance/atomic publication APIs, with native Qwen fit."""
    def __init__(self, feature_dim=2048, config=None, feature_spec_sha='', *,
                 feature_cache=None,feature_provider=None,device='cpu',block_state=None):
        self.config = config if isinstance(config,QwenBlockOnlineConfig) else QwenBlockOnlineConfig.from_mapping(config)
        if feature_dim != self.config.feature_dim:
            raise ValueError('QWEN_ONLINE_FEATURE_DIM_MISMATCH')
        self.feature_dim,self.feature_spec_sha,self.device = feature_dim,feature_spec_sha,torch.device(device)
        self._engine = QwenBlockWarmup.__new__(QwenBlockWarmup)
        self._engine.config,self._engine.device = self.config.core_config(),self.device
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(self.config.seed).get_state())
            self.model = QwenLastBlockQuestionProbe(self._engine.config)
        self.initial_block_proof = self.model.copy_official_block(block_state) if block_state is not None else {}
        self.initial_readout_proof = {}
        self.native_implementation = native_implementation_binding(self._engine.config)
        self.model.to(device=self.device,dtype=torch.float32)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.config.learning_rate)
        self.rng = torch.Generator(device='cpu').manual_seed(self.config.seed+1)
        self._engine.model,self._engine.optimizer,self._engine.rng = self.model,self.optimizer,self.rng
        self._engine.update_count,self._engine.fitted = 0,False
        self.replay_labels,self.replay_features = [],[]
        self.completed_outer,self.update_count,self.completed_attempts = 0,0,{}
        self.protocol,self.actor_snapshot_id = None,None
        self._initialization_provenance,self._fit_failed = {},False
        self._version_cache = None
        self.feature_provider = None
        if feature_provider is not None and feature_cache is not None and feature_provider is not feature_cache:
            raise ValueError('QWEN_ONLINE_CONFLICTING_FEATURE_PROVIDERS')
        provider = feature_provider if feature_provider is not None else feature_cache
        if provider is not None: self.attach_feature_provider(provider)

    @property
    def version(self):
        if self._version_cache is not None:
            return self._version_cache
        digest = hashlib.sha256(self.feature_spec_sha.encode())
        digest.update(_identity(asdict(self.config)).encode())
        digest.update(_identity(self.native_implementation).encode())
        for key,value in sorted(self.model.state_dict().items()):
            digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
        self._version_cache = f'qprobe-qwen:{self.completed_outer}:{self.update_count}:{digest.hexdigest()[:16]}'
        return self._version_cache

    def _prediction(self,qids):
        qids,lengths = self._resolve(qids)
        previous = self.model.training
        values = {key:[] for key in ('binary_logit','mixed_probability','predicted_statistic')}
        try:
            self.model.eval()
            with torch.no_grad():
                for index in torch.arange(len(qids)).split(self.config.micro_batch_size):
                    features,sizes = self._engine._batch(self.feature_provider,index,lengths,qids)
                    prediction = self.model(features,sizes)
                    for key in values: values[key].append(prediction[key].detach().cpu().clone())
        finally:
            self.model.train(previous)
        return {key:torch.cat(rows) for key,rows in values.items()}

    def predict(self, qids):
        if not self._engine.fitted or not self._initialization_provenance:
            raise ValueError('QWEN_ONLINE_REQUIRES_QUALIFIED_WARMUP_ADOPTION')
        return super().predict(qids)

    def bootstrap(self,*args,**kwargs):
        raise ValueError('QWEN_ONLINE_ADOPT_COMPLETED_WARMUP_WITHOUT_REFITTING')

    def _fit(self, outer):
        if self._fit_failed:
            raise ValueError('QWEN_ONLINE_FAILED_FIT_REQUIRES_EXPLICIT_TRANSACTION_RESTORE')
        qids,lengths = self._resolve(self.replay_features,self.replay_labels)
        counts = torch.tensor([row['correct_count'] for row in self.replay_labels],dtype=torch.long)
        ages = torch.tensor([outer-row['source_outer'] for row in self.replay_labels],dtype=torch.long)
        if bool(((ages < 0)|(ages >= 8)).any()): raise ValueError('QWEN_ONLINE_FORMAL_REPLAY_WINDOW')
        weights = formal_replay_weights(ages)  # Deliberately not renormalized.
        before,history,previous = self._adam_steps(),[],self.model.training
        self._version_cache = None
        try:
            self.model.train()
            for epoch in range(1,3):
                order = torch.randperm(len(qids),generator=self.rng)
                for batch_index,index in enumerate(order.split(self.config.batch_size)):
                    self.optimizer.zero_grad(set_to_none=True)
                    parts_sum = dict(binary_bce=0.,regression_mse_raw=0.,regression_mse_normalized=0.,regression_contribution=0.)
                    total,correct = 0.,0
                    for micro in index.split(self.config.micro_batch_size):
                        features,sizes = self._engine._batch(self.feature_provider,micro,lengths,qids)
                        prediction = self.model(features,sizes)
                        groups,parts = token_binary_group_loss(prediction,counts[micro],self.config)
                        age_weight = weights[micro].to(self.device)
                        loss = (groups*age_weight).sum()/len(index)
                        loss.backward(); total += float(loss.detach())
                        for key,value in parts.items():
                            parts_sum[key] += float((value.detach()*age_weight).sum()/len(index))
                        truth = ((counts[micro] > 0)&(counts[micro] < 5)).to(self.device)
                        correct += int(((prediction['mixed_probability'].detach() >= .5) == truth).sum())
                    row = dict(epoch=epoch,minibatch_index=batch_index,batch_size=len(index),
                        micro_batch_size=self.config.micro_batch_size,head_update_ordinal=len(history)+1,
                        head_adam_steps_before=self._adam_steps(),head_learning_rate=float(self.optimizer.param_groups[0]['lr']),
                        pre_step_weighted_loss=total,weight_reduction='mean(raw_age_weight_i*loss_i); no normalization',
                        binary_accuracy=correct/len(index),loss_parts=parts_sum)
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                        raise ValueError('QWEN_ONLINE_NONFINITE_GRADIENT')
                    self.optimizer.step(); self._engine.update_count += 1
                    if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                        raise ValueError('QWEN_ONLINE_NONFINITE_PARAMETERS')
                    row['head_adam_steps_after'] = self._adam_steps(); history.append(row)
            summary = self._training_summary(self._prediction(qids),counts,weights,2,len(history))
            summary['binary_accuracy'] = summary['binary_mixed']['accuracy']
            summary['teacher_replay_excluded'] = True
            summary['weight_reduction'] = 'mean(raw_age_weight_i*loss_i); no normalization'
            return dict(replay_groups=len(qids),head_optimizer_updates=len(history),
                replay_weighted_loss=summary['replay_weighted_loss'],total_loss=summary['total_loss'],
                loss_mode=HEAD_KIND,loss_identity=self.config.loss_identity,fit_epochs=2,
                head_batch_size=self.config.batch_size,head_learning_rate=float(self.optimizer.param_groups[0]['lr']),
                head_adam_steps_before=before,head_adam_steps_after=self._adam_steps(),
                training_summary=summary,epoch_history=[],minibatch_history=history,teacher_replay_excluded=True)
        except BaseException:
            self._fit_failed = True
            raise
        finally:
            self.model.train(previous)

    def update_completed_outer(self,outer,labels,features,**kwargs):
        if self._fit_failed: raise ValueError('QWEN_ONLINE_FAILED_FIT_REQUIRES_EXPLICIT_TRANSACTION_RESTORE')
        if not self._engine.fitted or not self._initialization_provenance:
            raise ValueError('QWEN_ONLINE_REQUIRES_QUALIFIED_WARMUP_ADOPTION')
        if type(outer) is not int or outer != self.completed_outer+1 or len(labels) != 512:
            raise ValueError('QWEN_ONLINE_REQUIRES_NEXT_COMPLETE_FORMAL_OUTER_512')
        return super().update_completed_outer(outer,labels,features,**kwargs)

    @classmethod
    def adopt_warmup(cls,path,labels,qids,*,expected_checkpoint_sha,expected_feature_spec_sha,
                     provenance,config=None,feature_cache=None,feature_provider=None,device='cpu'):
        digest = _sha(expected_checkpoint_sha,'warmup_checkpoint_sha')
        if _sha_file(path) != digest: raise ValueError('QPROBE_CHECKPOINT_SHA_MISMATCH')
        source = torch.load(path,map_location='cpu',weights_only=True)
        if source.get('token_feature_spec_sha256') != _sha(expected_feature_spec_sha,'feature_spec_sha'):
            raise ValueError('QPROBE_FEATURE_SPEC_MISMATCH')
        config = config if isinstance(config,QwenBlockOnlineConfig) else QwenBlockOnlineConfig.from_mapping(config)
        instance = cls(config.feature_dim,config,expected_feature_spec_sha,
            feature_cache=feature_cache,feature_provider=feature_provider,device=device)
        raw = source.get('head',{})
        if (raw.get('schema_version') != 1 or raw.get('role') != 'qwen_last_block_question_head' or
                raw.get('config') != asdict(config.core_config()) or raw.get('fitted') is not True or
                raw.get('update_count') != 200 or raw.get('loss_identity') != config.loss_identity or
                raw.get('native_implementation') != instance.native_implementation):
            raise ValueError('QWEN_ONLINE_UNQUALIFIED_NATIVE_WARMUP_STATE')
        provenance = copy.deepcopy(dict(provenance))
        actor_id = _required_text(provenance.get('actor_snapshot_id'),'actor_snapshot_id')
        valid,ordered,teacher_protocol = instance._bootstrap_labels(labels,qids)
        original_provenance = copy.deepcopy(source.get('label_producer_provenance'))
        if (isinstance(original_provenance,dict) and original_provenance.get('protocol') is not None and
                original_provenance['protocol'] != teacher_protocol):
            raise ValueError('QWEN_ONLINE_TEACHER_PROTOCOL_CHANGED')
        instance.initial_block_proof = copy.deepcopy(raw['initial_block_proof'])
        instance.initial_readout_proof = copy.deepcopy(raw['initial_readout_proof'])
        instance.model.load_state_dict(raw['model'],strict=True)
        instance.optimizer.load_state_dict(raw['optimizer']); instance.rng.set_state(raw['private_rng'])
        instance._engine.update_count,instance._engine.fitted = 200,True
        instance.actor_snapshot_id = actor_id
        # No teacher row or qid enters formal replay, including outer1's first fit.
        instance.replay_labels,instance.replay_features,instance.protocol = [],[],None
        instance._initialization_provenance = dict(provenance,kind='qualified_native_qwen_warmup_checkpoint_adoption',
            warmup_checkpoint_sha256=digest,warmup_head_adam_steps=200,adopted_at_completed_outer=0,
            bootstrap_groups=5120,teacher_replay_excluded=True,formal_replay_groups_at_adoption=0,
            formal_update_count=0,
            teacher_label_identity_sha256=_identity(dict(labels=valid,qids=ordered)),teacher_protocol=teacher_protocol,
            no_refit=True,initial_block_proof_sha256=_identity(instance.initial_block_proof),
            native_implementation_sha256=_identity(instance.native_implementation),
            original_label_producer_provenance=original_provenance)
        instance._validate_state()
        return instance

    def verify_native_provenance(self):
        proof = self.initial_block_proof
        expected = self.model.block.state_dict()
        provenance = self._initialization_provenance
        if (self.native_implementation != native_implementation_binding(self.config.core_config()) or
                proof.get('source_layer') != self.config.source_layer or proof.get('tensor_count') != len(expected) or
                proof.get('all_copied_parameters_trainable') is not True or set(proof.get('tensors',{})) != set(expected)):
            raise ValueError('QWEN_ONLINE_NATIVE_IMPLEMENTATION_OR_COPIED_BLOCK_PROOF_CHANGED')
        for key,value in expected.items():
            row = proof['tensors'][key]
            if (row.get('source_key') != f'model.layers.{self.config.source_layer}.'+key or
                    row.get('shape') != list(value.shape) or row.get('exact_cast_equal') is not True):
                raise ValueError('QWEN_ONLINE_COPIED_BLOCK_PARAMETER_BINDING_CHANGED')
            _sha(row.get('source_sha256'),'official_tensor_sha'); _sha(row.get('copied_fp32_sha256'),'initial_fp32_tensor_sha')
        if (not self.initial_readout_proof.get('fresh') or self.initial_readout_proof.get('seed') != self.config.seed or
                provenance.get('kind') != 'qualified_native_qwen_warmup_checkpoint_adoption' or
                provenance.get('warmup_head_adam_steps') != 200 or provenance.get('adopted_at_completed_outer') != 0 or
                provenance.get('bootstrap_groups') != 5120 or provenance.get('no_refit') is not True or
                provenance.get('teacher_replay_excluded') is not True or provenance.get('formal_replay_groups_at_adoption') != 0 or
                provenance.get('initial_block_proof_sha256') != _identity(proof) or
                provenance.get('native_implementation_sha256') != _identity(self.native_implementation)):
            raise ValueError('QWEN_ONLINE_WARMUP_ADOPTION_PROVENANCE_CHANGED')
        _sha(provenance.get('warmup_checkpoint_sha256'),'warmup_checkpoint_sha')
        _sha(provenance.get('teacher_label_identity_sha256'),'teacher_label_identity_sha')
        _sha(self.initial_readout_proof.get('weight_sha256'),'initial_readout_weight_sha')
        _sha(self.initial_readout_proof.get('bias_sha256'),'initial_readout_bias_sha')
        return dict(accepted=True,teacher_replay_excluded=True,warmup_head_adam_steps=200,
            initial_block_proof_sha256=_identity(proof),native_implementation_sha256=_identity(self.native_implementation))

    def _validate_state(self):
        super()._validate_state()
        window = min(self.completed_outer,8)
        expected_steps = 200+4*(window*(window+1)//2+8*max(0,self.completed_outer-8))
        if (self.update_count != self.completed_outer or not self._engine.fitted or
                self._engine.update_count != expected_steps or len(self.completed_attempts) != self.update_count or
                type(self._fit_failed) is not bool or
                any(row['source_outer'] <= 0 for row in self.replay_labels)):
            raise ValueError('QWEN_ONLINE_INVALID_FORMAL_COUNTERS_OR_TEACHER_REPLAY')
        self.verify_native_provenance()

    def state_dict(self):
        state = super().state_dict()
        state.update(head_kind=HEAD_KIND,config=asdict(self.config),native_implementation=copy.deepcopy(self.native_implementation),
            initial_block_proof=copy.deepcopy(self.initial_block_proof),initial_readout_proof=copy.deepcopy(self.initial_readout_proof),
            model_training=self.model.training)
        return state

    def load_state_dict(self,state):
        if (state.get('schema_version') != 1 or state.get('head_kind') != HEAD_KIND or
                QwenBlockOnlineConfig.from_mapping(state.get('config')) != self.config or
                state.get('feature_dim') != self.feature_dim or state.get('feature_spec_sha') != self.feature_spec_sha or
                state.get('loss_identity') != self.config.loss_identity or
                state.get('native_implementation') != self.native_implementation or
                state.get('replay_identity_sha256') != _identity(dict(labels=state['replay_labels'],qids=state['replay_features']))):
            raise ValueError('QWEN_ONLINE_CHECKPOINT_NAMESPACE_OR_REPLAY_MISMATCH')
        self.model.load_state_dict(state['model'],strict=True); self.optimizer.load_state_dict(state['optimizer'])
        self._version_cache = None
        self.rng.set_state(state['rng'])
        for key in ('replay_labels','replay_features','completed_outer','update_count','completed_attempts','protocol','actor_snapshot_id',
                    'initial_block_proof','initial_readout_proof'):
            setattr(self,key,copy.deepcopy(state[key]))
        self._engine.update_count,self._engine.fitted = state['head_adam_updates'],state['fitted']
        self._fit_failed = state['fit_failed']
        self._initialization_provenance = copy.deepcopy(state['initialization_provenance'])
        self.model.train(state['model_training'])
        self._validate_state()

    def save(self,path,*,actor_snapshot_id):
        receipt = super().save(path,actor_snapshot_id=actor_snapshot_id)
        receipt.update(head_kind=HEAD_KIND,loss_mode=HEAD_KIND,teacher_replay_excluded=True,
                       native_provenance=self.verify_native_provenance())
        return receipt
