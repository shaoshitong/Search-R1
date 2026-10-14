"""Online fitting/checkpoint protocol for frozen-token three-block Qwen probes."""
import copy
from dataclasses import asdict
import torch
from search_g1.qprobe_model import OnlineQProbe, QProbeConfig, validate_label, count_moment_loss, S_MAX
from torch.nn import functional as F
from search_g1.qprobe_sequence import SequenceFeatures
from search_g1.qprobe_transformer import QwenBlockCountProbe, load_official_initialization


def canonical_qwen_config(config):
    value=config.to_dict()
    # Transformers4.46.3 from_dict injects this null alias. The real attention
    # implementation is separately bound and must still compare exactly.
    if value.get('attn_implementation') is None:value.pop('attn_implementation',None)
    return value


class OnlineSequenceQProbe(OnlineQProbe):
    checkpoint_schema_version = 2
    def __init__(self, model, config=None, feature_spec_sha='', device='cpu', train_micro=4, predict_micro=32):
        if not isinstance(model, QwenBlockCountProbe):raise TypeError('QPROBE_SEQUENCE_MODEL_REQUIRED')
        # Set up the existing private RNG, replay/transaction metadata; no old
        # checkpoint or learned pooled head enters this initialization.
        super().__init__(model.feature_dim, config, feature_spec_sha)
        self.model=model.to(device=device,dtype=torch.float32)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.config.head_learning_rate)
        self.replay_features=SequenceFeatures([],[])
        self.initialization=copy.deepcopy(model.initialization)
        if type(train_micro) is not int or train_micro<1 or type(predict_micro) is not int or predict_micro<1:raise ValueError('QPROBE_SEQUENCE_MICRO_INVALID')
        self.train_micro=train_micro;self.predict_micro=predict_micro

    def set_device(self, device):
        if next(self.model.parameters()).device == torch.device(device):return
        state=self.optimizer.state_dict();self.model.to(device=device,dtype=torch.float32)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.config.head_learning_rate)
        self.optimizer.load_state_dict(state)

    @classmethod
    def fresh(cls, cache, config, *, device='cpu'):
        path=cache.path/'initialization.pt'
        proof=cache.manifest['files'].get('initialization.pt')
        if not proof:raise ValueError('QPROBE_SEQUENCE_INITIALIZER_MISSING')
        model=load_official_initialization(path,expected_file_sha256=proof['sha256'],
            expected_encoder_files_sha256=cache.spec['encoder_files_sha256'])
        return cls(model,config,cache.spec_sha,device=device,train_micro=config.get('sequence_train_micro',4),predict_micro=config.get('sequence_predict_micro',32))

    def _validate_features(self, features):
        if not isinstance(features,SequenceFeatures) or not len(features) or features.feature_dim!=self.feature_dim:
            raise ValueError('QPROBE_SEQUENCE_FEATURE_LAYOUT_REQUIRED')
        return features.detach().cpu()
    def _concat_features(self,batches):return SequenceFeatures.cat(batches)
    def _state_features(self):return self.replay_features.state_dict()
    def _restore_features(self,state):return SequenceFeatures.from_state(state).cpu()
    def _forward(self,features):
        features=self._validate_features(features)
        parameter=next(self.model.parameters())
        hidden,mask,positions=features.padded(parameter.device,parameter.dtype)
        return self.model(hidden,mask,positions)
    def predict(self,features):
        self._validate_features(features);self.model.eval();outputs=[]
        # Never pad all5120 candidates/replay rows into one allocation.
        with torch.no_grad():
            for start in range(0,len(features),self.predict_micro):
                out=self._forward(features[start:start+self.predict_micro])
                outputs.append({k:v.detach().cpu() for k,v in out.items()})
        return {k:torch.cat([item[k] for item in outputs],dim=0) for k in outputs[0]}
    def _fit(self,outer):
        if not self.replay_labels:raise ValueError('QPROBE_CANNOT_FIT_WITHOUT_REAL_LABELS')
        counts=torch.tensor([x['correct_count'] for x in self.replay_labels],dtype=torch.long)
        weights=torch.tensor([2**(-(outer-x['source_outer'])/self.config.replay_half_life_outer) for x in self.replay_labels])
        weights=weights*(len(weights)/weights.sum());updates=0
        self.model.train()
        for _ in range(self.config.head_epochs_per_outer):
            order=torch.randperm(len(counts),generator=self.rng)
            for start in range(0,len(order),self.config.head_batch_size):
                batch=order[start:start+self.config.head_batch_size];self.optimizer.zero_grad(set_to_none=True)
                for offset in range(0,len(batch),self.train_micro):
                    index=batch[offset:offset+self.train_micro];out=self._forward(self.replay_features[index])
                    target=counts[index].to(out['logits'].device)
                    count_moment_loss(out,target,self.config)
                    loss=F.cross_entropy(out['logits'],target,reduction='none')
                    loss+=self.config.mean_loss_weight*(out['mean']-target.float()/5).square()
                    std=(target.float()*(5-target.float())/20).sqrt()
                    loss+=self.config.std_loss_weight*((out['expected_std']-std)/S_MAX).square()
                    # Sum over physical micro, divide by original logical batch:
                    # exactly the upstream global age-weighted mean estimator.
                    scalar=(loss*weights[index].to(loss.device)).sum()/len(batch)
                    if not torch.isfinite(scalar):raise ValueError('QPROBE_NONFINITE_TRAINING_LOSS')
                    scalar.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):raise ValueError('QPROBE_NONFINITE_GRADIENT')
                self.optimizer.step();updates+=1
                if any(not torch.isfinite(p).all() for p in self.model.parameters()):raise ValueError('QPROBE_NONFINITE_PARAMETERS')
        pred=self.predict(self.replay_features);objective,parts=count_moment_loss(pred,counts,self.config,weights)
        return dict(replay_groups=len(counts),head_optimizer_updates=updates,replay_weighted_loss=float(objective),**{k:float(v) for k,v in parts.items()})
    def state_dict(self):
        state=super().state_dict()
        state.update(probe_architecture='qwen2_decoder3_count6',sequence_train_micro=self.train_micro,sequence_predict_micro=self.predict_micro,qwen_config=canonical_qwen_config(self.model.config),
            attention_implementation=self.model.config._attn_implementation,initialization=copy.deepcopy(self.initialization))
        return state
    def _restore_state(self,state):
        if state.get('probe_architecture')!='qwen2_decoder3_count6' or state.get('initialization')!=self.initialization:
            raise ValueError('QPROBE_SEQUENCE_ARCHITECTURE_OR_SOURCE_CHANGED')
        if state.get('qwen_config')!=canonical_qwen_config(self.model.config) or state.get('attention_implementation')!=self.model.config._attn_implementation:raise ValueError('QPROBE_SEQUENCE_QWEN_CONFIG_CHANGED')
        if state.get('sequence_train_micro')!=self.train_micro or state.get('sequence_predict_micro')!=self.predict_micro:raise ValueError('QPROBE_SEQUENCE_MICRO_CHANGED')
        super()._restore_state(state)

    @classmethod
    def from_checkpoint_state(cls,state,*,expected_actor_snapshot_id=None,expected_feature_spec_sha=None,device='cpu'):
        if state.get('schema_version')!=2 or state.get('probe_architecture')!='qwen2_decoder3_count6':raise ValueError('QPROBE_SEQUENCE_CHECKPOINT_SCHEMA')
        if not state.get('feature_spec_sha') or not state.get('actor_snapshot_id'):raise ValueError('QPROBE_UNBOUND_SEQUENCE_CHECKPOINT')
        if expected_actor_snapshot_id is not None and state['actor_snapshot_id']!=expected_actor_snapshot_id:raise ValueError('QPROBE_ACTOR_BINDING_MISMATCH')
        if expected_feature_spec_sha is not None and state['feature_spec_sha']!=expected_feature_spec_sha:raise ValueError('QPROBE_FEATURE_SPEC_MISMATCH')
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2RMSNorm
        config=Qwen2Config.from_dict(copy.deepcopy(state['qwen_config']));config._attn_implementation=state['attention_implementation']
        with torch.random.fork_rng(devices=[]):
            source=Qwen2DecoderLayer(config,layer_idx=config.num_hidden_layers-1)
            norm=Qwen2RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
            model=QwenBlockCountProbe(source,norm,config)
        model.initialization=copy.deepcopy(state['initialization'])
        instance=cls(model,state['config'],state['feature_spec_sha'],device=device,train_micro=state['sequence_train_micro'],predict_micro=state['sequence_predict_micro'])
        instance._restore_state(state)
        if len(instance.replay_labels)!=len(instance.replay_features):raise ValueError('QPROBE_REPLAY_COUNT_MISMATCH')
        if type(instance.completed_outer) is not int or instance.completed_outer<0 or type(instance.update_count) is not int or instance.update_count<0:raise ValueError('QPROBE_INVALID_CHECKPOINT_COUNTERS')
        for label in instance.replay_labels:
            outer=label.get('source_outer')
            if type(outer) is not int or not 0<=outer<=instance.completed_outer:raise ValueError('QPROBE_CHECKPOINT_CONTAINS_FUTURE_LABELS')
            validate_label(label,outer=outer,**{k:label.get(k) for k in ('run_id','attempt_id','policy_snapshot_id','sampling_contract_sha','strict_scoring_contract_sha','retriever_version')})
            if {k:label[k] for k in ('sampling_contract_sha','strict_scoring_contract_sha','retriever_version')}!=instance.protocol:raise ValueError('QPROBE_CHECKPOINT_REPLAY_PROTOCOL_MISMATCH')
        if len(instance.replay_features):instance._validate_features(instance.replay_features)
        if any(not torch.isfinite(v).all() for v in instance.model.state_dict().values()):raise ValueError('QPROBE_NONFINITE_CHECKPOINT')
        for optimizer_state in instance.optimizer.state.values():
            if any(isinstance(v,torch.Tensor) and not torch.isfinite(v).all() for v in optimizer_state.values()):raise ValueError('QPROBE_NONFINITE_OPTIMIZER_CHECKPOINT')
        return instance
