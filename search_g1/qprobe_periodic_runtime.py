"""Periodic native probe transactions; ordinary actor outers never fit it."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from search_g1.qprobe_periodic_model import HEAD_KIND, REFIT_OUTERS, PeriodicQwenProbe, core_config
from search_g1.qprobe_periodic_features import QuestionMetadata, encode
from search_g1.qprobe_runtime import QProbeRuntime, _plain_config, _token_cache_binding, _sha, _write_json, _atomic_save


def validate_config(config):
    config = _plain_config(config)
    required = dict(head_kind=HEAD_KIND,candidate_questions=5120,selected_questions=512,
        responses_per_question=5,exploration_questions=0,ranking='binary_mixed_probability',
        loss_mode='token_binary_std_v1',head_epochs_per_outer=0,replay_window_outer=0,
        bootstrap_head_epochs=20,refit_every_outer=25,refit_epochs=20,refit_groups=5120,
        online_student_features=True,inherit_probe_optimizer=True,std_loss_weight=.3,
        mean_loss_weight=0.,count_aux_loss_weight=0.,target_statistic='sample_std',
        max_tokens=244,feature_dim=2048,source_layer=35,num_hidden_layers=36,
        head_batch_size=256,micro_batch_size=32,head_learning_rate=.0003,
        hidden_size=2048,num_attention_heads=16,num_key_value_heads=2,intermediate_size=11008,
        rms_norm_eps=1e-6,rope_theta=1000000.,max_position_embeddings=32768,
        attention_implementation='sdpa',head_seed=1729)
    if any(config.get(k) != v or isinstance(config.get(k),bool) != isinstance(v,bool)
           for k,v in required.items()):
        raise ValueError('PERIODIC_EFFECTIVE_RECIPE_CHANGED')
    if core_config(config).epochs != 20:
        raise ValueError('PERIODIC_NATIVE_EPOCH_BUDGET_CHANGED')
    return config


class PeriodicRuntime(QProbeRuntime):
    def __init__(self, trainer, config, base_dir, tokenizer=None):
        self.trainer, self.config = trainer, validate_config(config)
        self.base = Path(base_dir).resolve()
        self.output_dir = Path(self.config['output_dir']).resolve()
        if not self.output_dir.is_relative_to(self.base):
            raise ValueError('PERIODIC_OUTPUT_ESCAPES_OWN_NAMESPACE')
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.cache = QuestionMetadata(self.config['cache_dir'])
        self.feature_spec_sha = self.cache.spec_sha
        self.cache_binding = _token_cache_binding(self.cache)
        self.token_binary, self.native_qwen, self.head_kind = True, False, HEAD_KIND
        self.tokenizer, self.pending = tokenizer, None
        self.selection_rng = np.random.default_rng(self.config.get('seed',20261013))
        self.refit_rng = np.random.default_rng(self.config.get('seed',20261013)+25)
        snapshot = self.config.get('snapshot_path')
        if snapshot:
            state = torch.load(snapshot,map_location='cpu',weights_only=False)
            initial = Path(snapshot).parent/state['head_file']
            expected = state['head_sha256']
        else:
            initial = self.config['initial_head_checkpoint']
            expected = self.config['initial_head_checkpoint_sha256']
        self.head = PeriodicQwenProbe.load(initial,expected_checkpoint_sha=expected,
            expected_feature_spec_sha=self.feature_spec_sha,device=self.config.get('head_device','cpu'))
        if self.head.config != core_config(self.config):
            raise ValueError('PERIODIC_RESTORED_HEAD_CONFIG_CHANGED')
        self.completed_outer = self.head.completed_outer
        self.completed_attempts = set()
        self.initial_head_adoption = None
        self.execution_phase = 'STUDENT_READY'

    def bootstrap_initial_head(self):
        if self.head._adam_steps() != 400 or self.completed_outer != 0:
            raise ValueError('PERIODIC_INITIAL_WARMUP20_NOT_COMPLETE')
        ready_path = Path(self.config['initial_head_ready'])
        if _sha(ready_path) != self.config['initial_head_ready_sha256']:
            raise ValueError('PERIODIC_INITIAL_WARMUP_READY_SHA_CHANGED')
        self.initial_head_adoption = dict(logical_outer=0,head_adam_steps=400,
            no_head_refit=True,optimizer_reset=False,student_parameters_copied=False)
        return copy.deepcopy(self.initial_head_adoption)

    def score(self, questions):
        questions = list(questions)
        # The fixed initial bank is metadata only here; no feature-cache reader.
        provider = encode(self.trainer,questions,self.cache)
        self.head.attach_feature_provider(provider)
        try:
            return super().score(questions)
        finally:
            self.head.attach_feature_provider(None)
            del provider

    def complete_outer(self, outer, actor_snapshot_id, attempt_identity=None):
        pending = self.pending
        if (pending is None or pending['outer'] != outer or pending['labels'] is None or
                outer != self.completed_outer+1 or self.version != pending['prediction_version'] or
                pending['attempt_id'] in self.completed_attempts):
            raise ValueError('PERIODIC_INCOMPLETE_OR_REPEATED_ACTOR_OUTER')
        if attempt_identity is not None and attempt_identity['attempt_id'] != pending['attempt_id']:
            raise ValueError('PERIODIC_ATTEMPT_ID_CHANGED')
        event = dict(stage='question_probe_outer_committed',outer=outer,
            attempt_id=pending['attempt_id'],head_version=self.version,
            successful_actor_snapshot_id=actor_snapshot_id,prequential=pending['prequential'],
            head_update=dict(head_optimizer_updates=0,head_adam_steps_after=self.head._adam_steps(),
                             fit_epochs=0,replay_groups=0))
        _write_json(self.output_dir/'completed'/(pending['attempt_id']+'.json'),event)
        self.completed_outer = self.head.completed_outer = outer
        self.completed_attempts.add(pending['attempt_id'])
        self.pending = None
        self.post_refit_state_path = None
        self.execution_phase = 'PROBE_PENDING' if outer in REFIT_OUTERS else (
            'FINAL_EVALUATION_PENDING' if outer==200 else 'STUDENT_READY')
        return event

    def save(self,path,actor_snapshot_id,candidate_state=None,reward_probe_version=None):
        if self.pending is not None:
            raise ValueError('PERIODIC_CANNOT_SAVE_PENDING_ACTOR_LABELS')
        path = Path(path)
        head_path = path.with_name(path.name+'.head.pt')
        head_receipt = self.head.save(head_path,actor_snapshot_id=actor_snapshot_id)
        state = dict(schema_version=1,config=self.config,feature_spec_sha=self.feature_spec_sha,
            actor_snapshot_id=actor_snapshot_id,completed_outer=self.completed_outer,
            completed_attempts=sorted(self.completed_attempts),head_version=self.version,
            selection_rng=copy.deepcopy(self.selection_rng.bit_generator.state),
            refit_rng=copy.deepcopy(self.refit_rng.bit_generator.state),execution_phase=self.execution_phase,
            head_file=head_path.name,head_sha256=_sha(head_path),candidate_state=candidate_state,
            reward_probe_version=reward_probe_version,head_receipt=head_receipt,
            head_kind=HEAD_KIND,cache_binding=self.cache_binding,initial_head_adoption=self.initial_head_adoption)
        receipt = _atomic_save(path,state)
        return dict(receipt,actor_snapshot_id=actor_snapshot_id,completed_outer=self.completed_outer,
            head_version=self.version,feature_spec_sha=self.feature_spec_sha,head=head_receipt,
            execution_phase=self.execution_phase)

    def restore(self,path,expected_actor_snapshot_id):
        state = super().restore(path,expected_actor_snapshot_id)
        self.refit_rng.bit_generator.state = state['refit_rng']
        self.execution_phase = state['execution_phase']
        return state

    def refit_after_checkpoint(self,manager,checkpoint):
        if self.completed_outer not in REFIT_OUTERS or self.completed_outer==0:
            return None
        from search_g1.qprobe_periodic_refit import run
        return run(self.trainer,manager,self,Path(checkpoint))
