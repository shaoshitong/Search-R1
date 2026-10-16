"""Driver transaction for four-rank trajectory EM probing and paired snapshots."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from search_g1.prefix20_contract import CONTRACT,mixed_probability,probe_batch_plan

ARCHITECTURE='qwen2_prefix20_trajectory_em_v1'
FEATURE_SPEC_SHA=hashlib.sha256(json.dumps(dict(
    encoder_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',feature_layer=27,
    feature_boundary='post_block',policy_tokens=20,head='official_block35_linear20',
    question_context=True,question_hidden_passed_to_head=False,encoder_frozen=True),
    sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_config(config):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(config):config=OmegaConf.to_container(config,resolve=True)
    except ImportError:pass
    config=dict(config)
    expected=dict(probe_architecture=ARCHITECTURE,candidate_questions=5120,selected_questions=512,
        exploration_questions=0,responses_per_question=5,head_learning_rate=1e-6,
        head_batch_size=256,head_epochs_per_outer=1,replay_window_outer=0)
    fields=set(expected)|{'model_dir','head_safetensors','probe_config','output_dir','snapshot_path','seed'}
    if set(config)-fields:raise ValueError('PREFIX20_UNKNOWN_CONFIGURATION_FIELDS')
    for key,value in expected.items():
        if config.get(key,value)!=value or isinstance(config.get(key,value),bool):
            raise ValueError('PREFIX20_AUTHORIZED_SETTING_CHANGED: '+key)
        if type(value) is int and type(config.get(key,value)) is not int:
            raise ValueError('PREFIX20_INTEGER_BUDGET_REQUIRED: '+key)
        config[key]=value
    for key in ('model_dir','head_safetensors','probe_config','output_dir'):
        if not isinstance(config.get(key),str) or not Path(config[key]).is_absolute():
            raise ValueError('PREFIX20_EXPLICIT_REMOTE_PATH_REQUIRED: '+key)
    if type(config.get('seed',1729)) is not int:raise ValueError('PREFIX20_INTEGER_SEED_REQUIRED')
    config.setdefault('seed',1729);config.setdefault('snapshot_path',None)
    return config


def worker_transport(group):
    if group.world_size!=4 or len(group.workers)!=4:raise ValueError('PREFIX20_FOUR_ACTOR_WORKERS_REQUIRED')
    import ray
    endpoints=[]
    for worker in group.workers:
        methods=set(getattr(worker,'_ray_method_signatures',{}))
        name=('actor_rollout_prefix20_probe_operation' if 'actor_rollout_prefix20_probe_operation' in methods
              else 'prefix20_probe_operation' if 'prefix20_probe_operation' in methods else None)
        if name is None:raise ValueError('PREFIX20_NATIVE_REGISTERED_ENDPOINT_MISSING')
        endpoints.append(getattr(worker,name))
    def transport(operation,payloads):
        if len(payloads)!=4:raise ValueError('PREFIX20_EXPLICIT_FOUR_RANK_PAYLOADS_REQUIRED')
        refs=[endpoint.remote(operation,payload) for endpoint,payload in zip(endpoints,payloads)]
        results=[];failures=[]
        # All rank calls are dispatched first. Drain every result even when one
        # fails, so the caller cannot overlap the next operation with live work.
        for ref in refs:
            try:results.append(ray.get(ref))
            except Exception as error:results.append(None);failures.append(type(error).__name__)
        if failures:raise RuntimeError('PREFIX20_WORKER_OPERATION_FAILED: '+str(failures))
        if [r['rank'] for r in results]!=list(range(4)):raise RuntimeError('PREFIX20_WORKER_RANK_ORDER_CHANGED')
        return results
    return transport


def binary_metrics(probabilities,labels):
    p=np.asarray(probabilities,dtype=np.float64);y=np.asarray(labels,dtype=np.float64)
    if p.shape!=y.shape or p.ndim!=1 or not len(p) or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
        raise ValueError('PREFIX20_INVALID_BINARY_METRIC_ROWS')
    clipped=np.clip(p,1e-12,1-1e-12)
    return dict(binary_accuracy=float(((p>=.5)==y).mean()),brier=float(((p-y)**2).mean()),
                bce=float(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)).mean()),
                positive_fraction=float(y.mean()),trajectories=len(y))


class Prefix20Runtime:
    def __init__(self,config,base_dir,worker_group=None,*,transport=None):
        self.config=validate_config(config);self.base=Path(base_dir).resolve()
        self.output_dir=Path(self.config['output_dir']).resolve()
        if not self.output_dir.is_relative_to(self.base):raise ValueError('PREFIX20_OUTPUT_MUST_BE_INSIDE_NEW_RUN')
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.transport=transport or worker_transport(worker_group)
        self.completed_outer=0;self.pending=None;self.completed_attempts=set()
        self.selection_rng=np.random.default_rng(self.config['seed'])
        self.feature_spec_sha=FEATURE_SPEC_SHA
        init={k:self.config[k] for k in ('model_dir','head_safetensors','probe_config')}
        receipts=self.transport('init',[dict(init) for _ in range(4)])
        if any(r['world_size']!=4 or r['completed_outer']!=0 or r['actual_steps']!=0 or not r['pretrained_head_loaded'] for r in receipts):
            raise RuntimeError('PREFIX20_PRETRAINED_INITIALIZATION_INCOMPLETE')

    @property
    def version(self):return 'prefix20-epoch68-outer-%04d'%self.completed_outer

    def score(self,rows):
        ids=[r['trajectory_id'] for r in rows]
        if len(ids)!=len(set(ids)) or any('strict_em' in r for r in rows):
            raise ValueError('PREFIX20_PREDICTION_ID_OR_LABEL_LEAK')
        before=self.version
        receipts=self.transport('predict',[dict(rows=rows[rank::4]) for rank in range(4)])
        if any(r['completed_outer']!=self.completed_outer or r['actual_steps']!=self.completed_outer*10 for r in receipts):
            raise RuntimeError('PREFIX20_PREDICTION_CHANGED_PROBE_BOUNDARY')
        values=[p for r in receipts for p in r['predictions']]
        mapping={r['trajectory_id']:r['em_probability'] for r in values}
        if len(values)!=len(mapping) or set(mapping)!=set(ids):raise RuntimeError('PREFIX20_INCOMPLETE_PREDICTIONS')
        if any(isinstance(p,bool) or not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1 for p in mapping.values()):
            raise RuntimeError('PREFIX20_NONFINITE_PREDICTIONS')
        if self.version!=before:raise RuntimeError('PREFIX20_PREDICTION_MUTATED_VERSION')
        return {tid:float(mapping[tid]) for tid in ids}

    def begin_outer(self,outer,attempt_id,rows,probabilities,policy_snapshot_id):
        if self.pending is not None or outer!=self.completed_outer+1 or not 1<=outer<=200:
            raise ValueError('PREFIX20_PENDING_OR_OUTER_MISMATCH')
        ids=[r['trajectory_id'] for r in rows]
        if len(ids)!=2560 or len(set(ids))!=2560 or any(type(r.get('strict_em')) is not int or r['strict_em'] not in (0,1) for r in rows):
            raise ValueError('PREFIX20_EXACT_NEW_SELECTED_G5_LABELS_REQUIRED')
        if set(probabilities)!=set(ids):raise ValueError('PREFIX20_SELECTED_PREDICTION_IDENTITIES_CHANGED')
        if (not isinstance(attempt_id,str) or len(attempt_id)!=32 or
                any(c not in '0123456789abcdef' for c in attempt_id) or attempt_id in self.completed_attempts):
            raise ValueError('PREFIX20_INVALID_OR_REPEATED_ATTEMPT_ID')
        groups=[rows[i:i+5] for i in range(0,2560,5)]
        if (len({g[0]['qid'] for g in groups})!=512 or any(
                len({r['qid'] for r in g})!=1 or [r['response_index'] for r in g]!=list(range(5)) for g in groups)):
            raise ValueError('PREFIX20_SELECTED_GROUP_ORDER_CHANGED')
        self.pending=dict(outer=outer,attempt_id=attempt_id,rows=copy.deepcopy(rows),
            probabilities=dict(probabilities),policy_snapshot_id=policy_snapshot_id,prediction_version=self.version)

    def abort_outer(self,outer,attempt_identity=None,reason='actor outer incomplete'):
        if self.pending is None:return None
        if outer!=self.pending['outer'] or (attempt_identity is not None and attempt_identity['attempt_id']!=self.pending['attempt_id']):
            raise ValueError('PREFIX20_ABORTED_ATTEMPT_MISMATCH')
        from search_g1.qprobe_runtime import _write_json
        event=dict(outer=outer,attempt_id=self.pending['attempt_id'],reason=reason,probe_updated=False)
        _write_json(self.output_dir/'aborted'/(self.pending['attempt_id']+'.json'),event)
        self.pending=None;return event

    def complete_outer(self,outer,actor_snapshot_id,attempt_identity=None):
        p=self.pending
        if p is None or p['outer']!=outer or outer!=self.completed_outer+1:
            raise ValueError('PREFIX20_COMPLETE_OUTER_WITHOUT_OWN_LABELS')
        if attempt_identity is not None and attempt_identity['attempt_id']!=p['attempt_id']:
            raise ValueError('PREFIX20_ACTOR_AND_PROBE_ATTEMPTS_DIFFER')
        rows=p['rows'];ids=[r['trajectory_id'] for r in rows];by_id={r['trajectory_id']:r for r in rows}
        plan=probe_batch_plan(ids,[int(x) for x in self.selection_rng.permutation(2560)])
        payloads=[dict(outer=outer,batches=[[by_id[tid] for tid in step[rank]] for step in plan]) for rank in range(4)]
        receipts=self.transport('update',payloads)
        if any(r['completed_outer']!=outer or r['actual_steps']!=outer*10 or r['local_trajectories']!=640 for r in receipts):
            raise RuntimeError('PREFIX20_ACTUAL_FOUR_RANK_UPDATE_INCOMPLETE')
        self.completed_outer=outer
        unlabeled=[{k:v for k,v in r.items() if k!='strict_em'} for r in rows]
        after=self.score(unlabeled);labels=[r['strict_em'] for r in rows]
        before_metrics=binary_metrics([p['probabilities'][tid] for tid in ids],labels)
        after_metrics=binary_metrics([after[tid] for tid in ids],labels)
        groups=[labels[i:i+5] for i in range(0,2560,5)]
        statistics=dict(selected_mixed_fraction=sum(0<sum(g)<5 for g in groups)/512,
            selected_mean_true_sample_std=sum(math.sqrt(sum(g)*(5-sum(g))/20) for g in groups)/512)
        event=dict(outer=outer,attempt_id=p['attempt_id'],actor_snapshot_id=actor_snapshot_id,
            label_policy_snapshot_id=p['policy_snapshot_id'],prediction_version=p['prediction_version'],
            updated_probe_version=self.version,head_update=dict(optimizer_steps_per_rank=10,
                actual_adam_steps_per_rank=outer*10,local_batch=64,global_batch=256,new_trajectories=2560,
                learning_rate=1e-6,epochs=1,replay_trajectories=0,**statistics,
                **{'before_'+k:v for k,v in before_metrics.items()},**{'after_'+k:v for k,v in after_metrics.items()}),
            rank_receipts=receipts,metrics_scope='selected new trajectories; before prediction is prequential, after fit is training-set fit')
        from search_g1.qprobe_runtime import _write_json
        _write_json(self.output_dir/'completed'/('outer-%04d.json'%outer),event)
        self.completed_attempts.add(p['attempt_id']);self.pending=None
        return event

    def save_checkpoint(self,directory,completed_outer,candidate_state,reward_probe_version,actor_snapshot_id=None):
        if self.pending is not None or completed_outer!=self.completed_outer:
            raise ValueError('PREFIX20_CANNOT_SAVE_PARTIAL_OR_UNPAIRED_PROBE')
        if not isinstance(actor_snapshot_id,str) or not actor_snapshot_id or not 1<=completed_outer<=200:
            raise ValueError('PREFIX20_EXPLICIT_COMPLETED_ACTOR_BINDING_REQUIRED')
        from search_g1.training_state import atomic_save
        directory=Path(directory);states=self.transport('state_dict',[{} for _ in range(4)])
        files=[]
        for rank,state in enumerate(states):
            if state['rank']!=rank or state['completed_outer']!=completed_outer or state['actual_steps']!=10*completed_outer:
                raise ValueError('PREFIX20_SAVED_WORKER_BOUNDARY_DIFFERS')
            receipt=atomic_save(directory/('prefix20-rank-%d.pt'%rank),state);files.append(dict(receipt,rank=rank))
        state=dict(schema_version=3,kind='prefix20_trajectory_probe',config=self.config,contract=CONTRACT,
            feature_spec_sha=self.feature_spec_sha,actor_snapshot_id=actor_snapshot_id,completed_outer=completed_outer,
            completed_attempts=sorted(self.completed_attempts),head_version=self.version,
            selection_rng=copy.deepcopy(self.selection_rng.bit_generator.state),rank_files=files,
            head_file=Path(files[0]['path']).name,head_sha256=files[0]['sha256'],
            candidate_state=candidate_state,reward_probe_version=reward_probe_version)
        path=directory/'question-probe-state.pt'
        receipt=atomic_save(path,state)
        validate_saved_state(state,path,actor_snapshot_id)
        return dict(receipt,kind=state['kind'],completed_outer=completed_outer,head_version=self.version,
                    actor_snapshot_id=actor_snapshot_id,feature_spec_sha=self.feature_spec_sha,rank_files=files)

    def restore(self,path,expected_actor_snapshot_id):
        if self.pending is not None:raise ValueError('PREFIX20_RESTORE_DURING_PENDING_OUTER')
        state=torch.load(path,map_location='cpu',weights_only=False)
        validate_saved_state(state,path,expected_actor_snapshot_id)
        ignore={'model_dir','head_safetensors','probe_config','output_dir','snapshot_path'}
        if {k:v for k,v in state['config'].items() if k not in ignore}!={k:v for k,v in self.config.items() if k not in ignore}:
            raise ValueError('PREFIX20_RECOVERY_SCIENTIFIC_SETTINGS_CHANGED')
        states=[torch.load(Path(path).parent/Path(r['path']).name,map_location='cpu',weights_only=False) for r in state['rank_files']]
        receipts=self.transport('load_state_dict',[dict(state=r) for r in states])
        if any(not r.get('model_optimizer_rng_restored') or not r.get('actual_readback_equal') or
                not r.get('actor_rng_unchanged') or r['actual_steps']!=10*state['completed_outer'] for r in receipts):
            raise RuntimeError('PREFIX20_ACTUAL_RESTORE_INCOMPLETE')
        from search_g1.full_state_restore import equal_tree
        actual=self.transport('state_dict',[{} for _ in range(4)])
        for expected,readback in zip(states,actual):equal_tree(expected,readback)
        self.completed_outer=state['completed_outer'];self.completed_attempts=set(state['completed_attempts'])
        self.selection_rng.bit_generator.state=copy.deepcopy(state['selection_rng'])
        equal_tree(state['selection_rng'],self.selection_rng.bit_generator.state)
        self.restore_receipts=receipts
        return state


def validate_saved_state(state,path,actor_snapshot_id):
    from reproduction.prefix20_joint4.assets import sha
    from search_g1.full_state_restore import equal_tree
    if state.get('schema_version')!=3 or state.get('kind')!='prefix20_trajectory_probe' or state['contract']!=CONTRACT:
        raise ValueError('PREFIX20_CHECKPOINT_SCHEMA_OR_CONTRACT_DIFFERS')
    validate_config(state['config']);outer=state['completed_outer']
    if state['actor_snapshot_id']!=actor_snapshot_id or state['feature_spec_sha']!=FEATURE_SPEC_SHA or not 1<=outer<=200:
        raise ValueError('PREFIX20_CHECKPOINT_ACTOR_BINDING_DIFFERS')
    attempts=state['completed_attempts']
    if (state['head_version']!='prefix20-epoch68-outer-%04d'%outer or len(attempts)!=outer or
            len(set(attempts))!=outer or any(not isinstance(a,str) or len(a)!=32 or
                any(c not in '0123456789abcdef' for c in a) for a in attempts)):
        raise ValueError('PREFIX20_CHECKPOINT_VERSION_OR_ATTEMPT_COUNT_DIFFERS')
    if [r['rank'] for r in state['rank_files']]!=list(range(4)):
        raise ValueError('PREFIX20_CHECKPOINT_FOUR_RANK_STATES_MISSING')
    if state['head_file']!='prefix20-rank-0.pt' or state['head_sha256']!=state['rank_files'][0]['sha256']:
        raise ValueError('PREFIX20_CHECKPOINT_HEAD_ALIAS_DIFFERS')
    reference=None
    for rank,record in enumerate(state['rank_files']):
        if Path(record['path']).name!='prefix20-rank-%d.pt'%rank:
            raise ValueError('PREFIX20_CHECKPOINT_RANK_FILENAME_DIFFERS')
        target=Path(path).parent/Path(record['path']).name
        if not target.is_file() or target.is_symlink():raise ValueError('PREFIX20_UNSAFE_CHECKPOINT_RANK_FILE')
        if target.stat().st_size!=record['bytes'] or sha(target)!=record['sha256']:
            raise ValueError('PREFIX20_CHECKPOINT_RANK_FILE_SHA_DIFFERS')
        saved=torch.load(target,map_location='cpu',weights_only=False)
        if saved['rank']!=rank or saved['completed_outer']!=outer or saved['actual_steps']!=10*outer:
            raise ValueError('PREFIX20_CHECKPOINT_RANK_COUNTER_DIFFERS')
        if not saved['optimizer']['state'] or any(float(g['lr'])!=1e-6 for g in saved['optimizer']['param_groups']):
            raise ValueError('PREFIX20_CHECKPOINT_ADAM_MISSING_OR_LR_CHANGED')
        for value in saved['optimizer']['state'].values():
            if (float(value['step'])!=outer*10 or not torch.isfinite(value['exp_avg']).all() or
                    not torch.isfinite(value['exp_avg_sq']).all()):
                raise ValueError('PREFIX20_CHECKPOINT_ADAM_STEP_OR_FINITE_STATE_DIFFERS')
        if not saved['model'] or any(not torch.isfinite(t).all() for t in saved['model'].values()):
            raise ValueError('PREFIX20_CHECKPOINT_NONFINITE_OR_EMPTY_MODEL')
        if set(saved['rng'])!={'python','numpy','torch_cpu','torch_cuda'}:
            raise ValueError('PREFIX20_CHECKPOINT_PRIVATE_RNG_MISSING')
        if reference is None:reference=dict(model=saved['model'],optimizer=saved['optimizer'])
        else:equal_tree(reference,dict(model=saved['model'],optimizer=saved['optimizer']))
    return True
