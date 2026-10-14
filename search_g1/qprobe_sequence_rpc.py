"""Sequence probe on one existing actor worker; no extra GPU or Ray actor."""
import copy
from dataclasses import asdict
import torch
from search_g1.qprobe_model import OnlineQProbe, QProbeConfig
from search_g1.qprobe_sequence import SequenceFeatures, SequenceCache
from search_g1.qprobe_sequence_model import OnlineSequenceQProbe
from search_g1.training_state import cpu_tree


OPERATIONS=frozenset(('init','predict','update_completed_outer','bootstrap','state_dict','load_state_dict','bind_actor'))


def execute_probe(head, operation, payload):
    """Same local head protocol; payload is internal, not a scoring HTTP API."""
    if operation not in OPERATIONS-{'init'}:raise ValueError('QPROBE_RPC_OPERATION_INVALID')
    if operation=='predict':
        features=SequenceFeatures.from_state(payload['features'])
        before=head.version;result=head.predict(features)
        if head.version!=before:raise RuntimeError('QPROBE_RPC_PREDICTION_MUTATED_HEAD')
    elif operation in ('update_completed_outer','bootstrap'):
        values=dict(payload);values['features']=SequenceFeatures.from_state(values['features'])
        if operation=='update_completed_outer':result=head.update_completed_outer(**values)
        else:result=head.bootstrap(**values)
    elif operation=='state_dict':result=cpu_tree(head.state_dict())
    elif operation=='load_state_dict':head.load_state_dict(payload['state']);result=True
    elif operation=='bind_actor':head.actor_snapshot_id=payload['actor_snapshot_id'];result=True
    return dict(result=result,version=head.version,completed_outer=head.completed_outer,
        update_count=head.update_count,actor_snapshot_id=head.actor_snapshot_id)


def worker_operation(worker, operation, payload):
    import torch.distributed as dist
    if dist.get_world_size()!=8 or dist.get_rank()!=7 or not worker._is_actor:
        raise RuntimeError('QPROBE_RPC_REQUIRES_EXISTING_ACTOR_RANK7_OF8')
    if not torch.cuda.is_available():raise RuntimeError('QPROBE_RPC_GPU_NOT_AVAILABLE')
    if operation=='init':
        if getattr(worker,'_sequence_qprobe',None) is not None:raise RuntimeError('QPROBE_RPC_ALREADY_INITIALIZED')
        cache=SequenceCache(payload['cache_dir'])
        if cache.spec_sha!=payload['feature_spec_sha']:raise ValueError('QPROBE_RPC_CACHE_SPEC_MISMATCH')
        config=payload['config'];device='cuda:'+str(torch.cuda.current_device())
        head=OnlineSequenceQProbe.fresh(cache,config,device=device)
        worker._sequence_qprobe=head
        from reproduction.common8.native_allocation import cuda_uuids
        uuids=cuda_uuids(count=torch.cuda.device_count())
        return dict(feature_dim=head.feature_dim,feature_spec_sha=head.feature_spec_sha,
            config=asdict(head.config),train_micro=head.train_micro,predict_micro=head.predict_micro,
            initialization=head.initialization,version=head.version,completed_outer=head.completed_outer,
            update_count=head.update_count,actor_snapshot_id=head.actor_snapshot_id,
            worker_rank=dist.get_rank(),world_size=dist.get_world_size(),actual_probe_device=str(next(head.model.parameters()).device),
            gpu_uuid=uuids[torch.cuda.current_device()])
    head=getattr(worker,'_sequence_qprobe',None)
    if head is None:raise RuntimeError('QPROBE_RPC_NOT_INITIALIZED')
    return execute_probe(head,operation,payload)


def worker_transport(group):
    """Call only rank7; do not broadcast replay/optimizer payload to eight ranks."""
    if group.world_size!=8 or len(group.workers)!=8:raise ValueError('QPROBE_RPC_REQUIRES_8_WORKERS')
    worker=group.workers[7]
    import ray
    # The normal common8 group is spawned from a colocated WorkerDict, whose
    # registered methods are prefixed by actor_rollout. Direct groups are also
    # explicit; never retry an uncertain operation under a different name.
    method_names=set(getattr(worker,'_ray_method_signatures',{}))
    if 'actor_rollout_qprobe_operation' in method_names:name='actor_rollout_qprobe_operation'
    elif 'qprobe_operation' in method_names:name='qprobe_operation'
    else:raise ValueError('QPROBE_RPC_REGISTERED_METHOD_MISSING')
    method=getattr(worker,name)
    return lambda operation,payload:ray.get(method.remote(operation,payload))


class RemoteSequenceQProbe(OnlineSequenceQProbe):
    """Driver metadata proxy. All block/head computation lives on allocated rank7."""
    def __init__(self,cache,config,transport):
        self.transport=transport
        receipt=transport('init',dict(cache_dir=str(cache.path),config=config,feature_spec_sha=cache.spec_sha))
        if receipt['worker_rank']!=7 or receipt['world_size']!=8 or not receipt['actual_probe_device'].startswith('cuda:'):
            raise RuntimeError('QPROBE_RPC_ACTUAL_GPU_OWNER_INVALID')
        self.config=QProbeConfig.from_mapping(receipt['config']);self.feature_dim=receipt['feature_dim'];self.feature_spec_sha=receipt['feature_spec_sha']
        self.train_micro=receipt['train_micro'];self.predict_micro=receipt['predict_micro'];self.initialization=copy.deepcopy(receipt['initialization'])
        self.worker_receipt=receipt;self._accept(receipt)
    def _accept(self,receipt):
        self._version=receipt['version'];self.completed_outer=receipt['completed_outer'];self.update_count=receipt['update_count'];self.actor_snapshot_id=receipt['actor_snapshot_id']
    def _request(self,operation,payload):
        receipt=self.transport(operation,payload);self._accept(receipt);return receipt['result']
    @property
    def version(self):return self._version
    def set_device(self,device):
        if device!='allocated_rank7':raise ValueError('QPROBE_RPC_OWNER_RELOCATION_FORBIDDEN')
    def predict(self,features):
        before=self.version;result=self._request('predict',{'features':features.detach().cpu().state_dict()})
        if self.version!=before:raise RuntimeError('QPROBE_RPC_PREDICTION_VERSION_CHANGED')
        return result
    def update_completed_outer(self,outer,labels,features,**kwargs):
        return self._request('update_completed_outer',dict(outer=outer,labels=labels,features=features.detach().cpu().state_dict(),**kwargs))
    def bootstrap(self,labels,features,**kwargs):
        return self._request('bootstrap',dict(labels=labels,features=features.detach().cpu().state_dict(),**kwargs))
    def state_dict(self):return self._request('state_dict',{})
    def load_state_dict(self,state):return self._request('load_state_dict',{'state':state})
    def save(self,path,*,actor_snapshot_id):
        receipt=OnlineQProbe.save(self,path,actor_snapshot_id=actor_snapshot_id)
        self._request('bind_actor',{'actor_snapshot_id':actor_snapshot_id})
        return receipt
