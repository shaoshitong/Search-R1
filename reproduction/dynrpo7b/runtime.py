"""Driver-side RPC facade; actual probe tensors live on allocated workers."""
import copy
import json
from pathlib import Path

from .recipe import probe_update, prefix_tokens, fit_prefix_tokens


class ProbeRuntime:
    def __init__(self,config,base,worker_group):
        self.config=dict(config)
        self.base=Path(base)
        self.output_dir=Path(config['output_dir'])
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.workers=worker_group
        self.completed_outer=0
        self.pending=None
        self.version='seed1-E9-raw'
        self.last_probe_checkpoint=None
        self.last_rollout_ref=None
        self.rollout_checkpoint_reference=None
        self._initialised=False
        self._probe_ref=None

    def rpc(self,operation,payload):
        return self.workers.dynrpo7b_probe_operation(operation,payload)

    def initialise(self):
        if not self._initialised:
            self.rpc('init',self.config)
            self._initialised=True

    def score_groups(self,groups,*,prefix):
        self.initialise()
        receipts=self.rpc('predict',dict(groups=groups,prefix=prefix))
        if any(row.get('position_capacity_changed') for row in receipts):
            self._probe_ref=None
            self.version += '-positions-%d'%max(row['position_capacity'] for row in receipts)
        predicted={row['qid']:float(row['score']) for receipt in receipts for row in receipt['predictions']}
        scores=[predicted.get(g['qid']) for g in groups]
        if any(value is None for value in scores):
            raise RuntimeError('Four-rank probe scoring did not return all candidate groups')
        return scores

    def begin_outer(self,outer,attempt_id,fit_groups,actor_version,directory):
        self.rollout_checkpoint_reference=self.last_rollout_ref
        self.pending=dict(outer=outer,attempt_id=attempt_id,fit_groups=fit_groups,
                          actor_version=actor_version,directory=str(directory))

    def abort_outer(self,outer,attempt_identity,reason):
        # A rejected actor attempt never fits or publishes this probe data.
        self.pending=None

    def complete_outer(self,outer,actor_snapshot_id,attempt_identity):
        if self.pending is None or self.pending['outer']!=outer:
            raise RuntimeError('Actor boundary differs from pending same-G5 collection')
        self.completed_outer=outer
        self.pending['actor_snapshot_id']=actor_snapshot_id
        return dict(head_update=dict(probe_fit_due=int(probe_update(outer)),
                                     probe_fit_completed=0))

    def save_checkpoint(self,directory,completed_outer,candidate_state,reward_probe_version,actor_snapshot_id=None):
        from search_g1.training_state import atomic_save
        self.initialise()
        directory=Path(directory)
        path=self.output_dir/('probe-state-outer-%04d.pt'%completed_outer)
        if self._probe_ref is None:
            receipts=self.rpc('state_dict',dict(path=str(path)))
            self._probe_ref=receipts[0]['reference']
        reference=self._probe_ref
        path=Path(reference['path'])
        metadata=dict(kind='dynrpo7b_group_variance',completed_outer=completed_outer,
            head_version=self.version,actor_snapshot_id=actor_snapshot_id,
            candidate_state=candidate_state,reward_probe_version=reward_probe_version,
            reference=reference,probe_pending=probe_update(completed_outer) and self.pending is not None,
            fit_prefix_tokens=fit_prefix_tokens(completed_outer),
            fit_groups=None if self.pending is None else self.pending['fit_groups'],
            fit_output_dir=(None if self.pending is None else self.pending.get('fit_output_dir',
                str(self.output_dir/('update-%04d'%completed_outer)))),
            fit_actor_checkpoint_ref=(None if self.pending is None else self.pending.get('actor_checkpoint_ref',
                dict(path=str(directory.parent)))),
            last_rollout_ref=self.last_rollout_ref)
        receipt=atomic_save(directory/'question-probe-state.pt',metadata)
        return dict(receipt,kind=metadata['kind'],completed_outer=completed_outer,
                    head_version=self.version,actor_snapshot_id=actor_snapshot_id,head_file=path.name,head_sha256=reference['sha256'],
                    reference=reference,fit_completed_epochs=(3 if probe_update(completed_outer)
                       and self.pending is None else 0))

    def fit_after_checkpoint(self,trainer,checkpoint):
        outer=self.completed_outer
        if not probe_update(outer):
            self.pending=None
            return None
        self.initialise()
        data=self.pending
        if data is None:
            return dict(already_published=True,completed_update_outer=outer)
        output_dir=Path(data.get('fit_output_dir',self.output_dir/('update-%04d'%outer)))
        actor_ref=data.get('actor_checkpoint_ref',dict(path=str(checkpoint)))
        # ProbeUpdater authenticates the dataset/actor binding and loads the
        # highest durable epoch once. The driver does not reload it again.
        receipts=self.rpc('fit',dict(groups=data['fit_groups'],prefix=fit_prefix_tokens(outer),
            completed_actor_outer=outer,rollout_actor_outer=outer-1,epochs=3,
            actor_checkpoint_ref=actor_ref,output_dir=str(output_dir)))
        from .checkpoint import RuntimeCheckpoint
        run=json.loads((self.base/'logs/run-manifest.json').read_text())
        journal=RuntimeCheckpoint(self.base/'preparation/dynrpo7b-checkpoints',run['run_id'])
        root=receipts[0]
        reference=root.get('checkpoint')
        if reference is None:
            reference=json.loads((output_dir/'published.json').read_text())['checkpoint']
        summaries={}
        for epoch in range(1,4):
            path=output_dir/('epoch-%02d.json'%epoch)
            if path.exists():
                summary=json.loads(path.read_text())
                summaries[summary['epoch']]=summary
        summaries.update({summary['epoch']:summary for summary in root.get('epochs',[])})
        for epoch in sorted(summaries):
            summary=summaries[epoch]
            if summary['epoch']>journal.state['probe_completed_epoch']:
                journal.probe_epoch(summary['epoch'],summary['checkpoint'])
        self.version='seed1-E9-raw-online-%04d'%outer
        self.pending=None
        self._probe_ref=reference
        self.last_probe_checkpoint=dict(reference,fit_completed_epochs=3,
                                       head_version=self.version)
        return receipts

    def restore(self,path,expected_actor_snapshot_id):
        import torch
        saved=torch.load(path,map_location='cpu',weights_only=False)
        self.initialise()
        self.rpc('load_state_dict',dict(reference=saved['reference']))
        self._probe_ref=saved['reference']
        self.completed_outer=saved['completed_outer']
        self.version=saved['head_version']
        self.last_rollout_ref=saved.get('last_rollout_ref')
        if saved.get('probe_pending'):
            self.pending=dict(outer=self.completed_outer,fit_groups=saved['fit_groups'],
                actor_snapshot_id=expected_actor_snapshot_id,
                fit_output_dir=saved.get('fit_output_dir'),
                actor_checkpoint_ref=saved.get('fit_actor_checkpoint_ref',dict(path=str(Path(path).parent.parent))))
            if self.pending['fit_output_dir'] is None:
                # Compatibility for the first local implementation snapshot.
                origin=next((p for p in Path(path).parents if (p/'logs/run-manifest.json').exists()),None)
                old=json.loads((origin/'logs/run-manifest.json').read_text()) if origin else {}
                old_output=(old.get('question_probe') or {}).get('output_dir',str(self.output_dir))
                self.pending['fit_output_dir']=str(Path(old_output)/('update-%04d'%self.completed_outer))
        return saved
