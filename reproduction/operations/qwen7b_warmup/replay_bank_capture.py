"""Independent production-scope causal replay of prepared matched banks.

Reuses unmodified qualified18c46aff replay normalization/capture/writer and
ReplayFeatureReader. Captures all20/23/26 together, uses no EM and generates no
new responses. This bank is a NEW HF-backend experiment for ALL variants, never
an accepted bit-exact replacement of actual-vLLM features. No scheduler calls.
"""
import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import platform
import time
from . import replay_probe_layers as r

STATUS='MATCHED_REAL_G5_MULTILAYER_HF_CAUSAL_REPLAY_SHARD_COMPLETE'


def load_fixture(selection,synthetic_fixture=False):
    """No128 qualification cap is relaxed in the old module; this is separate."""
    declared=selection['trajectories'];wanted={v['trajectory_id']:v['qid'] for v in declared}
    if len(wanted)!=len(declared) or not wanted:raise ValueError('Unique real selected trajectories required')
    if not synthetic_fixture and (selection['questions']!=2272 or len(declared)!=11360 or
            selection['role_counts']!={'fit':1632,'diagnostic512':128,'selection2048':512} or
            selection['capture_layers']!=[20,23,26] or selection['model_revision']!=r.REVISION or
            selection['label_blind_selection'] is not True):
        raise ValueError('Exact four-way matched9088q/G5 fixture scope required')
    found={};proofs=[]
    for item in selection['prefix_files']:
        # Synthetic small CPU tests use local metadata files only.
        if synthetic_fixture:
            proof=r.fingerprints(item['path'])
            if proof!=item:raise ValueError('Synthetic metadata fixture SHA differs')
        else:proof=r.checked_file(item)
        proofs.append(proof)
        with gzip.open(item['path'],'rt') as f:
            for line in f:
                row=json.loads(line);tid=row['trajectory_id']
                if tid not in wanted:raise ValueError('Prepared subset contains an undeclared trajectory')
                if tid in found:raise ValueError('Repeated prepared source trajectory')
                found[tid]=r.normalize_prefix(row,tid,wanted[tid],True)
    if set(found)!=set(wanted):raise ValueError('Missing actual saved causal prefix; no generation fallback')
    selected=[found[v['trajectory_id']] for v in declared]
    groups={}
    for row in selected:groups.setdefault(row['qid'],[]).append(row)
    if len(groups)!=selection['questions'] or any(len(g)!=5 for g in groups.values()):
        raise ValueError('Every selected question must retain exactly its original G5')
    for rows in groups.values():
        if any(row['initial_prompt_token_ids']!=rows[0]['initial_prompt_token_ids'] for row in rows):
            raise ValueError('Same-question actual prompt differs across G5')
    return selected,proofs


def verify_completed_chunk(path,expected_ids):
    complete=r.read(path/'complete.json')
    if complete['trajectory_ids']!=expected_ids or complete['source_runtime']!='huggingface_frozen_causal_prefill':
        raise ValueError('Existing replay chunk belongs to another exact input list')
    for layer,ref in complete['feature_manifests'].items():
        r.checked_file(ref);manifest=r.read(ref['path'])
        if int(layer)!=manifest['capture_layer']:raise ValueError('Existing chunk capture layer changed')
        for item in manifest['files'].values():r.checked_file(item)
    return complete


def capture(base,selection_path,selection_sha,model_dir,device_name):
    import torch
    if platform.system()!='Linux' or not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Actual owned remote Slurm CUDA allocation required')
    source=r.remote(selection_path);base=r.remote(base);proof=r.fingerprints(source)
    if proof['sha256']!=selection_sha:raise ValueError('Prepared selection SHA changed')
    selection=r.read(source)
    if r.fingerprints(r.__file__)['sha256']!=selection['source_model_binding']['legacy_replay_module_sha256']:
        raise ValueError('Replay primitive is not the exact18c46aff/Q1-qualified code')
    if str(Path(__file__).resolve().parents[3])!=selection['expected_code_root']:
        raise ValueError('Run only from the explicit isolated immutable source root')
    if str(model_dir)!=selection['source_model_binding']['official_view']:
        raise ValueError('Reuse the verified Q1 official model view')
    base.mkdir(exist_ok=True,parents=True)
    with (base/'capture.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (base/'complete.json').exists():
            done=r.read(base/'complete.json')
            if done['source_selection']!=proof:raise ValueError('Existing completed bank selection mismatch')
            return done
        selected,prefix_proofs=load_fixture(selection)
        model_proofs,config=r.verify_model(model_dir,selection)
        if any(t>=config['vocab_size'] for row in selected for seg in row['segments'] for t in seg['context_tokens']):
            raise ValueError('Actual prefix token outside official vocabulary')
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        device=torch.device(device_name);torch.cuda.set_device(device)
        import transformers
        from transformers import AutoModel
        model=AutoModel.from_pretrained(str(model_dir),torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',local_files_only=True,trust_remote_code=False,
            low_cpu_mem_usage=True).eval().requires_grad_(False).to(device)
        if len(model.layers)!=28 or model.config.hidden_size!=r.DIM or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Frozen official7B model geometry/gradients differ')
        passes=0
        def causal(context,positions):
            nonlocal passes
            if positions!=list(range(len(context))):raise ValueError('Actual causal positions changed')
            ids=torch.tensor([context],dtype=torch.long,device=device)
            with torch.inference_mode():
                out=model(input_ids=ids,attention_mask=torch.ones_like(ids),
                    position_ids=torch.tensor([positions],dtype=torch.long,device=device),
                    use_cache=False,output_hidden_states=True,return_dict=True)
                if len(out.hidden_states)!=29:raise ValueError('HF hidden-state boundary differs')
                result={layer:out.hidden_states[layer+1][0].detach().to('cpu',dtype=torch.bfloat16,copy=True) for layer in r.LAYERS}
            passes+=1;return result
        # Durable completed chunks allow later time-limit continuation without
        # overwriting old banks. Unknown partial chunks are preserved/refused.
        chunks=[];started=time.monotonic();chunk_size=64*5
        for start in range(0,len(selected),chunk_size):
            rows=selected[start:start+chunk_size];root=base/('chunk-%05d'%(start//chunk_size))
            tids=[row['trajectory_id'] for row in rows]
            if (root/'complete.json').exists():
                old=verify_completed_chunk(root,tids)
                if old['source_selection']!=proof:raise ValueError('Existing chunk selection changed')
                chunks.append(r.fingerprints(root/'complete.json'));continue
            if root.exists():raise ValueError('Unknown partial replay chunk preserved; inspect before a new isolated attempt')
            root.mkdir();writers={layer:r.ReplayLayerWriter(root/('layer-%02d'%layer),layer,rows) for layer in r.LAYERS}
            for index,row in enumerate(rows):
                values=r.replay_features(row,causal)
                for layer in r.LAYERS:writers[layer].write(row,values[layer])
                r.atomic(base/'progress.json',dict(status='ACTUAL_MATCHED_MULTILAYER_HF_CAUSAL_REPLAY_ADVANCING',
                    trajectories=start+index+1,expected_trajectories=len(selected),questions=len(selected)//5,
                    source_selection=proof,causal_prefill_passes_this_process=passes,
                    elapsed_seconds=time.monotonic()-started,actor_updates=0,time=time.time()))
            manifests={str(layer):writers[layer].finish() for layer in r.LAYERS}
            r.atomic(root/'complete.json',dict(source_selection=proof,trajectory_ids=tids,
                feature_manifests=manifests,source_runtime='huggingface_frozen_causal_prefill',
                actor_updates=0,native_bitexact_claim=False))
            chunks.append(r.fingerprints(root/'complete.json'))
        layers={str(layer):[] for layer in r.LAYERS}
        all_tids=[]
        for ref in chunks:
            own=verify_completed_chunk(Path(ref['path']).parent,r.read(ref['path'])['trajectory_ids'])
            all_tids+=own['trajectory_ids']
            for layer in layers:layers[layer].append(own['feature_manifests'][layer])
        if all_tids!=[row['trajectory_id'] for row in selected]:raise ValueError('Whole shard trajectory coverage differs')
        # Actual production reader initializes on all file hashes and exact row
        # geometry. No claim of equality to native-vLLM features is made.
        from .replay_feature_reader import ReplayFeatureReader
        for layer,refs in layers.items():
            reader=ReplayFeatureReader(refs,int(layer))
            if set(reader.records)!=set(all_tids):raise ValueError('Actual replay reader trajectory coverage differs')
            reader.get([all_tids[0],all_tids[-1]],'cpu');reader.close()
        result=dict(status=STATUS,source_selection=proof,trajectories=len(selected),questions=len(selected)//5,
            feature_manifests_by_capture_layer=layers,chunks=chunks,capture_layers=list(r.LAYERS),
            human_block_layers=[21,24,27],model_revision=r.REVISION,model_files=model_proofs,
            source_prefix_files=prefix_proofs,actor_frozen=True,actor_updates=0,new_responses_generated=0,
            source_runtime='huggingface_frozen_causal_prefill',attention_implementation='flash_attention_2',
            torch_version=torch.__version__,transformers_version=transformers.__version__,
            native_bitexact_equivalence_claim=False,accepted_for_production=False,
            permitted_use='ALL training/evaluation variants share this NEW HF replay backend',
            runtime_code=r.fingerprints(__file__),legacy_replay_code=r.fingerprints(r.__file__),
            job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,time=time.time())
        r.atomic(base/'complete.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True);p.add_argument('--selection-sha256',required=True)
    p.add_argument('--output',required=True);p.add_argument('--model',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    try:
        result=capture(a.output,a.selection,a.selection_sha256,a.model,a.device)
        print(json.dumps({k:result[k] for k in ('status','questions','trajectories','capture_layers','native_bitexact_equivalence_claim','accepted_for_production')}),flush=True)
    except BaseException as error:
        root=Path(a.output)
        if root.exists():r.atomic(root/'failure.json',dict(type=type(error).__name__,message=str(error),time=time.time()))
        raise


if __name__=='__main__':main()
