"""Independent qualified batched replay of the exact accepted peer8192 G10.

The legacy identity-only projector and causal-slot writer are reused without
modification. Only frozen HF context prefill batching differs; no tokens are
regenerated and all old5/new5 retain their own actual contexts/positions.
"""
import argparse,fcntl,json,os,time
from pathlib import Path
from . import peer_newseed_replay as old
from . import peer_newseed_ingest as raw
from . import replay_probe_layers as r
from .replay_batched_context import BatchedContextCapture
from .replay_bank_capture import verify_completed_chunk
TARGET=old.A/'peer-newseed8192-g10-batched-replay-20261021'
STATUS=old.STATUS


def prepare(base,raw_complete,controlled_plan,code_root,commit,golden,batch_size):
    base=r.remote(base)
    if base!=TARGET or batch_size not in (16,32):raise ValueError('Explicit qualified independent peer8192 branch required')
    proof=r.checked_file(golden);qualification=r.read(proof['path'])
    if qualification['status']!='REAL_SCALAR_BATCHED_HF_REPLAY_AND_E4_GOLDEN_COMPARISON_COMPLETE' or str(batch_size) not in qualification['downstream']:
        raise ValueError('Actual selected batch golden completion required')
    config=r.read(r.checked_file(qualification['config'])['path']);pub=r.read(r.checked_file(config['source_publication'])['path'])
    expected=pub['files']['reproduction/operations/qwen7b_warmup/replay_batched_context.py']['sha256']
    from . import replay_batched_context as core
    if r.fingerprints(core.__file__)['sha256']!=expected:raise ValueError('Batched core differs from actual golden GPU source')
    # Selection/input output is still owned by this experiment. No old GPU
    # capture was submitted; use the exact legacy projector at its fixed target
    # with the NEW immutable capture code root explicitly bound in selections.
    source=old.prepare(old.TARGET,raw_complete,controlled_plan,code_root,commit)
    own=dict(source,status='QUALIFIED_BATCHED_PEER8192_REAL_G10_PREFIX_INPUTS_PREPARED',
        base=str(base),source_prefix_preparation=r.fingerprints(old.TARGET/'preparation.json'),
        batching=dict(batch_size=batch_size,padded_token_budget=16384,window_trajectories=batch_size,
            golden=proof,core_sha256=expected,acceptance='Explicit operator selected actual scalar/batched golden evidence; no native bitexact claim'))
    base.mkdir(parents=True,exist_ok=True)
    raw.immutable(base/'preparation.json',own)
    return own


def capture(selection_path,selection_sha,output,preparation,preparation_sha):
    import torch
    from transformers import AutoModel
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Owned remote CUDA Slurm allocation required')
    selected_ref=r.fingerprints(r.remote(selection_path));prep_ref=r.fingerprints(r.remote(preparation))
    if selected_ref['sha256']!=selection_sha or prep_ref['sha256']!=preparation_sha:raise ValueError('Actual preparation/selection SHA differs')
    prep=raw.read(preparation);selection=raw.read(selection_path);batching=prep['batching'];code_root=Path(__file__).resolve().parents[3]
    if selection['expected_code_root']!=str(code_root) or selected_ref not in prep['selections']:raise ValueError('Wrong immutable source/selection')
    from . import replay_batched_context as core
    if r.fingerprints(core.__file__)['sha256']!=batching['core_sha256']:raise ValueError('Qualified batched core changed')
    r.checked_file(batching['golden']);base=r.remote(output);base.mkdir(parents=True,exist_ok=True)
    with (base/'capture.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (base/'complete.json').exists():raise ValueError('Existing actual shard completion preserved')
        selected=old.load_selection(selection);model_dir=selection['source_model_binding']['official_view'];proofs,config=r.verify_model(model_dir,selection)
        device=torch.device('cuda:0');torch.cuda.set_device(device)
        model=AutoModel.from_pretrained(model_dir,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2',local_files_only=True,trust_remote_code=False,low_cpu_mem_usage=True).eval().requires_grad_(False).to(device)
        torch.cuda.reset_peak_memory_stats(device);started=time.monotonic();chunks=[]
        engine=BatchedContextCapture(model,batch_size=batching['batch_size'],max_context_tokens=batching['padded_token_budget'],device=device)
        for start in range(0,len(selected),256):
            own=selected[start:start+256];root=base/('chunk-%05d'%(start//256));tids=[row['trajectory_id'] for row in own]
            if (root/'complete.json').exists():
                done=verify_completed_chunk(root,tids)
                if done['source_selection']!=selected_ref or done['source_preparation']!=prep_ref:raise ValueError('Completed chunk belongs to different actual inputs')
                chunks.append(r.fingerprints(root/'complete.json'));continue
            if root.exists():raise ValueError('Unknown partial chunk preserved; explicit isolated recovery required')
            root.mkdir();writers={layer:r.ReplayLayerWriter(root/('layer-%02d'%layer),layer,own) for layer in r.LAYERS};chunk_started=time.monotonic()
            for index,(row,values) in enumerate(engine.iter_replay(own,window_trajectories=batching['window_trajectories'])):
                for layer in r.LAYERS:writers[layer].write(row,values[layer])
                raw.atomic(base/'progress.json',dict(status='ACTUAL_QUALIFIED_BATCHED_PEER_G10_REPLAY_ADVANCING',trajectories=start+index+1,expected_trajectories=len(selected),elapsed_seconds=time.monotonic()-started,max_cuda_memory_allocated=torch.cuda.max_memory_allocated(device),batching=engine.stats(),time=time.time()))
            manifests={str(layer):writers[layer].finish() for layer in r.LAYERS}
            raw.immutable(root/'complete.json',dict(source_selection=selected_ref,source_preparation=prep_ref,trajectory_ids=tids,feature_manifests=manifests,source_runtime='huggingface_frozen_causal_prefill',actor_updates=0,native_bitexact_claim=False,batching=batching,actual_chunk_seconds=time.monotonic()-chunk_started,trajectories_per_second=len(own)/(time.monotonic()-chunk_started),max_cuda_memory_allocated=torch.cuda.max_memory_allocated(device)))
            chunks.append(r.fingerprints(root/'complete.json'))
        layers={str(layer):[] for layer in r.LAYERS};ids=[]
        for item in chunks:
            done=verify_completed_chunk(Path(item['path']).parent,raw.read(item['path'])['trajectory_ids']);ids.extend(done['trajectory_ids'])
            for layer in layers:layers[layer].append(done['feature_manifests'][layer])
        if ids!=[row['trajectory_id'] for row in selected]:raise ValueError('Actual peer G10 feature coverage incomplete')
        from .replay_feature_reader import ReplayFeatureReader
        for layer,refs in layers.items():
            reader=ReplayFeatureReader(refs,int(layer));assert set(reader.records)==set(ids);reader.get([ids[0],ids[-1]],'cpu');reader.close()
        value=dict(status=STATUS,source_selection=selected_ref,source_preparation=prep_ref,questions=selection['questions'],trajectories=len(selected),feature_manifests_by_capture_layer=layers,chunks=chunks,capture_layers=list(r.LAYERS),human_block_layers=[21,24,27],source_runtime='huggingface_frozen_causal_prefill',attention_implementation='flash_attention_2',model_revision=r.REVISION,model_files=proofs,domain_scope=selection['domain_scope'],actor_frozen=True,actor_updates=0,probe_updates=0,new_responses_generated=0,native_bitexact_equivalence_claim=False,batching=batching,actual_batching=engine.stats(),max_cuda_memory_allocated=torch.cuda.max_memory_allocated(device),job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,runtime_code=r.fingerprints(__file__),qualified_primitive=r.fingerprints(r.__file__),time=time.time())
        raw.immutable(base/'complete.json',value);return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','capture'),required=True);p.add_argument('--base',default=str(TARGET));p.add_argument('--raw-complete');p.add_argument('--controlled-plan',default=str(old.A/'matched-loss128-v2/plan.json'));p.add_argument('--code-root');p.add_argument('--commit');p.add_argument('--golden');p.add_argument('--batch-size',type=int,default=16);p.add_argument('--selection');p.add_argument('--selection-sha256');p.add_argument('--output');p.add_argument('--preparation');p.add_argument('--preparation-sha256');a=p.parse_args()
    try:
        if a.action=='prepare':v=prepare(a.base,r.fingerprints(a.raw_complete),r.fingerprints(a.controlled_plan),a.code_root,a.commit,r.fingerprints(a.golden),a.batch_size)
        else:v=capture(a.selection,a.selection_sha256,a.output,a.preparation,a.preparation_sha256)
    except BaseException as exc:
        if a.action=='capture' and a.output:
            base=Path(a.output);base.mkdir(exist_ok=True,parents=True);raw.atomic(base/'failure.json',dict(error_type=type(exc).__name__,error=str(exc)[:1000],time=time.time()))
        raise
    print(json.dumps(v,sort_keys=True),flush=True)
if __name__=='__main__':main()
