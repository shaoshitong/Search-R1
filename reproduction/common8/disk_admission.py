"""Reserve space for every requested full checkpoint before expensive GPU work."""
import json,math,shutil,struct,time
from pathlib import Path
from reproduction.common8.verify_data import digest
from reproduction.common8.stage_retrieval import atomic
from reproduction.common8.formal_contract import SAVE_STEPS, assert_formal_run


def checkpoint_floor(model):
    index=json.loads((model/'model.safetensors.index.json').read_text())
    tensors={}
    for filename in sorted(set(index['weight_map'].values())):
        with (model/filename).open('rb') as f:
            size=struct.unpack('<Q',f.read(8))[0];assert 0<size<16*2**20
            header=json.loads(f.read(size))
        for key,item in header.items():
            if key=='__metadata__':continue
            assert key not in tensors and index['weight_map'][key]==filename
            tensors[key]=item['shape']
    assert set(tensors)==set(index['weight_map'])
    config=json.loads((model/'config.json').read_text())
    # Runtime canonical saver may serialize the verified tied lm_head alias.
    if config.get('tie_word_embeddings') and 'lm_head.weight' not in tensors:
        tensors['lm_head.weight']=tensors['model.embed_tokens.weight']
    per_save=sum(math.prod(shape)*4 for shape in tensors.values())+512*2**20
    return dict(serialized_tensor_shapes=tensors,upper_bound_dtype='float32',save_steps=list(SAVE_STEPS),per_save_allowance_bytes=per_save,other_run_reserve_bytes=8*2**30,required_free_bytes=len(SAVE_STEPS)*per_save+8*2**30,scope='conservative all-checkpoint space allowance; not measured checkpoint sizes')


def admit(base):
    run=json.loads((base/'logs/run-manifest.json').read_text());assert run['profile']=='common8'
    assert_formal_run(run)
    model=Path(run['initial_model'])
    for name,sha in run['initial_model_sha256'].items():assert digest(model/name)==sha
    proof=checkpoint_floor(model);proof.update(run_id=run['run_id'],actual_free_bytes=shutil.disk_usage(base).free,checked_unix=time.time())
    if run.get('question_probe',{}).get('head_kind')=='qwen_native_periodic_lastblock_binary_v1':
        steps=sorted(set(SAVE_STEPS)|set(range(25,201,25)))
        shapes=dict(proof['serialized_tensor_shapes'])
        if json.loads((model/'config.json').read_text()).get('tie_word_embeddings'):
            shapes.pop('lm_head.weight',None)  # Adam owns the tied parameter once.
        adam_bytes=sum(math.prod(s)*8 for s in shapes.values())
        initial_head=Path(run['question_probe']['initial_head_checkpoint']) if run['question_probe'].get('initial_head_checkpoint') else None
        head_bytes=initial_head.stat().st_size if initial_head is not None else 2*2**30
        per_save=proof['per_save_allowance_bytes']+adam_bytes+head_bytes+2*2**30
        # Epoch recovery overwrites only latest-state; pre/post pairs retain
        # separate probe snapshots. Preserve the existing real-rollout audit.
        artifacts=3*head_bytes*8+256*2**30
        proof.update(save_steps=steps,per_save_allowance_bytes=per_save,
            full_rank_local_adam_allowance_bytes=adam_bytes,probe_state_allowance_bytes=head_bytes,
            other_run_reserve_bytes=artifacts,required_free_bytes=len(steps)*per_save+artifacts,
            scope='nine complete student/model/Adam/probe checkpoints, eight probe phases and retained rollout audit; filesystem capacity, not personal quota proof')
    proof['accepted']=proof['actual_free_bytes']>=proof['required_free_bytes']
    atomic(base/'logs/checkpoint-disk-admission.json',proof)
    assert proof['accepted'], 'Insufficient space for all five requested full checkpoints; do not drop saves or start training'
    return proof
