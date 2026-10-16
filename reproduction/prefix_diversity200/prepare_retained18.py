"""Prepare an isolated config from existing same-cluster shared inputs.

Use the existing retained18 CUDA runtime/model/index. Only missing CPU reward
libraries are installed in this task's own directory. Never mutate either
existing experiment, its environment, or any global mount/SSH configuration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .actor import sha, atomic_json

TREE = Path('/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008')
SHARED8 = '/mnt/shared/workspace/training_data_node_8/'


def alias(value):
    return SHARED8 + value[len('/mnt/data/'):] if value.startswith('/mnt/data/') else value


def prepare(template, run_id):
    if len(run_id) != 32 or any(c not in '0123456789abcdef' for c in run_id):
        raise ValueError('Explicit new run identity required')
    root = Path('/mnt/data/shaoshitong/search-g1-prefix-diversity-assets-' + run_id)
    ready = root / 'ready.json'
    if ready.exists():
        return json.loads(ready.read_text())
    root.mkdir(exist_ok=False)
    c = dict(template)
    for key, value in c.items():
        if isinstance(value, str) and value.startswith('/mnt/data/'):
            c[key] = alias(value)
    base = '/mnt/data/shaoshitong/search-g1-prefix-diversity200-20261016-' + run_id[:12]
    official = TREE / 'instruct300/assets/qwen2.5-3b-instruct-aa8e725'
    for name, expected in c['model_files_sha256'].items():
        if sha(official / name) != expected:
            raise ValueError('Retained18 official weights/tokenizer differ: ' + name)
    for key in ('model_dir', 'official_model_dir', 'encoder_model_dir', 'actor_model_dir'):
        c[key] = str(official)
    c.update(base=base, run_id=run_id, node='h100-10',
        actor_version=run_id+'-official0', selected_questions=1024, actor_selected_questions=512,
        probe_updates_per_outer=20, environment_root=str(TREE/'env'),
        runtime_extra_packages=str(root/'cpu-reward-packages'),
        retrieval_index=str(TREE/'assets-root/assets/wiki/e5_Flat.index'),
        retrieval_corpus=str(TREE/'assets-root/assets/wiki/wiki-18.jsonl'),
        retrieval_encoder=str(TREE/'assets-root/assets/e5-base-v2'),
        retriever_gpu_lib=str(TREE/'retriever-gpu-lib'),
        pool_state_path=base+'/question-pool/state.json', prefix_probe_receipt=base+'/probe/latest.json',
        validation_questions=SHARED8+'shaoshitong/g1f-ef498ac3/data/validation.parquet',
        final_questions=SHARED8+'shaoshitong/g1f-ef498ac3/data/final.parquet',
        retrieval_port=18221, search_url='http://127.0.0.1:18221/retrieve', grounding_probe_base_port=18320)
    source = Path(c['grounding_manifest'])
    if sha(source) != template['grounding_manifest_sha256']:
        raise ValueError('Original grounding soft reward manifest changed')
    original = json.loads(source.read_text()); mapped = dict(original)
    mapped.update(encoder=str(official), commit=alias(original['commit']), boundary=alias(original['boundary']))
    for name, key in [('commit_probe.pkl','commit'), ('boundary_probe.pkl','boundary')]:
        if sha(mapped[key]) != original['head_sha256'][name]:
            raise ValueError('Original grounding soft reward head changed')
    atomic_json(root/'grounding-manifest.json', mapped)
    c.update(grounding_manifest=str(root/'grounding-manifest.json'),
        grounding_manifest_sha256=sha(root/'grounding-manifest.json'),
        original_grounding_manifest=str(source), original_grounding_manifest_sha256=sha(source),
        grounding_path_remap_only=True)
    for key in ('probe_source_checkpoint', 'probe_source_cpu_proof', 'canonical_train', 'source_question_pool',
                'fixed_test_features', 'retrieval_index', 'retrieval_corpus', 'retrieval_encoder',
                'validation_questions', 'final_questions'):
        if not Path(c[key]).exists():
            raise FileNotFoundError(key + ': ' + c[key])
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    gpu = [row.split(',') for row in rows]
    if len(gpu) != 8 or any(int(row[2]) < 40000 for row in gpu):
        raise ValueError('Actual eight H100 memory headroom is insufficient; no foreign cleanup')
    c['physical_gpu_uuids'] = [row[1].strip() for row in sorted(gpu,key=lambda row:int(row[0]))]
    deps = Path(c['runtime_extra_packages']); deps.mkdir()
    command = [str(TREE/'env/bin/python'), '-m', 'pip', 'install', '--no-deps', '--no-cache-dir',
        '--target', str(deps), 'joblib==1.4.2', 'scikit-learn==1.5.2', 'scipy==1.13.1', 'threadpoolctl==3.6.0']
    env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK='1')
    with (root/'cpu-reward-packages-install.log').open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=180)
    if result.returncode:
        raise RuntimeError('Isolated CPU reward dependencies failed; preserve install log')
    atomic_json(root/'launch-config-template.json', c)
    result = dict(status='RETAINED18_INPUTS_AND_ISOLATED_CPU_REWARD_RUNTIME_READY',
        config=c, template_path=str(root/'launch-config-template.json'),
        template_sha256=sha(root/'launch-config-template.json'),
        source_weights_all_full_sha_match=True, source_assets_modified=False,
        new_warmup=False, new_actor_initialization='official', large_assets_via_Mac=False)
    atomic_json(ready, result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--template',required=True); p.add_argument('--run-id',required=True)
    args=p.parse_args(); print(json.dumps(prepare(json.loads(Path(args.template).read_text()),args.run_id)))
