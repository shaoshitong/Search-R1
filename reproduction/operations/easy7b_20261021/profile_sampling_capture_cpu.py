#!/usr/bin/env python3
"""CPU-only cost probe of the existing request capture path; never a rollout.

The engine, sequence, request IDs and completion are constructed CPU objects.
When installed, the real vLLM SamplingParams and its generation-config update
are used. Existing GroupCapture bodies run unchanged. Only this process's
callable_source binding is temporarily replaced for the explicit cached-source
counterfactual. No production process, source file, model, RNG or job is touched.
"""
import argparse
import ast
import copy
import functools
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from types import SimpleNamespace as S


class FixtureParams(S):
    """Optional dependency-free fixture; not vLLM stop execution."""
    def update_from_generation_config(self, fields, eos):
        ids = set(fields.get('eos_token_id', []))
        self._all_stop_token_ids = set(self.stop_token_ids) | ids | {eos}
        if not self.ignore_eos:
            self.stop_token_ids = sorted(set(self.stop_token_ids) | (ids - {eos}))

    @property
    def all_stop_token_ids(self):
        return self._all_stop_token_ids


def metadata_helper(repo):
    path = repo / 'verl/workers/rollout/vllm_rollout/vllm_rollout.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'actual_sampling_metadata')
    namespace = dict(copy=copy)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[node.name]


def make_fixture(capture, args, repo):
    sampling_args = dict(n=1, best_of=1, temperature=1., top_p=1., top_k=-1,
        min_p=0., presence_penalty=0., frequency_penalty=0., repetition_penalty=1.,
        logprobs=1, max_tokens=500, min_tokens=0, ignore_eos=False,
        stop=[], stop_token_ids=[], detokenize=False)
    if args.fixture_sampling:
        params = FixtureParams(**sampling_args)
        params_kind = 'constructed CPU metadata fixture, not installed vLLM'
    else:
        from vllm import SamplingParams
        params = SamplingParams(**sampling_args)
        params_kind = 'actual installed vLLM SamplingParams; mock engine and sequence'
    metadata = metadata_helper(repo)(params, True)
    fields = (json.loads(Path(args.generation_config).read_text())
              if args.generation_config else dict(eos_token_id=[7, 9]))
    raw_eos = fields.get('eos_token_id', [7, 9])
    eos_ids = [raw_eos] if isinstance(raw_eos, int) else list(raw_eos)
    eos = eos_ids[0]
    model = (Path(args.generation_config).resolve().parent
             if args.generation_config else Path('/constructed-cpu-profile/model'))
    # Schema-complete fixture identity. These fields deliberately do not claim
    # actual loader/run/GPU provenance, which this independent probe lacks.
    source = capture.callable_source(type(params).update_from_generation_config)
    generation = dict(schema_version=1, role=capture.GENERATION_ROLE,
        run_id='cpu-cost-probe-not-training', slurm_job_id=None, source_commit='0' * 40,
        source_map_sha256='0' * 64, model_path=str(model),
        generation_config_path=str(model / 'generation_config.json'),
        generation_config_sha256=capture.digest(fields),
        generation_config_eos_token_ids=sorted(eos_ids), loader_source=source,
        actual_loaded_fields_sha256=capture.digest(fields))
    engine = S(_common8_generation_provenance=generation,
               generation_config_fields=fields,
               tokenizer=S(eos_token_id=eos, pad_token_id=eos))
    return engine, params, metadata, params_kind, source


def lifecycle(capture, engine, params, metadata, tokens, index):
    request_id = 'cpu-request-' + str(index)
    row_id = 'cpu-row-' + str(index)
    call_id = 'cpu-call-' + str(index)
    group_capture = capture.GroupCapture(engine, params, metadata, call_id, [row_id], [tokens])
    seq = S(seq_id=index, eos_token_id=engine.tokenizer.eos_token_id)
    group_capture.before(request_id, seq, params)
    effective = copy.deepcopy(params)
    effective.update_from_generation_config(engine.generation_config_fields, seq.eos_token_id)
    group = S(request_id=request_id, prompt_token_ids=tokens,
              sampling_params=effective, get_seqs=lambda: [seq])
    group_capture.after(request_id, seq, params, group)
    behavior = [dict(request_id=request_id, actual_request_token_ids=tokens)]
    group_capture.attach(behavior)
    return behavior[0]['effective_stop_provenance']


def measured(capture, engine, params, metadata, tokens, requests, repeats, cached):
    original = capture.callable_source
    cached_source = functools.lru_cache(maxsize=None)(original)
    capture.callable_source = cached_source if cached else original
    try:
        lifecycle(capture, engine, params, metadata, tokens, 0)
        times = []
        proof = None
        for _ in range(repeats):
            began = time.perf_counter()
            for i in range(requests):
                value = lifecycle(capture, engine, params, metadata, tokens, i)
                if i == 0:
                    proof = value
            times.append(time.perf_counter() - began)
        return dict(seconds=times, median_seconds=statistics.median(times),
                    per_request_ms=statistics.median(times) * 1000 / requests,
                    first_proof_sha256=capture.digest(proof),
                    source_cache=(cached_source.cache_info()._asdict() if cached else None))
    finally:
        capture.callable_source = original


def attributed(capture, engine, params, metadata, tokens, requests):
    originals = {name: getattr(capture, name) for name in ('callable_source', '_read')}
    totals = {name: dict(calls=0, inclusive_seconds=0., bytes=0) for name in originals}
    def wrapper(name):
        original = originals[name]
        def call(*a, **kw):
            began = time.perf_counter()
            try:
                value = original(*a, **kw)
                if name == '_read':
                    totals[name]['bytes'] += len(value)
                return value
            finally:
                totals[name]['calls'] += 1
                totals[name]['inclusive_seconds'] += time.perf_counter() - began
        return call
    for name in originals:
        setattr(capture, name, wrapper(name))
    try:
        began = time.perf_counter()
        for i in range(requests):
            lifecycle(capture, engine, params, metadata, tokens, i)
        elapsed = time.perf_counter() - began
    finally:
        for name, value in originals.items():
            setattr(capture, name, value)
    return dict(requests=requests, total_seconds=elapsed,
                measured_subcalls=totals,
                note='Inclusive timers nest: _read is inside callable_source, do not add them.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--requests', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--prompt-lengths', type=int, nargs='+', default=[128, 1024, 3072])
    parser.add_argument('--outer-actions', type=int, default=38000)
    parser.add_argument('--ranks', type=int, default=4)
    parser.add_argument('--generation-config', help='Optional real small generation_config.json only')
    parser.add_argument('--fixture-sampling', action='store_true',
                        help='Allow stdlib CPU fixture on a machine without installed vLLM')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from reproduction.common8 import sampling_capture as capture
    engine, params, metadata, params_kind, source = make_fixture(capture, args, repo)
    rows = []
    for length in args.prompt_lengths:
        tokens = [10 + (i % 10000) for i in range(length)]
        baseline = measured(capture, engine, params, metadata, tokens, args.requests, args.repeats, False)
        cached = measured(capture, engine, params, metadata, tokens, args.requests, args.repeats, True)
        components = attributed(capture, engine, params, metadata, tokens, min(args.requests, 256))
        serial = baseline['per_request_ms'] * args.outer_actions / 1000
        cached_serial = cached['per_request_ms'] * args.outer_actions / 1000
        rows.append(dict(prompt_tokens=length, original_source_reads=baseline,
            cached_source_only=cached, instrumented_attribution=components,
            equal_first_proof=baseline['first_proof_sha256'] == cached['first_proof_sha256'],
            projected_cpu_work_for_outer_actions=dict(actions=args.outer_actions,
                serial_original_seconds=serial, serial_cached_seconds=cached_serial,
                ideal_even_rank_cpu_seconds=serial / args.ranks,
                ideal_even_rank_cached_cpu_seconds=cached_serial / args.ranks,
                scope='CPU work arithmetic only, not production wall time or a measured upper bound')))
    result = dict(format='independent_sampling_capture_cpu_cost_v1',
        epoch=time.time(), host=platform.node(), python=sys.version, repo=str(repo),
        requests_per_repeat=args.requests, repeats=args.repeats,
        params_scope=params_kind, callable_source=source,
        callable_source_bytes=Path(source['path']).stat().st_size,
        generation_config=args.generation_config, fixture_run_identity=True,
        actual_production_capture_bodies=True, production_runtime_modified=False,
        model_loaded=False, cuda_allocated=False, remote_commands_called=False,
        rows=rows,
        limitations=['Mock engine/sequence; no real vLLM sampling, GPU, Ray, retrieval, serialization or driver assembly.',
                    'The cached-source replacement exists only in this single probe process.',
                    'Prompt-length sweep is constructed, not the empirical production prompt distribution.',
                    'CPU timings depend on host and filesystem; rank division assumes equal independent work and is not a wall-time bound.'])
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
