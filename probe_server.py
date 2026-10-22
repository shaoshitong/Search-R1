"""Search-G1 representation-readout server with hot reload.

Serves /judge_probe -> {flip, boundary} for each rollout (prompt, response).
A background thread watches a manifest JSON; when the co-evolution controller writes
a new round (new policy checkpoint + freshly refit probes), the server reloads BOTH
the frozen policy-encoder snapshot and the two readout heads, then atomically swaps
them in.  Reward calls return detached scalars, so no gradient enters the evaluator.

manifest JSON:
  {"round": int, "encoder": <hf_ckpt_dir>, "commit": <pkl>, "boundary": <pkl>}
"""
from __future__ import annotations
import os
for _k in list(os.environ.keys()):
    if "proxy" in _k.lower():
        del os.environ[_k]

import argparse, gc, json, re, threading, time
from pathlib import Path
from reproduction.a800.prepare_run import digest
from typing import List
import numpy as np
import torch
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from search_g1.signals import contexts
from search_g1.resource import evaluator_gpu_lock

DTYPE = torch.bfloat16
app = FastAPI()
LOCK = threading.Lock()
STATE = {"round": -1, "encoder": None, "model": None, "tok": None,
         "commit": None, "boundary": None, "version": None}
MANIFEST = None


def _load(encoder, commit_pkl, boundary_pkl):
    tok = AutoTokenizer.from_pretrained(encoder, padding_side="left")
    tok.truncation_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        encoder, torch_dtype=DTYPE, device_map="auto",
        low_cpu_mem_usage=True, attn_implementation="sdpa").eval()
    commit = joblib.load(commit_pkl)
    boundary = joblib.load(boundary_pkl)
    return tok, model, commit, boundary


def _apply_unlocked(manifest_path, force=False):
    if not os.path.exists(manifest_path):
        return
    with open(manifest_path) as f:
        m = json.load(f)
    if (not force) and m.get("round", -1) == STATE["round"]:
        return
    enc = m["encoder"]
    for name,expected in m['encoder_sha256'].items():
        assert digest(Path(enc)/name)==expected, 'Frozen encoder hash mismatch'
    for name,path in [('commit_probe.pkl',m['commit']),('boundary_probe.pkl',m['boundary'])]:
        assert digest(Path(path))==m['head_sha256'][name], 'Readout hash mismatch'
    reload_model = force or STATE["model"] is None or (enc != STATE["encoder"])
    print(f"[probe] reloading round={m.get('round')} encoder={enc} "
          f"(model_reload={reload_model})", flush=True)
    tok = STATE["tok"]; model = STATE["model"]
    if reload_model:
        tok, model, commit, boundary = _load(enc, m["commit"], m["boundary"])
    else:
        commit = joblib.load(m["commit"]); boundary = joblib.load(m["boundary"])
    old = STATE["model"] if reload_model else None
    version = m["version"]
    assert isinstance(version, str) and version
    for readout in (commit, boundary):
        assert list(readout["clf"].classes_) == [0, 1]
        assert np.isfinite(readout["auc"])
    STATE.update({"round": m["round"], "encoder": enc, "version": version,
                  "tok": tok, "model": model, "commit": commit, "boundary": boundary})
    if old is not None and old is not model:
        del old
        torch.cuda.empty_cache()
    print(f"[probe] now serving round={STATE['round']} "
          f"commit(L{commit['layer']}/{commit.get('model')}) boundary(L{boundary['layer']})",
          flush=True)


def _apply(manifest_path, force=False):
    # Hooked inference and replacement loading must never overlap.
    with LOCK, evaluator_gpu_lock():
        _apply_unlocked(manifest_path, force)


def _watcher():
    while True:
        try:
            _apply(MANIFEST)
        except Exception as e:
            print(f"[probe] watcher error: {e}", flush=True)
        time.sleep(15)


class Item(BaseModel):
    id: str
    model_ranges: List[List[int]]
    prompt: str
    response: str = ""


class EvaluatorControl(BaseModel):
    version: str


@app.post('/suspend')
def suspend(req: EvaluatorControl):
    with LOCK, evaluator_gpu_lock():
        assert req.version == STATE['version'], 'Cannot suspend another evaluator version'
        STATE['model'] = None
        STATE['tok'] = None
        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated()
        assert allocated < 256 * 1024 * 1024, 'Frozen evaluator memory retained; refuse calibration'
        return {'version': STATE['version'], 'suspended': True, 'allocated_bytes_after_release': allocated}


@app.post('/resume')
def resume(req: EvaluatorControl):
    with LOCK, evaluator_gpu_lock():
        manifest = json.loads(Path(MANIFEST).read_text())
        assert req.version == manifest['version'], 'Cannot resume an unbound evaluator version'
        if STATE['model'] is None:
            _apply_unlocked(MANIFEST, force=True)
        assert STATE['model'] is not None and STATE['version'] == req.version
        return {'version': STATE['version'], 'resumed': True}


class ProbeRequest(BaseModel):
    items: List[Item]


def _release_judge_cuda_cache(version):
    """Called under both judge locks, after all returned features are on CPU."""
    torch.cuda.synchronize()
    def memory():
        free, total = torch.cuda.mem_get_info()
        return dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                    device_free_bytes=free, device_total_bytes=total)
    before = memory()
    torch.cuda.empty_cache()
    after = memory()
    record = dict(role='probe_post_judge_cuda_cache_release', pid=os.getpid(), version=version,
        cuda_device=torch.cuda.current_device(), checked_unix=time.time(), before=before, after=after,
        released_reserved_bytes=before['reserved_bytes']-after['reserved_bytes'],
        cuda_synchronized=True, scope='current probe process unused cache only; encoder weights remain resident')
    print('SEARCH_G1_PROBE_CACHE_RELEASE='+json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


@torch.no_grad()
def _last_hidden(texts, layer, model, tok, micro=4, max_len=4096):
    base = model.model
    cap = {}
    h = base.layers[layer - 1].register_forward_hook(
        lambda mod, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    out = []
    try:
        for s in range(0, len(texts), micro):
            enc = tok(texts[s:s + micro], return_tensors="pt", add_special_tokens=False,
                      padding=True, truncation=True, max_length=max_len).to(base.device)
            base(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                 use_cache=False)
            out.append(cap["h"][:, -1, :].float().cpu().numpy())
    finally:
        h.remove()
    return np.concatenate(out) if out else np.zeros((0, 1))


def _commit_ctx(prompt, response, model_ranges, position):
    ctx = contexts(prompt, response, model_ranges)
    if ctx is None:
        return prompt + response  # invalid outputs get fixed floor, not reliance credit
    if position == 'Xc':
        return ctx['commit']
    if position == 'Xa':
        return ctx['answer_state']
    raise ValueError('unsupported commit position')


@app.post("/judge_probe")
def judge_probe(req: ProbeRequest):
    with LOCK, evaluator_gpu_lock():
        model, tok = STATE['model'], STATE['tok']
        commit, boundary, version = STATE['commit'], STATE['boundary'], STATE['version']
        if model is None or version is None:
            raise RuntimeError('No complete calibrated evaluator loaded')
        if not req.items:
            return {'results': [], 'version': version}
        cc = [_commit_ctx(it.prompt, it.response, it.model_ranges, commit['position']) for it in req.items]
        bc = [it.prompt for it in req.items]
        Hc = _last_hidden(cc, commit['layer'], model, tok)
        Hb = _last_hidden(bc, boundary['layer'], model, tok)
        flip = commit['clf'].predict_proba(Hc)[:, 1]
        bnd = boundary['clf'].predict_proba(Hb)[:, 1]
        assert np.isfinite(flip).all() and np.isfinite(bnd).all()
        assert ((0 <= flip) & (flip <= 1)).all() and ((0 <= bnd) & (bnd <= 1)).all()
        result = {'version': version, 'results': [dict(id=it.id, version=version,
                  flip=float(flip[i]), boundary=float(bnd[i])) for i, it in enumerate(req.items)]}
        _release_judge_cuda_cache(version)
        return result


@app.get("/health")
def health():
    return {"ok": STATE["model"] is not None, "round": STATE["round"],
            "encoder": STATE["encoder"], "version": STATE["version"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--port", type=int, default=8002)
    args = ap.parse_args()
    MANIFEST = args.manifest
    print(f"[probe] manifest={MANIFEST} waiting for first round...", flush=True)
    for _ in range(600):
        if os.path.exists(MANIFEST):
            break
        time.sleep(2)
    _apply(MANIFEST, force=True)
    threading.Thread(target=_watcher, daemon=True).start()
    print(f"[probe] serving on 0.0.0.0:{args.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
