"""Generate calibration data for Search-G1 representation readouts.

Generation (rollouts / closed-book / flippability re-decode) runs on a vLLM engine
with batched multi-turn search; hidden-state extraction at the commit token runs on a
separate HF forward pass (single forward, no generation). Both share one GPU.

For each rollout we dump:
  - answer-delimiter and final answer-token hidden states at selected layers
  - prompt-last-token hidden states (per-question boundary probe input)
  - EM correctness (free label)
  - evidence-deletion sensitivity label: re-decode with <information> emptied;
        an answer change gives the operational positive label (1).
  - per-question closed-book label    : greedy decode with NO search; EM = inside boundary.

Usage:
    python ts_probe_dump.py --model <ckpt> --data data/nq_search/train.parquet \
        --tag nq --cap 100 --G 5 --layers 21,24,27 --out dumps/base250
"""
from __future__ import annotations
import argparse, json, os, re, time, hashlib, gc
from search_g1.signals import contexts, model_search_count
from search_g1.calibration_evidence import invalid_action_feedback, evidence_deleted_prefix, validate_provenance
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

DTYPE = torch.bfloat16
RETRIEVER_URL = os.environ.get("RETRIEVER_URL", "http://127.0.0.1:8000/retrieve")
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
INFO_RE = re.compile(r"<information>.*?</information>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# ----------------------------- helpers -----------------------------
def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


# Local service calls should not be routed through an outbound HTTP proxy.
_SESSION = requests.Session()
_SESSION.trust_env = False


def retrieve_batch(queries, topk=3):
    """Batch retrieval. Returns list[str] (joined docs) aligned with queries."""
    if not queries:
        return []
    chunk_size = max(1, int(os.environ.get("RETRIEVER_BATCH_SIZE", "32")))
    if len(queries) > chunk_size:
        out = []
        for s in range(0, len(queries), chunk_size):
            out.extend(retrieve_batch(queries[s:s + chunk_size], topk=topk))
        return out
    try:
        resp = _SESSION.post(RETRIEVER_URL, json={"queries": queries, "topk": topk},
                             timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            res = data.get("result", data.get("results")) if isinstance(data, dict) else data
            out = []
            for docs in res:
                joined = []
                for d in docs[:topk]:
                    if isinstance(d, str):
                        joined.append(d)
                    elif isinstance(d, dict):
                        joined.append(d.get("contents", d.get("text", "")))
                out.append("\n".join(joined))
            if len(out) != len(queries) or any(not text for text in out):
                raise RuntimeError("Missing calibration retrieval evidence")
            return out
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError("Calibration retrieval failed; refuse synthetic empty evidence") from e
    raise RuntimeError("Invalid retrieval response")


from verl.utils.reward_score.qa_em import normalize_answer, em_check
from search_r1.qa_protocol import parse_model_action


def normalize(s: str) -> str:
    return normalize_answer(s)


def em_score(pred, golds) -> int:
    return int(bool(normalize(pred)) and em_check(pred, golds))


def contains_gold(pred, golds) -> int:
    """Relaxed match for closed-book boundary label: gold appears in the answer text."""
    p = normalize(pred)
    if not p:
        return 0
    return int(any(normalize(g) and normalize(g) in p for g in golds))


QUESTION_RE = re.compile(r"Question:\s*(.*?)\s*$", re.DOTALL)


def counterfactual_answer(text: str) -> str:
    """The deletion prompt already prefills <answer>; only its payload is generated.

    Preserve the source's forced-close convention for budget-truncated output,
    while rejecting nested/search actions instead of extracting an inner answer.
    """
    action=parse_model_action('<answer>'+text)
    if action is None and '</answer>' not in text:
        action=parse_model_action('<answer>'+text+'</answer>')
    return action[1] if action is not None and action[0]=='answer' else ''


def counterfactual_labels(outputs, original_answers):
    """Source operational flip: losing a usable answer also counts as change.

    Keep invalid interventions visible; this is a proxy, not proof of causal
    evidence use or a label restricted to two valid changed answers.
    """
    if len(outputs) != len(original_answers):
        raise ValueError('Counterfactual answer alignment mismatch')
    parsed = [counterfactual_answer(text) for text in outputs]
    valid = [bool(answer) for answer in parsed]
    return [int(normalize(answer) != normalize(original))
            for answer, original in zip(parsed, original_answers)], valid


def gen_batch(llm, prompts, max_tokens, sample, stop):
    limit = getattr(llm, '_g1_calibration_prompt_limit', None)
    if getattr(llm, '_g1_calibration_total_context', None) is not None:
        # Closed-book and evidence-deletion interventions keep their original
        # label recipe inside the opt-in native4096 context.
        limit = llm._g1_calibration_total_context - max_tokens
    if limit is not None:
        capacity = llm.llm_engine.model_config.max_model_len
        if limit + max_tokens > capacity:
            raise ValueError('Calibration prompt/output budget exceeds engine context')
        tok = llm._g1_calibration_tokenizer
        ids = [tok.encode(p, add_special_tokens=False) for p in prompts]
        llm._g1_calibration_context_stats.append(dict(
            count=len(ids), max_original_tokens=max(map(len, ids), default=0),
            truncated=sum(len(x) > limit for x in ids), prompt_limit=limit,
            max_new=max_tokens))
        prompts = [{'prompt_token_ids': x[-limit:]} for x in ids]
    sp = SamplingParams(
        temperature=1.0 if sample else 0.0, top_p=1.0 if sample else 1.0,
        max_tokens=max_tokens, stop=stop, include_stop_str_in_output=True)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    return [o.outputs[0].text for o in outs]


def rollouts_batched(llm, prompts0, sample, max_turns=4, max_new=256, return_ranges=False, tokenizer=None, return_counts=False, correction_feedback=False, return_provenance=False):
    """Multi-turn search generation over a flat list of prompts. Returns generated texts."""
    N = len(prompts0)
    gen = [""] * N
    model_ranges = [[] for _ in range(N)]
    evidence_ranges = [[] for _ in range(N)]
    correction_ranges = [[] for _ in range(N)]
    actual_calls = [0] * N
    done = [False] * N
    for turn in range(max_turns + 1):
        idx = [i for i in range(N) if not done[i]]
        if not idx:
            break
        inputs = [prompts0[i] + gen[i] for i in idx]
        texts = gen_batch(llm, inputs, max_new, sample, ["</search>", "</answer>"])
        pending = []  # (seq_index, query)
        for k, i in enumerate(idx):
            t = texts[k]
            action = parse_model_action(t)
            begin = len(gen[i]); gen[i] += t
            model_ranges[i].append([begin, len(gen[i])])
            if action is not None and action[0] == 'search' and turn < max_turns:
                pending.append((i, action[1]))
            elif action is None and correction_feedback and turn < max_turns:
                feedback = invalid_action_feedback()
                start = len(gen[i]); gen[i] += feedback
                correction_ranges[i].append([start, len(gen[i])])
            else:
                # The historical default terminates malformed actions. Opt-in
                # corrections cannot extend the final answer-only turn.
                done[i] = True
        if pending:
            docs = retrieve_batch([q for _, q in pending])
            for (i, _), d in zip(pending, docs):
                observation=f"\n<information>{d}</information>\n"
                if tokenizer is not None:
                    observation=tokenizer.decode(tokenizer.encode(observation,add_special_tokens=False)[:500])
                start = len(gen[i]); gen[i] += observation
                evidence_ranges[i].append([start, len(gen[i])])
                actual_calls[i] += 1
    if return_provenance:
        for i in range(N):
            validate_provenance(gen[i], model_ranges[i], evidence_ranges[i], correction_ranges[i])
        return gen, model_ranges, actual_calls, evidence_ranges, correction_ranges
    if return_counts:
        return gen, model_ranges, actual_calls
    return (gen, model_ranges) if return_ranges else gen


@torch.no_grad()
def hidden_batch(model, tok, texts, layers, device, micro=8, max_len=4096):
    """Return {layer: np.array[n,d]} hidden of the last token for each text.

    Uses forward hooks on the specific decoder layers so we never materialize all
    37 hidden-state tensors (which OOMs). output_hidden_states index L corresponds
    to the output of decoder layer (L-1)."""
    base = model.model  # Qwen2Model
    captured = {}
    hooks = []

    def mk(L):
        def hook(mod, inp, out):
            captured[L] = (out[0] if isinstance(out, tuple) else out)
        return hook

    for L in layers:
        hooks.append(base.layers[L - 1].register_forward_hook(mk(L)))
    res = {L: [] for L in layers}
    try:
        for s in range(0, len(texts), micro):
            chunk = texts[s:s + micro]
            enc = tok(chunk, return_tensors="pt", add_special_tokens=False, padding=True,
                      truncation=True, max_length=max_len).to(device)
            # call the BASE transformer (no lm_head) -> avoids materializing the huge
            # [batch, seq, vocab] logits tensor (~10GB) that caused OOM.
            base(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                 use_cache=False)
            for L in layers:
                res[L].append(captured[L][:, -1, :].float().cpu().numpy())
            captured.clear()
    finally:
        for h in hooks:
            h.remove()
    return {L: np.concatenate(res[L]) if res[L] else np.zeros((0, 1)) for L in layers}


def parse_row(row, tokenizer):
    pd_ = row["prompt"]
    if isinstance(pd_, np.ndarray):
        pd_ = pd_.tolist()
    if isinstance(pd_, str):
        pd_ = json.loads(pd_)
    ptext = pd_[0]["content"] if isinstance(pd_, (list, tuple)) else (
        pd_.get("content", str(pd_)) if isinstance(pd_, dict) else str(pd_))
    rm = row["reward_model"]
    if isinstance(rm, np.ndarray):
        rm = rm.tolist()
    if isinstance(rm, str):
        rm = json.loads(rm)
    golds = rm["ground_truth"]["target"]
    if isinstance(golds, np.ndarray):
        golds = golds.tolist()
    if isinstance(golds, str):
        golds = [golds]
    chat = pd_ if isinstance(pd_, (list, tuple)) else [{"role": "user", "content": ptext}]
    model_prompt = (tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
                    if tokenizer.chat_template else ptext)
    return model_prompt, golds, ptext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--G", type=int, default=5)
    ap.add_argument("--layers", default="21,24,27")
    ap.add_argument("--max_turns", type=int, default=4)
    ap.add_argument("--rollout_max_new", type=int, default=256)
    ap.add_argument("--correction_feedback", action="store_true")
    ap.add_argument("--gpu_mem", type=float, default=0.45)
    ap.add_argument("--out", default="dumps/run")
    ap.add_argument("--interaction-config", help="Use the canonical new train/QA interaction for grounding rollouts")
    args = ap.parse_args()
    if args.max_turns != 4 or args.rollout_max_new not in (256, 500):
        raise ValueError('Expected four search-capable turns and a declared 256/500 action budget')

    layers = [int(x) for x in args.layers.split(",")]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    df = pd.read_parquet(args.data)
    if args.offset:
        df = df.iloc[args.offset:]
    if args.cap > 0:
        df = df.head(args.cap)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tok.truncation_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [parse_row(r, tok) for _, r in df.iterrows()]
    prompts = [p for p, _, _ in rows]
    golds_all = [g for _, g, _ in rows]
    # clean closed-book question prompts (no search instructions) for the boundary label
    questions = []
    for _, _, p in rows:
        starts = list(re.finditer(r"Question:\s*", p))
        questions.append(p[starts[-1].end():].strip() if starts else p)
    nq = len(rows)
    print(f"[dump] model={args.model} tag={args.tag} n_q={nq} G={args.G} layers={layers}",
          flush=True)

    # vLLM engine for generation
    interaction = None
    if args.interaction_config:
        from reproduction.prefix_selection200.interaction_protocol import BUDGET, CONTRACT as INTERACTION_CONTRACT
        interaction = json.loads(Path(args.interaction_config).read_text())
        if (interaction.get('interaction_protocol', INTERACTION_CONTRACT) != INTERACTION_CONTRACT
                or args.rollout_max_new != 500 or not args.correction_feedback):
            raise ValueError('Shared grounding requires new protocol/action500/correction labels')
        if interaction.get('calibration_model', args.model) != args.model:
            raise ValueError('Grounding interaction config names another actual actor model')
        interaction['calibration_model'] = args.model
    engine_context = BUDGET['total_tokens'] if interaction is not None else (4608 if args.rollout_max_new == 500 else 4096)
    engine_options = {} if interaction is None else dict(enable_chunked_prefill=True,
        enable_prefix_caching=True, disable_async_output_proc=True)
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=engine_context, enforce_eager=True, disable_log_stats=True, **engine_options)
    llm._g1_calibration_context_stats = []
    if interaction is not None:
        llm._g1_calibration_total_context = engine_context
    if args.rollout_max_new == 500:
        llm._g1_calibration_prompt_limit = 4096
        llm._g1_calibration_tokenizer = tok
    # HF model for hidden extraction
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tok.truncation_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print('[dump] vLLM loaded; HF extraction will follow generation', flush=True)
    t0 = time.time()

    # ---- per-question closed-book (boundary): answer directly, NO retrieval ----
    cb_prompt = [f"Answer the question directly with only the answer, no explanation.\n"
                 f"Question: {q}\nAnswer:" for q in questions]
    if tok.chat_template:
        cb_prompt = [tok.apply_chat_template([{"role":"user","content":p}],
                     add_generation_prompt=True,tokenize=False) for p in cb_prompt]
    cb_ans = gen_batch(llm, cb_prompt, 24, False, ["\n"])
    q_cb = [em_score(cb_ans[i], golds_all[i]) for i in range(nq)]
    question_ids = [hashlib.sha256(re.sub(r"\s+", " ", q).strip().casefold().encode()).hexdigest() for q in questions]
    print(f"[dump] closed-book done ({time.time()-t0:.0f}s) cb_rate={np.mean(q_cb):.3f}", flush=True)

    # ---- G rollouts per question (flat) ----
    flat_prompts = [prompts[i] for i in range(nq) for _ in range(args.G)]
    flat_qi = [i for i in range(nq) for _ in range(args.G)]
    if interaction is not None:
        from reproduction.prefix_selection200.calibration_interaction import rollouts as unified_rollouts
        gens, generated_ranges, actual_calls, evidence_ranges, correction_ranges = unified_rollouts(
            llm, flat_prompts, tok, args.G > 1, interaction)
        # Hidden contexts use the same admitted initial prompts as generation.
        prompts = [llm._g1_calibration_initial_prompt_texts[index * args.G] for index in range(nq)]
    else:
        gens, generated_ranges, actual_calls, evidence_ranges, correction_ranges = rollouts_batched(
                            llm, flat_prompts, sample=(args.G > 1), max_turns=args.max_turns,
                            max_new=args.rollout_max_new, tokenizer=tok,
                            correction_feedback=args.correction_feedback, return_provenance=True)
    print(f"[dump] rollouts done ({time.time()-t0:.0f}s)", flush=True)

    # build commit/answer-last contexts + 2 flippability contexts (valid rollouts only)
    valid, commit_ctx, ans_ctx, flip_ctx, flip2_ctx, answers, ems = [], [], [], [], [], [], []
    for j, g in enumerate(gens):
        ctx = contexts(prompts[flat_qi[j]], g, generated_ranges[j])
        if ctx is None or actual_calls[j] == 0:
            continue
        qi = flat_qi[j]
        ans = ctx['answer']
        commit_ctx.append(ctx['commit'])
        ans_ctx.append(ctx['answer_state'])
        src = ctx['prefix']
        # Remove only explicitly retrieved evidence. Correction feedback is
        # context, but must survive this intervention along with model reasoning.
        info_empty = evidence_deleted_prefix(g, len(src), generated_ranges[j],
                                             evidence_ranges[j], correction_ranges[j])
        flip_ctx.append(prompts[qi] + info_empty)
        flip2_ctx.append(prompts[qi] + THINK_RE.sub('', info_empty))
        answers.append(ans)
        ems.append(em_score(ans, golds_all[qi]))
        valid.append(j)
    print(f"[dump] valid rollouts={len(valid)}/{len(gens)}", flush=True)

    # ---- flippability re-decode (batched, greedy), two variants ----
    def flip_label(ctx_list):
        fa = gen_batch(llm, ctx_list, 32, False, ["</answer>"])
        labels, legal = counterfactual_labels(fa, answers)
        return fa, labels, legal
    flip_ans, flips, cf_valid = flip_label(flip_ctx)
    _, flips2, cf_valid2 = flip_label(flip2_ctx)
    print(f"[dump] flippability done ({time.time()-t0:.0f}s) flip={np.mean(flips):.3f} "
          f"flip_strict={np.mean(flips2):.3f}", flush=True)

    # Measure own allocator release; shared-service peaks require separate GPU samples.
    generation_allocated_before_release = torch.cuda.memory_allocated()
    calibration_context_stats = list(llm._g1_calibration_context_stats)
    interaction_stats = getattr(llm, '_g1_calibration_interaction_stats', None)
    # Dispose of generation engine before HF extraction: never overlap both caches.
    try:
        llm.llm_engine.model_executor.shutdown()
    except AttributeError:
        pass
    del llm
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    generation_allocated_after_release = torch.cuda.memory_allocated()
    assert generation_allocated_after_release < 256*1024*1024, 'Generation allocations remain before HF extraction'
    print('GENERATION_ALLOCATION_RELEASE=' + json.dumps(dict(before=generation_allocated_before_release,
          after=generation_allocated_after_release)), flush=True)
    hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=DTYPE, device_map="cuda",
             low_cpu_mem_usage=True, attn_implementation="sdpa").eval()
    # ---- hidden states (HF single forward, batched) ----
    Hc = hidden_batch(hf, tok, commit_ctx, layers, device)          # at "<answer>" tag token
    Ha = hidden_batch(hf, tok, ans_ctx, layers, device)             # at answer-span last token
    Hp_q = hidden_batch(hf, tok, prompts, layers, device)           # per-question prompt
    print(f"[dump] hidden extracted ({time.time()-t0:.0f}s)", flush=True)

    # ---- assemble ----
    y_em, y_flip, y_flip2, q_cb_r, gids, dss, recs = [], [], [], [], [], [], []
    Xc = {L: [] for L in layers}
    Xa = {L: [] for L in layers}
    Xp = {L: [] for L in layers}
    for k, j in enumerate(valid):
        qi = flat_qi[j]
        for L in layers:
            Xc[L].append(Hc[L][k]); Xa[L].append(Ha[L][k]); Xp[L].append(Hp_q[L][qi])
        y_em.append(ems[k]); y_flip.append(flips[k]); y_flip2.append(flips2[k])
        q_cb_r.append(q_cb[qi]); gids.append(qi); dss.append(args.tag)
        recs.append({"qi": qi, "em": ems[k], "flip": flips[k], "flip2": flips2[k],
                     "cb": q_cb[qi], "answer": answers[k], "flip_answer": flip_ans[k][:60], "counterfactual_valid": cf_valid[k], "counterfactual_strict_valid": cf_valid2[k]})

    out_npz = os.path.join(args.out, f"{args.tag}_{args.offset}.npz")
    save = {"y_em": np.array(y_em, np.int8), "y_flip": np.array(y_flip, np.int8),
            "y_flip2": np.array(y_flip2, np.int8),
            "counterfactual_valid": np.array(cf_valid, np.bool_),
            "counterfactual_strict_valid": np.array(cf_valid2, np.bool_),
            "q_cb": np.array(q_cb_r, np.int8), "gid": np.array(gids, np.int32),
            "question_id": np.array([question_ids[flat_qi[j]] for j in valid]),
            "boundary_question_id": np.array(question_ids),
            "boundary_q_cb": np.array(q_cb, np.int8),
            "ds": np.array(dss)}
    for L in layers:
        save[f"Xc_{L}"] = np.stack(Xc[L]).astype(np.float16) if Xc[L] else np.zeros((0, 1))
        save[f"Xa_{L}"] = np.stack(Xa[L]).astype(np.float16) if Xa[L] else np.zeros((0, 1))
        save[f"Xp_question_{L}"] = Hp_q[L].astype(np.float16)
        save[f"Xp_{L}"] = np.stack(Xp[L]).astype(np.float16) if Xp[L] else np.zeros((0, 1))
    np.savez_compressed(out_npz, **save)
    with open(os.path.join(args.out, f"{args.tag}_{args.offset}.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    proof={"actual_retrieval_counts":actual_calls,"authored_search_tag_counts":[model_search_count(g,spans) for g,spans in zip(gens,generated_ranges)],"generation_allocated_before_release":generation_allocated_before_release,"generation_allocated_after_release":generation_allocated_after_release,"model":args.model,"question_count":nq,"searched_valid_trajectory_count":len(valid),
           "closed_book_label":"exact_match_no_retrieval","reliance_label":"normalized_answer_change_after_evidence_deletion",
           "commit_position":"Xa","model_authored_answer_spans":True,"group_size":args.G,
           "question_ids":question_ids,"boundary_class_counts":np.bincount(q_cb,minlength=2).tolist(),
           "reliance_class_counts":np.bincount(y_flip,minlength=2).tolist(),
           "counterfactual_invalid_count":sum(not v for v in cf_valid),
           "counterfactual_strict_invalid_count":sum(not v for v in cf_valid2),
           "invalid_counterfactual_policy":"no_usable_answer_counts_as_flip_source_operational_proxy",
           "calibration_protocol":{"top_p":1.0,"rollout_max_new":args.rollout_max_new,"counterfactual_max_new":32,"closed_book_max_new":24,"correction_feedback":args.correction_feedback},
           "malformed_calibration_action_policy":"correction_within_four_turn_budget" if args.correction_feedback else "terminate_without_retrieval_or_correction_turn",
           "evidence_deletion":"typed_retrieval_ranges_only_preserve_model_and_correction",
           "retrieval_evidence_ranges":evidence_ranges,"correction_feedback_ranges":correction_ranges,
           "model_ranges":generated_ranges,"rollout_character_lengths":[len(g) for g in gens],
           "generation_context_policy":{"engine_max_tokens":engine_context,"prompt_max_tokens":4096 if args.rollout_max_new==500 else None,"truncation_side":"left","batches":calibration_context_stats},
           "boundary_all_reserved_questions":True,"observation_token_cap":500,"hidden_truncation_side":"left"}
    if interaction is not None:
        proof['interaction_protocol'] = INTERACTION_CONTRACT
        proof['interaction_config_sha256'] = file_sha(args.interaction_config)
        proof['interaction_execution'] = interaction_stats
        proof['grounding_source_actor_version'] = interaction.get('actor_version')
        proof['generation_context_policy']['prompt_max_tokens'] = INTERACTION_CONTRACT['max_start_tokens']
    with open(os.path.join(args.out,"calibration-label-proof.json"),"w") as f:json.dump(proof,f,indent=2)
    print(f"[dump] SAVED {out_npz} n={len(y_em)} EM={np.mean(y_em):.3f} "
          f"flip={np.mean(y_flip):.3f} cb={np.mean(q_cb_r):.3f} "
          f"total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
