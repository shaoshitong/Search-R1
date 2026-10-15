"""Shared answer-only EM/F1. Interaction format and algorithm reward are separate."""
import collections, hashlib, json, re, string
from pathlib import Path
VERSION='common-em-v4-final-model-action'
MILESTONES=(5,50,100,150,200)
def normalize(text):
    text=str(text).lower().translate(str.maketrans('','',string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())
def aliases(value):
    if isinstance(value,dict):return aliases(value['target'])
    if isinstance(value,str):return [value]
    if hasattr(value,'tolist'):return aliases(value.tolist())
    if isinstance(value,(list,tuple)):return [str(x) for x in value]
    raise ValueError('Missing reference aliases')
def extract(action):
    if not isinstance(action,str):return None,'missing_model_boundary'
    # Only the final top-level complete answer action is eligible. Earlier answer
    # actions, even in this generation, must neither supply nor veto the answer.
    start=action.rfind('<answer>')
    if start<0:return None,'missing_or_incomplete_answer'
    m=re.match(r'<answer>(.*?)</answer>\s*$',action[start:],re.S)
    if m is None:return None,'missing_incomplete_or_nonterminal_answer'
    if re.search(r'</?[A-Za-z_][^<>]*>',m.group(1)):return None,'nested_answer'
    stack=[]
    for tag in re.finditer(r'<(/?)([A-Za-z_][\w-]*)(?:\s[^<>]*)?\s*(/?)>',action[:start]):
        closing,name,selfclosing=tag.groups()
        if selfclosing or tag.group(0).endswith('/>'):continue
        if closing:
            if name in stack:
                # A closed parent ends that earlier action's scope. Its malformed
                # inner markup is a format diagnostic, not a veto on later EM.
                at=len(stack)-1-stack[::-1].index(name);stack=stack[:at]
            # Orphan closings are full-interaction-format diagnostics, not EM gates.
        else:stack.append(name)
    if stack:return None,'answer_inside_other_action'
    if not m.group(1).strip():return None,'empty_answer'
    return m.group(1),'legal_terminal_answer'
def score(action,references,boundary_verified=True):
    answer,reason=extract(action if boundary_verified else None)
    refs=aliases(references)
    em=int(answer is not None and any(normalize(answer)==normalize(x) for x in refs))
    def f1(ref):
        a=normalize(answer or '').split();b=normalize(ref).split()
        if not a or not b:return float(a==b and answer is not None)
        shared=sum((collections.Counter(a)&collections.Counter(b)).values())
        return 2*shared/(len(a)+len(b))
    return {'em':em,'f1':max(map(f1,refs),default=0.0) if answer is not None else 0.0,
            'answer':answer,'answer_format_legal':answer is not None,'answer_reason':reason,
            'boundary_verified':boundary_verified,'scorer_version':VERSION}
def identity(question,references):
    return hashlib.sha256(json.dumps([question,sorted(aliases(references))],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def summarize(rows):
    rows=list(rows);n=len(rows)
    if not n:raise ValueError('Empty denominator')
    grouped={}
    for ds in sorted(set(x['data_source'] for x in rows)):
        part=[x for x in rows if x['data_source']==ds]
        grouped[ds]={'questions':len(set(x.get('question_uid',x.get('question_answer_sha256',str(i))) for i,x in enumerate(part))),'trajectories':len(part),'correct':sum(x['em'] for x in part),'em':sum(x['em'] for x in part)/len(part),'f1':sum(x['f1'] for x in part)/len(part)}
    return {'questions':len(set(x.get('question_uid',x.get('question_answer_sha256',str(i))) for i,x in enumerate(rows))),'trajectories':n,'correct':sum(x['em'] for x in rows),'em':sum(x['em'] for x in rows)/n,'f1':sum(x['f1'] for x in rows)/n,'datasets':grouped,'macro_dataset_em':sum(x['em'] for x in grouped.values())/len(grouped),'weighted_overall_em':sum(x['em'] for x in rows)/n,'scorer_version':VERSION}
def mixed_first(candidate_rows,selected_rows):
    if len(selected_rows)!=2560:raise ValueError('Mixed-first update denominator must be2560')
    return {'candidate_pool':summarize(candidate_rows),'selected_updates':summarize(selected_rows)}
