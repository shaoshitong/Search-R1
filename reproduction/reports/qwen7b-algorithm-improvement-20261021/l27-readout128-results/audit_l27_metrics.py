"""Fresh CPU-only schema-specific L27/legacy prediction audit on HPC3.

No training, GPU operations, model tensors, remote writes, threshold fitting,
or untouched-final reads. Legacy baseline predictions deliberately use their
own sealed epoch/rank schema; L27 uses split-Adam/step-cursor schema.
"""
import gzip,hashlib,json,math,subprocess,time
from pathlib import Path

P=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=P/'algorithm-improvement-20261021'
NAMES=('last-prefix','mean-prefix','mean-question-prefix','attention-prefix','attention-question-prefix','flatten')


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def checked(ref):
    p=Path(ref['path']);assert str(p).startswith(str(P)+'/') and sha(p)==ref['sha256'],str(p)
    if 'bytes' in ref:assert p.stat().st_size==ref['bytes']
    return p
def gz(ref):
    with gzip.open(str(checked(ref)),'rt') as f:return [json.loads(line) for line in f]


def metric(rows):
    y=[0<sum(r['strict_em'])<5 for r in rows];scores=[r['predicted_mixed'] for r in rows]
    pred=[s>=.5 for s in scores];tp=sum(p and t for p,t in zip(pred,y));fn=sum(not p and t for p,t in zip(pred,y))
    fp=sum(p and not t for p,t in zip(pred,y));tn=sum(not p and not t for p,t in zip(pred,y))
    assert tp+fn and tn+fp
    pairs=sorted(zip(scores,y));i=0;rank=0.
    while i<len(pairs):
        j=i+1
        while j<len(pairs) and pairs[j][0]==pairs[i][0]:j+=1
        rank+=(i+1+j)*.5*sum(t for _,t in pairs[i:j]);i=j
    pos=tp+fn;neg=tn+fp;counts=[sum(r['strict_em']) for r in rows]
    classes={}
    for name,ix in [('allwrong',[i for i,k in enumerate(counts) if k==0]),('mixed',[i for i,k in enumerate(counts) if 0<k<5]),('allcorrect',[i for i,k in enumerate(counts) if k==5])]:
        called=sum(pred[i] for i in ix)
        classes[name]=dict(questions=len(ix),predicted_mixed=called,predicted_mixed_fraction=called/len(ix) if ix else None,
            correct_group_recall=(called if name=='mixed' else len(ix)-called)/len(ix) if ix else None)
    return dict(questions=len(rows),mixed_questions=pos,mixed_fraction=pos/len(rows),
        confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),mixed_recall=tp/pos,nonmixed_recall=tn/neg,
        mixed_balanced_accuracy=.5*(tp/pos+tn/neg),mixed_roc_auc=(rank-pos*(pos+1)/2)/(pos*neg),
        predicted_mixed_fraction=sum(pred)/len(rows),k_histogram=[counts.count(k) for k in range(6)],classwise=classes)


def verify_rows(rows,truth_rows,legacy_selection=False):
    truth={g['qid']:g for g in truth_rows};assert len(rows)==len(truth_rows)==len(truth)==len({r['qid'] for r in rows})
    assert {r['qid'] for r in rows}==set(truth)
    error=0.;normalized=[]
    for row in rows:
        own=truth[row['qid']];assert row['trajectory_ids']==own['trajectory_ids'] and len(own['labels'])==5
        if legacy_selection:
            assert set(row)=={'qid','trajectory_ids','p_j','predicted_mixed'}
        else:assert row['strict_em']==own['labels']
        assert all(type(v) is int and v in (0,1) for v in own['labels'])
        probabilities=row['p_j'];q=row['predicted_mixed']
        assert len(probabilities)==5 and all(math.isfinite(v) and 0<=v<=1 for v in probabilities+[q])
        ar=aw=1.
        for value in probabilities:ar*=value;aw*=1-value
        error=max(error,abs(1-ar-aw-q))
        normalized.append(dict(qid=row['qid'],strict_em=own['labels'],predicted_mixed=q))
    assert error<2e-6,error
    selected=sorted(normalized,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512]
    identity=digest(sorted([dict(qid=g['qid'],trajectory_ids=g['trajectory_ids'],labels=g['labels']) for g in truth_rows],key=lambda r:r['qid']))
    return dict(pool=metric(normalized),selected512=metric(selected),maximum_q_formula_error=error,
        own_EM_and_trajectory_IDs_verified=True,question_trajectory_EM_identity_sha256=identity,
        score_uses_no_labels=True,threshold=.5,threshold_fitted=False)


def compare_stored(actual,sealed):
    for part in ('pool','selected512'):
        for key in ('confusion','k_histogram'):assert actual[part][key]==sealed[part][key],(part,key)
        for key in ('mixed_balanced_accuracy','mixed_recall','nonmixed_recall','mixed_roc_auc','mixed_fraction'):
            assert abs(actual[part][key]-sealed[part][key])<1e-12,(part,key)


def new_variant(base,split_adam):
    complete=read(base/'complete.json');c=read(base/'config.json');assert sha(base/'config.json')==complete['config_sha256']
    assert complete['cursor']==128 and complete['actor_updates']==0
    if split_adam:
        assert complete['status']=='L27_READOUT128_COMPLETE' and complete['block_steps']==10592 and complete['head_steps']==128
        replay=read(base/'next-update-replay-cursor000000.json');assert replay['passed'] and replay['actual']==replay['expected']
        assert replay['block_steps']==10465 and replay['head_steps']==1
        restore=read(base/'restore-acceptance.json');assert restore['cursor']==0 and restore['block_steps']==10464 and restore['head_steps']==0
        curve=[json.loads(v) for v in (base/'curve.jsonl').read_text().splitlines()]
        assert [v['cursor'] for v in curve]==list(range(1,129))
        assert [v['block_steps'] for v in curve]==list(range(10465,10593)) and [v['head_steps'] for v in curve]==list(range(1,129))
    else:
        assert complete['status']=='MATCHED_PROBE_LOSS_ABLATION_COMPLETE' and complete['steps']==10592
        replay=read(base/'next-update-replay.json');assert replay['status']=='SINGLE_GPU_NEXT_UPDATE_REPLAY_PASSED' and replay['actual']==replay['expected']
    cp=complete['checkpoint'];assert cp==read(base/'latest.json') and cp['cursor']==128 and Path(cp['path']).stat().st_size==cp['bytes']
    assert not (base/'failure.json').exists()
    truth={'diagnostic512':read(checked(c['bank']))['diagnostic'],
        'selection2048':read(checked(c['selection_bank']))['selection_validation']}
    out={}
    for role,groups in truth.items():
        assert len(groups)==(512 if role=='diagnostic512' else 2048)
        sealed=complete['metrics'][role];actual=verify_rows(gz(sealed['predictions']),groups)
        compare_stored(actual,sealed);actual['prediction_ref']=sealed['predictions'];out[role]=actual
    return dict(base=str(base),config_sha256=complete['config_sha256'],complete_sha256=sha(base/'complete.json'),
        status=complete['status'],checkpoint=cp,checkpoint_tensor_reload_performed_here=False,
        fresh_checkpoint_content_SHA_performed_here=False,first_actual_update_replay_passed=True,
        split_Adam=split_adam,metrics=out)


def legacy_baseline(source):
    base=Path(source['base']);epoch=source['epoch'];c=read(base/'config.json');sealed=read(base/'probe'/('epoch-%04d.json'%epoch))
    assert sealed['epoch']==epoch and sealed['cursor']==0 and sealed['completed_epoch'] and sealed['independent_cpu_reload']
    assert sealed['sha256']==source['checkpoint_sha256'] and sealed['steps']==source['steps']
    bank=read(base/'bank/groups.json');assert sha(base/'bank/groups.json')==c['mixed_bank_sha256']
    diagnostic=[]
    for ref in source['diagnostic_prediction_files']:
        own=gz(ref)
        assert all(r['epoch']==epoch and r['split']=='diagnostic' and r['weight_kind']=='EMA0.99' for r in own)
        diagnostic.extend(own)
    diag=verify_rows(diagnostic,bank['diagnostic'])
    held=Path(c['selection_base']);proof=read(held/'bank/complete.json');assert sha(held/'bank/groups.json')==proof['groups_sha256']
    groups=read(held/'bank/groups.json')['selection_validation'];ref=sealed['metrics']['selection2048']['predictions']
    assert ref['sha256']==source['selection_prediction_sha256']
    selection=verify_rows(gz(ref),groups,legacy_selection=True)
    assert selection['selected512']['mixed_questions']==source['selected_mixed']
    diag['prediction_refs']=source['diagnostic_prediction_files'];selection['prediction_ref']=ref
    return dict(base=str(base),epoch=epoch,steps=sealed['steps'],checkpoint_sha256=sealed['sha256'],
        checkpoint_tensor_reload_performed_here=False,metrics=dict(diagnostic512=diag,selection2048=selection),
        prediction_schema='legacy four diagnostic rank files plus selection rows withoutEM; own bank labels joined after selection')


def audit(legacy):
    suite=A/'l27-readout128-v1';pub=read(suite/'source-publication.json')
    assert pub['commit']=='16bb5e97875bbe98097b157e84753ca7cecbcf60'
    for name,ref in pub['files'].items():assert sha(Path(pub['code_root'])/name)==ref['sha256']
    result=dict(status='FRESH_L27_READOUT_AND_BASELINE_OWN_EM_SHA_Q_METRICS_AUDITED',observed_epoch=time.time(),
        variants={n:new_variant(suite/n,True) for n in NAMES},
        baselines={'event128':new_variant(A/'matched-loss128-v2/event',False)},
        thresholds_tuned=False,final_pool_read=False,remote_writes=False,training_modified=False)
    for source in legacy:result['baselines']['E'+str(source['epoch'])]=legacy_baseline(source)
    all_results=list(result['variants'].values())+list(result['baselines'].values())
    for role in ('diagnostic512','selection2048'):
        identities={v['metrics'][role]['question_trajectory_EM_identity_sha256'] for v in all_results}
        assert len(identities)==1,'Different question/trajectory/EM pool in comparison'
    comparisons={}
    for name,value in result['variants'].items():
        own=value['metrics']['selection2048'];comparisons[name]={}
        for bname,baseline in result['baselines'].items():
            ref=baseline['metrics']['selection2048'];ba=own['pool']['mixed_balanced_accuracy'];top=own['selected512']['mixed_questions']
            oldba=ref['pool']['mixed_balanced_accuracy'];oldtop=ref['selected512']['mixed_questions']
            comparisons[name][bname]=dict(BA_delta_percentage_points=100*(ba-oldba),Mixed_at512_delta_questions=top-oldtop,
                BA_strictly_improved=ba>oldba,Mixed_at512_strictly_improved=top>oldtop,both_strictly_improved=ba>oldba and top>oldtop)
    result['comparisons']=comparisons
    p=subprocess.run(['sacct','-X','-n','-P','-j','743026,743027,743028,743029,743030,743031','--format=JobID,State,ExitCode,Elapsed'],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
    assert p.returncode==0 and all('|COMPLETED|0:0|' in line for line in p.stdout.splitlines()) and len(p.stdout.splitlines())==6
    result['fresh_sacct']=p.stdout
    return result
