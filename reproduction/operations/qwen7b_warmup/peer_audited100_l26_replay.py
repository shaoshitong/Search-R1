"""Typed fixed100-after366 cohort adapter; reuse qualified122 capture mathematics."""
import argparse,json,os,time
from pathlib import Path
from . import peer_audited122_l26_replay as capture_core
from . import replay_probe_layers as r
from . import peer_newseed_ingest as raw
TARGET=capture_core.old.A/'peer-audited100-after366-L26-replay-v1'
INGEST=capture_core.old.A/'peer-newseed-ingest-audited100-after366-20261021-636015f5-a02'
QUESTIONS=12800
STATUS='AUDITED100_AFTER366_REAL_G10_QUALIFIED_HF_LAYER26_SHARD_COMPLETE'
TAG='AUDITED100_AFTER366'
SELECTION_SHA='00e5021284cb8f6614670452b749c7add10c028e50578822432600762074662a'


def cohort_config(path,sha):
    ref=r.fingerprints(r.remote(path))
    if ref['sha256']!=sha or Path(ref['path'])!=TARGET/'cohort-config.json':raise ValueError('Exact published isolated cohort config required')
    c=r.read(ref['path'])
    if c['status']!='FROZEN_AUDITED100_AFTER366_ACCEPTED_G10_LAYER26_CONFIG' or c['questions']!=QUESTIONS or c['trajectories']!=QUESTIONS*10 or c['domains']!={'nq':128,'hotpotqa':12672}:raise ValueError('Exact100 accepted ownG10 budget/domains required')
    if c['source_selection']['sha256']!=SELECTION_SHA or c['raw_complete']['path']!=str(INGEST/'complete.json') or c['independent_join_audit']['path']!=str(INGEST/'independent-final-audit.json'):raise ValueError('Frozen100 selection and actual completed join required')
    if c['compute_layers']!=[20,23,26] or c['persist_layers']!=[26] or c['prefix_tokens']!=20 or c['question_max_tokens']!=273 or c['shards']!=4 or c['batch_size']!=32 or c['max_context_tokens']!=16384 or c['actor_updates']!=0 or c['new_rollouts_generated']!=0 or c['existing30016_changed'] is not False:raise ValueError('Qualified capture contract cannot change')
    for key in ('source_selection','raw_complete','raw_groups','source_events','independent_join_audit'):r.checked_file(c[key])
    return ref,c


def prepare(base,code_root,commit,golden,config_path,config_sha):
    ref,c=cohort_config(config_path,config_sha)
    cohort=dict(target=str(TARGET),ingest=str(INGEST),questions=QUESTIONS,raw_sha=c['raw_complete']['sha256'],audit_sha=c['independent_join_audit']['sha256'],group_sha=c['raw_groups']['sha256'],event_sha=c['source_events']['sha256'],domains=c['domains'],tag=TAG,audit_status='FROZEN_DELTA_RECEIVER_AND_ELIGIBLE_RAW_G10_INDEPENDENTLY_ACCEPTED',old_controlled_overlap=None,selection_description='all accepted fixed100-after366 delta; original ordered identity; labels/scores unused')
    done=capture_core.prepare(base,code_root,commit,golden,cohort=cohort)
    # A sidecar binds the unchanged producer preparation to the typed config.
    raw.immutable(TARGET/'cohort-binding.json',dict(config=ref,preparation=r.fingerprints(TARGET/'preparation.json'),source_selection=c['source_selection'],actor_updates=0))
    return done


def worker(preparation_path,preparation_sha,pair,config_path,config_sha):
    ref,c=cohort_config(config_path,config_sha)
    if pair not in (0,1) or os.environ.get('SLURM_NTASKS')!='2':raise ValueError('Explicit isolated two-task pair required')
    local=int(os.environ['SLURM_LOCALID'])
    if local not in (0,1):raise ValueError('One task per pair device required')
    p=r.fingerprints(r.remote(preparation_path));binding=r.read(TARGET/'cohort-binding.json')
    if p['sha256']!=preparation_sha or binding['preparation']!=p or binding['config']!=ref:raise ValueError('Actual preparation/config pair differs')
    done=r.read(p['path']);shard=pair*2+local;lookup={r.read(v['path'])['shard']:v for v in done['selections']}
    if shard not in lookup:raise ValueError('Original geometry blocked shard; no truncation')
    own=lookup[shard]
    return capture_core.capture(own['path'],own['sha256'],TARGET/('shard-%02d'%shard)/'capture',expected_questions=QUESTIONS//4,complete_status=STATUS)


def merge(config_path,config_sha):
    ref,c=cohort_config(config_path,config_sha)
    binding=r.read(TARGET/'cohort-binding.json')
    if binding['config']!=ref:raise ValueError('Actual accepted cohort config changed')
    from . import peer_audited122_l26_merge as merger
    return merger.merge(TARGET,cohort=dict(target=str(TARGET),questions=QUESTIONS,tag=TAG,shard_status=STATUS))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=('prepare','worker','merge'),required=True);p.add_argument('--base',default=str(TARGET));p.add_argument('--code-root');p.add_argument('--commit');p.add_argument('--golden');p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True);p.add_argument('--preparation');p.add_argument('--preparation-sha256');p.add_argument('--pair',type=int);a=p.parse_args()
    try:
        if a.action=='prepare':v=prepare(a.base,a.code_root,a.commit,r.fingerprints(a.golden),a.config,a.config_sha256)
        elif a.action=='merge':v=merge(a.config,a.config_sha256)
        else:v=worker(a.preparation,a.preparation_sha256,a.pair,a.config,a.config_sha256)
    except BaseException as e:
        out=TARGET/('shard-%02d'%(a.pair*2+int(os.environ.get('SLURM_LOCALID','0'))))/'capture' if a.action=='worker' else TARGET
        out.mkdir(parents=True,exist_ok=True);raw.atomic(out/'failure.json',dict(error_type=type(e).__name__,error=str(e)[:1000],time=time.time()));raise
    print(json.dumps(v,sort_keys=True),flush=True)
if __name__=='__main__':main()
