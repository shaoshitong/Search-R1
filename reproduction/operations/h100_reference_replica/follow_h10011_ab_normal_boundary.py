"""Read-only follow-up of the existing two native runs to a normal savepoint.

No submit, cancel, model load, training mutation, or primary-state write.
The previous through10 observer must have exited before starting this process.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
from .readonly_management_route import MODES, run_readonly

ROOT = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
RUNTIME = '2e3976230ac9425d66d8553afa9962821e964932'
WRAPPER = ('/mnt/shared/workspace/training_data_node_8/shaoshitong/'
           'search-g1-prefix5-reference-preparation-20261018/code/' + RUNTIME +
           '/repo/reproduction/operations/h100_reference_replica/h10011_private_paths.sh')
PATH_LAYOUTS = {
    'source8': WRAPPER,
    # CPU-qualified retained corpus/Arrow view; explicit selection only.
    'retained-cache-v2': ('/mnt/shared/workspace/training_data_node_10/shaoshitong/'
        'search-g1-prefix5-h10011-retained-fixed-assets-20261018/corpus-cache-relocations/'
        'ea94181843f69e23119e92af730895d8a32a0fa95abf8bff438ad9a2882d5d14/'
        'retained_cache_overlay_paths.sh'),
    # Already staged and CPU path-qualified; selecting this for a reader is not
    # a training handoff and never changes an existing mount namespace.
    'retained': ('/root/.local/share/search-g1/prefix5-retained-bootstrap/'
                 '6f970c9201962423daf2d463f9135f0f3cc6b16e5df9b9c12be24d5b7dbb3f60/'
                 'h10011_retained_paths.sh'),
}

INNER = r'''
import sys,json,time,hashlib
from pathlib import Path
ARMS=__ARMS__
VERIFIED=__VERIFIED__
sys.path.insert(0,next(iter(ARMS.values()))['base']+'/repo')
from reproduction.prefix_selection200.ab_control import observe
result={}
for arm,entry in ARMS.items():
 b=Path(entry['base']);v=observe(b,VERIFIED[arm]);n=v['complete_outer']
 stage_outer=(v['stage']or{}).get('outer',n)
 roots=sorted({n,stage_outer})
 v['partial_actor']=[]
 for p in sorted((b/'actor/outers').glob('outer-*-attempt-*/rank-0-updates.jsonl')):
  if not any(p.parent.name.startswith('outer-%04d-attempt-'%i)for i in roots):continue
  rows=[json.loads(line)for line in p.read_text().splitlines()]
  writes=[x for x in rows if x.get('event')=='ACTUAL_ADAM_WRITE']
  v['partial_actor'].append(dict(attempt=p.parent.name,writes=len(writes),actual=writes[-1]['actual_adam_steps']if writes else None))
 v['current_score_adoption']=None
 selection=b/'outers'/('outer-%06d'%stage_outer)/'probe-score/selection.json'
 previous=b/'outers'/('outer-%06d'%(stage_outer-1))/'complete.json'
 if selection.exists()and previous.exists():
  score=json.loads(selection.read_text());head=json.loads(previous.read_text())['probe']
  v['current_score_adoption']={key:score.get(key)for key in ('outer','actor_version','selection_model_kind','ema_updates','probe_before_checkpoint_sha256','outcome_labels_used')}
  v['current_score_adoption']['matches_previous_complete_head']=score['probe_before_checkpoint_sha256']==head['sha256']
 v['qa_pending']=None
 pending=b/'logs/qa-pending.json'
 if pending.exists():v['qa_pending']=json.loads(pending.read_text())
 v['grounding_refreshes']={}
 for i in (50,100,150):
  path=b/'outers'/('outer-%06d'%i)/'grounding-refresh.json'
  if path.exists():
   # Full refresh receipts embed the source actor state and can exceed8MB.
   # Keep them remote; normal milestone readers separately verify the original
   # QA/refit/adoption records. Repeating that payload in every status poll
   # makes an otherwise small read vulnerable to transfer timeouts.
   raw=path.read_bytes();refresh=json.loads(raw);ground=refresh['grounding']
   v['grounding_refreshes'][str(i)]=dict(
    metadata_summary_only=True,source_path=str(path),
    source_sha256=hashlib.sha256(raw).hexdigest(),
    outer=refresh['outer'],run_id=refresh['run_id'],
    required_before_outer=refresh['required_before_outer'],
    grounding={key:ground.get(key)for key in (
     'round','version','source_actor_version','manifest_path','manifest_sha256',
     'adoption_receipt','adoption_receipt_sha256','head_sha256')})
 if v['cleanup']or v['native'].get('leader_same_birth')is False:
  path=b/'logs/native-pipeline.log'
  with path.open('rb')as stream:
   stream.seek(max(0,path.stat().st_size-6000));v['terminal_log_tail']=stream.read().decode(errors='replace')
 result[arm]=v
print(json.dumps(result,allow_nan=False))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--until', type=int, default=50, choices=(50,100,150,200))
    parser.add_argument('--verified', type=int, default=10)
    parser.add_argument('--verified-a',type=int)
    parser.add_argument('--verified-b',type=int)
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--transport', choices=MODES, default='retained18')
    parser.add_argument('--path-layout', choices=tuple(PATH_LAYOUTS), default='source8')
    args = parser.parse_args()
    assert 0 <= args.verified < args.until and args.interval >= 60
    lock = (ROOT/'work/prefix5-ab-h10011-normal-boundary-observer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    arms = json.loads((ROOT/'work/prefix5-ab-assigned-arms.json').read_text())
    assert set(arms) == {'A','B'} and all(x['node']=='h100-11'for x in arms.values())
    spec = importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py')
    ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)
    verified = {'A':args.verified if args.verified_a is None else args.verified_a,
                'B':args.verified if args.verified_b is None else args.verified_b}
    if not all(0<=value<args.until for value in verified.values()):raise ValueError('Each arm requires its own actual verified boundary below target')
    facts = {s:[] for s in arms}
    last_key = None
    while True:
        tag = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        inner = INNER.replace('__ARMS__',repr(arms)).replace('__VERIFIED__',repr(verified))
        script = ('import subprocess,os\np=subprocess.run('+repr([
            'unshare','--mount','--propagation','private','bash',PATH_LAYOUTS[args.path_layout],
            '/dataset/shaoshitong/Search-G1-common8/env/bin/python','-c',inner])+
            ',env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1"),'
            'text=True,capture_output=True,timeout=120)\n'
            'assert p.returncode==0,p.stderr\nprint(p.stdout)')
        try:
            response, transport = run_readonly(ctl,'h100-11','/usr/bin/python3 -',
                input_text=script,timeout=155,mode=args.transport)
            if response.returncode: raise RuntimeError(response.stderr or response.stdout)
            raw = json.loads(response.stdout)
            evidence = ROOT/('work/prefix5-ab-h10011-normal-follow-'+tag+'-actual.json')
            evidence.write_text(json.dumps(raw,indent=2)+'\n')
            compact, attention = {}, []
            for arm,v in raw.items():
                for fact in v['facts']:
                    facts[arm].append(fact); verified[arm]=max(verified[arm],fact['outer'])
                compact[arm]={k:v[k]for k in ('base','mode','complete_outer','actor_adam','probe_adam',
                    'stage','native','cleanup','pipeline_complete','partial_actor','prefix','continuation',
                    'qa','qa_pending','current_score_adoption','grounding_refreshes')}
                if ((v['cleanup']or v['native'].get('leader_same_birth')is False)
                        and not v['pipeline_complete']): attention.append(arm)
            value=dict(observed_utc=tag,node='h100-11',arms=compact,verified=verified,
                original=str(evidence),original_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
                terminal_attention=attention,training_modified=False,observation_transport=transport)
            value['observation_namespace'] = dict(path_layout=args.path_layout,
                wrapper=PATH_LAYOUTS[args.path_layout],live_training_changed=False)
            (ROOT/'work/prefix5-ab-latest-compact.json').write_text(json.dumps(value,indent=2)+'\n')
            (ROOT/'work/prefix5-ab-h10011-normal-follow-actual-facts.json').write_text(
                json.dumps(dict(facts=facts,last_observation=str(evidence)),indent=2)+'\n')
            key=[(arm,v['complete_outer'],v['stage'],v['partial_actor'])for arm,v in raw.items()]
            if key != last_key:
                print(json.dumps(value),flush=True); last_key=key
            if attention:
                print(json.dumps({arm:raw[arm].get('terminal_log_tail')for arm in attention}),flush=True)
                return 3
            if all(v>=args.until for v in verified.values()):
                print('BOTH_NORMAL_SAVEPOINT_%d_OBSERVED_ROOT_CONTINUES_QA_REFIT_AND_STATE_ACCEPTANCE'%args.until,flush=True)
                return 0
            if args.once: return 0
        except (RuntimeError,ValueError,KeyError,AssertionError,subprocess.TimeoutExpired)as error:
            print(json.dumps(dict(observation_error=repr(error),tag=tag,training_modified=False)),flush=True)
            if args.once: return 2
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
