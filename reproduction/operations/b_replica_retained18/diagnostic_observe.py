"""Verify saved fixed512 G5 labels remotely and copy only a small summary."""
import argparse,json,time
from pathlib import Path
from .control import OUT,remote
from .shared_observe import BASE

SCRIPT=r'''
from pathlib import Path
from collections import defaultdict
import json,time,hashlib
p=Path(BASE)
def physical(s):
 prefix='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'
 return Path('/mnt/data/shaoshitong')/s[len(prefix):]if s.startswith(prefix)else Path(s)
def load(n):
 d=json.loads((p/'outers'/('outer-%06d'%n)/'diagnostic-complete.json').read_text())
 f=physical(d['full_complete']);raw=f.read_bytes()
 assert hashlib.sha256(raw).hexdigest()==d['full_complete_sha256']
 v=json.loads(raw);raw=physical(v['own_em']['path']).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==v['own_em']['sha256']
 rows=[json.loads(s)for s in raw.splitlines()];groups=defaultdict(list)
 for r in rows:
  assert r['strict_em']in(0,1)and r['actor_version']==d['actor_version']
  groups[r['qid']].append(r['strict_em'])
 assert len(rows)==2560 and len(groups)==512 and all(len(g)==5 for g in groups.values())
 assert len({r['trajectory_id']for r in rows})==2560
 exits=v['owned_native_worker_exit_codes']
 assert set(exits)=={str(i)for i in range(8)}and all(x==0 for x in exits.values())
 correct=sum(r['strict_em']for r in rows);mixed=sum(min(g)!=max(g)for g in groups.values())
 result=dict(completed=True,periodic_outer=n,actor_version=d['actor_version'],
  actor_outer=int(d['actor_version'].rsplit('actor',1)[1]),questions=512,trajectories=2560,
  strict_correct=correct,trajectory_mean_em=correct/2560,mixed_questions=mixed,
  question_mixed_fraction=mixed/512,source_sha256=d['full_complete_sha256'],
  own_em_sha256=v['own_em']['sha256'],worker_exit_codes=exits,
  fixed_identity_sha256=d['fixed_test_identity_sha256'],f1_available=False,is_final_51713=False)
 return result,set(groups)
result,qids=load(OUTER)
previous_path=p/'outers'/('outer-%06d'%(OUTER-10))/'diagnostic-complete.json'
if previous_path.exists():
 previous,old_qids=load(OUTER-10)
 assert qids==old_qids and result['fixed_identity_sha256']==previous['fixed_identity_sha256']
 result.update(same_actual_question_ids_as_previous=True,same_fixed_identity_as_previous=True,
  previous={k:previous[k]for k in('periodic_outer','actor_outer','trajectory_mean_em','strict_correct')},
  delta_em_percentage_points=100*(result['trajectory_mean_em']-previous['trajectory_mean_em']))
result['observed']=time.time()
print(json.dumps(result))
'''

def main():
 parser=argparse.ArgumentParser();parser.add_argument('outer',type=int)
 n=parser.parse_args().outer
 if not 70<=n<=200 or n%10:raise ValueError('Existing periodic70..200 receipt required')
 script='BASE='+repr(BASE)+'\nOUTER='+str(n)+'\n'+SCRIPT
 v=json.loads(remote(script,OUT/('diagnostic%d-%d-transport.json'%(n,time.time_ns())),target='h100-10',established_controller=True))
 root=OUT.parents[1];name='DIAGNOSTIC%d_DIRECT_EM.json'%n
 for dest in(OUT,root/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/b-replica-retained18-20261018',Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/b-replica-retained18-20261018')):
  (dest/name).write_text(json.dumps(v,ensure_ascii=False,indent=2))
 print(json.dumps(v))

if __name__=='__main__':main()
