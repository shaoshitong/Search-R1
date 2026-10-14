from pathlib import Path
import json,hashlib,xml.etree.ElementTree as E,datetime
D=Path(__file__).resolve().parent;out=Path('/Users/shaoshitong/Downloads/WWW_2026_PAPER/drawio_figure/reward-selection-comparison');v=json.load((D/'hard-conditioned-soft-data-20261015.json').open());old={r['step']:r for r in json.load((D.parent/'g1-diversity-20261014/distributions-and-evidence.json').open())['old_soft']['rows']}
for r in v['steps']:
 z=r['conditions']['all_correct'];p=r['conditions']['hard_std_positive'];wrong=r['conditions']['all_wrong'];base=old[r['step']]
 assert z['groups']+p['groups']+wrong['groups']==512 and z['trajectories']+p['trajectories']+wrong['trajectories']==2560
 assert z['groups']+wrong['groups']==round(base['C_zero_group_fraction']*512) and z['hard_em']==1 and wrong['hard_em']==0
 for key,oldkey in [('soft_reward_hist','R_hist'),('soft_advantage_hist','R_A_hist')]:
  assert z[key]['edges']==p[key]['edges']==base[oldkey]['edges']
  assert [a+b+c for a,b,c in zip(z[key]['counts'],p[key]['counts'],wrong[key]['counts'])]==base[oldkey]['counts']
 for c in [z,p]:
  for key in ['soft_reward_hist','soft_centered_delta_hist','soft_advantage_hist']:assert sum(c[key]['counts'])==c['trajectories']
files=[]
for name in ['reward-std-and-advantage-distributions.drawio','hard-conditioned-soft-reward-amplification.drawio']:
 path=out/name;root=E.parse(path).getroot();assert len(root.findall('diagram'))==3
 for page in root.findall('diagram'):
  cells=page.findall('.//mxCell');ids=[c.get('id') for c in cells];assert len(ids)==len(set(ids));assert all('image=' not in c.get('style','') for c in cells)
  values=[c.get('value','') for c in cells]
  if name.startswith('reward-std'):
   assert not any('Per-question reward std' in x or '(d) Advantages' in x for x in values)
   assert any('Soft advantage · all-correct vs mixed' in x for x in values)
 if name.startswith('reward-std'):
  values=[c.get('value','') for c in root.find('diagram').findall('.//mxCell')];assert 'Hard Reward (Search-R1)' in values and 'Soft Reward (Search-G1)' in values;assert not any('Reward distributions · iteration 20' in x for x in values)
 files.append({'file':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pages':3,'native_editable':True})
proof={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':v['sha256'],'conditions':'all-correct (Hard std=0, Hard mean=1) versus mixed (Hard std>0)','all_wrong_excluded_from_every_panel':True,'soft_std_zero_groups_not_used_as_selection_criterion':True,'included_plus_excluded_histograms_sum_to_original_histograms':True,'user_legend_and_removed_title_preserved':True,'figures':files,'interpretation':'All-correct Soft advantages are broadly distributed after removing all-wrong groups; group normalization does not mathematically guarantee a smooth population distribution.'};(out/'hard-conditioned-verification.json').write_text(json.dumps(proof,indent=2));(D/'hard-conditioned-verification.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
