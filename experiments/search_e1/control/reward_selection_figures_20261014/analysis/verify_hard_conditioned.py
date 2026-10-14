from pathlib import Path
import json,hashlib,xml.etree.ElementTree as E,datetime
D=Path(__file__).resolve().parent;out=Path('/Users/shaoshitong/Downloads/WWW_2026_PAPER/drawio_figure/reward-selection-comparison');v=json.load((D/'hard-conditioned-soft-data-20261015.json').open());old={r['step']:r for r in json.load((D.parent/'g1-diversity-20261014/distributions-and-evidence.json').open())['old_soft']['rows']}
for r in v['steps']:
 z=r['conditions']['hard_std_zero'];p=r['conditions']['hard_std_positive'];base=old[r['step']]
 assert z['groups']+p['groups']==512 and z['trajectories']+p['trajectories']==2560
 assert z['groups']==round(base['C_zero_group_fraction']*512)
 for key,oldkey in [('soft_reward_hist','R_hist'),('soft_advantage_hist','R_A_hist')]:
  assert z[key]['edges']==p[key]['edges']==base[oldkey]['edges']
  assert [a+b for a,b in zip(z[key]['counts'],p[key]['counts'])]==base[oldkey]['counts']
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
   assert any('Soft advantage by Hard std' in x for x in values)
 if name.startswith('reward-std'):
  values=[c.get('value','') for c in root.find('diagram').findall('.//mxCell')];assert 'Hard Reward (Search-R1)' in values and 'Soft Reward (Search-G1)' in values;assert not any('Reward distributions · iteration 20' in x for x in values)
 files.append({'file':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pages':3,'native_editable':True})
proof={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':v['sha256'],'condition_defined_only_by_hard_std':True,'soft_std_zero_groups_retained':True,'conditional_histograms_sum_to_original_histograms':True,'user_legend_and_removed_title_preserved':True,'figures':files,'interpretation':'Nonzero soft ordering and normalization amplification in EM-invariant groups; does not alone establish an erroneous gradient.'};(out/'hard-conditioned-verification.json').write_text(json.dumps(proof,indent=2));(D/'hard-conditioned-verification.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
