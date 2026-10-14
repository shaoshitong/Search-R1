from pathlib import Path
import json,hashlib,datetime,math,xml.etree.ElementTree as E
D=Path(__file__).resolve().parent;out=Path('/Users/shaoshitong/Downloads/WWW_2026_PAPER/drawio_figure/reward-selection-comparison');data=json.load((D.parent/'g1-diversity-20261014/distributions-and-evidence.json').open());curve=json.load((D/'fixed512-curve-data.json').open());docs=[]
for s in curve['series']:
 assert [v['iteration'] for v in s['points']]==[50,100,150,200]
 for p in s['points']:assert p['n']==512 and p['em']==p['correct']/512
for run in data['runs']:
 for r in run['rows']:
  assert sum(r['correct_count_hist'])==512
  assert sum(r['C_A_hist']['counts'])==2560
  expected=sum(n*math.sqrt(k*(5-k)/20) for k,n in enumerate(r['correct_count_hist']))/512
  assert abs(expected-r['C_std']['mean'])<1e-12
  assert sum(r['C_active_A_hist']['counts'])==5*(512-r['correct_count_hist'][0]-r['correct_count_hist'][5])
for r in data['old_soft']['rows']:
 assert sum(r['R_hist']['counts'])==2560 and sum(r['R_A_hist']['counts'])==2560 and sum(r['R_std_hist']['counts'])==512
for p in out.glob('*.drawio'):
 root=E.parse(p).getroot();cells=root.findall('.//mxCell');assert all('image=' not in c.get('style','') for c in cells);docs.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'pages':len(root.findall('diagram')),'native_editable_cells':len(cells),'embedded_bitmap':False})
progress=json.load((D/'final3395-progress-second.json').open());published=json.load((D/'publication-verified.json').open())
v={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'documents':docs,'twelve_em_points_verified':True,'histogram_counts_verified':True,'native_cli_export_checked':True,'group_std_correction':1,'per_question_group_size':5,'advantage_epsilon':1e-6,'original_training_modified':False,'final_eval_current':progress,'independent_reward_replay':{'accepted':False,'error':'Original final evaluation acquired GPUs; guard exited before model load','process_terminated':True,'online_rewards_not_persisted':True},'publication':{k:published[k] for k in ['source_commit','remote_sha','branch','verified_utc']}}
(D/'drawio-verification.json').write_text(json.dumps(v,indent=2));(out/'audit.json').write_text(json.dumps(v,indent=2))
(out/'data-notes.txt').write_text('''Data definition and coverage

Fixed512 EM curves: all three native evaluations use the same512 ordered question IDs, at50/100/150/200. See fixed512-curve-data.json for exact correct counts and source identities.

Names: oldSearch-G1 = Soft Reward; G1->R1 = Soft Reward -> Hard Reward; mixed-first = Easy Selection+Soft Reward.

Diversity = sample std within each question's five responses (correction1). Average question std and overall pooled reward std are different statistics. All histogram counts were checked; every full round has512 groups and2560 trajectories. Standardized scalar advantage A=(r-group_mean)/(sample_std+1e-6), equal trajectory weighting. This is not gradient clipping or PPO probability ratios.

actual-mean-reward-curves.drawio uses native actual-updates critic/rewards/mean. Curves have different reward scales; they do not provide paired reward std. Easy Selection current root retains completed rounds6..200; earlier1..5 are not invented in this chart.

hard-reward-distributions-three-methods.drawio: iter100 and200; each method's actual selected original training rollouts. Hard Reward here means strictEM0/1. These are EM-component normalized advantages, not actual soft-policy training advantages. Training selections differ; the three methods' sampled training questions are not identical. The fixed512 evaluation questions are identical.

reward-std-and-advantage-distributions.drawio: paired Hard/Soft values for identical original groups from the oldSoft Reward run, pagesiter20/100/140. Original frozen historical encoders and classifier heads were replayed. These are reconstructed rewards, not guaranteed bitwise recovery of online rewards.

question-level-reward-std.drawio: paired per-question std at oldSoft Reward iter100, separately including and excluding exactly constant groups. Hard mean std0.090635, Soft mean std0.131545. The histogram's first bin also includes small positive std; zero-std counts are separately reported.

Missing: Easy Selection per-trajectory Soft Reward was not persisted. Its exact-input replay could decode and validate C but was blocked before GPU model loading when original final job3395 acquired all8 GPUs. No aggregate Soft Reward from that failed replay enters any figure. G1->R1 post-switch counterfactual Soft Reward is also unavailable. Missing distributions are not replaced with batch averages.

Latest status: all3 training chains completed200 iterations. Easy Selection fixed512 iter200 =270/512 =52.734375%. Its51713-question final job3394 failed due to worker constructor arguments; original agent's replacement3395 is genuinely advancing, observed1280 ->1792 questions by2026-10-14 23:55 China time, without a complete final score.

Raw tokens, model weights and full evaluation outputs stay remote. No original trainer, source manifest, configuration, probe service, or other agent was changed or contacted. All draw.io cells are native editable vector/line/text objects, no embedded chart bitmap.
''')
print(json.dumps({'documents':[(x['file'],x['pages'],x['native_editable_cells']) for x in docs],'checks':'PASS'}))
