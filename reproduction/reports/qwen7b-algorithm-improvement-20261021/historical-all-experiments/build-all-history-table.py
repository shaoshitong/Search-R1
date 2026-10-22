"""Compile aggregate-only historical records; never load model/tensor/prediction files."""
import collections
import copy
import datetime
import hashlib
import json
from pathlib import Path

H = Path(__file__).resolve().parent
R = H.parent
REPO = R.parents[2]
rows = []
inputs = []


def read(path):
    path = Path(path)
    data = path.read_bytes()
    inputs.append({'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    return json.loads(data)


def add(row, file, pointer=None):
    row = copy.deepcopy(row)
    row['aggregate_file'] = str(file)
    if pointer:
        row['aggregate_pointer'] = pointer
    for k in ['train_BA', 'test_BA']:
        v = row.get(k)
        assert v is None or 0 <= v <= 1, (k, row)
    if row.get('top_mixed') is not None:
        assert row.get('top_k') and 0 <= row['top_mixed'] <= row['top_k']
        row['top_fraction'] = row['top_mixed'] / row['top_k']
    row['id'] = f'R{len(rows)+1:03d}'
    rows.append(row)


for name in ['early-warmup-finetune', 'direct-mixed-and-sampling',
             'earlier-feature-layer-pilots', 'count-capacity-and-calibration']:
    path = H / (name + '.json')
    for i, row in enumerate(read(path)['rows']):
        add(row, path, f'/rows/{i}')

for extra_name in ['extra-historical-diagnostics.json', 'extra-G20-history.json']:
    extra = H / extra_name
    if extra.exists():
        for i, row in enumerate(read(extra)['rows']):
            add(row, extra, f'/rows/{i}')

path = H / 'recent-all-pilot-epochs-remote.json'
recent = read(path)
for key, value in recent['pilots'].items():
    name = key.rsplit('_E', 1)[0]
    for st, val in value['metrics']['fixed_validation2048'].items():
        train = value['metrics']['fixed_train2048'][st]
        add(dict(experiment=f'8192题条件均衡G5 / {name}', checkpoint=f'E{value["epoch"]}',
                 state=st, train_BA=train['BA'], test_BA=val['BA'],
                 train_BCE=train['BCE'], test_BCE=val['BCE'],
                 top_mixed=val['Top512_Mixed'], top_k=512, pool_questions=2048,
                 pool_identity='当前独立HF固定G5 / 2048题 / 663 Mixed',
                 train_pool='固定训练B2048 / 888 Mixed；8192题pilot，每epoch随机G5',
                 source=str(path), status='MEASURED',
                 comparability_note='E0为初始评估；E1–3各64次更新。训练池不同于全量直接Mixed的训练A2048。'),
            path, f'/pilots/{key}/metrics/fixed_validation2048/{st}')

# Fresh E3 of the uniform full-data fork, absent from the earlier source snapshot.
u = recent['uniform']
if 'independent3' in u and 'training3' in u:
    for st, values in u['independent3']['metrics'].items():
        val = values['fixed_selection2048']
        train = u['training3']['metrics'][st]['fixed_train2048']
        add(dict(experiment='corrected-labelblind-uniformG5-classweighted749120', checkpoint='E3', state=st,
                 train_BA=train['BA'], test_BA=val['BA'], train_BCE=train['BCE'], test_BCE=val['BCE'],
                 top_mixed=val['Top512_Mixed'], top_k=512, pool_questions=2048,
                 pool_identity='当前独立HF固定G5 / 2048题 / 663 Mixed',
                 train_pool='固定训练A2048 / 1082 Mixed；全169028题，label-blind随机G5',
                 source=str(path), status='MEASURED',
                 comparability_note='已实际完成E3/cursor5219；更新旧的“E3未完成”状态。'),
            path, f'/uniform/independent3/metrics/{st}/fixed_selection2048')

path = R / 'natural-g5-online-8192/actual-matched-metric-summary-complete.json'
natural = read(path)['rows']
for i, value in enumerate(natural):
    if value['pool'] != 'fixed_validation2048':
        continue
    train = next(v for v in natural if v['epoch'] == value['epoch'] and
                 v['layer'] == value['layer'] and v['pool'] == 'fixed_train2048')
    for st, models in value['models'].items():
        val, tr = models['natural'], train['models'][st]['natural']
        add(dict(experiment=f'8192题天然随机G5 / {value["layer"]}', checkpoint=f'E{value["epoch"]}', state=st,
                 train_BA=tr['balanced_accuracy'], test_BA=val['balanced_accuracy'],
                 train_BCE=tr['binary_Mixed_BCE'], test_BCE=val['binary_Mixed_BCE'],
                 top_mixed=val['selected512']['Mixed'], top_k=512, pool_questions=2048,
                 pool_identity='当前独立HF固定G5 / 2048题 / 663 Mixed',
                 train_pool='固定训练B2048 / 888 Mixed；8192题天然随机G5 pilot',
                 source=str(path), status='MEASURED',
                 comparability_note='与条件均衡G5 pilot共享题目、架构、预算；不同G5抽样。'),
            path, f'/rows/{i}/models/{st}/natural')


def variance_rows(cell, experiment, path, base_pointer, full):
    for key, value in cell.items():
        if not key.startswith('evaluation-epoch-') or not key.endswith('.json'):
            continue
        val_pool = value['metrics']['fixed_validation2048_G5']
        train_pool = value['metrics']['fixed_train2048_G5']
        for st, val in val_pool.items():
            train = train_pool[st]
            add(dict(experiment=experiment, checkpoint=f'E{value["epoch"]}', state=st,
                     train_BA=train['predeclared_G5_classification_diagnostic']['balanced_accuracy'],
                     test_BA=val['predeclared_G5_classification_diagnostic']['balanced_accuracy'],
                     train_variance_MSE=train['variance_MSE'], test_variance_MSE=val['variance_MSE'],
                     test_variance_MAE=val['variance_MAE'], test_Spearman=val['variance_Spearman'],
                     top_mixed=val['Top512']['Mixed'], top_k=512, pool_questions=2048,
                     pool_identity='当前独立HF固定G5 / 2048题 / 663 Mixed',
                     train_pool='固定训练B2048 / 888 Mixed', source=str(path), status='MEASURED',
                     comparability_note=('全169028题/10epoch在跑，EMA0.9995。' if full else '8192题/3epoch已完成，EMA0.99。') +
                     '回归population方差；BA仅为预设方差阈值0.08的诊断，非Mixed概率阈值0.5；排名按方差。'),
                path, f'{base_pointer}/{key}/metrics/fixed_validation2048_G5/{st}')


path = R / 'packed-variance/actual-live-latest.json'
variance_rows(read(path), 'packed变量M3–10方差MSE / 8192题', path, '', False)
path = R / 'packed-variance-full169028/actual-live-latest.json'
live = read(path)
for loss, cell in live['experiments'].items():
    variance_rows(cell, f'packed变量M3–10方差 / 全169028 / {loss}', path, f'/experiments/{loss}', True)

measured = [v for v in rows if v.get('test_BA') is not None or v.get('top_mixed') is not None]
missing = [v for v in rows if v not in measured]
groups = collections.OrderedDict()
for row in measured:
    groups.setdefault(row['experiment'], []).append(row)
out = dict(scope='本线程Qwen2.5-7B探针研究历史；不是3B RL/整篇论文全部baseline',
           generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           latest_remote_observation_epoch=live['time'], rows=rows, input_proofs=inputs,
           counts={'records': len(rows), 'measured_records': len(measured),
                   'missing_or_unrun_records': len(missing), 'named_measured_families': len(groups)},
           counts_are_not_independent_experiment_counts=True,
           final4096_read=False, new_GPU_evaluation=False,
           comparability='Different pools, backends, G5 labels and decision rules are retained. Repeated references/threshold variants are not independent experiments.')
(H / 'ALL-HISTORICAL-RESULTS.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')


def pct(v):
    return '—' if v is None else f'{100*v:.2f}%'


def clean(v):
    if isinstance(v, dict):
        return str(v.get('name', v))
    return str(v).replace('|', '\\|').replace('\n', ' ')


def cp(v):
    value = v.get('checkpoint')
    return Path(value).name if value and '/' in str(value) else str(value or '未产生')


def loss(v, prefix):
    for key, name in [(f'{prefix}_BCE', 'BCE'), (f'{prefix}_Mixed_BCE', 'MixedBCE'),
                      (f'{prefix}_variance_MSE', 'vMSE')]:
        if v.get(key) is not None:
            return f'{name}={v[key]:.6f}'
    return '—'


md = ['# 7B探针历史实验完整记录', '',
      f'汇总时间：{out["generated_at"]}。共 **{len(rows)}条 checkpoint×权重×评测池/判定规则记录**，其中{len(measured)}条有实际指标，{len(missing)}条明确未完成或未留存指标。这些条目包含同一模型的参考复评，不能当作独立实验数量。', '',
      'BA指Mixed均衡准确率：0.5×(Mixed召回率＋nonMixed召回率)，不使用轨迹准确率替代。raw为实际训练权重，EMA为滑动平均权重；缺失记“—”。', '',
      '旧独立2048池有641道Mixed；当前HF固定G5独立2048池有663道Mixed。内部holdout的问题可能被原始warmup见过，不是未知问题验证。不同G5、特征后端、阈值和数据池不可混排行。表内保留来源中的原始池名称。当前2048池已用于多次探索，不是最终无偏测试；最终4096仍未评分。', '',
      '全量直接Mixed固定训练池A为2048题/1082 Mixed；新layer/prefix及方差pilot固定训练池B为2048题/888 Mixed。训练BA跨A/B不能直接比较。方差模型BA使用事先固定的v≥0.08诊断，其输出不是Mixed概率。', '',
      '完整机器记录含精确小数、各项loss、召回率/误报率（原始报告存在时）、checkpoint引用、原始文件及JSON pointer。此表只读小型已有汇总；没有重新做GPU评分或读取最终4096。', '',
      '[机器可读完整记录](ALL-HISTORICAL-RESULTS.json)', '']
for family, records in groups.items():
    md.extend([f'## {family}', '',
               '| 编号 | checkpoint | 权重 | 训练Mixed BA | 验证Mixed BA | 真实Mixed / TopK | 比例 | 候选池题数 | 训练loss | 验证loss |',
               '|---|---|---|---:|---:|---:|---:|---:|---|---|'])
    for v in records:
        top = f'{v["top_mixed"]}/{v["top_k"]}' if v.get('top_mixed') is not None else '—'
        md.append('| ' + ' | '.join([v['id'], clean(cp(v)), clean(v['state']), pct(v.get('train_BA')),
                  pct(v.get('test_BA')), top, pct(v.get('top_fraction')), str(v.get('pool_questions') or '—'),
                  loss(v, 'train'), loss(v, 'test')]) + ' |')
    md.append('')
    for pool in dict.fromkeys(clean(v.get('pool_identity', '未标记')) for v in records):
        md.append(f'- 评测池：{pool}')
    for note in dict.fromkeys(v.get('comparability_note', '') for v in records):
        if note:
            md.append('- ' + clean(note))
    for src in dict.fromkeys(v.get('source', v['aggregate_file']) for v in records):
        file, _, pointer = str(src).partition('#')
        p = Path(file) if Path(file).is_absolute() else REPO / file
        md.append(f'- [原始汇总：{p.name}]({p})' + (f'，JSON位置 `{pointer}`' if pointer else ''))
    md.append('')
md.extend(['## 未完成、未评分或没有保留该指标的记录', '',
           '这些记录保留以免把“提交/准备/局部进度”误写成已完成实验。', '',
           '| 编号 | 实验 | checkpoint | 权重 | 原始状态 |', '|---|---|---|---|---|'])
for v in missing:
    md.append('| ' + ' | '.join([v['id'], clean(v['experiment']), clean(cp(v)), clean(v['state']), clean(v.get('status', '未知'))]) + ' |')
md.append('')
(H / 'ALL-HISTORICAL-RESULTS.md').write_text('\n'.join(md))
print(json.dumps(out['counts'], ensure_ascii=False))
