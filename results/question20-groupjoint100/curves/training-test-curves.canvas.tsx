import React, { useEffect, useMemo, useRef, useState } from "react";
import curveMetrics from "./curve-metrics.json";

type MetricName = "accuracy" | "balanced_accuracy" | "trajectory_bce" | "group_bce" | "total_loss" | "mixed_recall" | "nonmixed_recall";
type Metrics = Partial<Record<MetricName, number | null>> & { count?: number; mixed_count?: number; nonmixed_count?: number };
type Row = { epoch: number; source_data_epoch?: number; observed_utc?: string; actual_lr?: number; train: Metrics; test: Metrics };
type Snapshot = { run_id?: string; job?: string | number; node?: string; snapshot_utc?: string; complete_epoch?: number; target_epochs?: number; rows: Row[] };
type Point = { epoch: number; value: number | null };
type Series = { label: string; color: string; split: "train" | "test"; metric: MetricName };
type PlotSpec = { key: string; title: string; description: string; percentage: boolean; series: Series[]; reference?: { value: number; label: string }; note?: string };

const DATA = curveMetrics as unknown as Snapshot;
const WINDOW = 10;
const TEST_MAJORITY = 329 / 512;
const COLORS = { train: "var(--qjc-train)", test: "var(--qjc-test)", nonmixed: "var(--qjc-nonmixed)" };
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const trainTest = (metric: MetricName): Series[] => [
  { label: "训练", split: "train", metric, color: COLORS.train },
  { label: "测试", split: "test", metric, color: COLORS.test },
];

const PLOTS: PlotSpec[] = [
  { key: "balanced", title: "混合判别 · balanced accuracy", description: "混合与非混合两类 recall 的均值", percentage: true,
    series: trainTest("balanced_accuracy"), reference: { value: .5, label: "50% 均衡参考" } },
  { key: "accuracy", title: "混合判别 · ordinary accuracy", description: "正确分类的问题数 / 完整 G5 问题数", percentage: true,
    series: trainTest("accuracy"), reference: { value: TEST_MAJORITY, label: "固定测试多数类 64.2578%" },
    note: "虚线只代表固定测试集。每轮训练批的类别比例不同。" },
  { key: "trajectory", title: "Trajectory BCE", description: "单条轨迹自身严格 EM 的二分类损失", percentage: false, series: trainTest("trajectory_bce") },
  { key: "group", title: "G5 mixed BCE", description: "同题五条轨迹是否混合正确的二分类损失", percentage: false, series: trainTest("group_bce") },
  { key: "total", title: "Total loss", description: "Trajectory BCE + 1 × G5 mixed BCE", percentage: false, series: trainTest("total_loss") },
  { key: "recall", title: "固定测试 · 两类 recall", description: "在各自真实类别内，被正确识别的问题比例", percentage: true,
    series: [
      { label: "混合类", split: "test", metric: "mixed_recall", color: COLORS.test },
      { label: "非混合类", split: "test", metric: "nonmixed_recall", color: COLORS.nonmixed },
    ] },
];

/** A value exists only after ten consecutive, complete, finite epochs. */
export function completeTrailingMean(points: Point[], window = WINDOW): Point[] {
  return points.map((point, index) => {
    if (index < window - 1) return { epoch: point.epoch, value: null };
    const tail = points.slice(index - window + 1, index + 1);
    const valid = tail.every((item, j) => finite(item.value) && item.epoch === point.epoch - window + 1 + j);
    return { epoch: point.epoch, value: valid ? tail.reduce((sum, item) => sum + (item.value as number), 0) / window : null };
  });
}

function normalizedRows(snapshot: Snapshot): Row[] {
  const byEpoch = new Map<number, Row>();
  for (const row of snapshot.rows || []) {
    if (!Number.isInteger(row.epoch) || row.epoch < 1 || !row.train || !row.test) continue;
    if (finite(snapshot.complete_epoch) && row.epoch > snapshot.complete_epoch) continue;
    if (byEpoch.has(row.epoch)) throw new Error(`Duplicate completed epoch ${row.epoch}`);
    byEpoch.set(row.epoch, row);
  }
  return Array.from(byEpoch.values()).sort((left, right) => left.epoch - right.epoch);
}

const ROWS = normalizedRows(DATA);
const formatValue = (value: number | null | undefined, percentage = false) => finite(value) ? percentage ? `${(value * 100).toFixed(2)}%` : value.toFixed(4) : "—";
const formatLR = (value: number | undefined) => finite(value) ? value.toExponential(3) : "—";

function useActualWidth() {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(420);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const update = () => {
      const next = Math.floor(element.getBoundingClientRect().width);
      if (next > 0) setWidth(old => old === next ? old : next);
    };
    update();
    if (typeof ResizeObserver !== "undefined") {
      const observer = new ResizeObserver(update);
      observer.observe(element);
      return () => observer.disconnect();
    }
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);
  return { ref, width };
}

function linePath(points: Point[], x: (epoch: number) => number, y: (value: number) => number): string {
  let previous: number | null = null;
  let path = "";
  for (const point of points) {
    if (!finite(point.value)) { previous = null; continue; }
    const connected = previous !== null && point.epoch === previous + 1;
    path += `${connected ? "L" : "M"}${x(point.epoch).toFixed(2)},${y(point.value).toFixed(2)} `;
    previous = point.epoch;
  }
  return path.trim();
}

function nearestEpoch(rows: Row[], target: number): number {
  return rows.reduce((best, row) => Math.abs(row.epoch - target) < Math.abs(best - target) ? row.epoch : best, rows[0].epoch);
}

function Plot({ spec, rows, activeEpoch, pinned, onHover, onPin, onMove }: {
  spec: PlotSpec; rows: Row[]; activeEpoch: number; pinned: boolean;
  onHover: (epoch: number | null) => void; onPin: (epoch: number) => void; onMove: (direction: number) => void;
}) {
  const { ref, width } = useActualWidth();
  const height = 248;
  const margin = { left: 64, right: 14, top: 21, bottom: 39 };
  const plotWidth = Math.max(1, width - margin.left - margin.right);
  const plotHeight = height - margin.top - margin.bottom;
  const first = rows[0].epoch, last = rows[rows.length - 1].epoch;
  const values = spec.series.map(series => {
    const raw = rows.map(row => ({ epoch: row.epoch, value: finite(row[series.split][series.metric]) ? row[series.split][series.metric] as number : null }));
    return { ...series, raw, mean: completeTrailingMean(raw) };
  });
  const maximum = Math.max(.1, ...values.flatMap(series => series.raw.filter(point => finite(point.value)).map(point => point.value as number)));
  const yMax = spec.percentage ? 1 : Math.ceil(maximum * 1.08 * 10) / 10;
  const x = (epoch: number) => margin.left + (last === first ? .5 : (epoch - first) / (last - first)) * plotWidth;
  const y = (value: number) => margin.top + plotHeight * (1 - value / yMax);
  const yTicks = Array.from({ length: 5 }, (_, index) => index * yMax / 4);
  const tickCount = width <= 360 ? 4 : 6;
  const xTicks = Array.from(new Set(Array.from({ length: tickCount }, (_, index) => Math.round(first + index * (last - first) / (tickCount - 1)))));
  const readEpoch = (event: React.MouseEvent<SVGSVGElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const position = (event.clientX - bounds.left) * width / bounds.width;
    const fraction = Math.max(0, Math.min(1, (position - margin.left) / plotWidth));
    return nearestEpoch(rows, first + fraction * (last - first));
  };
  return <figure className="qjc-card" data-chart={spec.key}>
    <figcaption><h2>{spec.title}</h2><p>{spec.description}</p></figcaption>
    <div className="qjc-chart-legend">{values.map(series => <span key={series.label}><i style={{ background: series.color }}/>{series.label}</span>)}
      {spec.reference && <span className="qjc-reference-legend"><i/>{spec.reference.label}</span>}
    </div>
    <div className="qjc-plot" ref={ref}>
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} role="img" tabIndex={0}
        aria-label={`${spec.title}，已完成 epoch ${first}至${last}，当前查看 epoch ${activeEpoch}。可用左右方向键切换。`}
        onMouseMove={event => onHover(readEpoch(event))} onMouseLeave={() => onHover(null)}
        onClick={event => onPin(readEpoch(event))} onKeyDown={event => {
          if (event.key === "ArrowLeft" || event.key === "ArrowRight") { event.preventDefault(); onMove(event.key === "ArrowLeft" ? -1 : 1); }
        }}>
        <title>{spec.title}：淡线为每轮原始值，粗线为完整十轮 trailing mean；点击固定查看轮次。</title>
        <text x={margin.left} y={12} className="qjc-axis-unit">{spec.percentage ? "比例（%）" : "平均损失（自然对数）"}</text>
        {yTicks.map(value => <g key={value}>
          <line x1={margin.left} y1={y(value)} x2={width - margin.right} y2={y(value)} className="qjc-grid-line"/>
          <text x={margin.left - 8} y={y(value) + 4} textAnchor="end" className="qjc-axis-label">{spec.percentage ? Math.round(value * 100) : value.toFixed(yMax < 1 ? 2 : 1)}</text>
        </g>)}
        {xTicks.map(epoch => <g key={epoch}>
          <line x1={x(epoch)} y1={height - margin.bottom} x2={x(epoch)} y2={height - margin.bottom + 4} className="qjc-axis-line"/>
          <text x={x(epoch)} y={height - margin.bottom + 18} textAnchor="middle" className="qjc-axis-label">{epoch}</text>
        </g>)}
        <text x={margin.left + plotWidth / 2} y={height - 3} textAnchor="middle" className="qjc-axis-label">已完成 epoch</text>
        {spec.reference && <line x1={margin.left} x2={width - margin.right} y1={y(spec.reference.value)} y2={y(spec.reference.value)} className="qjc-reference-line"/>}
        {values.map(series => <g key={series.label}>
          <path d={linePath(series.raw, x, y)} fill="none" stroke={series.color} strokeWidth={1.25} strokeOpacity={.25} strokeLinejoin="round"/>
          <path d={linePath(series.mean, x, y)} fill="none" stroke={series.color} strokeWidth={2.7} strokeLinejoin="round" strokeLinecap="round"/>
        </g>)}
        <line x1={x(activeEpoch)} x2={x(activeEpoch)} y1={margin.top} y2={height - margin.bottom} className={`qjc-cursor${pinned ? " is-pinned" : ""}`}/>
        {values.map(series => {
          const point = series.raw.find(row => row.epoch === activeEpoch);
          return finite(point?.value) ? <circle key={series.label} cx={x(activeEpoch)} cy={y(point!.value as number)} r={3.5} fill={series.color} stroke="var(--qjc-bg)" strokeWidth={1.6}><title>{series.label} E{activeEpoch}：{formatValue(point!.value, spec.percentage)}</title></circle> : null;
        })}
      </svg>
    </div>
    <div className="qjc-values" aria-label={`${spec.title} epoch ${activeEpoch} 的指标`}>
      <div className="qjc-values-head"><span>E{activeEpoch}</span><span>原始值</span><span>完整 10 轮均值</span></div>
      {values.map(series => <div key={series.label}><span><i style={{ background: series.color }}/>{series.label}</span>
        <strong>{formatValue(series.raw.find(point => point.epoch === activeEpoch)?.value, spec.percentage)}</strong>
        <strong>{formatValue(series.mean.find(point => point.epoch === activeEpoch)?.value, spec.percentage)}</strong>
      </div>)}
    </div>
    {spec.note && <p className="qjc-chart-note">{spec.note}</p>}
  </figure>;
}

const STYLES = String.raw`
#qjc-training-test-curves {
  --qjc-bg:var(--background); --qjc-fg:var(--foreground);
  --qjc-muted:var(--muted-foreground); --qjc-border:var(--border);
  --qjc-train:var(--viz-series-1);
  --qjc-test:var(--viz-series-2);
  --qjc-nonmixed:var(--viz-series-3);
  color:var(--qjc-fg); width:100%; min-width:0; max-width:1320px; margin:0 auto;
  font-family:inherit; font-size:13px; line-height:1.5; font-variant-numeric:tabular-nums;
}
#qjc-training-test-curves * { box-sizing:border-box; }
#qjc-training-test-curves h1,#qjc-training-test-curves h2,#qjc-training-test-curves p { margin:0; }
#qjc-training-test-curves h1 { font-size:24px; font-weight:600; letter-spacing:-.025em; line-height:1.25; }
#qjc-training-test-curves h2 { font-size:15px; font-weight:600; line-height:1.4; }
#qjc-training-test-curves .qjc-heading { display:grid; gap:9px; padding:4px 0 18px; }
#qjc-training-test-curves .qjc-eyebrow { font-size:11px; letter-spacing:.13em; color:var(--qjc-muted); }
#qjc-training-test-curves .qjc-subtitle { color:var(--qjc-muted); font-size:13px; max-width:900px; }
#qjc-training-test-curves .qjc-metadata { display:flex; flex-wrap:wrap; gap:6px 18px; font-size:12px; color:var(--qjc-muted); }
#qjc-training-test-curves .qjc-metadata strong { color:var(--qjc-fg); font-weight:500; }
#qjc-training-test-curves .qjc-controls { padding:14px 0 17px; border-top:1px solid var(--qjc-border); border-bottom:1px solid var(--qjc-border); display:grid; gap:11px; }
#qjc-training-test-curves .qjc-controls-top { display:flex; flex-wrap:wrap; gap:12px 24px; justify-content:space-between; align-items:center; }
#qjc-training-test-curves .qjc-epoch { display:flex; flex-wrap:wrap; align-items:baseline; gap:7px 15px; }
#qjc-training-test-curves .qjc-epoch strong { font-size:19px; font-weight:600; }
#qjc-training-test-curves .qjc-epoch span { font-size:12px; color:var(--qjc-muted); }
#qjc-training-test-curves .qjc-buttons { display:flex; gap:6px; }
#qjc-training-test-curves button { border:1px solid var(--qjc-border); color:var(--qjc-fg); background:var(--qjc-bg); min-height:32px; border-radius:5px; padding:4px 10px; cursor:pointer; font:inherit; font-size:12px; }
#qjc-training-test-curves button:disabled { opacity:.4; cursor:default; }
#qjc-training-test-curves button:focus-visible,#qjc-training-test-curves input:focus-visible,#qjc-training-test-curves svg:focus-visible { outline:2px solid var(--qjc-train); outline-offset:3px; }
#qjc-training-test-curves .qjc-slider { display:grid; grid-template-columns:max-content minmax(80px,1fr) max-content; gap:13px; align-items:center; }
#qjc-training-test-curves input[type=range] { width:100%; margin:0; accent-color:var(--qjc-train); }
#qjc-training-test-curves .qjc-slider span { color:var(--qjc-muted); font-size:11px; }
#qjc-training-test-curves .qjc-explanation { display:flex; flex-wrap:wrap; gap:8px 23px; color:var(--qjc-muted); font-size:12px; }
#qjc-training-test-curves .qjc-explanation i { display:inline-block; width:24px; height:0; margin-right:6px; vertical-align:middle; border-top:2.7px solid var(--qjc-fg); }
#qjc-training-test-curves .qjc-explanation .qjc-raw-example { border-top-width:1.25px; opacity:.3; }
#qjc-training-test-curves .qjc-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:22px 24px; padding-top:23px; }
#qjc-training-test-curves .qjc-card { min-width:0; margin:0; padding:15px 15px 13px; }
#qjc-training-test-curves figcaption { display:grid; gap:4px; }
#qjc-training-test-curves figcaption p { font-size:12px; color:var(--qjc-muted); }
#qjc-training-test-curves .qjc-chart-legend { display:flex; flex-wrap:wrap; min-height:27px; align-items:center; gap:4px 15px; padding:9px 0 2px; font-size:11px; }
#qjc-training-test-curves .qjc-chart-legend span { display:inline-flex; align-items:center; gap:5px; }
#qjc-training-test-curves .qjc-chart-legend i,#qjc-training-test-curves .qjc-values i { display:inline-block; width:8px; height:8px; border-radius:50%; flex:none; }
#qjc-training-test-curves .qjc-chart-legend .qjc-reference-legend { color:var(--qjc-muted); }
#qjc-training-test-curves .qjc-reference-legend i { width:18px; height:0; border-radius:0; border-top:1px dashed var(--qjc-muted); }
#qjc-training-test-curves .qjc-plot { width:100%; min-width:0; }
#qjc-training-test-curves svg { display:block; width:100%; height:248px; overflow:visible; cursor:crosshair; }
#qjc-training-test-curves .qjc-grid-line { stroke:var(--qjc-border); stroke-width:1; opacity:.6; }
#qjc-training-test-curves .qjc-axis-line { stroke:var(--qjc-muted); stroke-width:1; }
#qjc-training-test-curves .qjc-axis-label,#qjc-training-test-curves .qjc-axis-unit { font-size:12px; fill:var(--qjc-muted); }
#qjc-training-test-curves .qjc-reference-line { stroke:var(--qjc-muted); stroke-width:1.1; stroke-dasharray:5 4; opacity:.85; }
#qjc-training-test-curves .qjc-cursor { stroke:var(--qjc-muted); stroke-width:1; stroke-dasharray:2 4; opacity:.65; }
#qjc-training-test-curves .qjc-cursor.is-pinned { stroke-dasharray:3 3; }
#qjc-training-test-curves .qjc-values { border-top:1px solid var(--qjc-border); padding-top:8px; display:grid; gap:5px; font-size:12px; }
#qjc-training-test-curves .qjc-values>div { display:grid; grid-template-columns:minmax(66px,1fr) minmax(70px,.9fr) minmax(86px,1fr); gap:7px; align-items:center; }
#qjc-training-test-curves .qjc-values>div>span:first-child { display:flex; gap:6px; align-items:center; }
#qjc-training-test-curves .qjc-values-head { color:var(--qjc-muted); font-size:11px; }
#qjc-training-test-curves .qjc-values>div>:not(:first-child) { text-align:right; font-weight:500; }
#qjc-training-test-curves .qjc-chart-note { color:var(--qjc-muted); font-size:11px; margin-top:9px; }
#qjc-training-test-curves .qjc-footnote { color:var(--qjc-muted); font-size:12px; display:grid; gap:5px; margin-top:19px; padding-top:14px; border-top:1px solid var(--qjc-border); }
@media(max-width:780px) { #qjc-training-test-curves .qjc-grid { grid-template-columns:minmax(0,1fr); gap:16px; } }
@media(max-width:380px) { #qjc-training-test-curves h1 { font-size:21px; } #qjc-training-test-curves .qjc-card { padding:12px 10px; } #qjc-training-test-curves .qjc-controls-top { gap:9px; } }
`;

export default function TrainingTestCurves() {
  const rows = ROWS;
  const lastEpoch = rows.length ? rows[rows.length - 1].epoch : 0;
  const [selectedEpoch, setSelectedEpoch] = useState(lastEpoch);
  const [hoverEpoch, setHoverEpoch] = useState<number | null>(null);
  const byEpoch = useMemo(() => new Map(rows.map(row => [row.epoch, row])), [rows]);
  if (!rows.length) return <div id="qjc-training-test-curves"><style dangerouslySetInnerHTML={{ __html: STYLES }}/><p>尚无完整 epoch 指标，未绘制未完成轮次。</p></div>;
  const activeEpoch = hoverEpoch ?? selectedEpoch;
  const row = byEpoch.get(activeEpoch) || rows[rows.length - 1];
  const pin = (epoch: number) => { setSelectedEpoch(nearestEpoch(rows, epoch)); setHoverEpoch(null); };
  const move = (direction: number) => {
    const position = rows.findIndex(item => item.epoch === selectedEpoch);
    pin(rows[Math.max(0, Math.min(rows.length - 1, position + direction))].epoch);
  };
  return <div id="qjc-training-test-curves" data-complete-epoch={lastEpoch} data-selected-epoch={selectedEpoch} data-active-epoch={activeEpoch}>
    <style dangerouslySetInnerHTML={{ __html: STYLES }}/>
    <header className="qjc-heading">
      <span className="qjc-eyebrow">SEARCH-G1 · QUESTION + 20 TOKENS · JOINT BCE</span>
      <h1>混合判别：训练与固定测试</h1>
      <p className="qjc-subtitle">训练是每轮对应的 5,120 题 × G5，在本轮更新后的完整复测；测试始终是固定 512 题 × G5。统一以 q ≥ 0.5 判为混合，真实标签为 0 &lt; ΣEM &lt; 5。</p>
      <div className="qjc-metadata"><span>已完成 <strong>{rows.length} / {DATA.target_epochs || 100}</strong> epoch</span>
        <span>横轴截止 <strong>E{lastEpoch}</strong></span><span>作业 <strong>{String(DATA.job ?? "—")}</strong> · {DATA.node || "h100-8"}</span>
        {DATA.snapshot_utc && <span>快照 {DATA.snapshot_utc.replace("T", " ").replace(/\.\d+(?:Z|\+00:00)$/, " UTC").replace(/Z$/, " UTC")}</span>}
      </div>
    </header>
    <section className="qjc-controls" aria-label="选择已完成 epoch">
      <div className="qjc-controls-top"><div className="qjc-epoch"><strong>Epoch {activeEpoch}</strong><span>数据批 E{row.source_data_epoch ?? row.epoch}</span><span>实际 LR {formatLR(row.actual_lr)}</span></div>
        <div className="qjc-buttons"><button type="button" disabled={selectedEpoch === rows[0].epoch} onClick={() => move(-1)}>← 上一轮</button><button type="button" disabled={selectedEpoch === lastEpoch} onClick={() => move(1)}>下一轮 →</button><button type="button" disabled={selectedEpoch === lastEpoch} onClick={() => pin(lastEpoch)}>最新</button></div>
      </div>
      <label className="qjc-slider"><span>E{rows[0].epoch}</span><input type="range" min={rows[0].epoch} max={lastEpoch} step={1} value={selectedEpoch} aria-label="查看完整 epoch" onChange={event => pin(Number(event.target.value))}/><span>E{lastEpoch}</span></label>
      <div className="qjc-explanation"><span><i className="qjc-raw-example"/>原始每轮值</span><span><i/>完整 10 轮 trailing mean</span><span>悬停查看 · 点击固定 · 左右方向键切换</span></div>
    </section>
    <main className="qjc-grid">{PLOTS.map(spec => <Plot key={spec.key} spec={spec} rows={rows} activeEpoch={activeEpoch} pinned={hoverEpoch === null} onHover={setHoverEpoch} onPin={pin} onMove={move}/>)}</main>
    <footer className="qjc-footnote"><p>粗线只使用截至该轮的完整 10 轮：E1–E9 没有平滑值；缺轮或缺指标时断开，不补值、不外推。原始点始终保留。</p>
      <p>ordinary accuracy 的固定测试多数类基线为 329/512 = 64.2578%；balanced accuracy 的类别均衡参考为 50%。训练批每轮不同，不能把固定测试基线当作训练基线。</p>
      <p>图中训练与测试均来自相同的更新后 head；损失为自然对数。此文件静态读取同目录指标摘要，不自动联网或操作训练。</p>
    </footer>
  </div>;
}
