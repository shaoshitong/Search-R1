from pathlib import Path
import json
D=Path(__file__).resolve().parent;out=Path('/Users/shaoshitong/Downloads/WWW_2026_PAPER/drawio_figure/reward-selection-comparison');v=json.load((D/'hard-conditioned-soft-data-20261015.json').open());data=[]
for row in v['steps']:
 data.append({'step':row['step'],'conditions':{key:{k:z[k] for k in ['groups','trajectories','soft_reward_hist','soft_centered_delta_hist','soft_advantage_hist','active_soft_groups']} for key,z in row['conditions'].items() if key in ['hard_std_zero','hard_std_positive']}})
s=r'''import React,{useState} from 'react';
const rows=DATA;
const colors=['#8C6BB1','#168D7B'];
function Histogram({row,field,title,lo,hi,limit}:{row:any,field:string,title:string,lo:number,hi:number,limit:number}){
 const W=720,H=350,L=64,R=18,T=35,B=55,X=(x:number)=>L+(x-lo)/(hi-lo)*(W-L-R),Y=(y:number)=>H-B-Math.min(y,limit)/limit*(H-T-B);
 const series=[row.conditions.hard_std_zero,row.conditions.hard_std_positive];
 return <section><h2 style={{fontSize:19}}>{title}</h2><svg role="img" aria-label={title} viewBox={`0 0 ${W} ${H}`} style={{width:'100%'}}>
 {[0,.25,.5,.75,1].map(f=><g key={f}><line x1={L} x2={W-R} y1={Y(f*limit)} y2={Y(f*limit)} stroke="var(--border)"/><text x={L-8} y={Y(f*limit)+5} fontSize="13" textAnchor="end" fill="var(--secondary)">{(f*limit).toFixed(1)}</text></g>)}
 {[-1.5,-1,-.5,0,.5,1,1.5].filter(x=>x>=lo&&x<=hi).map(x=><text key={x} x={X(x)} y={H-B+25} textAnchor="middle" fontSize="13" fill="var(--secondary)">{x}</text>)}
 {series.map((s:any,j:number)=>{const h=s[field],n=s.trajectories;let pts=[[h.edges[0],0],...h.counts.flatMap((c:number,i:number)=>[[h.edges[i],c/n*100],[h.edges[i+1],c/n*100]]),[h.edges[h.edges.length-1],0]];return <g key={j}>
 {h.counts.map((c:number,i:number)=>c>0?<rect key={i} x={X(h.edges[i])} y={Y(c/n*100)} width={X(h.edges[i+1])-X(h.edges[i])} height={Y(0)-Y(c/n*100)} fill={colors[j]} opacity=".12"><title>{j===0?'Hard std=0':'Hard std>0'}; [{h.edges[i].toFixed(4)}, {h.edges[i+1].toFixed(4)}): {c}/{n} = {(c/n*100).toFixed(3)}%</title></rect>:null)}
 <path d={pts.map((p:number[],i:number)=>(i?'L':'M')+X(p[0])+','+Y(p[1])).join(' ')} fill="none" stroke={colors[j]} strokeWidth="2.4"/>
 {h.counts.map((c:number,i:number)=>c/n*100>limit?<text key={'over'+i} x={X((h.edges[i]+h.edges[i+1])/2)+(j===0?-7:7)} y={18+j*14} textAnchor={j===0?'end':'start'} fontSize="12" fill={colors[j]}>{(c/n*100).toFixed(1)}%↑</text>:null)}
 </g>})}<text transform={`rotate(-90 17 ${H/2})`} x="17" y={H/2} textAnchor="middle" fontSize="14" fill="var(--foreground)">Bin frequency (%)</text></svg></section>
}
export default function HardConditionedSoft(){const [step,setStep]=useState(100),[limit,setLimit]=useState(100);const row=rows.find(r=>r.step===step)!;return <main style={{maxWidth:1250,margin:'auto',padding:25,color:'var(--foreground)',fontFamily:'Georgia,serif'}}><h1>Hard Reward固定分组下的Soft分布</h1><p>始终按同题5条Hard Reward的std=0或&gt;0选题；两个子集内部都保留全部Soft轨迹。</p><div style={{display:'flex',gap:18,alignItems:'center'}}><label>iteration <select value={step} onChange={e=>setStep(+e.target.value)}>{rows.map(r=><option key={r.step}>{r.step}</option>)}</select></label><label>纵轴上限 <select value={limit} onChange={e=>setLimit(+e.target.value)}><option value={100}>100%（完整）</option><option value={10}>10%（放大尾部）</option><option value={3}>3%（放大尾部）</option></select></label></div><p style={{color:colors[0]}}>Hard std=0：{row.conditions.hard_std_zero.groups}题、{row.conditions.hard_std_zero.trajectories}轨迹；其中{row.conditions.hard_std_zero.active_soft_groups}题的Soft std&gt;0。</p><p style={{color:colors[1]}}>Hard std&gt;0：{row.conditions.hard_std_positive.groups}题、{row.conditions.hard_std_positive.trajectories}轨迹。</p>{limit<100&&<p>纵轴已放大；截去的峰值在图顶标注真实比例。数据和分母没有筛选或改变。</p>}<Histogram row={row} field="soft_reward_hist" title="原始Soft Reward r" lo={-.15} hi={1.65} limit={limit}/><Histogram row={row} field="soft_centered_delta_hist" title="同题中心化差值 δr=r−mean(r)" lo={-1.85} hi={1.85} limit={limit}/><Histogram row={row} field="soft_advantage_hist" title="标准化优势 A=δr/(sample std+1e−6)" lo={-1.85} hi={1.85} limit={limit}/><p>样本std使用correction=1；频率按轨迹数计算。Soft值来自历史冻结probe重建。分布可显示EM无区分时仍产生Soft优势与数值放大，不能单凭此判定错误梯度。</p></main>}
'''.replace('DATA',json.dumps(data,separators=(',',':')))
p=out/'hard-conditioned-soft-reward.canvas.tsx';p.write_text(s);assert 'dashed' not in s
for row in data:
 for key,vv in row['conditions'].items():
  for field in ['soft_reward_hist','soft_centered_delta_hist','soft_advantage_hist']:assert sum(vv[field]['counts'])==vv['trajectories']
print('Canvas written; exact counts and condition rules verified')
