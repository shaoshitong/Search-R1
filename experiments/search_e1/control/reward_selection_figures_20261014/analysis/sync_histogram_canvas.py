from pathlib import Path
import json
D=Path(__file__).resolve().parent;out=Path('/Users/shaoshitong/Downloads/WWW_2026_PAPER/drawio_figure/reward-selection-comparison');p=out/'em512-and-reward-std.canvas.tsx';s=p.read_text();a=s.index('const data=')+11;b=s.index(';\n',a);v=json.loads(s[a:b]);old={r['step']:r for r in json.load((D.parent/'g1-diversity-20261014/distributions-and-evidence.json').open())['old_soft']['rows']}
for r in v['old_soft']:
 source=old[r['step']]
 for k in ['C_active_A_hist','R_active_A_hist','C_zero_group_fraction','R_zero_group_fraction']:r[k]=source[k]
s=s[:a]+json.dumps(v,separators=(',',':'))+s[b:]
s=s.replace("strokeDasharray={i===1?'8 5':undefined}","strokeDasharray={s.dashed?'8 5':undefined}")
a=s.index('function fromHist(');b=s.index('export default',a)
s=s[:a]+'''function fromHist(h:any,name:string,color:string){let n=h.counts.reduce((a:number,b:number)=>a+b,0);return {name,color,points:[[h.edges[0],0],...h.counts.flatMap((c:number,i:number)=>[[h.edges[i],c/n*100],[h.edges[i+1],c/n*100]]),[h.edges[h.edges.length-1],0]]}}\n'''+s[b:]
s=s.replace("['adv','标准化优势']","['adv','(c) 全部轨迹优势'],['active_adv','(d) std>0题组优势']")
s=s.replace("name:s.name,color:colors[i],points", "name:s.name,color:colors[i],dashed:i===1,points")
s=s.replace('Soft Reward（蓝） · Hard Reward（紫虚线）','Soft Reward（蓝实线） · Hard Reward（橙实线）')
s=s.replace("'Hard Reward',colors[1]","'Hard Reward','#B46F26'")
s=s.replace("fromHist(r.R_A_hist,'Soft Reward'", "fromHist(mode==='active_adv'?r.R_active_A_hist:r.R_A_hist,'Soft Reward'").replace("fromHist(r.C_A_hist,'Hard Reward'", "fromHist(mode==='active_adv'?r.C_active_A_hist:r.C_A_hist,'Hard Reward'")
s=s.replace('xmin={-1.81} xmax={1.81} ymin={0} ymax={100}', "xmin={-1.85} xmax={1.85} ymin={0} ymax={mode==='active_adv'?40:100}")
s=s.replace("<p>Soft Reward来自冻结历史probe重建", "{mode==='adv'&&<p>(c)：512组共2560条轨迹；std=0的整组也保留，其优势为0。</p>}{mode==='active_adv'&&<p>(d)：Hard、Soft分别保留各自std&gt;0的题组，分别重新归一化。Hard {(1-r.C_zero_group_fraction)*512}组、{(1-r.C_zero_group_fraction)*2560}条；Soft {(1-r.R_zero_group_fraction)*512}组、{(1-r.R_zero_group_fraction)*2560}条。不是删除单条A=0；此图仍可能出现A=0。</p>}<p>Soft Reward来自冻结历史probe重建")
p.write_text(s);print('Canvas preview: closed solid histogram boundaries and explicit c/d modes')
