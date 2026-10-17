"""Self-contained three-tab Canvas/offline browser; no remote assets or runtime."""
from pathlib import Path
import re,json,hashlib,datetime,xml.etree.ElementTree as E,shutil
D=Path(__file__).resolve().parent;O=D/'figures'
report=json.loads((D/'all-seven-results.json').read_text())
subtitle=report['viewer_subtitle']
entries=[('train','训练EM曲线','training-em-curves.svg'),('test','固定512题柱状图','fixed512-em-bars.svg'),('final','51,713题终评表','final51713-seven-dataset-table.svg')]
figures=[]
for key,label,name in entries:
 raw=(O/name).read_text();s=re.sub(r'^<\?xml[^>]*>\s*','',raw);s=re.sub(r'<!DOCTYPE[^>]*>\s*','',s)
 s=re.sub(r'(<svg\b[^>]*?)\swidth="[^"]*"',r'\1 width="100%"',s,count=1)
 s=re.sub(r'(<svg\b[^>]*?)\sheight="[^"]*"',r'\1 style="display:block;width:100%;height:auto"',s,count=1)
 root=E.fromstring(s);assert not root.findall('.//{http://www.w3.org/2000/svg}image')
 figures.append({'key':key,'label':label,'svg':s})
data=json.dumps(figures,ensure_ascii=False,separators=(',',':'))
code='''import React, {useState} from 'react';
const figures=DATA;
export default function ExperimentResults(){
 const [tab,setTab]=useState('train'),[zoom,setZoom]=useState(100);
 const figure=figures.find(v=>v.key===tab)!;
 return <main style={{padding:20,color:'var(--foreground)',background:'var(--background)',fontFamily:'system-ui'}}>
 <h1 style={{fontSize:24,margin:'0 0 7px'}}>实验结果 · 2026年10月17日</h1>
 <p style={{color:'var(--secondary)',fontSize:13}}>SUBTITLE</p>
 <div style={{display:'flex',flexWrap:'wrap',gap:9,alignItems:'center',marginBottom:14}}>
 {figures.map(v=><button key={v.key} onClick={()=>{setTab(v.key);setZoom(100)}} style={{padding:'9px 14px',borderRadius:8,border:'1px solid var(--border)',background:tab===v.key?'#304958':'transparent',color:tab===v.key?'white':'var(--foreground)',cursor:'pointer'}}>{v.label}</button>)}
 <label style={{marginLeft:'auto',fontSize:13}}>缩放 <input type="range" min="70" max="220" step="10" value={zoom} onChange={e=>setZoom(+e.target.value)} aria-label="图表缩放"/> {zoom}%</label>
 <button onClick={()=>setZoom(100)}>适合窗口</button></div>
 <div style={{overflow:'auto',border:'1px solid var(--border)',borderRadius:10}}><div role="img" aria-label={figure.label} style={{width:zoom+'%',minWidth:800,background:'white',margin:'auto'}} dangerouslySetInnerHTML={{__html:figure.svg}}/></div>
 </main>;
}
'''.replace('DATA',data).replace('SUBTITLE',subtitle)
p=D/'results.canvas.tsx';p.write_text(code)
html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>实验结果 ·10月17日</title><style>body{font:15px system-ui;margin:0;background:#f3f6f8;color:#263b49}main{max-width:1800px;margin:auto;padding:24px}h1{margin:0 0 9px}p{color:#72838e}nav{display:flex;flex-wrap:wrap;gap:9px;margin:18px 0}button{padding:10px 15px;border:1px solid #ccd8df;background:white;border-radius:8px;color:#304958;cursor:pointer}button.active{background:#304958;color:white}.viewer{overflow:auto;border:1px solid #dce5ea;border-radius:12px;background:white}#figure{min-width:800px;margin:auto}label{margin-left:auto;align-self:center}</style><main><h1>实验结果 ·2026年10月17日</h1><p>SUBTITLE</p><nav id="tabs"></nav><div class="viewer"><div id="figure"></div></div></main><script>const figures=DATA;let current=DEFAULT,zoom=100;function render(){document.querySelector('#tabs').innerHTML=figures.map(v=>`<button class="${v.key===current?'active':''}" onclick="show('${v.key}')">${v.label}</button>`).join('')+`<label>缩放 <input type="range" min="70" max="220" value="${zoom}" step="10" oninput="scale(+this.value)"> <span id="z">${zoom}%</span></label><button onclick="scale(100)">适合窗口</button>`;document.querySelector('#figure').innerHTML=figures.find(v=>v.key===current).svg;scale(zoom)}function show(k){current=k;zoom=100;render()}function scale(z){zoom=z;document.querySelector('#figure').style.width=z+'%';document.querySelector('#z').textContent=z+'%'}render();</script></html>'''.replace('DATA',data).replace('SUBTITLE',subtitle)
for name,tab in [('interactive.html','train'),('ranking.html','test'),('final-seven.html','final')]:
 (D/name).write_text(html.replace('DEFAULT',json.dumps(tab)))
# Keep established in-app URLs useful, preserving the previous served pages.
served=D.parent/'training-and-fixed512-em-20261012-2245'
if served.exists():
 backup=served/'backups'/'before-20261017-1230-refresh';backup.mkdir(parents=True,exist_ok=True)
 for name in ['interactive.html','ranking.html','final-seven.html']:
  dest=served/name
  if dest.exists() and not (backup/name).exists():shutil.copy2(dest,backup/name)
  shutil.copy2(D/name,dest)
audit={'created_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'canvas':str(p),'canvas_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'tabs':[v['key'] for v in figures],'self_contained':True,'raster_images':0,'remote_assets':0,'old_browser_pages_backed_up':str(served/'backups/before-20261017-1230-refresh'),'png_artifacts':[str(x) for x in O.glob('*.png')],'scope':'output figures and owned result viewer only; paper drawio not modified'}
(D/'canvas-verification.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
