"""A self-contained Codex Canvas preview of the verified editable drawing."""
from pathlib import Path
import json, re, hashlib, xml.etree.ElementTree as E
D=Path(__file__).resolve().parent
O=Path('/Users/shaoshitong/Downloads/WWW_2026_PAPER/drawio_figure/reward-selection-comparison')
svg=(O/'hard-conditioned-soft-reward-amplification.svg').read_text()
# Draw.io embeds the complete XML in its SVG content attribute. Strip that
# duplicate payload; the visible native SVG remains identical.
svg=re.sub(r'\scontent="[^"]*"','',svg,count=1)
# Native SVG paths and XHTML text render directly in the Canvas browser. Draw.io
# also adds raster fallbacks for each text label; omit those duplicate fallbacks.
svg=re.sub(r'<image\b[^>]*/>','',svg)
svg=re.sub(r'^<\?xml[^>]*>\s*','',svg)
svg=re.sub(r'<!DOCTYPE[^>]*>\s*','',svg)
svg=re.sub(r'(<svg\b[^>]*?)\swidth="[^"]*"',r'\1 width="100%"',svg,count=1)
svg=re.sub(r'(<svg\b[^>]*?)\sheight="[^"]*"',r'\1 height="auto"',svg,count=1)
assert len(E.fromstring(svg).findall('.//{http://www.w3.org/2000/svg}image'))==0
code='''import React, {useState} from 'react';
const figure = SVG_DATA;
export default function RewardSelectionStory(){
 const [scale,setScale]=useState(100);
 return <main style={{padding:18,color:'var(--foreground)',background:'var(--background)'}}>
  <div style={{display:'flex',alignItems:'center',gap:14,marginBottom:12,fontFamily:'system-ui',fontSize:14}}>
   <strong>Reward selection · 三图递进版</strong>
   <label style={{marginLeft:'auto'}}>缩放 <input type="range" min="70" max="180" step="10" value={scale} onChange={e=>setScale(+e.target.value)} aria-label="图形缩放"/> {scale}%</label>
   <button onClick={()=>setScale(100)}>适合窗口</button>
  </div>
  <div style={{overflow:'auto',border:'1px solid var(--border)',borderRadius:12}}>
   <div style={{width:scale+'%',minWidth:650,background:'#ffffff',margin:'auto'}} dangerouslySetInnerHTML={{__html:figure}}/>
  </div>
 </main>;
}
'''.replace('SVG_DATA',json.dumps(svg,ensure_ascii=False))
p=O/'reward-selection-storyboard.canvas.tsx';p.write_text(code)
audit=json.loads((D/'selection-storyboard-verification.json').read_text())
audit['canvas_preview']={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'self_contained':True,'raster_images':0,'source_svg_sha256':hashlib.sha256((O/'hard-conditioned-soft-reward-amplification.svg').read_bytes()).hexdigest()}
(D/'selection-storyboard-verification.json').write_text(json.dumps(audit,indent=2))
print(p)
