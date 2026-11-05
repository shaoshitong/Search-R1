"""Compile the reviewed Chinese report with vector figures and readable tables."""
from pathlib import Path
import json
import subprocess
import sys

root=Path(__file__).resolve().parents[2]
source=Path(sys.argv[1]).resolve()
target=Path(sys.argv[2]).resolve()
report=Path(__file__).with_name("report")
body=report/"body.typ"
subprocess.run(["pandoc",str(source),"-f","markdown+tex_math_dollars","-t","typst",
                "--wrap=none","-o",str(body)],check=True)
text=body.read_text().replace('"../src/v8/report/figures/','"figures/')
# Give definitions the width they need; Pandoc defaults these to equal columns.
for marker in ['table.header([术语]', 'table.header([符号]', 'table.header([问题]']:
    position=text.find(marker)
    if position>=0:
        start=text.rfind('columns: (50%, 50%)',0,position)
        if start>=0:
            text=text[:start]+text[start:].replace('columns: (50%, 50%)','columns: (27%, 73%)',1)
body.write_text(text)
title=next(line.lstrip('# ').strip() for line in source.read_text().splitlines() if line.startswith('# '))
wrapper='''#import "style.typ": *
#show: setup
#set document(title: TITLE)
#set par(leading:0.55em,spacing:0.6em)
#show heading.where(level:1): it => block(above:3mm,below:4mm)[#text(font:sansfont,size:23pt,weight:"bold",fill:navy,it.body)]
#show heading.where(level:2): it => block(above:5mm,below:2.5mm)[#text(font:sansfont,size:14pt,weight:"bold",fill:navy,it.body)]
#show heading.where(level:3): it => block(above:3mm,below:1.5mm)[#text(font:sansfont,size:11.5pt,weight:"bold",fill:teal,it.body)]
#show table: set text(font:sansfont,size:9pt)
#set table(inset:(x:4pt,y:3pt),stroke:(bottom:0.35pt+rgb("d4dfe3")),fill:(x,y)=>if y==0 {pale} else {white})
#show figure.where(kind:table): it => block(breakable:true,width:100%,it.body)
#set page(header:context {
  if counter(page).get().first()>1 {
    set text(font:sansfont,size:7.3pt,fill:muted)
    grid(columns:(1fr,auto),[内容平台的曝光与创作者竞争],[研究与实验总结])
    v(2mm)
    line(length:100%,stroke:0.4pt+rgb("d4dfe3"))
  }
},footer:context {
  set text(font:sansfont,size:7.4pt,fill:muted)
  grid(columns:(1fr,auto),[课程研究项目],[#counter(page).display("1") / #counter(page).final().first()])
})
#include "body.typ"
'''.replace('TITLE',json.dumps(title,ensure_ascii=False))
(report/"report.typ").write_text(wrapper)
subprocess.run(["typst","compile","--root",str(root),str(report/"report.typ"),str(target)],check=True)
print(target,target.stat().st_size)
