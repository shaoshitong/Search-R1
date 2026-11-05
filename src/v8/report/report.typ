#import "style.typ": *
#show: setup
#set document(title: "旧作品、账号信息与新生产：内容平台的曝光实验与创作者博弈")
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
