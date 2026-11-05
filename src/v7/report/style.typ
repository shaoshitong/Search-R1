#let navy = rgb("123047")
#let teal = rgb("087f8c")
#let gold = rgb("d9a441")
#let red = rgb("bc5a45")
#let ink = rgb("243947")
#let muted = rgb("657783")
#let pale = rgb("eef4f5")
#let warm = rgb("f8f6f1")

#let bodyfont = "Songti SC"
#let sansfont = "Hiragino Sans GB"
#let latinfont = "Avenir Next"

#let setup(body) = {
  set document(title: "旧作品存量、账号资本与动态身份竞争", author: "AI内容平台与创作者博弈研究项目", date: datetime(year:2026,month:11,day:5))
  set text(font:(bodyfont,"Libertinus Serif"),size:11pt,lang:"zh",region:"CN",fill:ink)
  set par(justify:true,leading:0.7em,spacing:0.7em,first-line-indent:2em)
  set page(paper:"a4",margin:(top:20mm,bottom:19mm,left:22mm,right:22mm),
    header:context {
      if counter(page).get().first()>1 {
        set text(font:sansfont,size:7.3pt,fill:muted)
        grid(columns:(1fr,auto),[AI内容平台与创作者博弈],[V7 · 状态模型与证据对应])
        v(2mm)
        line(length:100%,stroke:0.4pt+rgb("d4dfe3"))
      }
    },
    footer:context {
      if counter(page).get().first()>1 {
        set text(font:sansfont,size:7.4pt,fill:muted)
        grid(columns:(1fr,auto),[课程研究报告 · 2026年11月5日],[#counter(page).display("1") / #counter(page).final().first()])
      }
    })
  set heading(numbering:none)
  show heading.where(level:1): it => block(above:0pt,below:6mm)[#text(font:sansfont,size:20pt,weight:"bold",fill:navy,it.body)]
  show heading.where(level:2): it => block(above:4mm,below:2.5mm)[#text(font:sansfont,size:11.5pt,weight:"bold",fill:teal,it.body)]
  show math.equation: set text(font:"STIX Two Math",size:11pt)
  set math.equation(numbering:none)
  set figure(gap:2.5mm,supplement:[图])
  show figure.caption: it => {
    set par(first-line-indent:0pt,leading:0.45em,justify:true)
    text(font:sansfont,size:8.5pt,fill:muted,it)
  }
  show link: set text(fill:teal)
  body
}

#let kicker(label) = block(above:0pt,below:3mm)[#text(font:latinfont,size:8.5pt,weight:600,fill:teal,tracking:1.1pt,label)]
#let chapter(n,title,sub:none) = {
  kicker("RESEARCH NOTE / "+str(n))
  heading(level:1,title)
  if sub != none { block(above:-2mm,below:5mm)[#text(font:sansfont,size:9.4pt,fill:muted,sub)] }
}

#let note(title,body,kind:"teal") = {
  let accent=if kind=="red" {red} else if kind=="gold" {gold} else {teal}
  block(width:100%,fill:if kind=="gold" {warm} else {pale},stroke:(left:2.5pt+accent),inset:(x:4mm,y:3.2mm),above:3mm,below:4mm)[
    #set par(first-line-indent:0pt,leading:0.45em,spacing:0.4em)
    #text(font:sansfont,size:9.2pt,weight:"bold",fill:navy,title)
    #v(1mm)
    #text(size:9.5pt,body)
  ]
}

#let source(body) = block(above:1mm,below:3mm)[
  #set par(first-line-indent:0pt,leading:0.3em)
  #text(font:sansfont,size:7.5pt,fill:muted,body)
]

#let tab(columns,headers,rows) = {
  set par(first-line-indent:0pt,leading:0.4em)
  set text(font:sansfont,size:8.7pt)
  table(columns:columns,inset:(x:2.6mm,y:2.2mm),stroke:none,
    fill:(x,y)=>if y==0 {pale} else if calc.odd(y) {white} else {rgb("f7f9fa")},
    table.hline(stroke:0.7pt+navy),
    table.header(..headers.map(h=>text(weight:"bold",fill:navy,h))),
    ..rows.flatten(),
    table.hline(stroke:0.45pt+rgb("cbd7dc")))
}

#let metric(value,label,detail) = block(fill:pale,inset:4mm,radius:2mm,width:100%)[
  #set par(first-line-indent:0pt,leading:0.45em,spacing:0pt)
  #text(font:latinfont,size:23pt,weight:600,fill:teal,value)
  #v(2mm)
  #text(font:sansfont,size:9.5pt,weight:"bold",fill:navy,label)
  #v(1.2mm)
  #text(size:8.6pt,fill:muted,detail)
]

#let chart(name,caption,width:100%,draft:false) = figure(
  if draft {block(width:width,height:78mm,fill:pale,inset:5mm)[#text(font:sansfont,size:8.2pt,fill:muted,"图表排版占位 / "+name)]}
  else {image("../figures/"+name+".svg",width:width)},caption:caption)

#let tiny(body) = text(font:sansfont,size:8.2pt,fill:muted,body)
