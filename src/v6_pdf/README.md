# 中文图文 PDF 编译源

本目录把冻结的 V5 研究结果排成14页A4中文报告，嵌入六张矢量科学图。
没有重新估计模型、采集原始数据或改变V5结果。图内经验数值读取
`docs/V5_AUDIT/`的汇总JSON；理论图读取既有参数条件，明确标为模型示例。

## 编译

已安装环境：Typst 0.14.2，macOS的Songti SC、Hiragino Sans GB、Avenir Next，
以及Libertinus Serif和STIX Two Math。其他系统可在style.typ中替换为已安装的
中文字体。无需下载TeX环境；默认直接复用已经转路径的SVG，不依赖图内字体。

在仓库根目录执行：

```bash
bash src/v6_pdf/build.sh /absolute/path/to/report.pdf
```

`tex/report.typ`为正文、公式、图注、参考文献；`tex/style.typ`定义字体、页眉页脚、
表格、配色与排版。`figures/`包含六图的SVG、矢量PDF及图表来源/验收清单。

## 重新绘图

现有 `/Applications/miniconda3/bin/python` 提供matplotlib、numpy、PyMuPDF和Pillow。
不应在Mac下载大规模原始数据。仅使用以下小型汇总：

```bash
/Applications/miniconda3/bin/python src/v6_pdf/plotting/plot_v5.py \
  --data docs/V5_AUDIT --out src/v6_pdf/figures
/Applications/miniconda3/bin/python src/v6_pdf/plotting/qa_assets.py \
  --figures src/v6_pdf/figures
```

脚本还输出300dpi PNG便于预览；PNG预览未纳入源码归档，生成清单仍保留实际运行时的对应哈希。中文标签转为矢量路径，避免CFF字体子集编码错误；
分解图不添加未计算的误差条。图表、差值区间、预测性诊断及模型参数示例分别标明。

## 制作与核验记录

按用户建议实际调用过本地Claude Code，但其返回未登录、退出1，没有生成代码；
未修改登录或配置。随后直接编写绘图代码并正常完成。真实回执见audit/，不能将本图
声称为Claude Code成功绘制。

报告经逐页PNG视觉检查和独立数值校核。原草稿的流程括号、数学lr字面量、最优积分
上横线，以及理论示例参数说明均已修正。最终数值/视觉审查绑定最终PDF的SHA。
PDF中的V5源码版本是分析冻结版本；本次排版源码有单独的新提交与交付构建清单。

原始数据与逐作品预测继续留HKUST3。本目录仅保留必要排版/绘图源、矢量图和小型审计。
