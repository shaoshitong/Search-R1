"""Run after native draw.io PDF export; preserve vectors and verify final files."""
from pathlib import Path
import json,hashlib,math,xml.etree.ElementTree as E
import fitz
P=Path(__file__).resolve().parent
pdf=P/'token-nq-em-iter200.pdf'
doc=fitz.open(pdf);page=doc[0]
original_text=page.get_text()
boxes=[fitz.Rect(b[:4]) for b in page.get_text('blocks')]
for d in page.get_drawings():
    rect=d['rect']
    # Only the white artboard is excluded from content bounds.
    if rect.width>page.rect.width*.99 and rect.height>page.rect.height*.99:
        continue
    boxes.append(rect)
bounds=fitz.Rect(boxes[0])
for b in boxes[1:]:bounds|=b
crop=(bounds+(-4,-4,4,4)) & page.rect
page.set_cropbox(crop)
temp=P/'token-nq-em-iter200-crop-tmp.pdf'
doc.save(temp,garbage=4,deflate=True);doc.close();temp.replace(pdf)
doc=fitz.open(pdf);page=doc[0]
assert original_text==page.get_text()
assert len(page.get_images())==0
assert len(page.get_drawings())>=40
page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(P/'token-nq-em-iter200.png')
audit=json.loads((P/'data-audit.json').read_text())
root=E.parse(P/'token-nq-em-iter200.drawio')
assert root.find(".//mxCell[@id='iteration-key']") is None
assert root.find(".//mxCell[@id='method-legend-background']") is None
assert 'Iteration 200' not in page.get_text()
region=root.find(".//mxCell[@id='dynrpo-dominated-region']/mxGeometry")
dyn=audit['points'][-1]
assert math.isclose(float(region.get('x')),110+(dyn['mean_tokens']-800)/2000*1790,abs_tol=1e-8)
assert math.isclose(float(region.get('y')),495-(dyn['em_percent']-28)/22*460,abs_tol=1e-8)
assert audit['dominated_region']['methods']==['Search-G1','TOP-D alpha=0.1','OPD alpha=1','OPSD3304']
notice=root.find(".//mxCell[@id='dominated-region-description']")
assert notice.get('value')=='More tokens, lower EM!'
assert 'fontColor=#AE2928;' in notice.get('style')
assert 'fontStyle=3;' in notice.get('style')
assert 'fontSize=40;' in notice.get('style')
assert len(set(d['color'] for d in audit['marker_designs'].values()))==7
assert len(set(d['shape'] for d in audit['marker_designs'].values()))==7
topd_leader=root.find(".//mxCell[@id='leader-B']/mxGeometry")
assert topd_leader is not None
assert topd_leader.find("mxPoint[@as='targetPoint']").get('x')=='1677'
points=[]
for r in audit['points']:
    c=root.find(f".//mxCell[@id='point-{r['code']}']/mxGeometry")
    x=float(c.get('x'))+float(c.get('width'))/2
    y=float(c.get('y'))+float(c.get('height'))/2
    decoded_x=800+(x-110)/1790*2000
    decoded_y=28+(495-y)/460*22
    assert math.isclose(decoded_x,r['mean_tokens'],abs_tol=1e-8)
    assert math.isclose(decoded_y,r['em_percent'],abs_tol=1e-8)
    assert r['label'] in page.get_text()
    assert root.find(f".//mxCell[@id='label-{r['code']}']").get('value')==r['label']
    marker=root.find(f".//mxCell[@id='point-{r['code']}']")
    color=audit['marker_designs'][r['code']]['color']
    assert f'fillColor={color};' in marker.get('style')
    assert 'fillStyle=solid;' in marker.get('style')
    points.append(dict(code=r['code'],verified_x=decoded_x,verified_y=decoded_y))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(verified=True,point_coordinate_checks=points,
    dominance_region_boundary_verified=True,dominated_methods=audit['dominated_region']['methods'],
    direct_labels_verified=True,iteration_heading_and_legend_removed=True,
    seven_distinct_solid_marker_designs_verified=True,topd_visible_leader_verified=True,
    bold_red_exclamation_verified=True,
    all_pdf_text_preserved=True,page_size=[page.rect.width,page.rect.height],
    native_vector_paths=len(page.get_drawings()),raster_images=len(page.get_images()),
    files={p.name:sha(p) for p in P.iterdir() if p.suffix in ['.pdf','.png','.drawio','.py','.json'] and p.name!='verification.json'})
(P/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
