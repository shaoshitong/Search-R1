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
    points.append(dict(code=r['code'],verified_x=decoded_x,verified_y=decoded_y))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(verified=True,point_coordinate_checks=points,
    all_pdf_text_preserved=True,page_size=[page.rect.width,page.rect.height],
    native_vector_paths=len(page.get_drawings()),raster_images=len(page.get_images()),
    files={p.name:sha(p) for p in P.iterdir() if p.suffix in ['.pdf','.png','.drawio','.py','.json'] and p.name!='verification.json'})
(P/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
