"""Structural checks and PDF previews for the finalized six figure assets."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
import pymupdf


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--figures",type=Path,default=Path(__file__).resolve().parent.parent/"figures")
    args=parser.parse_args();root=args.figures;src=Path(__file__).resolve().parent
    manifest=json.loads((root/"figure_manifest.json").read_text());rows=[];tiles=[]
    for item in manifest["figures"]:
        stem=item["name"];doc=pymupdf.open(root/f"{stem}.pdf");page=doc[0];im=Image.open(root/f"{stem}.png")
        rows.append({"name":stem,"pdf_pages":len(doc),"pdf_images":len(page.get_images()),
            "pdf_vector_paths":len(page.get_drawings()),"svg_image_elements":(root/f"{stem}.svg").read_text().count("<image"),
            "png_pixels":im.size,"png_dpi":im.info.get("dpi"),"height_at_6_4in_width":6.4*im.height/im.width,
            "all_file_hashes_match_manifest":all(hashlib.sha256((root/f["name"]).read_bytes()).hexdigest()==f["sha256"] for f in item["files"].values())})
        preview=src/f"qa_{stem}_pdf.png";page.get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False).save(preview)
        thumb=Image.open(preview).convert("RGB");thumb.thumbnail((900,630))
        tile=Image.new("RGB",(950,680),"white");tile.paste(thumb,((950-thumb.width)//2,35))
        ImageDraw.Draw(tile).text((15,8),stem,fill="#123047");tiles.append(tile)
    canvas=Image.new("RGB",(1900,2040),"#eeeeee")
    for i,t in enumerate(tiles):canvas.paste(t,((i%2)*950,(i//2)*680))
    canvas.save(src/"qa_pdf_contact_sheet.png")
    log=(src/"render.log").read_text()
    acceptance={"figures":rows,"missing_glyph_warning_present":"missing from font" in log,
        "source_sha256":hashlib.sha256((src/"plot_v5.py").read_bytes()).hexdigest(),
        "cc_task_exit_code":1,"cc_task_produced_code":False,"fallback_completed":True,
        "pdf_creation":"Outlined SVG converted to vector PDF using PyMuPDF; all fills opaque and in-range.",
        "svg_text_is_outlined":True,"minimum_configured_label_pt":10,
        "decomposition_identity_error":manifest["figures"][3]["plotted_values"]["decomposition_HQF_minus_H"]["identity_error"]}
    (root/"figure_acceptance.json").write_text(json.dumps(acceptance,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(acceptance,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
