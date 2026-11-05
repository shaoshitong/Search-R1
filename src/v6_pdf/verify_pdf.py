"""Read-only structural/text acceptance after manual PDF page inspection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pymupdf

parser=argparse.ArgumentParser()
parser.add_argument('pdf',type=Path)
parser.add_argument('--reproduced',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
document=pymupdf.open(args.pdf)
texts=[page.get_text() for page in document]
fonts={font[0]:font for page in document for font in page.get_fonts(full=True)}
font_info=[]
for xref,font in fonts.items():
    name,extension,kind,content=document.extract_font(xref)
    font_info.append({'name':name,'type':kind,'bytes':len(content),'embedded':bool(content)})
assert len(document)==14
assert all(text.strip() and '\ufffd' not in text for text in texts)
assert all('图表排版占位' not in text and '编译样张' not in text for text in texts)
assert all(x['embedded'] for x in font_info)
assert digest(args.pdf)==digest(args.reproduced)
assert sum(len(page.get_images()) for page in document)==0
assert 'lr' not in texts[7]
assert 'case12' in texts[9] and 'case148' in texts[10]
report={'checked_utc':datetime.now(timezone.utc).isoformat(),
        'pdf_sha256':digest(args.pdf),'pdf_bytes':args.pdf.stat().st_size,
        'pages':len(document),'all_page_text_extracts_without_replacement_characters':True,
        'embedded_fonts':font_info,'raster_image_count':0,
        'vector_scientific_figures':6,'deterministic_reproduction_sha_matches':True,
        'reproduced_pdf_sha256':digest(args.reproduced),
        'manual_visual_review':{'pages_1_to_7':'Independent AI review; corrected p3 re-render checked.',
             'pages_8_to_14':'Root visual inspection; corrected p8/p10/p11 also independently checked.',
             'not_external_human_peer_review':True},
        'numerical_experiments_rerun_for_pdf':False,'claude_code_plot_generation_succeeded':False,
        'actual_plot_generation':'Existing matplotlib/PyMuPDF after Claude Code returned not logged in.'}
args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='embedded_fonts'},ensure_ascii=False,indent=2))
