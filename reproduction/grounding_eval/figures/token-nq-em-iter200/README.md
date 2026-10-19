# NQ EM vs normal inference tokens, iter200

Seven paired measured results on the same NQ 3,610 questions.

- X: mean cumulative actual input + raw generated tokens per question; repeated input history and discarded native generation tails included.
- Y: normal inference NQ exact match (%) from the same evaluation as token accounting.
- Native implementations have different action-stop behavior. This is not a policy-only efficiency comparison. See data-audit.json for provenance and limitations.
- No CF, pilot, probing, or training token costs. No interpolated values or invented error bars.

Style follows the latest user-edited time-em-iter50-100 PDF. Original style files are preserved.

Build in the original artifact folder with `python3 build_figure.py`, then export the drawio file via draw.io desktop CLI using `--export --format pdf --crop`, then run `python3 finalize_figure.py`. Finalization requires PyMuPDF and checks all seven point coordinates, PDF text, and vector content. It also creates the PNG and verification record.

The repository copy is a publication/backup snapshot of code and audit data; the original artifact folder and local source-summary paths are recorded in the files. Full inference outputs remain remote.

Visual revision: removed the iteration heading and letter-code legend; enlarged direct labels and axes. The light-blue lower-right rectangle is bounded by the exact DynRPO coordinates and means higher measured token cost and lower EM. It does not encode latency. All seven numerical results remain unchanged.

Sketch revision: deeper blue shaded region; seven solid colored marker designs (circle, right triangle, diamond, square, hexagon, downward triangle, star); rough double-stroke axes and leader lines; TOP-D leader explicitly attaches next to the visible label; red bold `More tokens, lower EM!` notice. Numerical measurements unchanged.

Pale-red revision: changed only the dominance-area fill to #F4DCDD at 82% opacity and its sketch boundary to #C99191. Sketch outlines, direct labels, solid multicolor markers, red notice, coordinates and measurements preserved.
