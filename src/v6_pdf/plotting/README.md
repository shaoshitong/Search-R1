# V5 scientific figure assets

The actual Claude Code task was attempted with the existing configuration. It exited 1 with `Not logged in · Please run /login`, generated no plotting source, and recorded zero API tokens. See `claude_plot_response.json` and `cc_execution_receipt.json`. No login, proxy or credential settings were changed. Codex then wrote and executed `plot_v5.py` directly.

Use the already installed runtime; no package installation is needed:

```sh
MPLCONFIGDIR="$PWD/.mplcache" /Applications/miniconda3/bin/python plot_v5.py --data /absolute/path/to/outputs/V5_AUDIT --out /absolute/path/to/figures
/Applications/miniconda3/bin/python qa_assets.py --figures /absolute/path/to/figures
```

Dependencies: matplotlib, numpy and PyMuPDF; QA additionally uses Pillow. Font: the installed `/System/Library/Fonts/Hiragino Sans GB.ttc`.

Each of the six named assets is saved as vector SVG, vector PDF and 300-dpi PNG. All Chinese labels are paths in SVG/PDF, avoiding external font dependencies. Direct Matplotlib Type42 embedding of this CFF font produced incorrect glyphs in a real PDF QA render. The implemented workaround converts the already outlined SVG to PDF with PyMuPDF. Explicit pastel fills avoid SVG reader differences in opacity and clipping.

`figure_manifest.json` records input JSON hashes, exact plotted values, captions and output hashes. `figure_acceptance.json` records vector structure, PNG density and decomposition consistency. Actual rendering stdout is in `render.log`; process exit is recorded in `plot_execution_receipt.json`. QA preview PNGs are intermediate source-directory artifacts and are not report figures.

The plots preserve the distinction between predictive loss, observational bootstrap intervals, post hoc concentration diagnostics and theoretical parameter examples. The log-score decomposition is an identity, with no invented confidence intervals.
