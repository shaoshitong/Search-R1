# Search-G1 joint probe: final training and fixed-test curves

Final aggregate snapshot: 2026-10-16T10:51:20.003593+00:00; all completed epochs1-100 of100 from native job3412 onh100-8. These figures use only the small complete-epoch metric summaries. Native exit, checkpoint integrity and resource release are accepted separately; plotting does not perform or assert those checks.

- Train: each epoch evaluates its own5,120-question G5 source batch with the same post-update head. Sources1-70 are followed by sources1-30 again. Batches therefore change across epochs, but the final30 epochs deliberately replay earlier batches; this is not a fixed-training-set curve or100 independent question batches.
- Test: the same512 questions xG5 every epoch;183 mixed and329 nonmixed groups. Threshold0.5 is unchanged. Balanced accuracy constant-class reference is50%; ordinary test accuracy majority reference is329/512=64.2578125%. The fixed-test majority reference must not be interpreted as the changing train-batch baseline.
- Group truth:0<own strict-EM correct count<5. Group score:1-product(p)-product(1-p).
- Each loss is an independently averaged natural-log BCE; total=trajectory BCE+group BCE, weight1. All shown train/test losses and accuracies use the same complete post-update head, rather than online pre-update training predictions.
- Thin faint lines retain all100 observations. Thick curves are complete trailing10-epoch means starting at epoch10:91 valid means, no incomplete windows, no missing-epoch interpolation or extrapolation.
- Highest observed fixed-test balanced accuracy remains epoch45: 62.9619479463%. This is a metric maximum and does not assert that an epoch45 checkpoint was retained.
- Final epoch100: train ordinary accuracy74.06250000%, balanced73.9142497400%; fixed-test ordinary accuracy61.52343750%, balanced59.8759280482%. The final ten-epoch fixed-test balanced mean(E91-100) is58.2498712774%; the test curve is not a monotonic improvement.

Files: `accuracy-curves.png/pdf`, `loss-recall-curves.png/pdf`, interactive `training-test-curves.canvas.tsx`, reproducible `plot_curves.py`, aggregate `curve-metrics.json`.

QA: both final PNGs were opened with `view_image`; labels, axes, legends, final E100 endpoints and footnotes fit without clipping or overlap. Canvas TypeScript5.8.3 transpilation has zero syntax diagnostics; its10-epoch smoothing was checked for the first nine null values and91 valid final means. The native Codex Canvas preview was not visually inspected, and no such claim is made. Scientific PNG/PDF export uses Matplotlib; no remote queries, training, predictions or model files are read by the plotting code.

Source evidence SHA256: `b3bfa5abedae7730fe562a8fc2aad316de94e1003d6711d5b5d344ee66483359`. Each row retains its original remote metrics SHA. No checkpoint, trajectory, feature tensor or full prediction file is included. Only the eight existing files in this directory were updated for this final snapshot.
