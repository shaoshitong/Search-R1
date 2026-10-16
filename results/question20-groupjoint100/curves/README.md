# Search-G1 joint probe: training and fixed-test curves

Snapshot: 2026-10-16T10:08:33.543090+00:00; completed epochs 1–65 of 100. Native job 3412 on h100-8 continued healthy. Only aggregate metrics were copied locally.

- Train: each epoch uses a different source batch of 5,120 questions × G5. Same post-update head evaluates all 25,600 trajectories and 5,120 groups. This is not a fixed-training-set curve.
- Test: the same 512 questions × G5 every epoch; 183 mixed and 329 nonmixed groups. Threshold 0.5. Balanced accuracy constant-class baseline is 50%; ordinary test accuracy majority baseline is 329/512 = 64.2578125%.
- Group truth: 0 < own strict-EM correct count < 5. Group score: 1-product(p)-product(1-p).
- Each loss is an independently averaged BCE; total = trajectory BCE + group BCE, weight 1.
- Faint lines retain all observations. Thick curves are full trailing 10-epoch means, starting at epoch 10. No incomplete epochs or extrapolation.
- Best test balanced accuracy: epoch 45, 62.9619479463%. Last epoch 65: train 70.4549943440%, test 59.0579168535%. Last ten completed test epochs (56–65): mean balanced accuracy 56.20733%; the recent curve is not monotonic improvement.

Files: `accuracy-curves.png/pdf`, `loss-recall-curves.png/pdf`, interactive `training-test-curves.canvas.tsx`, reproducible `plot_curves.py`, aggregate `curve-metrics.json`.

Source evidence SHA256: `146fafa51df2bad78fb81fc7b64404a1ac49bc2cffe1b4cd583f5116efbfc575`. Each row retains its original remote metrics SHA. No checkpoint, trajectory, feature tensor, or full prediction file is included.
