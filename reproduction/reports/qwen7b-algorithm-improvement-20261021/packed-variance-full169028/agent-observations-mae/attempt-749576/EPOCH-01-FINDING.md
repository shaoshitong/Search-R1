MAE 749576 completed epoch 1 and entered epoch 2 on both ranks. Its first epoch visited all 169,028 questions once in 1,321 updates; the source records 1,036.89 seconds for the epoch. The live source and loss remain unchanged.

On the fixed independent 2,048-question G5 diagnostic, raw MAE predictions are nearly zero (mean 0.00018017, standard deviation 0.00005855), and Top512 contains 138 Mixed questions (26.95%) against 663/2048 (32.37%) in the pool. EMA Top512 contains 187 (36.52%). At the predeclared variance threshold 0.08, both raw and EMA have 50% balanced accuracy and zero Mixed recall. These are variance scores, not Mixed probabilities.

The raw variance MAE is 0.0623337; the diagnostic zero predictor has MAE 0.0622656. Thus the objective value is close to the trivial zero baseline and does not establish useful selection. Scores are not exactly identical or exactly zero, but their range has collapsed near zero. No threshold fitting or intervention was performed.

The same epoch, same fixed question/trajectory/label identities give MSE raw Top512 219/512 (42.77%) and z-plus-MSE-anchor raw 218/512 (42.58%). These comparisons are early diagnostics, not final candidate selection. Full epoch 10 and terminal checkpoint CPU acceptance are still pending. Final4096 was not accessed.
