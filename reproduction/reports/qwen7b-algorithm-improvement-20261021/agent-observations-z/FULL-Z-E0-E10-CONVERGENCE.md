# Full Z, all10epochs complete

Completed749577 with0:0;13210 actual Adam updates. Final CP fresh SHA, raw/EMA strict CPU reload, Adam clocks/moments, rank RNG, all10 own-M plans and global normalization source were accepted. This is CPU restoration, not a GPU next-update replay.

Loss is normalized MSE on predicted4v plus raw4v MSE; v is selected-own-M strict-EM population variance. The table reports original-variance MSE on two fixedG5 pools, not the training normalization loss. EMA.9995 is updated from raw Adam parameters. All selection ranks predicted variance; diagnostic BA uses the fixed.08 threshold.

|Epoch|State|Fixed train BA|Train variance MSE|Train Top512|Viewed dev BA|Dev variance MSE|Dev Spearman|Dev Top512|
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|0|raw|50.00%|0.0107068|231|50.00%|0.0119847|0.04119|175|
|0|EMA|50.00%|0.0107068|231|50.00%|0.0119847|0.04119|175|
|1|raw|56.99%|0.0087559|313|53.78%|0.0082889|0.19106|218|
|1|EMA|57.38%|0.0087949|291|57.27%|0.0088321|0.15684|215|
|2|raw|60.53%|0.0084065|333|55.58%|0.0082382|0.20308|224|
|2|EMA|58.57%|0.0086343|320|54.94%|0.0083465|0.19149|223|
|3|raw|61.70%|0.0082276|340|54.91%|0.0082042|0.21665|217|
|3|EMA|60.04%|0.0084898|335|55.18%|0.0082266|0.21006|225|
|4|raw|67.07%|0.0075564|360|59.67%|0.0082786|0.23346|223|
|4|EMA|61.15%|0.0083270|349|55.01%|0.0081721|0.22296|220|
|5|raw|64.53%|0.0077955|399|54.44%|0.0083925|0.22668|224|
|5|EMA|63.40%|0.0080088|365|55.56%|0.0081574|0.23362|221|
|6|raw|68.17%|0.0070802|426|54.67%|0.0087535|0.22620|225|
|6|EMA|67.35%|0.0074666|400|56.58%|0.0082125|0.23963|229|
|7|raw|75.68%|0.0057944|467|57.60%|0.0089637|0.22403|234|
|7|EMA|71.65%|0.0067040|433|57.20%|0.0083848|0.23878|230|
|8|raw|79.92%|0.0048084|478|59.32%|0.0096351|0.22778|227|
|8|EMA|76.48%|0.0058328|465|57.53%|0.0086686|0.23564|235|
|9|raw|81.14%|0.0043105|497|59.06%|0.0097861|0.20947|237|
|9|EMA|79.97%|0.0050153|488|58.72%|0.0089973|0.23147|237|
|10|raw|82.17%|0.0041685|500|59.30%|0.0095700|0.21620|245|
|10|EMA|81.74%|0.0043959|500|58.55%|0.0092455|0.22880|235|

AtE10, both states select500/512 trueMixed in the fixed training probe, versus245raw and235EMA on the already viewed independent-question dev pool. Dev Top selection is above its663/2048 population proportion, but the large training gap and nonmonotonic dev loss remain. The best single observed dev Top is245/512raw atE10; that is a post-hoc viewed-dev result, not a frozen final-confirmation outcome. No final4096 selection/scoring has occurred.

MSE andMAE arm completion is outside this Z-only receipt; the common verifier skipped their active checkpoints in this pass.
