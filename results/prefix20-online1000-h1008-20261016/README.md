# Prefix20 online1000: first fresh epoch2 evidence

This records job3408 on h100-8, run8a6b24dc19664cb3a3fb97101f21c54e. The executed runtime remains frozen at ca1c60f49a29c30ca299c1d6073ee9a4bb86655b; this publication adds only audit code and small facts.

Epoch2 actually completed 5120 distinct questions × G5 =25600 fresh complete trajectories. Head training uses global batch256, eight ranks × local32, no gradient accumulation, one natural without-replacement pass:100 synchronized Adam updates per rank, from100 to200. The fixed diagnostic has512 questions × G5 =2560 trajectories; epoch3 was observed starting. This does not complete the1000-epoch goal or the five-fresh-epoch initial acceptance.

Console phase wall times include process startup, polling, shutdown and publication: rollout401.3454s, pack6.6147s, encode28.5221s, training phase28.5851s. The training metric separately reports the update loop4.8394s and diagnostic loop0.1335s; these are not the complete phase wall time.

Question/selection and other bounded JSON/ledger files received independent complete SHA readback in the audit. Compact/full rollout payload paths, byte sizes and hashes come from their producer receipts; those payloads were not independently rehashed here. The mutable latest.pt model hash is likewise a producer receipt value, not an independent model readback. No model/tensor/raw rollout payload was downloaded to Mac and no GPU work, remote file change or training change was performed.

The archived audit script was consumed once and is for reproducibility, not a request to replay it blindly. Large artifacts and reusable questions/compact prefixes remain at the remote paths recorded in epoch2-actual-facts.json.
