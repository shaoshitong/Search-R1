# Single leading full169028 L15/P80 MSE ownG5 configuration

Independent full-data runtime copied from accepted `92ec87b0c1a02a068517c8ed84aa4b8f5727599c`; the existing pilot and active seeded jobs are unchanged. Two independent fresh-official replications use init/plan seeds9137/2026102227 and19137/2026102302. Both use the same authentic ownG10 dataset, sample five trajectories per question per epoch, and train the copied official block15 plus the packed2048-position head with population-variance MSE, LR5e-6, Adam and EMA.9995. No pilot checkpoint is resumed.

Every epoch visits all169028 questions once in an independently shuffled seed/epoch order. Each rank handles84514 questions:1320 batches of64 and a final34. The final global68-question update uses five micros8/8/8/8/2 per rank and exact local-mean losses before DDP averaging. There are1321 Adam updates per epoch,13210 across10 epochs. Full raw/EMA/Adam/both-rank RNG/sampling-plan and explicit complete epoch cursor states are saved every epoch.

The original fixed trainB2048 and independent viewed2048 ownG5 are unchanged. The primary metric is viewed Top512 true Mixed. Epoch and raw/EMA state are chosen using seed1 development results only, then the identical choice is applied to seed2. Seed2 is not used to choose the candidate. Final4096 remains blind.

The independent fixture is `A/full169028-policy80-ownG10-20261023-v1/fixture`. Source staging does not make a training-ready claim. The `qualify` action requires the complete fixture and its SHA-bound independent accepted.json; the runtime enforces the same gate. No training is submitted by this deployer. Both entries use normal/acd_u,2H100,12CPU,128GiB and12h. Root owns submission after fresh input acceptance.

The full fixture must additionally have `fresh_full_P20_prefix_prompt_and_own_EM_exact=true` and a verified `P20_alignment` report, comparing every1690280 own training trajectory against the previously accepted fullP20 bank. Producer-only acceptance cannot start training. The final immutable two-seed entry roots use suffix-v2 and fixture-base ends in `/fixture`; earlier v1 source-only entries were never submitted.
