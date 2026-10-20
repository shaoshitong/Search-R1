# Frozen 8192 real-G10 causal prefix horizon diagnostic

The full read-only reconstruction completed: 8,192 fixed NQ training questions, 81,920 authentic original5+newseed5 trajectories, 244 accepted aggregate units, 377.107 seconds, peak RSS 389,226,496 bytes (371.2 MiB). The process has exited; its terminal log reports successful complete, no failure receipt exists, and the independent CPU auditor exited 0. No GPU/model/probe/optimizer/actor updates, extra rollout, validation or final-pool reads occurred.

For all 81,920 trajectories, reconstructed Prefix20 policy token IDs, positions and every sealed causal context segment match the existing prefix exactly. Prefix20 duplicate/conflict counts also exactly reproduce the separately accepted earlier P20 study. Every source batch input/prefix/full file was SHA-checked before journal reconstruction; all source file identity/size/timestamp stamps remained unchanged at completion. The independent audit freshly hashes every aggregate unit and metadata source; it does not claim a second bulk rehash of all raw full journals.

The denominator for opposite-EM pairs is the 57,765 within-question correct/wrong pairs in authentic G10; full duplicate pairs use all 368,640 within-question pairs. “Distinguishable” means the declared causal token/context/position inputs differ, not that a probe can learn to distinguish correctness.

| Actual policy prefix | Identical causal pair count / all pairs | Opposite-EM identical causal pairs | Fraction of correct/wrong pairs | Input-distinguishable correct/wrong | Mixed questions containing a conflict /3489 | Naturally shorter trajectories | With past retrieval context |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 114665 /368640 (31.1049%) | 16350 | 28.30434% | 71.69566% | 2926 (83.8636%) | 0 /81920 (0.0000%) | 0 /81920 (0.0000%) |
| 20 | 3665 /368640 (0.9942%) | 505 | 0.87423% | 99.12577% | 307 (8.7991%) | 1 /81920 (0.0012%) | 84 /81920 (0.1025%) |
| 40 | 262 /368640 (0.0711%) | 27 | 0.04674% | 99.95326% | 22 (0.6306%) | 1626 /81920 (1.9849%) | 14677 /81920 (17.9163%) |
| 80 | 64 /368640 (0.0174%) | 0 | 0.00000% | 100.00000% | 0 (0.0000%) | 38641 /81920 (47.1692%) | 70804 /81920 (86.4307%) |
| 160 | 64 /368640 (0.0174%) | 0 | 0.00000% | 100.00000% | 0 (0.0000%) | 72308 /81920 (88.2666%) | 72736 /81920 (88.7891%) |

Prefix5 has severe collisions in this cohort. Prefix20 removes most: only 0.87423% of correct/wrong pairs still have exactly identical inputs. Thus the current Prefix20 low within-question AUC cannot mainly be attributed to literally identical causal inputs. Prefix40 removes another 478 conflicting pairs; Prefix80 removes the remaining 27. Prefix160 provides no additional collision reduction here. This supports a bounded longer-prefix experiment as an information-availability test if later authorized; it does not establish better fit, Mixed BA, AUC or selection utility.

The observation budget also changes. Prefix20 includes actual past retrieval context for 0.10254% of trajectories; Prefix40 for 17.9163%; Prefix80 for 86.4307%; Prefix160 for 88.7891%. Policy-token-only signatures and full causal signatures produce the same within-question partitions in this cohort, but context alignment was still checked rather than assumed. Prefix80 is naturally short for 47.1692% and Prefix160 for 88.2666%; no padding was represented as generated evidence. A later model may pad these missing slots under an explicit mask, but this study trained no long-prefix model.

Longer contexts are reconstructed only as actual_request_token_ids + raw_generated_token_ids[:selected_retained_count] for the selected turn. Native chunk requests, retained cuts, prior retained policy and prior environment observations reconstruct the full journal; discarded raw tails and observations after the selected cutoff never enter a retained segment. Prior-turn retrieval observations are legitimate causal inputs, and their increased coverage is an intervention in available information and inference cost rather than a free token-length change. “Multi-turn” here means selected policy tokens span more than one actual behavior turn; past retrieval coverage is separately counted.

The exact original8192 identities and own strict EM remain fixed. Original-G5 and newseed-G5 summaries are retained separately in the independent audit; they are not presented as unseen performance. No G5 multiset Mixed-conflict enumeration or new GPU job was started, and ongoing Prefix20 append-only training is unchanged.

Producer complete: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer8192-long-prefix-causal-diagnostic-v1/complete.json`, 93778 bytes, SHA256 `542f3928efdce42358ced2d3ae645fe2fd4364502d6711102d328609346ad0af`.

Independent audit: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer8192-long-prefix-causal-diagnostic-v1/independent-audit.json`, 9209 bytes, SHA256 `ee47e18b59c645340121f9b9fad345b4de33e3f0fca77b8f6d2578ea8b75c9dd`.

Config: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer8192-long-prefix-causal-diagnostic-v1/config.json`, 124446 bytes, SHA256 `0870307e3286a8926c5e5842c7076c2bebce4281544be87625cd7aad1685ee94`.

Producer source `992da5465e0b6a728b2e400fe4958798a5c81322`; independent auditor source `9d029bda6e49bcde6da98f8060093bc7470b75a7`; 8 CPU tests passed. Shared main HEAD/index and active training runtime were not modified.
