# Cumulative768 / 512 additional updates

Job743517 retained its full CP768 before scoring and continued. Independent own-ID/strict-EM/score/full-checkpoint-SHA acceptance passed; the same frozen HF2048 pool contains663 trueMixed questions (32.373%).

|Measure|CP768|same-HF E2|same-HF E4|
|---|---:|---:|---:|
|Mixed balanced accuracy|55.2308%|56.2505%|55.7714%|
|Mixed recall|72.6998%|74.8115%|67.5716%|
|NonMixed recall|37.7617%|37.6895%|43.9711%|
|AUC|57.0472%|—|57.6476%|
|Top512 trueMixed|195 (38.0859%)|181|196|

AllWrong falsePositive rate54.8815%; AllCorrect falsePositive rate85.7576%. SelectedK0..5=[189,39,37,52,67,128]. Both fairness frontiers are required; this checkpoint does not pass the joint improvement criterion.

Fixed9137/2000 paired question bootstrap (descriptive, no tuning): candidate−E2 BA−1.0197pp,95%CI[−2.8953,+0.9327], TopMixed+14 CI[−8,+35]; candidate−E4 BA−0.5406pp CI[−2.5384,+1.4265], TopMixed−1 CI[−26,+19]. Final4096 remains unopened.

Full checkpoint remote SHA005abf05aa1d18bddc796c8f784987e162f658bc54d3eed7bda8b70bd20febef (3745837338B). Independent milestone receipt f941b5b6474c5b44d0b49f6933bc83a296d49f4a20b5f3820f12deecdead7593. Paired result85ce67011598c22127c9ce16dc4c67ad1b23a50b0a4c0151476a7c9a89b002d1; paired independent receipt e279387838541c3bb7a198a6755a13e0aa42325b0db49060286ac722a292e21a. All artifacts remain under remote peer-expanded-append-only-train-v2 or peer-expanded-append-stability-1a0f5b4/cursor-000768-v2.
