# Controller-only identity reference adapter

The full-bank producer binds the qualification-v3 fixed6656 identity document; the official source/eval contract binds the main-run document. These are separately SHA-bound metadata files with equivalent qid/question_hash identities. Preparation now freshly checks both descriptors, requires the exact role/count and compares all6656 qid/hash pairs, rather than requiring equal path strings. No identity, label or evaluation pool is changed.

The deployed21-file GPU runtime remains code-dd7ecdf7060f and source-publication6458B SHAcdae3137aec68c9521afe2bb11e60fb67fc5922b7db00bb076b0e2f787bfce80. The new controller pins that publication commit explicitly, records its own distinct current commit/hash in config/submission receipts, and never renames or overwrites the immutable source root/publication. Stage is idempotent and re-verifies the existing runtime; it does not republish it with the controller's identity.

The first local prepare attempt after staging hit a transient GitHub SSH closure before any remote prepare action; a bounded retry returned WAITING_FOR_ACTUAL_FULL169028_SOURCE_RESOLVED_BANK with config_written=false and GPU_submitted=false. No actual model/update/GPU submission has occurred.
