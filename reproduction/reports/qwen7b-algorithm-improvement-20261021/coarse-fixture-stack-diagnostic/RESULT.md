# Focused immutable synthetic-test diagnosis

The original single fixture was run under a33second wall bound; no active
source, original test, model, cache or GPU training was changed. No matching
old worker or held code-qualifier lock existed before either bounded attempt.

The initial3GiB RLIMIT_AS attempt exited127 after8.49s with a dynamic-loader
thread-local allocation abort, before Python could emit its10s stack. This
virtual-address restriction is not a3GiB RSS cap and yielded no valid test
qualification or blocked-line evidence.

Using the existing743978 allocation with CPU2/gpu0 and --mem=3G, without the
virtual-address limit, the exact same immutable fixture ended within21.26s.
The first10s stack was shared-environment torch import bytecode I/O. The
fixture then completed its body in8.48s and failed in TemporaryDirectory exit:
`test_coarse_event_cached_train.py:22 -> shutil.rmtree -> Errno39 checkpoints`.
No state-update or restore assertion error was observed. The diagnostic
process returned0 but unittest success wasfalse; this is NOT a PASS receipt.

The owning group agent received the exact stack/log refs and is limiting its
fix to the synthetic fixture temp-directory/NFS mmap cleanup and its resource
harness. This diagnostic made no source fix and did not repeat more tests.
The real Count source805, cached features, checkpoints and scoring stay intact.
