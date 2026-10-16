# Sigma3 native4 archive150 recovery

This change repairs the actual pre-copy archive150 failure from source
`5f0065b98d1239c994addc19fd0b7e32f506d7fd`. The original CPU archive child
returned held exit1 at 1792116692.8873627; its follower returned1 at
1792116701.0708904. No archive150 payload was created. The original complete
DU failed with permission errors; its numerical stdout is never used as
storage usage. The immutable failure fixtures contain all24 small original
files, plus the five actual held files of the successful original archive100.

The four-GPU producer721485, its sigma3 objective, original RAW supervisor,
checkpoint/source/model/cache/retrieval assets and other experiments remain
unchanged. In particular, private behavior-trace directories require mode700
on every write; changing them to750 is not a valid storage repair.

## Complete storage coverage

The new archive child holds the existing archive-storage lock. It requests a
fresh JXU GNU du of the entire experiments root, excluding only exact absolute
MOU roots identified by real permission errors. A separate MOU supervisor
holds its own actual owner child and that child's real GNU du to completion.
The exclusion set must exactly equal the MOU scan set, with disjoint roots,
identical root inode/device/UID/GID/mode before and after, complete stdout
root sets, empty stderr and actual child wait0 at both levels. No raw trace
contents, file modes or directory ownership are modified.

The sum is a conservative allocated upper bound. Cross-partition hard links
may be counted twice. It is neither an atomic filesystem snapshot nor a hard
quota measurement. Copy admission still requires:

```
complete allocated upper bound + 2 * all dynamic source-record bytes + 1 GiB <= 2e12 bytes
```

Admission refuses results older than600 seconds or a transaction older than
900 seconds. Both original children must be terminal before a stale complete
transaction can be retained and a new numbered full transaction started.
Failed, partial, unrecognized, omitted-root or permission-error scans never
contribute a usable number. Historical final consumption checks the retained
measurement's before/after and held-byte evidence; it does not require an
unrelated temporary directory to remain present forever. Historical receipts
cannot be reused for new copy admission.

## Once handoff and remaining acceptance

`recover_sigma3_archive150_first` authenticates all original24 failure files,
the unchanged native4 producer/sourcepair150, dead original follower/archive
runner/child, reaped failed DU observer, healthy original RAW and zero150
payload. It preserves only the failed canonical150 control by an exclusive
intent and one fsynced atomic rename. The stopped old follower remains in
place. Existing intents or unknown outcomes are inspect-only, never replayed.

The successful original100 remains bound to source5f and controller560250;
it is not attributed to the new follower. Before inheriting5/50/100, the new
detached follower freshly reads every source/archive/target byte and exact
dynamic file set for all three milestones. Only new150/200 archives use the
two-owner protocol. The same five milestone checks, native4 restore-only
final audit, independent RAW consumption, exact native51713/seven-source
EM/F1 and no-update201 requirements remain mandatory.

The independent final consumer needs a separate explicit archive150 recovery
source registration, retaining the original archive100 source pin. Publishing
this runtime alone does not prove final acceptance or authorize claiming the
goal complete.

## Validation

Local tests consume the original failure bytes and held100 files, attack
same-size byte/inode changes and wrong inheritance, verify one-time atomic
preservation, reject stale/missing/failed held storage proofs, and preserve
private700. Linux tests additionally execute real GNU du on directory-block,
sparse-file, symlink and cross-partition-hardlink fixtures and verify real
child reaping even after binding publication failure. These Linux tests must
run without skips on the fully published immutable remote source before use.
