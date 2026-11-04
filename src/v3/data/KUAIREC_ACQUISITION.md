# KuaiRec daily exposure metadata

The working data is remote:
`hkust3:/data/shaoshitong/ai_creator_game_20261104/v3/kuairec/item_daily_features.csv`.
No viewer raw features, video media, or row-level metadata were copied to the Mac.

## Acquisition and integrity

The official Zenodo record is open:
https://zenodo.org/records/18164998 . The official 431,964,858-byte archive lists
MD5 `261550d472c48eff4990fb13c0e5bcf7`. Its slow full download failed with curl18
after 3,591,120 bytes; the partial and byte-growth/failure audit were retained.
No other process was stopped. The stop check found that download already exited.

The required single CSV was acquired from the public, non-gated third-party
repository `hiiamkik/miniOneRec-kuairec`, pinned to commit
`c644945dfa71b5d6f67ae9c7aff0eae40caf06b8`, through hf-mirror.com.
It has 85,855,213 bytes and SHA256:
`45943d63c44652b6403f3a4f78c7225e1afe7916bab17d9a674d7979245e085b`.
The whole file matches the replica repository's LFS hash. Separately, a successful
HTTP206 request of the official ZIP's final 65,536 bytes exposed its central
directory: the official `KuaiRec 2.0/data/item_daily_features.csv` has the same
85,855,213 bytes and CRC32 `12e08ed8`, which the downloaded CSV also matches.
This is not a completed whole-official-ZIP MD5 verification.

Zenodo declares CC-BY-4.0; the linked GitHub repository LICENSE declares
CC-BY-SA-4.0. Both declarations are retained in the provenance audit.

## Actual schema and coverage

The CSV has 343,341 unique video-date observations and 58 columns. It covers
10,728 videos from 8,369 author account IDs over 63 dates, 2020-07-05 through
2020-09-05. `show_cnt` and `show_user_num` are the documented daily exposure
count and recipient count. `author_id` and `upload_dt` are present, as are
plays, valid plays, likes, comments, shares, follows and related outcomes.

There are no duplicate video-date rows or changing author IDs within a video.
`show_cnt` is present in every observed row: 842 zero, 342,499 positive, and
no negative values. The unbalanced item-date grid has 332,523 unobserved pairs;
they must not be converted to zero exposures. Source `visible_status` is
described as current APP state, so it is only an observed-status inventory proxy.

1,071 accounts have at least two observed videos, and 1,049 have at least two
public-label observed videos on some common date. This is enough to construct
within-account multiple-item examples, but the set is not the complete inventory
of each creator. The data have no AI label, economic-person ownership map,
production-cost shock or known ranking exponent.

## Scripts

- `acquire_kuairec.py`: original official-archive attempt, preserving progress.
- `acquire_item_daily_replica.py`: successful single metadata replica acquisition
  with LFS SHA and official ZIP-member CRC/size verification.
- `summarize_item_daily.py`: schema, author/item/date coverage, zero/missing
  exposure and observed-inventory summaries. It fits no regressions.

Small audit copies are delivered in `outputs/V3_AUDIT/`; raw metadata remains
remote. The 2026 package release date is not the date of the 2020 observations.
