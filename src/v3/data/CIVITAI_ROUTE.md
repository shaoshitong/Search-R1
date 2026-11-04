# Civitai public metadata route and source discovery

## Result in this research round

Civitai robots and anonymous SFW API probes did not return HTTP responses:
HKUST3 timed out connecting; the Mac encountered a TLS connection failure.
No exit changes, alternate-domain circumvention, login, cookies, CAPTCHA
handling or additional bulk requests were used. No first/second live snapshot
was obtained, and no watcher was created. The status archive distinguishes
network failures from an HTTP refusal and distinguishes archival UTC from a
missing exact request timestamp.

`probe_public_civitai.py` preserves a future single robots probe with exact UTC.
It has not been rerun against the failed target. It is not a bulk crawler.

## Official public interfaces

Docs repository commit: `7ae23923c333688ad70a7a4b3ee49afd62e60f82`.

- [Images](https://github.com/civitai/civitai-developer-docs/blob/7ae23923c333688ad70a7a4b3ee49afd62e60f82/site/reference/images.md):
  anonymous `/api/v1/images`, SFW browsing level, image/post IDs, creation time,
  prompt metadata, interaction counts and cursor. Do not assume impressions
  exist merely because reactions are exposed.
- [Posts](https://github.com/civitai/civitai-developer-docs/blob/7ae23923c333688ad70a7a4b3ee49afd62e60f82/site/reference/posts.md):
  public `/api/v1/posts/{id}` returns `user.id` and username. It is rate-limited;
  respect `Retry-After`. Public responses have 300-second edge caching.
- [Users](https://github.com/civitai/civitai-developer-docs/blob/7ae23923c333688ad70a7a4b3ee49afd62e60f82/site/reference/users.md):
  public `/api/v1/users?query=` or `?ids=` resolves stable platform account IDs.
  `/me` is authenticated and is outside this collection route.
- [Official CLI](https://github.com/civitai/cli), Apache-2.0: public read commands
  support `--json --anon`. The latter is essential because a logged-in CLI
  otherwise automatically sends a stored token. No installation or execution
  of this CLI was necessary for the probes.
- [CivitScraper](https://github.com/moonwhaler/CivitScraper), AGPL-3.0: can disable
  optional image downloads, but mainly scans local model files by hash. It is
  less suitable than the direct public REST/official CLI route for image panels.

A successful future panel should retain fixed image/post IDs, stable `user.id`,
source request UTC, cache headers and outcome fields. A short unchanged interval
can reflect caching or low activity, not collection stagnation.

## MM2024 and newer paper discovery

[Wei et al., MM2024](https://tysong.github.io/files/MM24-Civitai.pdf), section 3.1,
describes REST API plus Selenium and records views and tips. Neither the paper,
its author's linked publication page, nor the inspected public author GitHub
repositories exposed a downloadable dataset or dedicated collector repository.
This search result is not a claim that no such source can exist.
Views also should not be silently equated with recommendation impressions.

The same author's 2026 paper
[Navigating the Open-Source Model Ecosystem](https://arxiv.org/abs/2607.10538)
links [Pixiv-AI-generation-metadata](https://huggingface.co/datasets/Wei-Yiluo/Pixiv-AI-generation-metadata).
The current API reports `gated=auto`; the page requires accepting sharing of
Hugging Face information. Files were excluded, including via mirrors. The public
card describes 2023-01-01 to 2026-01-01 metadata, but this is not an acquired source.

## Existing stable IDs recovered

The already acquired public Shiba 2026 snapshot has usable root `userId` in
72,949 of 73,020 models' raw JSON, even though its simplified SQL columns show
only creator usernames.
`audit_existing_user_ids.py` extracts this account-level field and joins it to
model versions without changing V1/V2. It still supplies no economic owner map,
production cost, observed ranking exponent, or recommendation impressions.
