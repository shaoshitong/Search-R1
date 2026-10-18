# Retained corpus cache metadata relocation

The first B continuation exited before new Adam writes because the retained archive has a different mtime from the original archive recorded in corpus.json. The existing independent archive and extracted-corpus content SHA evidence remains valid.

A separate versioned cache view now preserves all original and retained payloads. Large immutable cache files are hardlinked to their previously fully verified retained copies; small metadata files are copied privately. Only the private corpus receipt changes its archive timestamp to the actual retained archive identity. The original corpus.json and archive timestamp are unchanged. No large content hash or extraction is repeated.

The new overlay is restricted to a fresh private mount namespace. The unchanged frozen retriever prepare_corpus entry and native datasets cached loader actually passed on CPU, reusing28 Arrow shards and21,015,324 documents in7.82 seconds. This proves the actual corpus/Arrow path, not GPU retrieval or model forward. Future normal restore must follow the latest actual complete joint, preserve A/B science and B's current prefix width, and verify normal next-iteration progress. No active trainer was stopped, modified or switched to this overlay.

Explicit readonly control layout `retained-cache-v2` is now available. Actual training continues on the original source8 layout. The eventual source8-independent full training handoff is still pending; neither this CPU pass nor the existing data copies marks the overall200-outer/QA/final goal complete.
