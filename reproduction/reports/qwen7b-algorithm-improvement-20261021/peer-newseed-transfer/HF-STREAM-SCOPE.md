# Peer new-seed raw data transport

Human authorization: use the new data generated in thread01a157ec-c811-7c51-b270-db2042410464 in this experiment. The old lite migration was the original HPC3 dataset sent to H100; it must not be transferred back as if new. New seed20261021/run70188b8e50fd4b45bc420ce58014f46e produces independent G5 responses for the same official7B actor. Source tensors are not cached; raw payload transfer alone does not qualify hidden features.

`peer_newseed_hf_stream.py` runs on mgmt-host. It reads only a complete immutable source snapshot from our NAS10 and streams archive bytes in at most8MiB reads to the existing authorized private dataset repository. No model/rollout payload is written on Mac or mgmt-host. Credentials enter encrypted stdin and remain in process RAM; archive size/SHA are checked through official HF LFS metadata. New snapshot-qualified paths preserve the old repository contents. Upload completion explicitly does not mean HPC3 acceptance.

`deploy_peer_newseed_hf.py` stages only published source in4096-byte chunks, validates all command sizes below16KB and protects launches with lock/intent/actual birth metadata. It uses the current successful existing management SSH route with strict host keys and no Mac proxy changes. Downstream remote downloads, two campus hops, safe tar extraction and fresh all-file target hashes are still required.

Four CPU checks cover real bounded seek/reopen IO, archive/member escape and hidden-array rejection, strict known-host behavior, source totals and actual control source compilation/size/secret-stdin contract. No uploaded data or destination acceptance is claimed by these tests.
