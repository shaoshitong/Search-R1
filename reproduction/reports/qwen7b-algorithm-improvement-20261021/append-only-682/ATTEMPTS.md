# Preparation attempts and actual launch

Attemptv1 runtime217f99e7 was staged without GPU submission. Its remote focused suite caught one NFS-specific toy-test cleanup failure: an open torch mmap prevented removing the temporary checkpoints directory (Errno39 Directory not empty). All other focused tests passed; no production checkpoint or feature writer was changed. The failed source and CPU log remain in the independentv1 directory.

Fixb8c3f8a7 explicitly releases the test mmap before temporary-directory cleanup and gates the first additional optimizer transaction on allocation-local independent CP257 CPU acceptance. Its new isolatedv2 runtime passed all8 focused tests and real metadata preparation (sourceCP256/Adam10720, exact old256-prefix+new682-tail,684 additional L26 manifests, frozenHF eval isolation). No production tensor was loaded during prepare.

ParentGO authorized the unchanged contract. Job743517 was submitted once with held identity acceptance, then released, normal/acd_u1H1008CPU64G2h/no dependencies, sourceb8c3f8a76fa195efc7c2e95cfa1a7d36669576ec. It allocatedACD1-22. Running and reader verification alone are not accepted training progress. The independent first realCP257, continuous5new updates plus the next, and complete938/fixed evaluations remain required.

Runtime root: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-expanded-append-only-train-v2/code-b8c3f8a76fa1`. Config15081B SHA`d56668bdd2e9839abe042c0d4f720225cb0cafd62f22924867dbfda883b17a80`; sbatch1219B SHA`3f6a48e1c2ce1394b7a99119d1345b20b879f9441ebef24c84592103f7ee33f4`.
