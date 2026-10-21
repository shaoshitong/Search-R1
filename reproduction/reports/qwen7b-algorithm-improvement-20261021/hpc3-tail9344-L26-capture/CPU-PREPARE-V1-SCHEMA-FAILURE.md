# Preserved first preparation failure

Immutable0d872886/v1 CPU projection failed before selecting any row or launching a GPU. The accepted physical prefix descriptor contains `{path,bytes,sha256,count}`; the first helper incorrectly compared it to an exact three-field dictionary. Actual fresh path/bytes/SHA all matched, including the first seal and prefix file. The v1 log and partials remain untouched.

The v2 helper verifies all three path/bytes/SHA fields strictly and preserves the additional count metadata. One narrow code-only test accepts extra count metadata while rejecting a wrong SHA. This changes metadata adaptation only; source raw0898b545, whole-execution source resolution, official model and replay mathematics are unchanged. No tensor/model update or new job occurred in v1.
