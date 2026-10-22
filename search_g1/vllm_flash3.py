"""Process-local FA3 prefill and paged decode for the pinned vLLM 0.6.3.

No installed package is edited. vLLM retains its scheduler, RoPE, KV writes,
sampling and log-probability implementation. Only its two attention functions
are replaced in the explicitly selected FLASH_ATTN backend.
"""
import importlib.metadata
import time


class Flash3Functions:
    def __init__(self, backend):
        self.backend = backend
        self.calls = dict(prefill=0, paged_prefill=0, decode=0)

    @staticmethod
    def options(alibi_slopes, dropout_p, window_size, softcap):
        if alibi_slopes is not None or dropout_p != 0:
            raise ValueError('FA3 rollout supports the current zero-dropout/no-ALiBi Qwen recipe')
        return dict(window_size=tuple(window_size or (-1, -1)), softcap=softcap or 0.)

    @staticmethod
    def output(result):
        return result[0] if isinstance(result, tuple) else result

    def varlen(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
               max_seqlen_q, max_seqlen_k, softmax_scale=None, causal=False,
               window_size=(-1, -1), alibi_slopes=None, softcap=0.,
               block_table=None, dropout_p=0.):
        options = self.options(alibi_slopes, dropout_p, window_size, softcap)
        if block_table is None:
            result = self.backend.module.flash_attn_varlen_func(
                q, k, v, cu_seqlens_q, cu_seqlens_k, None, None,
                max_seqlen_q, max_seqlen_k, softmax_scale=softmax_scale,
                causal=causal, **options)
            self.calls['prefill'] += 1
        else:
            # vLLM has already written the new KV rows into its original cache.
            # FA3's packed-query paged API uses sequence lengths, not cu-K.
            result = self.backend.module.flash_attn_with_kvcache(
                q, k, v, cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                cache_seqlens=cu_seqlens_k[1:] - cu_seqlens_k[:-1],
                page_table=block_table, softmax_scale=softmax_scale,
                causal=causal, **options)
            self.calls['paged_prefill'] += 1
        return self.output(result)

    def kvcache(self, q, k_cache, v_cache, block_table=None,
                cache_seqlens=None, softmax_scale=None, causal=False,
                alibi_slopes=None, softcap=0., **kwargs):
        options = self.options(alibi_slopes, 0., kwargs.pop('window_size', (-1, -1)), softcap)
        if kwargs:
            raise ValueError('Unqualified vLLM decode arguments: ' + ','.join(sorted(kwargs)))
        result = self.backend.module.flash_attn_with_kvcache(
            q, k_cache, v_cache, page_table=block_table,
            cache_seqlens=cache_seqlens, softmax_scale=softmax_scale,
            causal=causal, **options)
        self.calls['decode'] += 1
        return self.output(result)

    def report(self):
        return dict(calls=dict(self.calls), provenance=self.backend.provenance,
                    implementation='vllm063_process_local_flash3',
                    checked_unix=time.time(), installed_environment_modified=False)


def install_vllm_flash3(run, *, backend=None):
    """Call in each rollout worker before constructing its TP1 engine."""
    if importlib.metadata.version('vllm') != '0.6.3':
        raise ValueError('This adapter is scoped to the inspected vLLM 0.6.3 API')
    import vllm.attention.backends.flash_attn as target
    installed = getattr(target, '_search_g1_flash3_functions', None)
    if installed is not None:
        return installed
    if backend is None:
        from search_g1.flash3_attention import load_backend
        backend = load_backend(run)
    functions = Flash3Functions(backend)
    target.flash_attn_varlen_func = functions.varlen
    target.flash_attn_with_kvcache = functions.kvcache
    target._search_g1_flash3_functions = functions
    return functions
