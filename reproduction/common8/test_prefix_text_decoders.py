"""CPU tests for decoder selection, without importing training dependencies."""
import ast
from pathlib import Path
import unittest


def helper():
    path = Path(__file__).with_name('generation_behavior.py')
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'prefix_text_decoders')
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['prefix_text_decoders']


class PrefixDecoderTests(unittest.TestCase):
    def test_qualified_qwen_uses_same_native_decoder_with_both_special_modes(self):
        calls = []
        class Backend:
            def decode(self, tokens, skip_special_tokens):
                calls.append((tokens, skip_special_tokens))
                return ('雪\ufffd </search>\n' if not skip_special_tokens else '雪 </search>\n')
        qwen = type('Qwen2TokenizerFast', (), {
            '__module__': 'transformers.models.qwen2.tokenization_qwen2_fast',
            'is_fast': True, 'backend_tokenizer': Backend()})()
        visible, executor = helper()(qwen)
        self.assertEqual(visible([3, 4]), '雪\ufffd </search>\n')
        self.assertEqual(executor([3, 4]), '雪 </search>\n')
        self.assertEqual(calls, [([3, 4], False), ([3, 4], True)])

    def test_other_tokenizer_preserves_hf_options_and_text(self):
        calls = []
        class Other:
            def decode(self, tokens, **kwargs):
                calls.append((tokens, kwargs))
                return '  雪 </answer>\n'
        visible, executor = helper()(Other())
        self.assertEqual(visible([1]), executor([1]))
        self.assertEqual(calls, [([1], dict(skip_special_tokens=False, clean_up_tokenization_spaces=False)),
                                 ([1], dict(skip_special_tokens=True, clean_up_tokenization_spaces=False))])

    def test_same_named_foreign_class_does_not_bypass_custom_decoder(self):
        qwen = type('Qwen2TokenizerFast', (), {'is_fast': True,
            'decode': lambda self, tokens, **kwargs: 'custom text'})()
        visible, executor = helper()(qwen)
        self.assertEqual(visible([1]), 'custom text')
        self.assertEqual(executor([1]), 'custom text')


if __name__ == '__main__':
    unittest.main()
