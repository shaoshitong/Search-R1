"""Recover question metadata from the unchanged shared dataset prompt."""
import functools
import json
import re


def with_question(kwargs):
    extra = kwargs['extra_info']
    if 'question' in extra:
        if not isinstance(extra['question'], str) or not extra['question']:
            raise ValueError('Malformed explicit question metadata')
        return kwargs
    messages = list(kwargs['raw_prompt'])
    if len(messages) != 1 or messages[0]['role'] != 'user':
        raise ValueError('Question adapter requires one original user prompt')
    content = messages[0]['content']
    prefix, separator, question = content.partition('\nQuestion: ')
    if not separator or not prefix.startswith('Answer the given question.') or not question:
        raise ValueError('Original question delimiter missing')
    result = dict(kwargs)
    result['extra_info'] = dict(extra, question=question)
    return result


def install(loop):
    if getattr(loop, '_shared_question_adapter', False):
        return
    original = loop.run

    @functools.wraps(original)
    async def run(instance, sampling_params, **kwargs):
        params = dict(sampling_params)
        stops = params.get('stop') or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        if '</search>' not in stops:
            stops.append('</search>')
        params['stop'] = stops
        params['include_stop_str_in_output'] = True
        return await original(instance, params, **with_question(kwargs))

    loop.run = run
    loop._shared_question_adapter = True


def search_calls(text):
    if '<tool_call>' in text or '</tool_call>' in text:
        return None
    matches = list(re.finditer(r'<search>(.*?)</search>', text, re.DOTALL))
    if text.count('<search>') != len(matches) or text.count('</search>') != len(matches):
        return []
    return [{'name': 'search', 'arguments': json.dumps({'query': m.group(1).strip()}, ensure_ascii=False)}
            for m in matches if m.group(1).strip()]


def canonical_format_for_score(text):
    # This is a scoring copy only; never alter generated tokens or training data.
    if any(tag in text for tag in ('<reason>', '</reason>', '<tool_call>', '</tool_call>')):
        return text
    if text.count('<think>') != 1 or text.count('</think>') != 1:
        return text
    result = text.replace('<think>', '<reason>').replace('</think>', '</reason>')
    calls = search_calls(result)
    if calls:
        result = re.sub(r'<search>(.*?)</search>',
                        lambda m: '<tool_call>' + json.dumps({'name': 'search', 'arguments': {'query': m.group(1).strip()}}, ensure_ascii=False) + '</tool_call>',
                        result, flags=re.DOTALL)
    return result


def install_parser(parser, function_call):
    if getattr(parser, '_shared_search_adapter', False):
        return
    original = parser.extract_tool_calls

    @functools.wraps(original)
    async def extract(instance, response_ids):
        text, native = await original(instance, response_ids)
        fallback = search_calls(text)
        if fallback is None or native:
            return text, native
        return text, [function_call(**call) for call in fallback]

    parser.extract_tool_calls = extract
    parser._shared_search_adapter = True
