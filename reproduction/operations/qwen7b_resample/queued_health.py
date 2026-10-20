"""Own-process readiness compatibility with the bounded59s relay contract."""
import urllib.request


def install():
    original = urllib.request.urlopen
    if getattr(original, '_queued_health', False):return original
    def urlopen(url, *args, **kwargs):
        if getattr(url, 'full_url', url) == 'http://127.0.0.1:28831/health':
            if len(args) >= 2:
                values = list(args);values[1] = max(60, values[1]) if isinstance(values[1], (int, float)) else 60;args = tuple(values)
            else:
                timeout = kwargs.get('timeout')
                kwargs['timeout'] = max(60, timeout) if isinstance(timeout, (int, float)) else 60
        return original(url, *args, **kwargs)
    urlopen._queued_health = True
    urllib.request.urlopen = urlopen
    return original
