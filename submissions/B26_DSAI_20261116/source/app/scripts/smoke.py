"""Live HTTP E2E: python scripts/smoke.py http://127.0.0.1:8000"""
import json
import sys
from urllib.request import Request, urlopen

base = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8000'
call = {'call_id': 'smoke', 'started_at': '2026-11-13T12:00:00', 'turns': [{'idx': 7, 'speaker': 'client', 'text': 'Интернет не работает.'}]}
for path in ('/health', '/model/info', '/', '/static/app.js'):
    with urlopen(base + path, timeout=5) as r:
        assert r.status == 200
with urlopen(Request(base + '/extract', data=json.dumps(call).encode(), headers={'Content-Type': 'application/json'}), timeout=10) as r:
    result = json.load(r)
assert result['meta']['is_mock'] is True
fact = result['prediction']['facts'][0]
assert fact['source']['turn_idx'] == 7 and fact['source']['quote'] == call['turns'][0]['text']
assert fact['polarity'] == 1
print('PASS: health, model/info, UI assets, live extraction and source')
