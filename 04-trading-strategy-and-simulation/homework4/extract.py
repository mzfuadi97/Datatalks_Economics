import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

path = sys.argv[1]
keyword = sys.argv[2] if len(sys.argv) > 2 else None

nb = json.load(open(path, encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    s = ''.join(c['source'])
    if keyword and keyword not in s:
        continue
    print('CELL', i, c['cell_type'])
    print(s[:2000])
    print('--- outputs ---')
    for o in c.get('outputs', []):
        if 'text' in o:
            print(''.join(o['text'])[:4000])
        elif 'data' in o and 'text/plain' in o['data']:
            print(''.join(o['data']['text/plain'])[:4000])
        elif o.get('output_type') == 'error':
            print('ERROR:', o.get('ename'), o.get('evalue'))
    print('=====')
