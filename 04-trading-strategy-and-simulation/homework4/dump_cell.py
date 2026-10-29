import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
path = sys.argv[1]
idx = int(sys.argv[2])
nb = json.load(open(path, encoding='utf-8'))
c = nb['cells'][idx]
for o in c.get('outputs', []):
    if 'data' in o and 'text/plain' in o['data']:
        print(''.join(o['data']['text/plain']))
    elif 'text' in o:
        print(''.join(o['text']))
