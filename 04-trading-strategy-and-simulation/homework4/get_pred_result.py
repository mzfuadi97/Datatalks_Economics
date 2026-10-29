import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
path = sys.argv[1]
pred_name = sys.argv[2]

nb = json.load(open(path, encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    s = ''.join(c['source'])
    if f"isin(['{pred_name}'])" in s or f'df_sim1_results[df_sim1_results.prediction.isin' in s and pred_name in s:
        print('CELL', i)
        print(s[:300])
        for o in c.get('outputs', []):
            if 'data' in o and 'text/plain' in o['data']:
                print(''.join(o['data']['text/plain']))
        print('-----')
