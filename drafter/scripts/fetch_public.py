"""Fetch a seeded UltraChat-200k slice (train_sft for training, test_sft for eval) via the datasets-server API."""
import json, random, ssl, time, urllib.request, certifi, sys
ctx = ssl.create_default_context(cafile=certifi.where())
def get(u):
    for i in range(6):
        try: return json.load(urllib.request.urlopen(u, timeout=60, context=ctx))
        except Exception as e: print('retry', e, file=sys.stderr); time.sleep(3)
    raise SystemExit('fetch failed')
D = "HuggingFaceH4/ultrachat_200k"
sizes = {s['split']: s['num_rows'] for s in get(f"https://datasets-server.huggingface.co/size?dataset={D}")['size']['splits']}
print(sizes)
rng = random.Random(20261014)
out = {}
for split, n in (("train_sft", 6), ("test_sft", 1)):
    got = []
    for off in sorted(rng.sample(range(0, sizes[split] - 100), n)):
        for x in get(f"https://datasets-server.huggingface.co/rows?dataset={D}&config=default&split={split}&offset={off}&length=100")['rows']:
            row = x['row']
            got.append({"src": "ultrachat_200k/" + split, "row_idx": x['row_idx'], "prompt_id": row['prompt_id'], "prompt": row['prompt']})
    out[split] = got
    print(split, len(got))
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False)
