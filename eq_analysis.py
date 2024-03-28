import wandb, ipdb, pandas as pd, polars as pl

def extract_query_details(key):
    key = key.strip('held_out/').strip('r2score').rstrip(' ')
    parts = key.split('(')
    query_name = parts[0].strip()
    if len(parts) > 1:
        time_range = parts[1].strip(') ').strip()
    else:
        time_range = None
    return query_name, time_range

api = wandb.Api()
runx = api.run("payal-collabs/EveryQueryGPT/487l51nc")
summary = runx.summary
r2scores = []
for k in summary.keys(): 
    if k.startswith('held_out/') and k.endswith('r2score') and k!='held_out/r2score': 
        name, timing = extract_query_details(k)
        r2scores.append((name, timing, summary[k]))
r2scores = pl.DataFrame(r2scores, schema=['query','timing','r2score'])

ipdb.set_trace()
