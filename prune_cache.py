"""prune_cache.py — drop given feature indices from a prep_cache .pt (keeps edges/labels).
Usage: python prune_cache.py --cache IN.pt --out OUT.pt --drop 0,1,7
"""
import argparse, torch

ap = argparse.ArgumentParser()
ap.add_argument('--cache', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--drop', required=True, help='comma-separated feature indices to drop')
a = ap.parse_args()

c = torch.load(a.cache)
drop = {int(i) for i in a.drop.split(',')}
keep = [i for i in range(c['in_f']) if i not in drop]
c['X'] = [x[:, keep] for x in c['X']]
c['in_f'] = len(keep)
torch.save(c, a.out)
print(f"pruned {a.cache} -> {a.out}: dropped {sorted(drop)}, in_f -> {len(keep)}")
