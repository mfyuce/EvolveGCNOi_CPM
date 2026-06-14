"""Paired analysis for CPM (same harness as BuST). Parses 'RESULT expCcpm mode=.. seed=.. mcc=..'
lines; prints per-config mean±SEM and the SAME paired comparisons as the BuST analysis, so the two
datasets can be compared head-to-head under an identical pipeline."""
import re, sys
import numpy as np
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

text = open(sys.argv[1]).read()
data = {}
for m in re.finditer(r'RESULT expCcpm mode=(\w+) seed=(\d+) mcc=([\d.\-]+)', text):
    data.setdefault(m.group(1), {})[int(m.group(2))] = float(m.group(3))


def arr(cfg):
    d = data.get(cfg, {}); seeds = sorted(d)
    return seeds, np.array([d[s] for s in seeds], float)


print("=== CPM per-config MCC (mean±std, SEM), same harness as BuST ===")
for cfg in ['gnn_rel', 'gnn_eng', 'gnn_edge', 'static_rel', 'static_eng', 'rf_rel', 'rf_eng']:
    seeds, v = arr(cfg)
    if len(v) == 0:
        continue
    sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float('nan')
    print(f"{cfg:12s} {v.mean():5.1f} ± {v.std():4.1f}   SEM {sem:4.2f}   n={len(v)}")


def paired(a, b):
    sa, va = arr(a); sb, vb = arr(b)
    da = dict(zip(sa, va)); db = dict(zip(sb, vb))
    common = sorted(set(sa) & set(sb))
    if not common:
        print(f"  {a} - {b}: (no common seeds)"); return
    diff = np.array([da[s] - db[s] for s in common])
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = wilcoxon(diff).pvalue if (wilcoxon and np.any(diff != 0)) else float('nan')
    print(f"  {a:10s} - {b:10s}: Δ={diff.mean():+5.1f}  95%CI[{lo:+5.1f},{hi:+5.1f}]  "
          f"{(diff > 0).sum()}/{len(diff)} seeds>0  Wilcoxon p={p:.3f}")


print("\n=== paired: GNN vs RF (does CPM flip like BuST?) ===")
paired('gnn_eng', 'rf_eng')
paired('gnn_rel', 'rf_rel')
paired('gnn_edge', 'rf_rel')
print("\n=== paired: recurrence isolation ===")
paired('gnn_rel', 'static_rel')
paired('gnn_eng', 'static_eng')
print("\n=== paired: static-GCN vs RF ===")
paired('static_rel', 'rf_rel')
paired('static_eng', 'rf_eng')
