"""
aggregate_results.py — parse zoo_results/RESULTS.txt into mean±std tables.

Reads RESULT lines of the form
  RESULT ds=v1 model=gconvgru seed=3 ... mcc=.. macro_f1=.. mal_f1=.. ben_f1=..
         roc_auc=.. acc=.. thr=.. ...
and prints, per dataset, a per-model table of mean±std over seeds, sorted by MCC.

Usage
-----
  python aggregate_results.py [results_file]   (default zoo_results/RESULTS.txt)
"""
import sys, re
from collections import defaultdict
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "zoo_results/RESULTS.txt"
METRICS = ["mcc", "macro_f1", "mal_f1", "ben_f1", "roc_auc", "acc"]

# model display order (paper narrative)
ORDER = ["gconvgru", "tgcn", "static",
         "evolve_h", "evolve_h_shallow", "evolve_h_resid", "evolve_h_wide",
         "evolve_h_best", "evolve_h_jk", "evolve_o", "hybrid"]


def parse(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.startswith("RESULT"):
                continue
            kv = dict(re.findall(r"(\w+)=([^\s]+)", line))
            rows.append(kv)
    return rows


def main():
    rows = parse(PATH)
    # group by (ds, model)
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ds = r.get("ds", r.get("tag", "?"))
        m = r.get("model", "?")
        for met in METRICS:
            if met in r:
                try:
                    g[(ds, m)][met].append(float(r[met]))
                except ValueError:
                    pass

    datasets = sorted({k[0] for k in g})
    for ds in datasets:
        print(f"\n{'='*78}\nDATASET: {ds}\n{'='*78}")
        hdr = f"{'model':18s} " + " ".join(f"{m:>14s}" for m in METRICS) + "  n"
        print(hdr); print("-" * len(hdr))
        models = [m for m in ORDER if (ds, m) in g] + \
                 [m for (d, m) in g if d == ds and m not in ORDER]
        # sort by mean mcc desc
        def mean_mcc(m):
            v = g[(ds, m)].get("mcc", [])
            return np.mean(v) if v else -1e9
        for m in sorted(dict.fromkeys(models), key=mean_mcc, reverse=True):
            cells = []
            n = 0
            for met in METRICS:
                v = g[(ds, m)].get(met, [])
                n = max(n, len(v))
                if v:
                    cells.append(f"{np.mean(v):6.2f}±{np.std(v):4.2f}")
                else:
                    cells.append(f"{'—':>11s}")
            print(f"{m:18s} " + " ".join(f"{c:>14s}" for c in cells) + f"  {n}")


if __name__ == "__main__":
    main()
