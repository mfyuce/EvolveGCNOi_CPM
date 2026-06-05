"""
build_features_cpm.py — append engineered consistency features to a CPM cache.

On BurST-ADMA the strongest honest model was a static GCN fed hand-engineered
consistency features. This builds the CPM analogue: per-node, per-timestep
features that capture physical / temporal implausibility of the CPM-observed
state, which is what a misbehaving (spoofed) report violates.

Appended features (all z-scored over time before saving)
  d_x, d_y, d_heading, d_speed, d_acc   temporal |Δ| vs the node's previous
                                        active appearance (sudden jumps)
  pos_jump = sqrt(d_x²+d_y²)            combined position jump
  kin_resid = |d_speed − acc_t|         speed change vs reported acceleration
  in_degree                             # proximity detections (sensors seeing it)

in_f → in_f + 8.  Output: <cache>_aug.pt with the same structure.

Usage
-----
  python build_features_cpm.py --cache data/cache_v1.pt     --out data/cache_v1_aug.pt
  python build_features_cpm.py --cache data/v2/cache_v2.pt  --out data/v2/cache_v2_aug.pt
"""
import argparse, time
import numpy as np
import torch

# kinematic feature indices (shared by v1/v2 layouts)
IX, IY, IHEAD, ISPEED, IACC = 0, 1, 2, 3, 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    t0 = time.time()
    c = torch.load(args.cache)
    X, EI, active = c["X"], c["EI"], c["active"]
    n_nodes, T, lags, in_f = c["n_nodes"], c["T"], c["lags"], c["in_f"]
    n_steps = T - lags
    print(f"loaded {args.cache}: T={T} N={n_nodes} F={in_f} ({time.time()-t0:.0f}s)", flush=True)

    EXTRA = 8
    aug = [torch.zeros(n_nodes, EXTRA, dtype=torch.float32) for _ in range(n_steps)]
    last = torch.zeros(n_nodes, 5)          # last active [x,y,heading,speed,acc]
    seen = torch.zeros(n_nodes, dtype=torch.bool)

    for t in range(n_steps):
        x = X[t]
        act = active[t]
        kin = x[:, [IX, IY, IHEAD, ISPEED, IACC]]            # (N,5)
        d = (kin - last).abs()                                # |Δ| vs last active
        d[~seen] = 0.0                                         # no delta on first sight
        pos_jump = torch.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)    # combined position jump
        kin_resid = (d[:, 3] - x[:, IACC].abs()).abs()        # Δspeed vs |acc|

        # in-degree from proximity edges (src != dst)
        ei = EI[t]
        deg = torch.zeros(n_nodes)
        if ei.numel():
            src, dst = ei[0], ei[1]
            prox = src != dst
            if prox.any():
                deg.scatter_add_(0, dst[prox], torch.ones(prox.sum()))

        feat = torch.stack([d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4],
                            pos_jump, kin_resid, deg], dim=1)   # (N,8)
        feat[~act] = 0.0                                        # zero for inactive nodes
        aug[t] = feat

        # update last-active state
        upd = act
        last[upd] = kin[upd]
        seen[upd] = True

    # z-score the appended features over all active node-steps
    stacked = torch.cat([aug[t][active[t]] for t in range(n_steps)], dim=0)  # (M,8)
    mu = stacked.mean(0); sd = stacked.std(0); sd[sd == 0] = 1.0
    print(f"appended-feature means: {np.round(mu.numpy(),3)}", flush=True)
    print(f"appended-feature stds:  {np.round(sd.numpy(),3)}", flush=True)

    X_aug = []
    for t in range(n_steps):
        a = (aug[t] - mu) / sd
        a[~active[t]] = 0.0
        X_aug.append(torch.cat([X[t], a], dim=1))   # (N, in_f+8)

    c_out = dict(c)
    c_out["X"] = X_aug
    c_out["in_f"] = in_f + EXTRA
    torch.save(c_out, args.out)
    print(f"saved {args.out}  in_f {in_f}->{in_f+EXTRA}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
