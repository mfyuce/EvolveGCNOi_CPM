"""
models_cpm.py — model zoo for the CPM misbehavior-detection re-evaluation.

All models do node-level BINARY classification (2 logits per node).
Each model exposes a `kind` attribute that tells the driver how to call it:

  kind == "node_rec" : forward(x, ei, ew, H) -> (logits, H)   # node-state recurrence
  kind == "evolve"   : forward(x, ei, ew)    -> logits         # weight-state; has .reset()
  kind == "static"   : forward(x, ei, ew)    -> logits         # no state

Variants (paper "model fix" study)
----------------------------------
Reference / baselines
  GConvGRUNet   node-state recurrent (upper reference)
  TGCNNet       node-state recurrent
  StaticGCN     static GCN, configurable depth

Paper model + progressive fixes (keep the evolving-weight identity)
  EvolveHBaseline   evolve-conv + 3 GCN  (the published EvolveGCN-H wrapper)
  EvolveHShallow    M1: evolve-conv + 1 GCN          (attacks over-smoothing B4)
  EvolveHResidual   M2: evolve-conv + residual GCNs  (attacks B4)
  EvolveHWide       M3: project F->proj, evolve in proj-dim (attacks B1+B3)
  EvolveHFixedPool  M4: fixed TopK ratio             (attacks B1)
  EvolveHWideResid  M3+M2+M4 combined (strongest pure-EvolveGCN-H)
  EvolveHJK         JK-Net concat head (anti-over-smoothing alternative)

Hybrid / alternative
  EvolveONet        M5: stock EvolveGCN-O + GCN head
  EvolveHybrid      M6: evolving-weight conv -> node-state GConvGRU (cures B2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, GATv2Conv
from torch_geometric_temporal.nn.recurrent import GConvGRU, TGCN, EvolveGCNO
from graphs.recurrent.evolvegcnh_improved import EvolveGCNHImproved


# ───────────────────────── reference / baseline models ─────────────────────────
class GConvGRUNet(nn.Module):
    kind = "node_rec"
    def __init__(self, in_f, h=32, d=0.5, K=2):
        super().__init__()
        self.rec = GConvGRU(in_f, h, K)
        self.l1 = nn.Linear(h, h); self.cl = nn.Linear(h, 2); self.d = d
    def forward(self, x, ei, ew, H=None):
        H = self.rec(x, ei, ew, H)
        z = F.dropout(F.relu(self.l1(H)), p=self.d, training=self.training)
        return self.cl(z), H


class TGCNNet(nn.Module):
    kind = "node_rec"
    def __init__(self, in_f, h=32, d=0.5):
        super().__init__()
        self.rec = TGCN(in_f, h)
        self.l1 = nn.Linear(h, h); self.cl = nn.Linear(h, 2); self.d = d
    def forward(self, x, ei, ew, H=None):
        H = self.rec(x, ei, ew, H)
        z = F.dropout(F.relu(self.l1(H)), p=self.d, training=self.training)
        return self.cl(z), H


class StaticGCN(nn.Module):
    kind = "static"
    def __init__(self, in_f, h=32, d=0.5, depth=3):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_f if i == 0 else h, h) for i in range(depth)])
        self.cl = nn.Linear(h, 2); self.d = d
    def forward(self, x, ei, ew):
        h = x
        for i, c in enumerate(self.convs):
            h = F.relu(c(h, ei, ew))
            if i == len(self.convs) - 2:
                h = F.dropout(h, p=self.d, training=self.training)
        return self.cl(h)


# ───────────────────── EvolveGCN-H baseline + progressive fixes ─────────────────
def _make_evolve_h(n_nodes, in_f):
    """Build an EvolveGCNHImproved.

    NOTE — structural constraint: EvolveGCN-H pools the graph to exactly
    `in_f` nodes (ratio = in_f/n_nodes) because those pooled rows form the
    GRU's batch dimension, and the GRU hidden state IS the in_f×in_f weight
    matrix. So the number of nodes kept in the temporal summary is tied to
    the channel count — the only way to widen the summary is to widen the
    channels (the M3 'wide' variant projects F→proj first, raising kept
    nodes from ~in_f to proj)."""
    return EvolveGCNHImproved(n_nodes, in_f)


def _evolve_embed(rec, x, ei, ew):
    """Run one EvolveGCN-H step → (N, in_f) node embeddings, advancing the weight state."""
    X = rec.pooling_layer(x, ei)[0][None, :, :]
    if rec.weight is None:
        object.__setattr__(rec, "weight", rec.initial_weight)
    _, W = rec.recurrent_layer(X, rec.weight[None, :, :])
    W = W.squeeze(0)
    object.__setattr__(rec, "weight", W)
    return rec.conv_layer(W, x, ei, ew)


class EvolveHBaseline(nn.Module):
    """Published EvolveGCN-H wrapper: evolve-conv + 3 stacked GCN (4 hops → over-smooths)."""
    kind = "evolve"
    def __init__(self, n_nodes, in_f, h=32, d=0.5):
        super().__init__()
        self.rec = _make_evolve_h(n_nodes, in_f)
        self.c1 = GCNConv(in_f, h); self.c2 = GCNConv(h, h); self.c3 = GCNConv(h, h)
        self.cl = nn.Linear(h, 2); self.d = d
    def reset(self):
        object.__setattr__(self.rec, "weight", None)
    def forward(self, x, ei, ew):
        hh = _evolve_embed(self.rec, x, ei, ew)
        hh = F.relu(self.c1(hh, ei, ew)); hh = F.relu(self.c2(hh, ei, ew))
        hh = F.dropout(hh, p=self.d, training=self.training)
        return self.cl(F.relu(self.c3(hh, ei, ew)))


class EvolveHShallow(nn.Module):
    """M1: evolve-conv + a single GCN (2 hops). Tests the over-smoothing hypothesis."""
    kind = "evolve"
    def __init__(self, n_nodes, in_f, h=32, d=0.5):
        super().__init__()
        self.rec = _make_evolve_h(n_nodes, in_f)
        self.c1 = GCNConv(in_f, h); self.cl = nn.Linear(h, 2); self.d = d
    def reset(self):
        object.__setattr__(self.rec, "weight", None)
    def forward(self, x, ei, ew):
        hh = _evolve_embed(self.rec, x, ei, ew)
        hh = F.dropout(F.relu(self.c1(hh, ei, ew)), p=self.d, training=self.training)
        return self.cl(hh)


class EvolveHResidual(nn.Module):
    """M2: evolve-conv + residual GCN stack (skip connections fight over-smoothing)."""
    kind = "evolve"
    def __init__(self, n_nodes, in_f, h=32, d=0.5):
        super().__init__()
        self.rec = _make_evolve_h(n_nodes, in_f)
        self.proj = nn.Linear(in_f, h)
        self.c1 = GCNConv(h, h); self.c2 = GCNConv(h, h); self.c3 = GCNConv(h, h)
        self.cl = nn.Linear(h, 2); self.d = d
    def reset(self):
        object.__setattr__(self.rec, "weight", None)
    def forward(self, x, ei, ew):
        hh = _evolve_embed(self.rec, x, ei, ew)
        h0 = F.relu(self.proj(hh))
        h1 = F.relu(self.c1(h0, ei, ew)) + h0
        h2 = F.relu(self.c2(h1, ei, ew)) + h1
        h2 = F.dropout(h2, p=self.d, training=self.training)
        h3 = F.relu(self.c3(h2, ei, ew)) + h2
        return self.cl(h3)


class EvolveHWide(nn.Module):
    """M3: project F->proj first, run EvolveGCN-H in proj-dim (ratio proj/N, W proj×proj)."""
    kind = "evolve"
    def __init__(self, n_nodes, in_f, h=32, d=0.5, proj=32):
        super().__init__()
        self.pre = nn.Linear(in_f, proj)
        self.rec = _make_evolve_h(n_nodes, proj)
        self.c1 = GCNConv(proj, h); self.c2 = GCNConv(h, h)
        self.cl = nn.Linear(h, 2); self.d = d
    def reset(self):
        object.__setattr__(self.rec, "weight", None)
    def forward(self, x, ei, ew):
        xp = F.relu(self.pre(x))
        hh = _evolve_embed(self.rec, xp, ei, ew)
        hh = F.relu(self.c1(hh, ei, ew))
        hh = F.dropout(F.relu(self.c2(hh, ei, ew)), p=self.d, training=self.training)
        return self.cl(hh)


class EvolveHWideResid(nn.Module):
    """M3+M2: wide evolve (proj-dim) + residual head. Strongest pure-EvolveGCN-H."""
    kind = "evolve"
    def __init__(self, n_nodes, in_f, h=32, d=0.5, proj=32):
        super().__init__()
        self.pre = nn.Linear(in_f, proj)
        self.rec = _make_evolve_h(n_nodes, proj)
        self.proj2 = nn.Linear(proj, h)
        self.c1 = GCNConv(h, h); self.c2 = GCNConv(h, h)
        self.cl = nn.Linear(h, 2); self.d = d
    def reset(self):
        object.__setattr__(self.rec, "weight", None)
    def forward(self, x, ei, ew):
        xp = F.relu(self.pre(x))
        hh = _evolve_embed(self.rec, xp, ei, ew)
        h0 = F.relu(self.proj2(hh))
        h1 = F.relu(self.c1(h0, ei, ew)) + h0
        h1 = F.dropout(h1, p=self.d, training=self.training)
        h2 = F.relu(self.c2(h1, ei, ew)) + h1
        return self.cl(h2)


class EvolveHJK(nn.Module):
    """JK-Net: concatenate every hop's output (preserves shallow + deep features)."""
    kind = "evolve"
    def __init__(self, n_nodes, in_f, h=32, d=0.5, proj=32):
        super().__init__()
        self.pre = nn.Linear(in_f, proj)
        self.rec = _make_evolve_h(n_nodes, proj)
        self.proj2 = nn.Linear(proj, h)
        self.c1 = GCNConv(h, h); self.c2 = GCNConv(h, h)
        self.cl = nn.Linear(3 * h, 2); self.d = d   # concat of 3 hops
    def reset(self):
        object.__setattr__(self.rec, "weight", None)
    def forward(self, x, ei, ew):
        xp = F.relu(self.pre(x))
        hh = _evolve_embed(self.rec, xp, ei, ew)
        h0 = F.relu(self.proj2(hh))
        h1 = F.relu(self.c1(h0, ei, ew))
        h2 = F.relu(self.c2(h1, ei, ew))
        z = torch.cat([h0, h1, h2], dim=1)
        z = F.dropout(z, p=self.d, training=self.training)
        return self.cl(z)


# ─────────────────────────── hybrid / alternative ──────────────────────────────
class EvolveONet(nn.Module):
    """M5: stock EvolveGCN-O (weight evolves via GRU, no TopK pooling) + GCN head."""
    kind = "evolve"
    def __init__(self, n_nodes, in_f, h=32, d=0.5):
        super().__init__()
        self.rec = EvolveGCNO(in_f)
        self.c1 = GCNConv(in_f, h); self.c2 = GCNConv(h, h)
        self.cl = nn.Linear(h, 2); self.d = d
    def reset(self):
        # stock EvolveGCNO lazily builds self.weight; clearing forces re-init per window
        if hasattr(self.rec, "weight"):
            try:
                object.__setattr__(self.rec, "weight", None)
            except Exception:
                pass
    def forward(self, x, ei, ew):
        hh = F.relu(self.rec(x, ei, ew))
        hh = F.relu(self.c1(hh, ei, ew))
        hh = F.dropout(F.relu(self.c2(hh, ei, ew)), p=self.d, training=self.training)
        return self.cl(hh)


class EvolveHybrid(nn.Module):
    """M6: evolving-weight conv produces embeddings, then a node-state GConvGRU carries
    memory across time. Keeps the EvolveGCN prior but cures the weight-collapse (B2)."""
    kind = "node_rec"
    def __init__(self, n_nodes, in_f, h=32, d=0.5, proj=32, K=2):
        super().__init__()
        self.pre = nn.Linear(in_f, proj)
        self.rec = _make_evolve_h(n_nodes, proj)
        self.gru = GConvGRU(proj, h, K)
        self.l1 = nn.Linear(h, h); self.cl = nn.Linear(h, 2); self.d = d
        self._win_start = True
    def reset(self):
        object.__setattr__(self.rec, "weight", None)
    def forward(self, x, ei, ew, H=None):
        xp = F.relu(self.pre(x))
        emb = _evolve_embed(self.rec, xp, ei, ew)      # evolving-weight embedding
        H = self.gru(emb, ei, ew, H)                    # node-state recurrence
        z = F.dropout(F.relu(self.l1(H)), p=self.d, training=self.training)
        return self.cl(z), H


class LocalTemporalPreservingGNN(nn.Module):
    """
    Designed to prevent spatial smoothing from destroying local anomaly signals.
    Runs a purely local (per-node) GRUCell to capture vehicle-history dynamics,
    then uses GATv2Conv with edge features to aggregate neighbor anomalies/disagreements,
    and CONCATENATES both to classify.
    """
    kind = "node_rec"
    def __init__(self, in_f, h=32, d=0.5, edge_dim=1):
        super().__init__()
        self.local_gru = nn.GRUCell(in_f, h)
        self.edge_dim = edge_dim
        self.gat = GATv2Conv(h, h, heads=2, concat=False,
                             edge_dim=edge_dim, add_self_loops=False)
        self.lin1 = nn.Linear(h * 2, h)
        self.cl = nn.Linear(h, 2)
        self.d = d

    def forward(self, x, ei, ew, H=None):
        if H is None:
            H = torch.zeros(x.size(0), self.local_gru.hidden_size, device=x.device)
        H_new = self.local_gru(x, H)
        
        if ew is not None:
            if ew.ndim == 1:
                ea = ew.unsqueeze(-1)
            else:
                ea = ew
            if ea.size(-1) < self.edge_dim:
                padding = torch.zeros(ea.size(0), self.edge_dim - ea.size(-1), device=ea.device)
                ea = torch.cat([ea, padding], dim=-1)
            elif ea.size(-1) > self.edge_dim:
                ea = ea[:, :self.edge_dim]
        else:
            ea = None
            
        g = F.elu(self.gat(H_new, ei, edge_attr=ea))
        h_comb = torch.cat([H_new, g], dim=-1)
        h = F.relu(self.lin1(h_comb))
        z = F.dropout(h, p=self.d, training=self.training)
        return self.cl(z), H_new


# ─────────────────────────────── registry ──────────────────────────────────────
def build(name, n_nodes, in_f, h=32, d=0.5, **kw):
    name = name.lower()
    table = {
        "gconvgru":        lambda: GConvGRUNet(in_f, h, d),
        "tgcn":            lambda: TGCNNet(in_f, h, d),
        "static":          lambda: StaticGCN(in_f, h, d, depth=kw.get("depth", 3)),
        "evolve_h":        lambda: EvolveHBaseline(n_nodes, in_f, h, d),     # paper model
        "evolve_h_shallow":lambda: EvolveHShallow(n_nodes, in_f, h, d),     # M1
        "evolve_h_resid":  lambda: EvolveHResidual(n_nodes, in_f, h, d),    # M2
        "evolve_h_wide":   lambda: EvolveHWide(n_nodes, in_f, h, d, proj=kw.get("proj", 32)),  # M3
        "evolve_h_best":   lambda: EvolveHWideResid(n_nodes, in_f, h, d, proj=kw.get("proj", 32)),  # M3+M2
        "evolve_h_jk":     lambda: EvolveHJK(n_nodes, in_f, h, d, proj=kw.get("proj", 32)),   # JK
        "evolve_o":        lambda: EvolveONet(n_nodes, in_f, h, d),         # M5
        "hybrid":          lambda: EvolveHybrid(n_nodes, in_f, h, d, proj=kw.get("proj", 32)),  # M6
        "local_preserving": lambda: LocalTemporalPreservingGNN(in_f, h, d, edge_dim=kw.get("edge_dim", 1)),
    }
    if name not in table:
        raise ValueError(f"unknown model '{name}'. options: {list(table)}")
    return table[name]()


ALL_MODELS = [
    "gconvgru", "tgcn", "static",
    "evolve_h", "evolve_h_shallow", "evolve_h_resid", "evolve_h_wide",
    "evolve_h_best", "evolve_h_jk", "evolve_o", "hybrid", "local_preserving",
]
