# Build a CORRECT v2 cache = v1 edges (unchanged) + a 10th node feature
# (per-node mean incoming detection_confidence), z-scored per-node over time
# exactly like BurstAdmaDatasetLoader does. The previous v2 z-scored the
# self-loop EDGE weights to mean 0, producing negative weights that made
# GCNConv's deg^{-1/2} normalization NaN. Edges must stay positive, so we
# keep v1 edges verbatim and only add the confidence feature.
import torch, numpy as np, time
t0=time.time()
c = torch.load('data/cache_v1.pt')
X,EI,EW,active = c['X'],c['EI'],c['EW'],c['active']
n,T,lags,inf = c['n_nodes'],c['T'],c['lags'],c['in_f']
nsteps=T-lags
print(f'v1 cache loaded F={inf} ({time.time()-t0:.0f}s)',flush=True)
# raw per-node mean incoming detection_confidence (prox edges src!=dst, EW=confidence)
conf=torch.zeros(nsteps,n)
for t in range(nsteps):
    ei=EI[t]; ew=EW[t]
    if ei.numel():
        s,d=ei[0],ei[1]; prox=s!=d
        if prox.any():
            acc=torch.zeros(n); cnt=torch.zeros(n)
            acc.scatter_add_(0,d[prox],ew[prox]); cnt.scatter_add_(0,d[prox],torch.ones(int(prox.sum())))
            m=cnt>0; conf[t,m]=acc[m]/cnt[m]
# z-score per-node over time (match loader: mean/std over axis=0 = time)
mu=conf.mean(0,keepdim=True); sd=conf.std(0,keepdim=True); sd[sd==0]=1.0
confz=(conf-mu)/(sd+1e-10)
X2=[torch.cat([X[t],confz[t][:,None]],1) for t in range(nsteps)]
c['X']=X2; c['in_f']=inf+1
torch.save(c,'data/v2/cache_v2.pt')
print(f'saved corrected cache_v2.pt F={inf}->{inf+1} ({time.time()-t0:.0f}s)',flush=True)
print(f'conf nonzero%={float((conf>0).float().mean()*100):.1f} mean={float(conf[conf>0].mean()):.3f}',flush=True)
