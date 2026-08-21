"""Forensic validation of pipeline.local_dim estimators on synthetic ground truth.

Read-only w.r.t. the repo: imports pipeline code, writes nothing.

Experiments:
  E1  Known-dimension recovery: d-dim Gaussian embedded in D ambient.
      Does VGT (as implemented) recover d? Does PR?
  E2  Flagship regime: unit-norm isotropic vectors in D=2304.
      Distance concentration -> what does VGT report?
  E3  Geometry adjudication: tight near-duplicate clump vs single isolated
      outlier, on the same background cloud. Which one produces the
      low-VGT / huge-radius-span signature seen in the shipped data?
  E4  Mechanism check: is the log-count span constant (~3.1987)?
"""
import sys
sys.path.insert(0, "/home/user/workspace/repo")
import numpy as np
from scipy.spatial import KDTree
from pipeline.local_dim import _vgt_single, estimate_local_dim_pr

rng = np.random.default_rng(0)

def vgt_dims(X, max_k=50, n_radii=10):
    tree = KDTree(X)
    dists, _ = tree.query(X, k=max_k + 1, workers=-1)
    out = np.empty(len(X))
    spans = np.empty(len(X))
    logv_spans = np.empty(len(X))
    for i in range(len(X)):
        d_i = dists[i, 1:]
        dim, curve = _vgt_single(d_i, n_radii, True)
        out[i] = dim
        pos = d_i[d_i > 0]
        spans[i] = pos[-1] / pos[0] if len(pos) and pos[0] > 0 else np.nan
        lv = curve["log_v"]
        logv_spans[i] = (lv[-1] - lv[0]) if len(lv) >= 2 else np.nan
    return out, spans, logv_spans

def embed(X, D, rng):
    d = X.shape[1]
    Q, _ = np.linalg.qr(rng.standard_normal((D, d)))
    return X @ Q.T

print("=" * 72)
print("E1: known-dimension recovery (n=5000, ambient D=100, no noise)")
print(f"{'true d':>7} {'VGT med':>8} {'VGT p95':>8} {'PR med':>7}")
for d in [2, 5, 10, 20, 50]:
    n = 5000
    X = embed(rng.standard_normal((n, d)), 100, rng)
    v, _, _ = vgt_dims(X)
    pr = estimate_local_dim_pr(X, k=30, n_jobs=4)
    print(f"{d:>7} {np.median(v):>8.2f} {np.percentile(v,95):>8.2f} {np.median(pr):>7.2f}")

print("=" * 72)
print("E2: flagship regime — isotropic unit-norm, n=4000, D=2304 (true d≈2303)")
X = rng.standard_normal((4000, 2304))
X /= np.linalg.norm(X, axis=1, keepdims=True)
v, spans, lv = vgt_dims(X)
print(f"VGT median={np.median(v):.1f}  p95={np.percentile(v,95):.1f}  max={v.max():.1f}")
print(f"radius span (r_max/r_min) median={np.median(spans):.4f}")
print(f"log-count span: median={np.nanmedian(lv):.4f}  std={np.nanstd(lv):.4f}"
      f"  (log(49)-log(2)={np.log(49)-np.log(2):.4f})")

print("=" * 72)
print("E3: clump-vs-outlier adjudication (background n=3000, D=100, unit-norm)")
B = rng.standard_normal((3000, 100))
B /= np.linalg.norm(B, axis=1, keepdims=True)
# (a) tight near-duplicate clump of 10 (convergent features)
c = rng.standard_normal(100); c /= np.linalg.norm(c)
clump = c + 1e-4 * rng.standard_normal((10, 100))
clump /= np.linalg.norm(clump, axis=1, keepdims=True)
# (b) single isolated outlier, far from everything
outlier = 5.0 * (rng.standard_normal(100))
outlier /= np.linalg.norm(outlier); outlier = outlier * 1.0 + 3.0  # shift far away
X = np.vstack([B, clump, outlier[None, :]])
v, spans, _ = vgt_dims(X)
bg, cl, ol = v[:3000], v[3000:3010], v[3010]
bgs, cls_, ols = spans[:3000], spans[3000:3010], spans[3010]
print(f"background : VGT med={np.median(bg):8.2f}   span med={np.median(bgs):8.2f}")
print(f"CLUMP (10) : VGT med={np.median(cl):8.2f}   span med={np.median(cls_):8.2f}  <- convergence signature?")
print(f"OUTLIER (1): VGT   ={ol:8.2f}   span    ={ols:8.2f}  <- isolation signature?")
print("=" * 72)
print("E4: shipped-artifact signature = (low VGT + span >> median).")
print("Whichever of E3(a)/E3(b) reproduces it identifies the true geometry.")
