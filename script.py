import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Utilities
# ----------------------------
def make_low_rank_matrix(m, d, k, singular_values=None, device="cpu"):
    U, _ = torch.linalg.qr(torch.randn(m, k, device=device))
    V, _ = torch.linalg.qr(torch.randn(d, k, device=device))
    if singular_values is None:
        singular_values = torch.linspace(5.0, 1.0, k, device=device)
    S = torch.diag(singular_values)
    return U @ S @ V.T  # m x d

def effective_rank(s, eps=1e-12):
    p = s / (s.sum() + eps)
    H = -(p * (p + eps).log()).sum()
    return torch.exp(H).item()

@torch.no_grad()
def summarize_A(A, A_star, thresh=1e-2, topk=10):
    sv = torch.linalg.svdvals(A)
    sv = torch.sort(sv, descending=True).values
    fro = torch.linalg.norm(A, ord="fro").item()
    nuc = sv.sum().item()
    spec = sv[0].item()
    er = effective_rank(sv)
    num_rank = int((sv > thresh).sum().item())
    rel_err = ((A - A_star).norm() / (A_star.norm() + 1e-12)).item()
    return {
        "rel_err": rel_err,
        "fro": fro,
        "nuc": nuc,
        "spec": spec,
        "eff_rank": er,
        "num_rank": num_rank,
        "top_sv": sv[:topk].cpu(),
    }

def minibatches(X, Y, batch_size, generator=None):
    n = X.shape[0]
    idx = torch.randperm(n, generator=generator, device=X.device)
    for i in range(0, n, batch_size):
        j = idx[i:i+batch_size]
        yield X[j], Y[j]

# ----------------------------
# Experiment
# ----------------------------
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Dimensions
d = 50
m = 50
k = 5       # true rank
r = 30      # hidden dim (overparameterized)

# Data
n = 20000
X = torch.randn(n, d, device=device) / (d ** 0.5)  # roughly whitened
A_star = make_low_rank_matrix(m, d, k, device=device)
Y = X @ A_star.T

# Models
W = nn.Parameter(0.01 * torch.randn(m, r, device=device))
U = nn.Parameter(0.01 * torch.randn(r, d, device=device))

B = nn.Parameter(torch.zeros(m, d, device=device))  # shallow linear

# Optimizers
lr = 0.5
opt_deep = optim.SGD([W, U], lr=lr)
opt_shallow = optim.SGD([B], lr=lr)
loss_fn = nn.MSELoss()

# Logging / runtime knobs
epochs = 50
batch_size = 512
svd_every_epochs = 5     # compute SVD metrics only every few epochs
rank_thresh = 1e-2

print(f"device={device}, n={n}, d={d}, m={m}, true rank k={k}, hidden r={r}, lr={lr}")

with torch.no_grad():
    s_star = torch.sort(torch.linalg.svdvals(A_star), descending=True).values
    print("A* top-10 singular values:", s_star[:10].cpu().numpy())

    md0 = summarize_A(W @ U, A_star, thresh=rank_thresh)
    ms0 = summarize_A(B, A_star, thresh=rank_thresh)
    print("\n[Init]")
    print("  Deep   rel_err={:.3f} nuc={:.3f} effR={:.2f} numR={}".format(
        md0["rel_err"], md0["nuc"], md0["eff_rank"], md0["num_rank"]))
    print("  Shallow rel_err={:.3f} nuc={:.3f} effR={:.2f} numR={}".format(
        ms0["rel_err"], ms0["nuc"], ms0["eff_rank"], ms0["num_rank"]))

print("\nepoch | loss_deep | loss_shallow | rel_err_deep | rel_err_shallow | nuc_deep | nuc_shallow | effR_deep | effR_shallow | numR_deep | numR_shallow")

g = torch.Generator(device=device)
g.manual_seed(123)

for epoch in range(1, epochs + 1):
    # one epoch of SGD
    deep_loss_accum = 0.0
    shallow_loss_accum = 0.0
    steps = 0

    for xb, yb in minibatches(X, Y, batch_size=batch_size, generator=g):
        steps += 1

        # Deep update
        opt_deep.zero_grad(set_to_none=True)
        A_deep = W @ U
        yhat_deep = xb @ A_deep.T
        loss_deep = loss_fn(yhat_deep, yb)
        loss_deep.backward()
        opt_deep.step()

        # Shallow update
        opt_shallow.zero_grad(set_to_none=True)
        yhat_shallow = xb @ B.T
        loss_shallow = loss_fn(yhat_shallow, yb)
        loss_shallow.backward()
        opt_shallow.step()

        deep_loss_accum += loss_deep.item()
        shallow_loss_accum += loss_shallow.item()

    avg_deep_loss = deep_loss_accum / steps
    avg_shallow_loss = shallow_loss_accum / steps

    # expensive metrics less frequently
    if epoch == 1 or epoch % svd_every_epochs == 0 or epoch == epochs:
        with torch.no_grad():
            md = summarize_A(W @ U, A_star, thresh=rank_thresh)
            ms = summarize_A(B, A_star, thresh=rank_thresh)
        print(
            f"{epoch:5d} | {avg_deep_loss:9.3e} | {avg_shallow_loss:11.3e} | "
            f"{md['rel_err']:.3e} | {ms['rel_err']:.3e} | "
            f"{md['nuc']:.3f} | {ms['nuc']:.3f} | "
            f"{md['eff_rank']:.2f} | {ms['eff_rank']:.2f} | "
            f"{md['num_rank']:8d} | {ms['num_rank']:10d}"
        )

# Final spectra
with torch.no_grad():
    md = summarize_A(W @ U, A_star, thresh=rank_thresh, topk=10)
    ms = summarize_A(B, A_star, thresh=rank_thresh, topk=10)

print("\n[Final top-10 singular values]")
print("A*     :", torch.sort(torch.linalg.svdvals(A_star), descending=True).values[:10].cpu().numpy())
print("Deep A :", md["top_sv"].numpy())
print("Shallow:", ms["top_sv"].numpy())