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

lr = 0.5

# Logging / runtime knobs
epochs = 50
batch_size = 512
svd_every_epochs = 5     # compute SVD metrics only every few epochs
rank_thresh = 1e-2

def train_model(
    name,
    params,
    predict_fn,
    get_A_fn,
    X,
    Y,
    A_star,
    lr,
    epochs,
    batch_size,
    svd_every_epochs,
    rank_thresh,
    device,
    seed=123,
):
    opt = optim.SGD(params, lr=lr)
    loss_fn = nn.MSELoss()
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    with torch.no_grad():
        m0 = summarize_A(get_A_fn(), A_star, thresh=rank_thresh)
    print(f"\n[{name} init] rel_err={m0['rel_err']:.3f} nuc={m0['nuc']:.3f} effR={m0['eff_rank']:.2f} numR={m0['num_rank']}")
    print(f"{name} epoch | loss | rel_err | nuc | effR | numR")

    for epoch in range(1, epochs + 1):
        loss_accum = 0.0
        steps = 0

        for xb, yb in minibatches(X, Y, batch_size=batch_size, generator=g):
            steps += 1
            opt.zero_grad(set_to_none=True)
            yhat = predict_fn(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            opt.step()
            loss_accum += loss.item()

        avg_loss = loss_accum / steps

        if epoch == 1 or epoch % svd_every_epochs == 0 or epoch == epochs:
            with torch.no_grad():
                m = summarize_A(get_A_fn(), A_star, thresh=rank_thresh)
            print(
                f"{name:7s} {epoch:5d} | {avg_loss:9.3e} | {m['rel_err']:.3e} | "
                f"{m['nuc']:.3f} | {m['eff_rank']:.2f} | {m['num_rank']:4d}"
            )

    with torch.no_grad():
        return summarize_A(get_A_fn(), A_star, thresh=rank_thresh, topk=10)

print(f"device={device}, n={n}, d={d}, m={m}, true rank k={k}, hidden r={r}, lr={lr}")

with torch.no_grad():
    s_star = torch.sort(torch.linalg.svdvals(A_star), descending=True).values
    print("A* top-10 singular values:", s_star[:10].cpu().numpy())
md = train_model(
    name="Deep",
    params=[W, U],
    predict_fn=lambda xb: xb @ (W @ U).T,
    get_A_fn=lambda: W @ U,
    X=X,
    Y=Y,
    A_star=A_star,
    lr=lr,
    epochs=epochs,
    batch_size=batch_size,
    svd_every_epochs=svd_every_epochs,
    rank_thresh=rank_thresh,
    device=device,
)
ms = train_model(
    name="Shallow",
    params=[B],
    predict_fn=lambda xb: xb @ B.T,
    get_A_fn=lambda: B,
    X=X,
    Y=Y,
    A_star=A_star,
    lr=lr,
    epochs=epochs,
    batch_size=batch_size,
    svd_every_epochs=svd_every_epochs,
    rank_thresh=rank_thresh,
    device=device,
)

# Final spectra
print("\n[Final top-10 singular values]")
print("A*     :", torch.sort(torch.linalg.svdvals(A_star), descending=True).values[:10].cpu().numpy())
print("Deep A :", md["top_sv"].numpy())
print("Shallow:", ms["top_sv"].numpy())
