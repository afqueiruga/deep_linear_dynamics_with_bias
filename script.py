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
def make_error_projectors(A_star, k):
    # Build projectors onto the rank-k row/column support of A* from its SVD:
    # A* = U_k S_k V_k^T, with P_left = U_k U_k^T and P_right = V_k V_k^T.
    U, _, Vh = torch.linalg.svd(A_star, full_matrices=True)
    U_k = U[:, :k]
    V_k = Vh[:k, :].T
    P_left = U_k @ U_k.T
    P_right = V_k @ V_k.T
    I_left = torch.eye(A_star.shape[0], device=A_star.device, dtype=A_star.dtype)
    I_right = torch.eye(A_star.shape[1], device=A_star.device, dtype=A_star.dtype)
    P_left_perp = I_left - P_left
    P_right_perp = I_right - P_right
    return P_left, P_right, P_left_perp, P_right_perp

@torch.no_grad()
def summarize_A(
    A,
    A_star,
    thresh=1e-2,
    topk=10,
    P_left=None,
    P_right=None,
    P_left_perp=None,
    P_right_perp=None,
):
    # Generic spectrum/size diagnostics for an end-to-end map A.
    sv = torch.linalg.svdvals(A)
    sv = torch.sort(sv, descending=True).values
    fro = torch.linalg.norm(A, ord="fro").item()
    nuc = sv.sum().item()
    spec = sv[0].item()
    er = effective_rank(sv)
    num_rank = int((sv > thresh).sum().item())
    err = A - A_star
    rel_err = (err.norm() / (A_star.norm() + 1e-12)).item()
    out = {
        "rel_err": rel_err,
        "fro": fro,
        "nuc": nuc,
        "spec": spec,
        "eff_rank": er,
        "num_rank": num_rank,
        "top_sv": sv[:topk].cpu(),
    }
    if (
        P_left is not None
        and P_right is not None
        and P_left_perp is not None
        and P_right_perp is not None
    ):
        # Error decomposition for E = A - A*:
        # support component  : P_left E P_right
        # null component     : (I - P_left) E (I - P_right)
        # mixed component    : everything else (cross terms)
        err_support = P_left @ err @ P_right
        err_null = P_left_perp @ err @ P_right_perp
        err_mixed = err - err_support - err_null
        err_norm = err.norm() + 1e-12
        out.update(
            {
                "err_support": err_support.norm().item(),
                "err_null": err_null.norm().item(),
                "err_mixed": err_mixed.norm().item(),
                # Fraction of total error norm ||E||_F in each subspace component.
                # support_frac near 1.0 means most error lies inside A*'s support.
                # null_frac near 1.0 means most error lies in A*'s orthogonal nullspace.
                "err_support_frac": (err_support.norm() / err_norm).item(),
                "err_null_frac": (err_null.norm() / err_norm).item(),
                "err_mixed_frac": (err_mixed.norm() / err_norm).item(),
            }
        )
    return out

def minibatches(X, Y, batch_size, generator=None):
    n = X.shape[0]
    idx = torch.randperm(n, generator=generator, device=X.device)
    for i in range(0, n, batch_size):
        j = idx[i:i+batch_size]
        yield X[j], Y[j]

class DeepLinear(nn.Module):
    def __init__(self, d, m, r, init_scale=0.01, device="cpu"):
        super().__init__()
        self.W = nn.Parameter(init_scale * torch.randn(m, r, device=device))
        self.U = nn.Parameter(init_scale * torch.randn(r, d, device=device))

    def end_to_end(self):
        return self.W @ self.U

    def forward(self, x):
        return x @ self.end_to_end().T

# ----------------------------
# Experiment
# ----------------------------
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Dimensions
d = 50
m = 50
k = 5       # true rank
deep_widths = [k, d, 2 * d, 10 * d]

# Data
n = 20000
X = torch.randn(n, d, device=device) / (d ** 0.5)  # roughly whitened
A_star = make_low_rank_matrix(m, d, k, device=device)
Y = X @ A_star.T

# Models
shallow_model = nn.Linear(d, m, bias=False, device=device)
with torch.no_grad():
    nn.init.normal_(shallow_model.weight, mean=0.0, std=0.01)

# Logging / runtime knobs
# Trial hyperparameters (adjusted for stronger convergence)
deep_optimizer_name = "adam"
shallow_optimizer_name = "adam"
deep_lr = 2e-2
shallow_lr = 1e-2
epochs = 400
batch_size = n
svd_every_epochs = 20     # compute SVD metrics only every few epochs
rank_thresh = 1e-2
# Fixed projectors from A* used to track how training error moves across subspaces.
P_left, P_right, P_left_perp, P_right_perp = make_error_projectors(A_star, k)

def train_model(
    name,
    model,
    get_A_fn,
    X,
    Y,
    A_star,
    lr,
    optimizer_name,
    epochs,
    batch_size,
    svd_every_epochs,
    rank_thresh,
    device,
    P_left,
    P_right,
    P_left_perp,
    P_right_perp,
    seed=123,
):
    if optimizer_name.lower() == "sgd":
        opt = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "adam":
        opt = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer_name={optimizer_name}")
    loss_fn = nn.MSELoss()
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    with torch.no_grad():
        m0 = summarize_A(
            get_A_fn(model),
            A_star,
            thresh=rank_thresh,
            P_left=P_left,
            P_right=P_right,
            P_left_perp=P_left_perp,
            P_right_perp=P_right_perp,
        )
    print(
        f"\n[{name} init] rel_err={m0['rel_err']:.3f} nuc={m0['nuc']:.3f} "
        f"effR={m0['eff_rank']:.2f} numR={m0['num_rank']} "
        f"support={m0['err_support']:.3e} null={m0['err_null']:.3e}"
    )
    # support_frac = ||P_left E P_right||_F / ||E||_F
    # null_frac    = ||P_left_perp E P_right_perp||_F / ||E||_F
    print(f"{name} epoch | loss | rel_err | support_err | null_err | support_frac | null_frac")

    for epoch in range(1, epochs + 1):
        loss_accum = 0.0
        steps = 0

        for xb, yb in minibatches(X, Y, batch_size=batch_size, generator=g):
            steps += 1
            opt.zero_grad(set_to_none=True)
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            opt.step()
            loss_accum += loss.item()

        avg_loss = loss_accum / steps

        if epoch == 1 or epoch % svd_every_epochs == 0 or epoch == epochs:
            with torch.no_grad():
                m = summarize_A(
                    get_A_fn(model),
                    A_star,
                    thresh=rank_thresh,
                    P_left=P_left,
                    P_right=P_right,
                    P_left_perp=P_left_perp,
                    P_right_perp=P_right_perp,
                )
            print(
                f"{name:7s} {epoch:5d} | {avg_loss:9.3e} | {m['rel_err']:.3e} | "
                f"{m['err_support']:.3e} | {m['err_null']:.3e} | "
                f"{m['err_support_frac']:.2f} | {m['err_null_frac']:.2f}"
            )

    with torch.no_grad():
        return summarize_A(
            get_A_fn(model),
            A_star,
            thresh=rank_thresh,
            topk=10,
            P_left=P_left,
            P_right=P_right,
            P_left_perp=P_left_perp,
            P_right_perp=P_right_perp,
        )

print(
    f"device={device}, n={n}, d={d}, m={m}, true rank k={k}, deep_widths={deep_widths}, "
    f"deep_opt={deep_optimizer_name}, deep_lr={deep_lr}, "
    f"shallow_opt={shallow_optimizer_name}, shallow_lr={shallow_lr}, "
    f"epochs={epochs}, batch_size={batch_size}"
)

with torch.no_grad():
    s_star = torch.sort(torch.linalg.svdvals(A_star), descending=True).values
    print("A* top-10 singular values:", s_star[:10].cpu().numpy())
deep_results = {}
for r in deep_widths:
    deep_model = DeepLinear(d=d, m=m, r=r, init_scale=0.01, device=device)
    deep_results[r] = train_model(
        name=f"Deep(r={r})",
        model=deep_model,
        get_A_fn=lambda model: model.end_to_end(),
        X=X,
        Y=Y,
        A_star=A_star,
        lr=deep_lr,
        optimizer_name=deep_optimizer_name,
        epochs=epochs,
        batch_size=batch_size,
        svd_every_epochs=svd_every_epochs,
        rank_thresh=rank_thresh,
        device=device,
        P_left=P_left,
        P_right=P_right,
        P_left_perp=P_left_perp,
        P_right_perp=P_right_perp,
    )

ms = train_model(
    name="Shallow",
    model=shallow_model,
    get_A_fn=lambda model: model.weight,
    X=X,
    Y=Y,
    A_star=A_star,
    lr=shallow_lr,
    optimizer_name=shallow_optimizer_name,
    epochs=epochs,
    batch_size=batch_size,
    svd_every_epochs=svd_every_epochs,
    rank_thresh=rank_thresh,
    device=device,
    P_left=P_left,
    P_right=P_right,
    P_left_perp=P_left_perp,
    P_right_perp=P_right_perp,
)

# Final spectra
print("\n[Final top-10 singular values]")
print("A*     :", torch.sort(torch.linalg.svdvals(A_star), descending=True).values[:10].cpu().numpy())
for r in deep_widths:
    print(f"Deep(r={r}):", deep_results[r]["top_sv"].numpy())
print("Shallow:", ms["top_sv"].numpy())
print("\n[Final error decomposition]")
for r in deep_widths:
    mr = deep_results[r]
    print(
        "Deep(r={}) support={:.3e} null={:.3e} mixed={:.3e}".format(
            r, mr["err_support"], mr["err_null"], mr["err_mixed"]
        )
    )
print(
    "Shallow support={:.3e} null={:.3e} mixed={:.3e}".format(
        ms["err_support"], ms["err_null"], ms["err_mixed"]
    )
)
