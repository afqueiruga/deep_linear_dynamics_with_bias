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

def make_low_rank_inputs(n, d, kx, device="cpu"):
    # X has rank <= kx by construction: X = Z Vx^T with Vx orthonormal columns.
    Vx, _ = torch.linalg.qr(torch.randn(d, kx, device=device))
    Z = torch.randn(n, kx, device=device) / (kx ** 0.5)
    X = Z @ Vx.T
    return X, Vx

def add_label_noise(Y, noise_std, generator=None):
    if noise_std <= 0:
        return Y
    noise = noise_std * torch.randn(Y.shape, device=Y.device, dtype=Y.dtype, generator=generator)
    return Y + noise

def format_top_svs(sv, k=10):
    vals = sv[:k].tolist()
    return "[" + ", ".join(f"{v:.3e}" for v in vals) + "]"

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
        err_outside = err - err_support
        err_null = P_left_perp @ err @ P_right_perp
        err_mixed = err - err_support - err_null
        err_norm = err.norm() + 1e-12
        out.update(
            {
                "err_support": err_support.norm().item(),
                "err_outside": err_outside.norm().item(),
                "err_null": err_null.norm().item(),
                "err_mixed": err_mixed.norm().item(),
                # Fraction of total error norm ||E||_F in each subspace component.
                # support_frac near 1.0 means most error lies inside A*'s support.
                # outside_frac near 1.0 means most error lies outside A*'s support.
                # null_frac near 1.0 means most error lies in A*'s orthogonal nullspace.
                "err_support_frac": (err_support.norm() / err_norm).item(),
                "err_outside_frac": (err_outside.norm() / err_norm).item(),
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
deep_lr_decay_gamma = 0.99
shallow_lr_decay_gamma = 1.0
epochs = 400
batch_size = n
svd_every_epochs = 20     # compute SVD metrics only every few epochs
rank_thresh = 1e-2
run_experiment_1 = False  # disabled for efficiency; set True to re-enable
# Label-noise options (set to >0 to enable additive Gaussian noise on Y).
label_noise_std_exp1 = 0.0
# Experiment-2-specific schedule to probe slow implicit regularization.
epochs_lowX = 200
svd_every_epochs_lowX = 5
deep_lr_lowX = 1e-2
deep_lr_decay_gamma_lowX = 0.999
shallow_lr_lowX = 1e-2
shallow_lr_decay_gamma_lowX = 1.0
top_sv_every_epochs_lowX = svd_every_epochs_lowX
top_sv_k = 10
top_sv_method_lowX = "lowrank"  # "exact" or "lowrank"
label_noise_values_lowX = [0.0, 1e-2]
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
    lr_decay_gamma,
    epochs,
    batch_size,
    svd_every_epochs,
    rank_thresh,
    device,
    P_left,
    P_right,
    P_left_perp,
    P_right_perp,
    model_null_proj=None,
    model_null_label="model_null_norm",
    top_sv_every_epochs=None,
    top_sv_k=10,
    top_sv_method="exact",
    seed=123,
):
    if optimizer_name.lower() == "sgd":
        opt = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "adam":
        opt = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer_name={optimizer_name}")
    scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=lr_decay_gamma)
    loss_fn = nn.MSELoss()
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    with torch.no_grad():
        A0 = get_A_fn(model)
        m0 = summarize_A(
            A0,
            A_star,
            thresh=rank_thresh,
            P_left=P_left,
            P_right=P_right,
            P_left_perp=P_left_perp,
            P_right_perp=P_right_perp,
        )
        if model_null_proj is not None:
            m0[model_null_label] = (A0 @ model_null_proj).norm().item()
        if top_sv_method == "exact":
            sv0 = torch.sort(torch.linalg.svdvals(A0), descending=True).values[:top_sv_k]
        elif top_sv_method == "lowrank":
            q = min(max(top_sv_k + 5, top_sv_k), min(A0.shape))
            _, s_lr0, _ = torch.svd_lowrank(A0, q=q, niter=2)
            sv0 = torch.sort(s_lr0, descending=True).values[:top_sv_k]
        else:
            raise ValueError(f"Unknown top_sv_method={top_sv_method}")

    header = (
        "model          | epoch | loss      | lr        | rel_err   | fro       | nuc       | spec      | "
        "eff_rank | num_rank | err_support | err_outside | err_null  | err_mixed | "
        "supp_frac | out_frac | null_frac | mixed_frac"
    )
    if model_null_proj is not None:
        header += f" | {model_null_label}"
    header += " | top_sv"
    print(f"\n[{name}]")
    print(header)
    init_row = (
        f"{name:14s} | {0:5d} | {'-':>9s} | {lr:9.3e} | {m0['rel_err']:9.3e} | {m0['fro']:9.3e} | "
        f"{m0['nuc']:9.3e} | {m0['spec']:9.3e} | {m0['eff_rank']:8.3f} | {m0['num_rank']:8d} | "
        f"{m0['err_support']:11.3e} | {m0['err_outside']:11.3e} | {m0['err_null']:9.3e} | {m0['err_mixed']:9.3e} | "
        f"{m0['err_support_frac']:9.3f} | {m0['err_outside_frac']:8.3f} | {m0['err_null_frac']:9.3f} | {m0['err_mixed_frac']:10.3f}"
    )
    if model_null_proj is not None:
        init_row += f" | {m0[model_null_label]:11.3e}"
    init_row += f" | {format_top_svs(sv0, k=top_sv_k)}"
    print(init_row)

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

        scheduler.step()

        if epoch == 1 or epoch % svd_every_epochs == 0 or epoch == epochs:
            with torch.no_grad():
                A_cur = get_A_fn(model)
                m = summarize_A(
                    A_cur,
                    A_star,
                    thresh=rank_thresh,
                    P_left=P_left,
                    P_right=P_right,
                    P_left_perp=P_left_perp,
                    P_right_perp=P_right_perp,
                )
                if model_null_proj is not None:
                    m[model_null_label] = (A_cur @ model_null_proj).norm().item()
                print_top_sv = top_sv_every_epochs is None or epoch % top_sv_every_epochs == 0 or epoch == 1 or epoch == epochs
                sv_str = "-"
                if print_top_sv:
                    if top_sv_method == "exact":
                        sv_top = torch.sort(torch.linalg.svdvals(A_cur), descending=True).values[:top_sv_k]
                    elif top_sv_method == "lowrank":
                        q = min(max(top_sv_k + 5, top_sv_k), min(A_cur.shape))
                        _, s_lr, _ = torch.svd_lowrank(A_cur, q=q, niter=2)
                        sv_top = torch.sort(s_lr, descending=True).values[:top_sv_k]
                    else:
                        raise ValueError(f"Unknown top_sv_method={top_sv_method}")
                    sv_str = format_top_svs(sv_top, k=top_sv_k)
            row = (
                f"{name:14s} | {epoch:5d} | {avg_loss:9.3e} | {scheduler.get_last_lr()[0]:9.3e} | {m['rel_err']:9.3e} | {m['fro']:9.3e} | "
                f"{m['nuc']:9.3e} | {m['spec']:9.3e} | {m['eff_rank']:8.3f} | {m['num_rank']:8d} | "
                f"{m['err_support']:11.3e} | {m['err_outside']:11.3e} | {m['err_null']:9.3e} | {m['err_mixed']:9.3e} | "
                f"{m['err_support_frac']:9.3f} | {m['err_outside_frac']:8.3f} | {m['err_null_frac']:9.3f} | {m['err_mixed_frac']:10.3f}"
            )
            if model_null_proj is not None:
                row += f" | {m[model_null_label]:11.3e}"
            row += f" | {sv_str}"
            print(row)

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
    f"deep_decay_gamma={deep_lr_decay_gamma}, "
    f"shallow_opt={shallow_optimizer_name}, shallow_lr={shallow_lr}, "
    f"shallow_decay_gamma={shallow_lr_decay_gamma}, "
    f"epochs={epochs}, batch_size={batch_size}"
)

if run_experiment_1:
    g_noise_exp1 = torch.Generator(device=device)
    g_noise_exp1.manual_seed(2026)
    Y_train_exp1 = add_label_noise(Y, label_noise_std_exp1, generator=g_noise_exp1)
    with torch.no_grad():
        print("\n====================")
        print("Experiment 1: low-rank A*, full-rank X")
        print("====================")
        print(f"label_noise_std_exp1={label_noise_std_exp1}")
        s_star = torch.sort(torch.linalg.svdvals(A_star), descending=True).values
        print("A* top-10 singular values:", format_top_svs(s_star, k=10))
    deep_results = {}
    for r in deep_widths:
        deep_model = DeepLinear(d=d, m=m, r=r, init_scale=0.01, device=device)
        deep_results[r] = train_model(
            name=f"Deep(r={r})",
            model=deep_model,
            get_A_fn=lambda model: model.end_to_end(),
            X=X,
            Y=Y_train_exp1,
            A_star=A_star,
            lr=deep_lr,
            optimizer_name=deep_optimizer_name,
            lr_decay_gamma=deep_lr_decay_gamma,
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
        Y=Y_train_exp1,
        A_star=A_star,
        lr=shallow_lr,
        optimizer_name=shallow_optimizer_name,
        lr_decay_gamma=shallow_lr_decay_gamma,
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

    print("\n[Final top-10 singular values]")
    print("A*     :", format_top_svs(torch.sort(torch.linalg.svdvals(A_star), descending=True).values, k=10))
    for r in deep_widths:
        print(f"Deep(r={r}):", format_top_svs(deep_results[r]["top_sv"], k=10))
    print("Shallow:", format_top_svs(ms["top_sv"], k=10))
    print("\n[Final error decomposition]")
    for r in deep_widths:
        mr = deep_results[r]
        print(
            "Deep(r={}) support={:.3e} outside={:.3e} null={:.3e} mixed={:.3e}".format(
                r, mr["err_support"], mr["err_outside"], mr["err_null"], mr["err_mixed"]
            )
        )
    print(
        "Shallow support={:.3e} outside={:.3e} null={:.3e} mixed={:.3e}".format(
            ms["err_support"], ms["err_outside"], ms["err_null"], ms["err_mixed"]
        )
    )
else:
    print("\nExperiment 1 is disabled for efficiency (run_experiment_1=False).")

# ----------------------------
# Experiment 2
# A* full rank, X low rank. Only A* on span(X) is identifiable from Y.
# ----------------------------
kx = 5
A_star_full = make_low_rank_matrix(m, d, k=min(m, d), device=device)  # full-rank target map
X_low, _ = make_low_rank_inputs(n=n, d=d, kx=kx, device=device)

I_m = torch.eye(m, device=device)
I_d = torch.eye(d, device=device)

# Build support from SVD of the observed training data X_low.
# For X_low = U_x S_x V_x^T, the identifiable subspace for A is span(V_x) in input-feature space.
# (U_x is in sample space and does not define the domain subspace of A.)
_, sx_data, Vhx_data = torch.linalg.svd(X_low, full_matrices=False)
# Numerical-rank threshold: keep singular directions above relative noise floor.
sv_tol = 1e-6 * sx_data.max()
rank_x_emp = int((sx_data > sv_tol).sum().item())
Vx_data = Vhx_data[:rank_x_emp, :].T
P_x = Vx_data @ Vx_data.T
Q_x = I_d - P_x

# For this experiment:
# support  = error on identifiable input subspace: ||E P_x||_F
# outside  = error on input nullspace           : ||E Q_x||_F
# null     = set equal to outside by choosing P_left_perp = I_m.
# Learnable target component from data support:
# A*_learnable = A* P_x = A* U U^T, where U spans the input-data support.
#
# Here U is represented by Vx_data (right singular vectors of X_low), since
# support for A (m x d) lives in input-feature space.
A_star_learnable = A_star_full @ P_x
A_star_unlearnable = A_star_full @ Q_x

P_left2 = I_m
P_right2 = P_x
P_left2_perp = I_m
P_right2_perp = Q_x

with torch.no_grad():
    s_full = torch.sort(torch.linalg.svdvals(A_star_full), descending=True).values
    print("\n====================")
    print("Experiment 2: full-rank A*, low-rank X")
    print("====================")
    print(f"X rank proxy kx={kx}, empirical rank from SVD={rank_x_emp}")
    print(
        f"lowX schedule: epochs={epochs_lowX}, deep_lr={deep_lr_lowX}, "
        f"deep_decay_gamma={deep_lr_decay_gamma_lowX}, shallow_lr={shallow_lr_lowX}, "
        f"shallow_decay_gamma={shallow_lr_decay_gamma_lowX}, svd_every={svd_every_epochs_lowX}, "
        f"top_sv_every={top_sv_every_epochs_lowX}, top_sv_method={top_sv_method_lowX}"
    )
    print(f"label_noise_values_lowX={label_noise_values_lowX}")
    print("A* (full) top-10 singular values:", format_top_svs(s_full, k=10))
    print(
        "||A*_learnable||_F={:.3e}, ||A*_unlearnable||_F={:.3e}".format(
            A_star_learnable.norm().item(), A_star_unlearnable.norm().item()
        )
    )

for noise_idx, label_noise_std_lowX in enumerate(label_noise_values_lowX):
    Y_low = X_low @ A_star_full.T
    g_noise_lowX = torch.Generator(device=device)
    g_noise_lowX.manual_seed(2027 + noise_idx)
    Y_low = add_label_noise(Y_low, label_noise_std_lowX, generator=g_noise_lowX)

    print("\n--------------------")
    print(f"Experiment 2 run: label_noise_std_lowX={label_noise_std_lowX}")
    print("--------------------")

    deep_results_lowX = {}
    deep_models_lowX = {}
    for r in deep_widths:
        deep_model = DeepLinear(d=d, m=m, r=r, init_scale=0.01, device=device)
        deep_models_lowX[r] = deep_model
        deep_results_lowX[r] = train_model(
            name=f"LowX-Deep(r={r})",
            model=deep_model,
            get_A_fn=lambda model: model.end_to_end(),
            X=X_low,
            Y=Y_low,
            A_star=A_star_full,
            lr=deep_lr_lowX,
            optimizer_name=deep_optimizer_name,
            lr_decay_gamma=deep_lr_decay_gamma_lowX,
            epochs=epochs_lowX,
            batch_size=batch_size,
            svd_every_epochs=svd_every_epochs_lowX,
            rank_thresh=rank_thresh,
            device=device,
            P_left=P_left2,
            P_right=P_right2,
            P_left_perp=P_left2_perp,
            P_right_perp=P_right2_perp,
            model_null_proj=Q_x,
            model_null_label="model_nullX_norm",
            top_sv_every_epochs=top_sv_every_epochs_lowX,
            top_sv_k=top_sv_k,
            top_sv_method=top_sv_method_lowX,
        )

    shallow_model_lowX = nn.Linear(d, m, bias=False, device=device)
    with torch.no_grad():
        nn.init.normal_(shallow_model_lowX.weight, mean=0.0, std=0.01)

    ms_lowX = train_model(
        name="LowX-Shallow",
        model=shallow_model_lowX,
        get_A_fn=lambda model: model.weight,
        X=X_low,
        Y=Y_low,
        A_star=A_star_full,
        lr=shallow_lr_lowX,
        optimizer_name=shallow_optimizer_name,
        lr_decay_gamma=shallow_lr_decay_gamma_lowX,
        epochs=epochs_lowX,
        batch_size=batch_size,
        svd_every_epochs=svd_every_epochs_lowX,
        rank_thresh=rank_thresh,
        device=device,
        P_left=P_left2,
        P_right=P_right2,
        P_left_perp=P_left2_perp,
        P_right_perp=P_right2_perp,
        model_null_proj=Q_x,
        model_null_label="model_nullX_norm",
        top_sv_every_epochs=top_sv_every_epochs_lowX,
        top_sv_k=top_sv_k,
        top_sv_method=top_sv_method_lowX,
    )

    print(f"\n[Experiment 2 final spectra | noise={label_noise_std_lowX}]")
    print("A* (full):", format_top_svs(torch.sort(torch.linalg.svdvals(A_star_full), descending=True).values, k=10))
    for r in deep_widths:
        print(f"LowX-Deep(r={r}):", format_top_svs(deep_results_lowX[r]["top_sv"], k=10))
    print("LowX-Shallow:", format_top_svs(ms_lowX["top_sv"], k=10))

    print(f"\n[Experiment 2 identifiable vs null(X) decomposition | noise={label_noise_std_lowX}]")
    for r in deep_widths:
        A_hat = deep_models_lowX[r].end_to_end()
        support_fit_err = ((A_hat - A_star_full) @ P_x).norm().item()
        learnable_target_err = (A_hat @ P_x - A_star_learnable).norm().item()
        model_nullX_norm = (A_hat @ Q_x).norm().item()
        target_nullX_norm = A_star_unlearnable.norm().item()
        print(
            "LowX-Deep(r={}) support_fit_err={:.3e} learnable_target_err={:.3e} model_nullX_norm={:.3e} target_nullX_norm={:.3e}".format(
                r, support_fit_err, learnable_target_err, model_nullX_norm, target_nullX_norm
            )
        )

    A_hat_shallow_lowX = shallow_model_lowX.weight
    support_fit_err_shallow = ((A_hat_shallow_lowX - A_star_full) @ P_x).norm().item()
    learnable_target_err_shallow = (A_hat_shallow_lowX @ P_x - A_star_learnable).norm().item()
    model_nullX_norm_shallow = (A_hat_shallow_lowX @ Q_x).norm().item()
    target_nullX_norm_shallow = A_star_unlearnable.norm().item()
    print(
        "LowX-Shallow support_fit_err={:.3e} learnable_target_err={:.3e} model_nullX_norm={:.3e} target_nullX_norm={:.3e}".format(
            support_fit_err_shallow, learnable_target_err_shallow, model_nullX_norm_shallow, target_nullX_norm_shallow
        )
    )
