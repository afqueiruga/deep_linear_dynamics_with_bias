#!/usr/bin/env python3
"""
Symbolic proof sketch for low-rank training dynamics of a 2-layer deep linear model.

The script derives symbolic identities with SymPy MatrixSymbols and writes a
Markdown report containing LaTeX equations.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sympy as sp


def md_block_math(expr):
    return "$$\n" + sp.latex(expr) + "\n$$\n"


def md_equation(label, expr):
    return f"**{label}**\n\n{md_block_math(expr)}"


def write_equation_with_note(f, label, expr, note):
    f.write(md_equation(label, expr))
    f.write(note.strip() + "\n\n")


def prove_matrix_dynamics():
    n, r = sp.symbols("n r", integer=True, positive=True)
    W = sp.MatrixSymbol("W", n, r)
    U = sp.MatrixSymbol("U", r, n)
    A_star = sp.MatrixSymbol("A_star", n, n)

    A = W * U
    E = A - A_star

    # Gradient flow for L = 1/2 ||WU - A_*||_F^2 in the whitened setting.
    dW = -E * U.T
    dU = -W.T * E

    # Product rule: dA/dt = (dW/dt)U + W(dU/dt)
    dA = dW * U + W * dU

    # Equivalent compact form often used in analyses.
    dA_expected = -E * (U.T * U) - (W * W.T) * E

    # For these expressions, structural equality is enough because dA_expected
    # is exactly the expanded product-rule substitution of dW and dU.
    assert sp.srepr(dA) == sp.srepr(dA_expected)

    return {
        "A": A,
        "E": E,
        "dW": dW,
        "dU": dU,
        "dA": dA,
    }


def prove_mode_dynamics():
    # Single singular mode in aligned coordinates:
    # target singular value sigma, current factors w,u, end-to-end mode s = w*u.
    sigma = sp.symbols("sigma", real=True)
    w, u = sp.symbols("w u", real=True)
    s = sp.symbols("s", real=True)

    dw = -(s - sigma) * u
    du = -w * (s - sigma)
    ds = sp.simplify(dw * u + w * du)

    # General mode ODE: ds = -(w^2 + u^2)(s - sigma)
    ds_expected = -(w**2 + u**2) * (s - sigma)
    assert sp.simplify(ds - ds_expected) == 0

    # Balanced regime: w = u and s = w^2.
    ds_balanced = sp.simplify(ds_expected.subs({u: w, s: w**2}))
    ds_balanced_in_s = sp.simplify(ds_balanced.subs({w**2: s}))

    # Logistic-type mode growth law.
    logistic_form = 2 * s * (sigma - s)
    assert sp.simplify(ds_balanced_in_s - logistic_form) == 0

    # Zero modes are invariant: if s(0)=0 then ds/dt=0 at s=0.
    assert sp.simplify(logistic_form.subs({s: 0})) == 0

    return {
        "dw": dw,
        "du": du,
        "ds": ds_expected,
        "ds_balanced": ds_balanced_in_s,
    }


def prove_output_bias_dynamics():
    """
    Bias-at-output model:
        y_hat = W U x + w

    For population squared loss with input mean mu and covariance I:
        L = 1/2 E || (A-A_*)x + (w-w_*) ||^2
          = 1/2 ||A-A_*||_F^2
            + <A-A_*, (w-w_*) mu^T>
            + 1/2 ||w-w_*||^2
            + const
    where A = WU.
    """
    n, r = sp.symbols("n r", integer=True, positive=True)
    W = sp.MatrixSymbol("W", n, r)
    U = sp.MatrixSymbol("U", r, n)
    A_star = sp.MatrixSymbol("A_star", n, n)
    w = sp.MatrixSymbol("w", n, 1)
    w_star = sp.MatrixSymbol("w_star", n, 1)
    mu = sp.MatrixSymbol("mu", n, 1)

    A = W * U
    E = A - A_star
    e = w - w_star

    # Gradient wrt A and w for non-centered inputs (mean mu, covariance I).
    gA = E + e * mu.T
    gw = E * mu + e

    # Chain-rule to factors.
    dW = -gA * U.T
    dU = -W.T * gA
    dw = -gw
    dA = dW * U + W * dU
    dA_expected = -gA * (U.T * U) - (W * W.T) * gA
    assert sp.srepr(dA) == sp.srepr(dA_expected)

    # Zero-mean specialization (mu=0): bias decouples from factor dynamics.
    dW_mu0 = -E * U.T
    dU_mu0 = -W.T * E
    dw_mu0 = -e

    return {
        "A": A,
        "E": E,
        "e": e,
        "mu": mu,
        "gA": gA,
        "gw": gw,
        "dW": dW,
        "dU": dU,
        "dw": dw,
        "dA": dA,
        "dW_mu0": dW_mu0,
        "dU_mu0": dU_mu0,
        "dw_mu0": dw_mu0,
    }


def prove_inner_bias_dynamics():
    """
    Inner + output bias model:
        y_hat = W (U x + u) + w = A x + b
    with
        A = WU,  b = Wu + w.
    """
    n, r = sp.symbols("n r", integer=True, positive=True)
    W = sp.MatrixSymbol("W", n, r)
    U = sp.MatrixSymbol("U", r, n)
    u = sp.MatrixSymbol("u", r, 1)
    w = sp.MatrixSymbol("w", n, 1)
    A_star = sp.MatrixSymbol("A_star", n, n)
    b_star = sp.MatrixSymbol("b_star", n, 1)
    mu = sp.MatrixSymbol("mu", n, 1)

    A = W * U
    b = W * u + w
    E = A - A_star
    e = b - b_star

    # Population gradients for E[xx^T]=I, E[x]=mu.
    gA = E + e * mu.T
    gb = E * mu + e

    # Chain-rule gradients wrt factors and biases.
    dW = -(gA * U.T + gb * u.T)
    dU = -W.T * gA
    du = -W.T * gb
    dw = -gb

    dA = dW * U + W * dU

    db = dW * u + W * du + dw

    # Centered-input specialization.
    dW_mu0 = -E * U.T - e * u.T
    dU_mu0 = -W.T * E
    du_mu0 = -W.T * e
    dw_mu0 = -e

    # Special subcase: centered inputs and no output bias (w = 0).
    e_wo = W * u - b_star
    dW_mu0_w0 = -E * U.T - e_wo * u.T
    dU_mu0_w0 = -W.T * E
    du_mu0_w0 = -W.T * e_wo

    return {
        "A": A,
        "b": b,
        "E": E,
        "e": e,
        "gA": gA,
        "gb": gb,
        "dW": dW,
        "dU": dU,
        "du": du,
        "dw": dw,
        "dA": dA,
        "db": db,
        "dW_mu0": dW_mu0,
        "dU_mu0": dU_mu0,
        "du_mu0": du_mu0,
        "dw_mu0": dw_mu0,
        "dW_mu0_w0": dW_mu0_w0,
        "dU_mu0_w0": dU_mu0_w0,
        "du_mu0_w0": du_mu0_w0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown+LaTeX proof report for deep linear low-rank dynamics."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output markdown path. Defaults to outputs/theory_sympy_low_rank_proof_<timestamp>.md",
    )
    args = parser.parse_args()

    matrix_result = prove_matrix_dynamics()
    mode_result = prove_mode_dynamics()
    bias_result = prove_output_bias_dynamics()
    inner_bias_result = prove_inner_bias_dynamics()
    sigma, s = sp.symbols("sigma s", real=True)
    off_target_decay = sp.simplify(mode_result["ds_balanced"].subs({sigma: 0}))

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("outputs") / f"theory_sympy_low_rank_proof_{ts}.md"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# SymPy Proof: Low-Rank Dynamics for 2-Layer Deep Linear Model\n\n")
        f.write("Model:\n\n")
        f.write("$$\n\\hat{y} = W U x\n$$\n\n")
        f.write("Whitened squared-loss setting:\n\n")
        f.write("$$\nL(A) = \\frac{1}{2}\\|A - A_*\\|_F^2, \\quad A = WU\n$$\n\n")

        f.write("## Matrix-Valued Gradient Flow\n\n")
        write_equation_with_note(
            f,
            "A := WU",
            matrix_result["A"],
            (
                "Interpretation: `A` is the end-to-end linear map realized by the two factors. "
                "Any rank structure in `A` must emerge through coupled updates of `W` and `U`."
            ),
        )
        write_equation_with_note(
            f,
            "E := A - A_*",
            matrix_result["E"],
            (
                "Interpretation: `E` is the training error operator in parameter space. "
                "All subsequent dynamics are driven by this residual."
            ),
        )
        write_equation_with_note(
            f,
            "dW/dt",
            matrix_result["dW"],
            (
                "Interpretation: `W` is updated by right-multiplying the error with `U^T`. "
                "Only directions represented through the current `U` receive strong updates."
            ),
        )
        write_equation_with_note(
            f,
            "dU/dt",
            matrix_result["dU"],
            (
                "Interpretation: `U` is updated by left-multiplying the error with `W^T`. "
                "Only directions represented through the current `W` are strongly amplified."
            ),
        )
        write_equation_with_note(
            f,
            "dA/dt",
            matrix_result["dA"],
            (
                "Conclusion: end-to-end error is preconditioned by `U^T U` and `W W^T`, not by identity. "
                "This induces anisotropic learning and favors subspaces already activated by the factors."
            ),
        )

        f.write("## Singular-Mode Dynamics (Aligned + Balanced Regime)\n\n")
        write_equation_with_note(
            f,
            "dw/dt",
            mode_result["dw"],
            (
                "Interpretation: each factor component grows or shrinks based on signed mismatch `(s-\\sigma)` "
                "scaled by the other factor."
            ),
        )
        write_equation_with_note(
            f,
            "du/dt",
            mode_result["du"],
            (
                "Interpretation: the two factors are dynamically coupled; mode activation requires both "
                "`w` and `u` to become non-negligible."
            ),
        )
        write_equation_with_note(
            f,
            "ds/dt",
            mode_result["ds"],
            (
                "Interpretation: mode speed is proportional to `(w^2+u^2)` and residual `(\\sigma-s)`. "
                "Small factors imply slow early-time dynamics."
            ),
        )
        write_equation_with_note(
            f,
            "Balanced law: ds/dt",
            mode_result["ds_balanced"],
            (
                "Conclusion: in the balanced regime this is logistic growth for each target-aligned mode. "
                "Tiny modes grow slowly, then accelerate, then saturate near `\\sigma`."
            ),
        )
        write_equation_with_note(
            f,
            "Off-target case (sigma = 0)",
            off_target_decay,
            (
                "Conclusion: modes outside the target subspace decay (`ds/dt=-2s^2 <= 0`) in this reduction, "
                "so the same dynamics that slow small modes also suppress non-target modes."
            ),
        )

        f.write("**Final takeaway**\n\n")
        f.write(
            "The symbolic derivation supports a low-rank inductive bias mechanism: "
            "target-aligned modes grow with logistic-type dynamics, while off-target modes decay "
            "in the balanced/aligned reduction.\n\n"
        )

        f.write("## Output-Bias Model: $\\hat{y} = WUx + w$\n\n")
        f.write(
            "Assume population squared loss with `E[xx^T]=I` and mean `E[x]=\\mu`, target "
            "`y_* = A_*x + w_*`, and definitions `A=WU`, `E=A-A_*`, `e=w-w_*`.\n\n"
        )
        write_equation_with_note(
            f,
            "Gradient wrt A",
            bias_result["gA"],
            (
                "Interpretation: compared to the no-bias case, there is an extra rank-1 coupling term "
                "`e\\mu^T`. Nonzero bias error and nonzero input mean tilt the factor updates."
            ),
        )
        write_equation_with_note(
            f,
            "Gradient wrt w",
            bias_result["gw"],
            (
                "Interpretation: bias update sees both direct bias error `e` and projection of matrix "
                "error through mean input `E\\mu`."
            ),
        )
        write_equation_with_note(
            f,
            "dW/dt",
            bias_result["dW"],
            (
                "Conclusion: factor dynamics are now driven by `E + e\\mu^T`; this is the precise change "
                "from the pure deep linear model."
            ),
        )
        write_equation_with_note(
            f,
            "dU/dt",
            bias_result["dU"],
            (
                "Conclusion: same coupling appears in `U` updates, so bias mismatch can feed back into "
                "mode growth when `\\mu \\neq 0`."
            ),
        )
        write_equation_with_note(
            f,
            "dw/dt",
            bias_result["dw"],
            (
                "Conclusion: bias is not generally independent unless inputs are centered "
                "(`\\mu=0`)."
            ),
        )
        write_equation_with_note(
            f,
            "dA/dt",
            bias_result["dA"],
            (
                "Interpretation: the end-to-end map keeps the same Gram-preconditioned structure, but "
                "with effective residual `E + e\\mu^T`."
            ),
        )

        f.write("### Centered-input specialization (`\\mu = 0`)\n\n")
        write_equation_with_note(
            f,
            "dW/dt |_{mu=0}",
            bias_result["dW_mu0"],
            "Same as the original deep linear model.",
        )
        write_equation_with_note(
            f,
            "dU/dt |_{mu=0}",
            bias_result["dU_mu0"],
            "Same as the original deep linear model.",
        )
        write_equation_with_note(
            f,
            "dw/dt |_{mu=0}",
            bias_result["dw_mu0"],
            (
                "Bias decouples and follows first-order linear decay to `w_*`. If `w_*=0`, then "
                "`w(t)=w(0)e^{-t}`."
            ),
        )

        f.write("### Initialization cases (for `w_*=0`)\n\n")
        f.write(
            "1. `w(0)=0`: with centered inputs, `dw/dt=-w` implies `w(t)=0` for all `t`. "
            "So `W,U` dynamics are exactly unchanged from the no-bias deep linear model.\n\n"
        )
        f.write(
            "2. `w(0)` random small (same scale as factor init): with centered inputs, "
            "`w(t)=w(0)e^{-t}` and decays on O(1) time; `W,U` still follow the original deep linear "
            "dynamics. With non-centered inputs (`mu!=0`), this initial bias transient induces an extra "
            "term `e\\mu^T` in factor updates, creating temporary coupling between bias and low-rank mode "
            "evolution.\n\n"
        )
        f.write(
            "**What changes vs deep linear without bias?**\n\n"
            "- If inputs are centered, only an independent bias-decay channel is added.\n"
            "- If inputs are not centered, bias error couples into factor learning through `e\\mu^T`, "
            "so early dynamics can differ until bias error shrinks.\n\n"
        )

        f.write("## Inner-Bias Model: $\\hat{y} = W(Ux + u) + w$\n\n")
        f.write(
            "Use `A=WU`, `b=Wu+w`, `E=A-A_*`, `e=b-b_*`, with population assumptions "
            "`E[xx^T]=I`, `E[x]=\\mu`.\n\n"
        )
        write_equation_with_note(
            f,
            "g_A",
            inner_bias_result["gA"],
            "Matrix gradient matches the affine form: residual plus mean-coupled bias term.",
        )
        write_equation_with_note(
            f,
            "g_b",
            inner_bias_result["gb"],
            "Bias gradient mixes direct bias error and projected matrix error through `\\mu`.",
        )
        write_equation_with_note(
            f,
            "dW/dt",
            inner_bias_result["dW"],
            (
                "Key new term vs output-bias-only: `-gb u^T`. Because `b=Wu+w`, `W` now gets a second "
                "feedback path through inner bias `u`."
            ),
        )
        write_equation_with_note(
            f,
            "dU/dt",
            inner_bias_result["dU"],
            "Same structure as before, but driven by `g_A` that already contains bias coupling.",
        )
        write_equation_with_note(
            f,
            "du/dt",
            inner_bias_result["du"],
            (
                "Inner bias follows `-W^T g_b`: it is amplified/suppressed through current output "
                "directions encoded in `W`."
            ),
        )
        write_equation_with_note(
            f,
            "dw/dt",
            inner_bias_result["dw"],
            "Output bias still follows `-g_b`, but now shares responsibility for fitting `b_*` with `Wu`.",
        )
        write_equation_with_note(
            f,
            "dA/dt",
            inner_bias_result["dA"],
            (
                "Compared to deep linear, there is an extra coupling term `-g_b u^T U`. "
                "Thus inner bias can directly perturb singular-mode evolution of `A`."
            ),
        )
        write_equation_with_note(
            f,
            "db/dt",
            inner_bias_result["db"],
            (
                "Affine term dynamics are coupled through all parameters (`W,U,u,w`), so translation "
                "fitting is no longer an independent channel."
            ),
        )

        f.write("### Centered-input specialization (`\\mu = 0`)\n\n")
        write_equation_with_note(
            f,
            "dW/dt |_{mu=0}",
            inner_bias_result["dW_mu0"],
            (
                "Unlike output-bias-only, factor updates are still modified by bias error via "
                "`-e u^T`."
            ),
        )
        write_equation_with_note(
            f,
            "dU/dt |_{mu=0}",
            inner_bias_result["dU_mu0"],
            "Same as deep linear in form (`-W^T E`).",
        )
        write_equation_with_note(
            f,
            "du/dt |_{mu=0}",
            inner_bias_result["du_mu0"],
            "Inner bias remains coupled to current output basis (`W^T`).",
        )
        write_equation_with_note(
            f,
            "dw/dt |_{mu=0}",
            inner_bias_result["dw_mu0"],
            "Output bias still tracks total affine error `e`.",
        )

        f.write("### Inner-bias-only subcase (`w=0`, `\\mu=0`)\n\n")
        write_equation_with_note(
            f,
            "dW/dt |_{mu=0,w=0}",
            inner_bias_result["dW_mu0_w0"],
            "Even without output bias, inner bias affects factor learning through `-(Wu-b_*)u^T`.",
        )
        write_equation_with_note(
            f,
            "dU/dt |_{mu=0,w=0}",
            inner_bias_result["dU_mu0_w0"],
            "Unchanged deep-linear-form matrix channel.",
        )
        write_equation_with_note(
            f,
            "du/dt |_{mu=0,w=0}",
            inner_bias_result["du_mu0_w0"],
            "Inner bias update is shaped by `W^T` and current affine mismatch.",
        )

        f.write("### Initialization cases (centered inputs, `w=0`)\n\n")
        f.write(
            "1. `u(0)=0`: initially the extra `-e u^T` term in `dW/dt` is zero, so the matrix channel "
            "starts like deep linear; however `du/dt = -W^T(Wu-b_*)` is generally nonzero, so `u` turns on "
            "and coupling appears shortly after unless `W^T b_* = 0`.\n\n"
        )
        f.write(
            "2. `u(0)` random small (same scale as factor init): coupling is present from the start through "
            "`-e u^T` and `-g_b u^T U`; this can perturb early singular-mode ordering relative to deep "
            "linear, especially when affine target magnitude is non-negligible.\n\n"
        )
        f.write(
            "**What changes vs deep linear / output-bias-only?**\n\n"
            "- Output-bias-only adds an affine channel that can decouple when `\\mu=0`.\n"
            "- Inner bias introduces multiplicative coupling through `Wu`, so even at `\\mu=0` the factor "
            "dynamics are generally modified once `u` is nonzero.\n\n"
        )

        f.write("All symbolic checks passed.\n")

    print(f"Wrote markdown proof to: {output_path}")


if __name__ == "__main__":
    main()
