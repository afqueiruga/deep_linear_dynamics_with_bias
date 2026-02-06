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
        f.write(md_equation("A := WU", matrix_result["A"]) + "\n")
        f.write(md_equation("E := A - A_*", matrix_result["E"]) + "\n")
        f.write(md_equation("dW/dt", matrix_result["dW"]) + "\n")
        f.write(md_equation("dU/dt", matrix_result["dU"]) + "\n")
        f.write(md_equation("dA/dt", matrix_result["dA"]) + "\n")

        f.write("## Singular-Mode Dynamics (Aligned + Balanced Regime)\n\n")
        f.write(md_equation("dw/dt", mode_result["dw"]) + "\n")
        f.write(md_equation("du/dt", mode_result["du"]) + "\n")
        f.write(md_equation("ds/dt", mode_result["ds"]) + "\n")
        f.write(md_equation("Balanced law: ds/dt", mode_result["ds_balanced"]) + "\n")

        f.write("All symbolic checks passed.\n")

    print(f"Wrote markdown proof to: {output_path}")


if __name__ == "__main__":
    main()
