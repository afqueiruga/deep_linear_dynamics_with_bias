#!/usr/bin/env python3
"""
Symbolic proof sketch for low-rank training dynamics of a 2-layer deep linear model

    y_hat = W U x

under squared loss in the whitened setting:

    L(A) = 1/2 ||A - A_*||_F^2,  A = WU.

The script uses SymPy MatrixSymbol expressions to derive:
1) Gradient-flow updates for W and U.
2) The induced ODE for A = WU.
3) Per-singular-mode logistic dynamics in the balanced diagonal regime.
"""

import sympy as sp


def pprint_expr(label, expr):
    print(label)
    sp.pprint(expr, use_unicode=True)
    print()


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
    sp.init_printing(use_unicode=True)
    matrix_result = prove_matrix_dynamics()
    mode_result = prove_mode_dynamics()

    print("=== Matrix-valued gradient-flow proof (2-layer deep linear) ===")
    pprint_expr("A :=", matrix_result["A"])
    pprint_expr("E :=", matrix_result["E"])
    pprint_expr("dW/dt =", matrix_result["dW"])
    pprint_expr("dU/dt =", matrix_result["dU"])
    pprint_expr("dA/dt =", matrix_result["dA"])

    print("=== Singular-mode dynamics (aligned + balanced regime) ===")
    pprint_expr("dw/dt =", mode_result["dw"])
    pprint_expr("du/dt =", mode_result["du"])
    pprint_expr("ds/dt =", mode_result["ds"])
    pprint_expr("Balanced law: ds/dt =", mode_result["ds_balanced"])

    print("All symbolic checks passed.")


if __name__ == "__main__":
    main()
