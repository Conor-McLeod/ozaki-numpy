import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from fractions import Fraction

    from ozaki2 import ozaki_gemm
    from ozaki2.crt_tables import MODULI, crt_weights, log2P, modulus_product
    from ozaki2.inverse_scaling import crt_reconstruct, inverse_scaling
    from ozaki2.moduli import moduli_expand, moduli_gemm_loop
    from ozaki2.scaling import bound_gemm, extract_bound, refine_shifts, trunc_core

    np.set_printoptions(linewidth=120)
    return (
        Fraction,
        MODULI,
        bound_gemm,
        crt_reconstruct,
        crt_weights,
        extract_bound,
        inverse_scaling,
        log2P,
        mo,
        moduli_expand,
        moduli_gemm_loop,
        modulus_product,
        np,
        ozaki_gemm,
        refine_shifts,
        trunc_core,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ozaki Scheme II, step by step

    We want $C = AB$ for float64 matrices, but using only **int8 × int8 → int32**
    matrix multiplications (the fast tensor-core operation on GPUs).

    The plan:

    1. **Scale** each row of $A$ and each column of $B$ by a power of two, then
       truncate, giving integer matrices $A_\text{core}$ and $B_\text{core}$.
       The scales are chosen so that every entry of
       $X = A_\text{core} B_\text{core}$ satisfies $|X| < M/2$, where
       $M = p_0 p_1 \cdots p_{N-1}$ is the product of $N$ moduli.
    2. For each modulus $p_i$, take residues $A_\text{core} \bmod p_i$ and
       $B_\text{core} \bmod p_i$. They fit in int8, so the product
       $X \bmod p_i$ can be computed with **one int8 GEMM**.
    3. Rebuild $X$ from its $N$ residues with the **Chinese Remainder Theorem**,
       then undo the power-of-two scaling.

    Each cell below is one phase of `seq::ozaki_gemm` in par_gemmul8, calling
    the matching function from the `ozaki2` package. The docstrings there
    name the CUDA kernels.
    """)
    return


@app.cell
def _(mo):
    num_moduli_slider = mo.ui.slider(2, 20, value=8, label="num_moduli N", show_value=True)
    num_moduli_slider
    return (num_moduli_slider,)


@app.cell
def _(np):
    rng = np.random.default_rng(2003)
    A = 10.0 ** rng.uniform(-3, 3, size=(3, 4)) * rng.choice([-1, 1], size=(3, 4))
    B = 10.0 ** rng.uniform(-3, 3, size=(4, 2)) * rng.choice([-1, 1], size=(4, 2))
    print("A =\n", A)
    print("B =\n", B)
    return A, B


@app.cell
def _(MODULI, log2P, modulus_product, num_moduli_slider):
    N = num_moduli_slider.value
    M = modulus_product(N)
    print(f"moduli     : {MODULI[:N]}")
    print(f"M          = {M}  (~2^{M.bit_length() - 1})")
    print(f"log2P      = {log2P(N)}  (bit budget of one factor: log2(sqrt((M-1)/2)))")
    return M, N


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase A: first-pass shifts and int8 upper bounds

    For each row of $A$ (column of $B$), find the absolute max and choose
    `shift = 5 - floor(log2(amax))`, so that the max scaled by $2^\text{shift}$
    lands in $[32, 64)$. Then round $|A|$ **up** at that scale:
    `A_bound = ceil(|A| * 2^shift)`. That gives small nonnegative int8
    numbers, each at least as large as the scaled element it came from.
    """)
    return


@app.cell
def _(A, B, extract_bound):
    A_bound, shiftA0, B_bound, shiftB0 = extract_bound(A, B)
    print("shiftA0 =", shiftA0, "  shiftB0 =", shiftB0)
    print("A_bound =\n", A_bound)
    print("B_bound =\n", B_bound)
    return A_bound, B_bound, shiftA0, shiftB0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase B: bound the product with one int8 GEMM

    `C_hi = A_bound @ B_bound` bounds $|A|\,|B|$ entrywise (in the scaled units)
    and costs one cheap int8 GEMM instead of a floating-point one.
    """)
    return


@app.cell
def _(A_bound, B_bound, bound_gemm):
    C_hi_bound = bound_gemm(A_bound, B_bound)
    print("C_hi =\n", C_hi_bound)
    return (C_hi_bound,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase C: refine the shifts

    Each factor has a budget of `log2P` bits. Row $i$'s products use up to
    `log2(max C_hi[i, :])` of the combined budget, so row $i$ of $A$ can be
    scaled up by another `floor(log2P - log2(rowmax)/2)` bits (and column $j$ of
    $B$ likewise). This guarantees $|A_\text{core} B_\text{core}| < M/2$.

    **Sign flip:** after this phase, `shiftA` is stored *negated*. It is now the
    exponent that *undoes* the scaling: $A \approx A_\text{core} \cdot 2^{\text{shiftA}}$.
    """)
    return


@app.cell
def _(C_hi_bound, N, refine_shifts, shiftA0, shiftB0):
    shiftA, shiftB = refine_shifts(C_hi_bound, shiftA0, shiftB0, N)
    print("shiftA =", shiftA, "  shiftB =", shiftB)
    return shiftA, shiftB


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase D: truncate to integers

    `A_core = trunc(A * 2^-shiftA)`. These are big integers, stored exactly in
    float64 here (par_gemmul8 uses int32 / int64 / double depending on N).
    Truncation is where the scheme loses accuracy: everything below the last
    kept bit is dropped. More moduli → bigger $M$ → more bits kept.
    """)
    return


@app.cell
def _(A, B, M, np, shiftA, shiftB, trunc_core):
    A_core, B_core = trunc_core(A, shiftA, B, shiftB)
    # Shown as Python ints: they are exact integers, just stored in float64.
    _to_int = np.vectorize(int, otypes=[object])
    print("A_core =\n", _to_int(A_core))
    print("B_core =\n", _to_int(B_core))

    # Exact integer product, to check the |X| < M/2 guarantee.
    X_exact = _to_int(A_core) @ _to_int(B_core)
    print("\nmax |A_core @ B_core| / (M/2) =", float(max(abs(x) for x in X_exact.flat) / (M / 2)))
    return A_core, B_core, X_exact


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase E: residues

    For each modulus: `A_lo[i] = A_core mod p_i`, as the symmetric
    representative in $[-p_i/2, p_i/2]$, so it fits in an int8. $B$ likewise.
    Now $N$ small int8 matrices stand in for one big-integer matrix.
    """)
    return


@app.cell
def _(A_core, B_core, N, moduli_expand):
    A_lo, B_lo = moduli_expand(A_core, B_core, N)
    print("A_lo.shape =", A_lo.shape, " B_lo.shape =", B_lo.shape)
    print("A_lo[0] (mod 256) =\n", A_lo[0])
    print("A_lo[1] (mod 255) =\n", A_lo[1])
    return A_lo, B_lo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The moduli GEMM loop

    For each $i$: `C_hi = A_lo[i] @ B_lo[i]` (int8 GEMM, int32 result), then
    `C_mid[i] = C_hi mod p_i`. This is where almost all the time goes on a GPU.
    It's $N$ int8 GEMMs, independent of each other (which is what
    par_gemmul8's parallel version distributes across GPUs).

    Check: `C_mid[i]` really is the exact product `A_core @ B_core` mod $p_i$.
    """)
    return


@app.cell
def _(A_lo, B_lo, MODULI, X_exact, moduli_gemm_loop):
    C_mid = moduli_gemm_loop(A_lo, B_lo)
    print("C_mid.shape =", C_mid.shape)
    _ok = all(
        ((C_mid[_i].astype(object) - X_exact) % _p == 0).all()
        for _i, _p in enumerate(MODULI[: C_mid.shape[0]])
    )
    print("C_mid[i] == (A_core @ B_core) mod p_i for every i:", _ok)
    return (C_mid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inverse scaling: CRT, then undo the shifts

    With the CRT weights $q\Pi_i$ ($\equiv 1 \bmod p_i$, $\equiv 0 \bmod p_j$):

    $$S = \sum_i q\Pi_i \, c_i, \qquad X = S - M \cdot \operatorname{round}(S/M), \qquad C = X \cdot 2^{\text{shiftA}_i + \text{shiftB}_j}$$

    First with exact Python integers, then the way par_gemmul8 does it in
    doubles (double-double for $N > 6$).
    """)
    return


@app.cell
def _(C_mid, Fraction, M, N, X_exact, crt_reconstruct, crt_weights, np):
    _S = sum(_w * C_mid[_i].astype(object) for _i, _w in enumerate(crt_weights(N)))
    X_crt = np.vectorize(lambda s: s - M * round(Fraction(s, M)), otypes=[object])(_S)
    print("exact CRT == A_core @ B_core:", (X_crt == X_exact).all())

    X_float = crt_reconstruct(C_mid, N)
    _err = max(abs(Fraction(_g) - _x) for _g, _x in zip(X_float.flat, X_exact.flat))
    print("float CRT abs error:", float(_err), "  (relative to M:", float(_err / M), ")")
    return


@app.cell
def _(A, B, C_mid, N, inverse_scaling, np, ozaki_gemm, shiftA, shiftB):
    C = inverse_scaling(C_mid, shiftA, shiftB, N)
    print("C (Ozaki) =\n", C)
    print("A @ B     =\n", A @ B)
    # The phases above are exactly what ozaki_gemm chains together.
    assert np.array_equal(C, ozaki_gemm(A, B, N))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy vs number of moduli

    Relative error against the exact product (computed with fractions). Each
    extra modulus adds ~8 bits to $M$, so ~4 more bits per factor, until the
    float64 limit (~1e-16) is reached around $N \approx 14$.
    """)
    return


@app.cell
def _(A, B, Fraction, mo, np, ozaki_gemm):
    _F = np.vectorize(Fraction, otypes=[object])
    _exact = _F(A) @ _F(B)
    _scale = max(abs(_x) for _x in _exact.flat)

    def _relerr(C):
        return float(max(abs(Fraction(_c) - _x) for _c, _x in zip(C.flat, _exact.flat)) / _scale)

    _rows = [{"N": _n, "rel. error": f"{_relerr(ozaki_gemm(A, B, _n)):.2e}"} for _n in range(2, 21)]
    _rows.append({"N": "numpy A @ B", "rel. error": f"{_relerr(A @ B):.2e}"})
    mo.ui.table(_rows, selection=None)
    return


if __name__ == "__main__":
    app.run()
