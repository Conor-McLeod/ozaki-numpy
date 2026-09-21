import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from fractions import Fraction

    from ozaki2 import ozaki_gemm
    from ozaki2.crt_tables import MODULI, crt_weights, log2P, moduli_product
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
        moduli_product,
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

    1. **Scale** each row of $A$ and each column of $B$ by a power of two
       (scale vectors $\mu$, $\nu$), then truncate, giving integer matrices
       $A' = \operatorname{trunc}(\operatorname{diag}(\mu) A)$ and
       $B' = \operatorname{trunc}(B \operatorname{diag}(\nu))$. The scales are
       chosen so that $2 \sum_h |a'_{ih}| |b'_{hj}| < \mathcal{P}$, where
       $\mathcal{P} = p_1 p_2 \cdots p_N$ is the product of $N$ moduli.
    2. For each modulus $p_i$, take residues $A'_i = \operatorname{rmod}(A', p_i)$
       and $B'_i = \operatorname{rmod}(B', p_i)$. They fit in int8, so
       $A'B' \bmod p_i$ can be computed with **one int8 GEMM**.
    3. Rebuild $C'' = A'B'$ from its $N$ residues with the **Chinese Remainder
       Theorem**, then undo the scaling: $C = \operatorname{diag}(\mu^{-1})\, C'' \operatorname{diag}(\nu^{-1})$.

    The notation is the paper's (Uchino, Ozaki, Imamura, SC Workshops '25,
    `ozaki2.pdf`). Each cell below is one phase of `seq::ozaki_gemm` in
    par_gemmul8, calling the matching function from the `ozaki2` package. The
    variable names are par_gemmul8's (`A_core` for $A'$, `A_lo` for $A'_i$, …),
    and the docstrings name the CUDA kernels.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As part of the Chinese Remainder Theorem we use the . When we say "moduli", we are referring to the integers that may be used for $m$ in each congruence

    The moduli are taken from the existing set of: 256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193, 191, 181, 179, 173. These are largest 20 pairwise coprime moduli smaller than or equal to 256.

    They form a pairwise coprime set, meaning that no pair of integers taken from this set share any non-zero prime factors. This is a property required for the Chinese Remainder Theorem to hold.

    $\mathcal{P}$ is the extremely large number we get from multiplying all the pairwise coprime moduli together.
    $$\mathcal{P} = \prod_{i=1}^N p_i$$

    But why this set of pairwise coprime integers specifically? Theyy

    $\log_2(\sqrt{\frac{\mathcal{P}-1}{2}})$ is `log2P`.
    - So `log20` is the number you have to raise 2 to to get $\sqrt{\frac{\mathcal{P}-1}{2}}$.
    - So by definition if you raise 2 to $\log_2(\sqrt{\frac{\mathcal{P}-1}{2}})$ you get $\sqrt{\frac{\mathcal{P}-1}{2}}$.

    Scaling picks shifts so that each each entry of $A_{core} B_{core}$ is at most `2^log2P * 2^log2P`

    $$2^{\log_2(\sqrt{\frac{\mathcal{P}-1}{2}})} \cdot 2^{\log_2(\sqrt{\frac{\mathcal{P}-1}{2}})} = \sqrt{\frac{\mathcal{P}-1}{2}} \ \cdot \sqrt{\frac{\mathcal{P}-1}{2}} = \frac{\mathcal{P} - 1}{2}$$
    """)
    return


@app.cell
def _(MODULI, log2P, moduli_product, num_moduli_slider):
    N = num_moduli_slider.value
    # The paper's 𝒫 (positive). Not called P, because par_gemmul8's `P` table
    # holds -𝒫.
    calP = moduli_product(N)
    print(f"moduli     : {MODULI[:N]}")
    print(f"𝒫          = {calP}  (~2^{calP.bit_length() - 1})")
    print(f"log2P      = {log2P(N)}  (𝒫'_accu, bit budget of one factor: log2(sqrt((𝒫-1)/2)))")
    return N, calP


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase A: first-pass shifts and int8 upper bounds

    For each row of $A$ (column of $B$), find the absolute max and choose
    $\mu'_i = 2^{5 - \lfloor \log_2 \max_h |a_{ih}| \rfloor}$, so that the max
    scaled by $\mu'_i$ lands in $[32, 64)$. Then round $|A|$ **up** at that scale:
    $\bar{A} = \lceil \operatorname{diag}(\mu') |A| \rceil$. That gives small
    nonnegative int8 numbers, each at least as large as the scaled element it
    came from.

    par_gemmul8: `shiftA0` holds the exponent ($\mu'_i = 2^\text{shiftA0[i]}$),
    and `A_bound` is $\bar{A}$.
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

    $\bar{C} = \bar{A}\bar{B}$ (par_gemmul8: `C_hi`) bounds
    $\operatorname{diag}(\mu')\,|A|\,|B| \operatorname{diag}(\nu')$ entrywise, and
    costs one cheap int8 GEMM instead of a floating-point one.
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

    Each factor has a budget of $\mathcal{P}'_\text{accu}$ (`log2P`) bits. Row
    $i$'s products use up to $\log_2 \max_h \bar{c}_{ih}$ of the combined
    budget, so row $i$ of $A$ can be scaled up by more bits:

    $$\mu_i = \mu'_i \cdot 2^{\lfloor \mathcal{P}'_\text{accu} - 0.51 \log_2 \max_h \bar{c}_{ih} \rfloor}$$

    (and $\nu_j$ likewise). This guarantees $2 \sum_h |a'_{ih}| |b'_{hj}| < \mathcal{P}$.
    The paper uses 0.51. par_gemmul8's code uses $0.5 + 3 \cdot 2^{-23}$, which
    keeps a few more bits, and we follow the code.

    **Sign flip:** after this phase, `shiftA` is stored *negated*: it is the
    exponent of $\mu^{-1}$, the one that *undoes* the scaling
    ($\mu_i = 2^{-\text{shiftA[i]}}$, and $A \approx A' \cdot 2^{\text{shiftA}}$).
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

    $A' = \operatorname{trunc}(\operatorname{diag}(\mu) A)$ (par_gemmul8:
    `A_core`). These are big integers, stored exactly in float64 here
    (par_gemmul8 uses int32 / int64 / double depending on N). Truncation is
    where the scheme loses accuracy: everything below the last kept bit is
    dropped. More moduli → bigger $\mathcal{P}$ → more bits kept.
    """)
    return


@app.cell
def _(A, B, calP, np, shiftA, shiftB, trunc_core):
    A_core, B_core = trunc_core(A, shiftA, B, shiftB)
    # Shown as Python ints: they are exact integers, just stored in float64.
    _to_int = np.vectorize(int, otypes=[object])
    print("A_core =\n", _to_int(A_core))
    print("B_core =\n", _to_int(B_core))

    # Exact integer product A'B', and a check of the paper's eq. 3.
    X_exact = _to_int(A_core) @ _to_int(B_core)
    _abs_sum = abs(_to_int(A_core)) @ abs(_to_int(B_core))
    print("\nmax 2 * sum_h |a'_ih| |b'_hj| / 𝒫 =", float(max(2 * _x for _x in _abs_sum.flat) / calP), "(must be < 1)")
    return A_core, B_core, X_exact


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase E: residues

    For each modulus: $A'_i = \operatorname{rmod}(A', p_i)$ with
    $\operatorname{rmod}(x, p) = x - p \cdot \operatorname{round}(x/p)$, the
    representative in $[-p_i/2, p_i/2]$, so it fits in an int8. $B$ likewise
    (par_gemmul8: `A_lo[i]`, `B_lo[i]`). Now $N$ small int8 matrices stand in
    for one big-integer matrix.
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

    For each $i$: $C'_i = A'_i B'_i$ (int8 GEMM, int32 result; par_gemmul8:
    `C_hi`), then reduce it mod $p_i$ again (par_gemmul8: `C_mid[i]`). This is
    where almost all the time goes on a GPU. It's $N$ int8 GEMMs, independent
    of each other (which is what par_gemmul8's parallel version distributes
    across GPUs).

    The paper reduces to *unsigned* $U_i = \operatorname{mod}(C'_i, p_i) \in [0, p_i)$.
    par_gemmul8 uses the *signed* $\operatorname{rmod}$. Both are the same residue.

    Check: `C_mid[i]` really is the exact product $A'B'$ mod $p_i$.
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
    print("C_mid[i] == A'B' mod p_i for every i:", _ok)
    return (C_mid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inverse scaling: CRT, then undo the shifts

    With $q_i$ the inverse of $\mathcal{P}/p_i$ mod $p_i$, the CRT weights
    $\frac{\mathcal{P}}{p_i} q_i$ (par_gemmul8: `qPi`) are $\equiv 1 \bmod p_i$ and
    $\equiv 0 \bmod p_j$. So with $U_i$ = `C_mid[i]`:

    $$C' = \sum_i \frac{\mathcal{P}}{p_i} q_i \, U_i, \qquad C'' = C' - \mathcal{P} \cdot \operatorname{round}(C'/\mathcal{P}), \qquad C = \operatorname{diag}(\mu^{-1})\, C'' \operatorname{diag}(\nu^{-1})$$

    First with exact Python integers, then the way par_gemmul8 does it in
    doubles (split into two doubles, $s_{i1} + s_{i2}$, for $N > 6$).
    """)
    return


@app.cell
def _(C_mid, Fraction, N, X_exact, calP, crt_reconstruct, crt_weights, np):
    _C1 = sum(_w * C_mid[_i].astype(object) for _i, _w in enumerate(crt_weights(N)))
    _C2 = np.vectorize(lambda c: c - calP * round(Fraction(c, calP)), otypes=[object])(_C1)
    print("exact CRT C'' == A'B':", (_C2 == X_exact).all())

    _C2_float = crt_reconstruct(C_mid, N)
    _err = max(abs(Fraction(_g) - _x) for _g, _x in zip(_C2_float.flat, X_exact.flat))
    print("float CRT abs error:", float(_err), "  (relative to 𝒫:", float(_err / calP), ")")
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
    extra modulus adds ~8 bits to $\mathcal{P}$, so ~4 more bits per factor, until the
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
