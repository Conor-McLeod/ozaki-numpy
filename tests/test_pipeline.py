"""End-to-end checks of the invariants each phase is there to guarantee."""

import math
from fractions import Fraction

import numpy as np
import pytest

from ozaki2 import ozaki_gemm
from ozaki2.crt_tables import MODULI, crt_weights, modulus_product
from ozaki2.inverse_scaling import crt_reconstruct

NS = [2, 3, 6, 7, 10, 14, 20]


def wide_range(rng, shape, lo=-5, hi=5):
    """Random signed values spanning 10^lo .. 10^hi, as in ozaki_scheme.py."""
    return 10.0 ** rng.uniform(lo, hi, size=shape) * rng.choice([-1, 1], size=shape)


def exact_int_matmul(X, Y):
    """Exact product of integer-valued float matrices, as Python ints."""
    Xo = np.vectorize(int, otypes=[object])(X)
    Yo = np.vectorize(int, otypes=[object])(Y)
    return Xo @ Yo


def exact_matmul(A, B):
    """Exact A @ B as Fractions."""
    Af = np.vectorize(Fraction, otypes=[object])(A)
    Bf = np.vectorize(Fraction, otypes=[object])(B)
    return Af @ Bf


@pytest.fixture(params=NS)
def case(request):
    rng = np.random.default_rng(2003 + request.param)
    A = wide_range(rng, (7, 9))
    B = wide_range(rng, (9, 5))
    return request.param, A, B, ozaki_gemm(A, B, request.param, trace=True)


def test_bounds_bound(case):
    """Phase A/B: C_hi_bound >= |A| @ |B| in the phase-A scaled units."""
    n, A, B, tr = case
    Asc = np.ldexp(np.abs(A), tr.shiftA0[:, None].astype(int))
    Bsc = np.ldexp(np.abs(B), tr.shiftB0[None, :].astype(int))
    assert (tr.A_bound >= Asc).all() and (tr.B_bound >= Bsc).all()
    assert (tr.C_hi_bound >= Asc @ Bsc).all()


def test_core_product_below_half_M(case):
    """Phases A-D exist to make |A_core @ B_core| < M/2."""
    n, A, B, tr = case
    X = exact_int_matmul(tr.A_core, tr.B_core)
    M = modulus_product(n)
    assert all(2 * abs(x) < M for x in X.flat)


def test_residues(case):
    """Phase E + GEMM loop: C_mid[i] = (A_core @ B_core) mod p_i."""
    n, A, B, tr = case
    X = exact_int_matmul(tr.A_core, tr.B_core)
    for i, p in enumerate(MODULI[:n]):
        got = tr.C_mid[i].astype(object)
        assert all((g - x) % p == 0 for g, x in zip(got.flat, X.flat))
        # and they are symmetric residues stored in int8
        assert tr.C_mid[i].dtype == np.int8


def test_exact_crt_recovers_core_product(case):
    """With exact integers, the CRT gives back A_core @ B_core."""
    n, A, B, tr = case
    X = exact_int_matmul(tr.A_core, tr.B_core)
    M = modulus_product(n)
    S = sum(w * tr.C_mid[i].astype(object) for i, w in enumerate(crt_weights(n)))
    rec = np.vectorize(lambda s: s - M * round(Fraction(s, M)), otypes=[object])(S)
    assert (rec == X).all()


def test_float_crt_error_is_tiny_relative_to_M(case):
    """The float CRT in inverse scaling gives A_core @ B_core up to an error
    that is tiny *relative to M*: the rounding of the big sum S (N <= 6) or of
    S_lo (N > 6). For N <= 6 that is a few units, for X up to ~M/2."""
    n, A, B, tr = case
    X = exact_int_matmul(tr.A_core, tr.B_core)
    got = crt_reconstruct(tr.C_mid, n)
    M = modulus_product(n)
    for g, x in zip(got.flat, X.flat):
        assert abs(Fraction(g) - x) <= Fraction(M, 2**40) + Fraction(np.spacing(abs(float(x))))


def test_accuracy_improves_with_moduli():
    rng = np.random.default_rng(7)
    A = wide_range(rng, (6, 12), -2, 2)
    B = wide_range(rng, (12, 4), -2, 2)
    exact = exact_matmul(A, B)
    scale = float(max(abs(x) for x in exact.flat))

    def err(C):
        return max(abs(Fraction(c) - x) for c, x in zip(C.flat, exact.flat)) / scale

    errs = [float(err(ozaki_gemm(A, B, n))) for n in range(2, 21)]
    # 4 more moduli always buy a lot more accuracy, until the float64 floor.
    for a, b in zip(errs, errs[4:]):
        assert b < a / 100 or b < 1e-15
    assert errs[-1] <= 4 * float(err(A @ B)) + 1e-16


def test_zero_rows_and_columns():
    rng = np.random.default_rng(1)
    A = wide_range(rng, (4, 5))
    B = wide_range(rng, (5, 3))
    A[1] = 0.0
    B[:, 2] = 0.0
    for n in NS:
        C = ozaki_gemm(A, B, n)
        assert (C[1] == 0).all() and (C[:, 2] == 0).all()
        assert np.isfinite(C).all()


def test_huge_dynamic_range_in_one_row():
    """ldexp of 1e-300 at the row's scale underflows. The bound must still be 1.

    Every term of this product (1e300*1e-300, 1e-300*1e300, 1*1) is 1, so
    A @ B = 3, and a float64 GEMM gets 3 exactly. But Ozaki Scheme II gives
    each row of A and column of B a *single* scale, set by its largest
    entry, and keeps a fixed number of bits (~77 at 20 moduli) below it. The
    1e-300 and 1.0 entries sit ~2000 bits below the 1e300 in the same
    row/column, so they truncate to zero and C = 0.

    The error is small relative to rowmax|A| * colmax|B| (1e600 here), not
    relative to |A| @ |B|. That's an inherent weakness of the scheme on
    inputs whose rows/columns span a huge exponent range.
    """
    A = np.array([[1e300, 1e-300, 1.0]])
    B = np.array([[1e-300], [1e300], [1.0]])
    tr = ozaki_gemm(A, B, 20, trace=True)
    assert (tr.A_bound[0] >= 1).all()
    assert tr.C[0, 0] == 0.0
    # 1e600 overflows a double, so check the bound in exact arithmetic.
    bound = Fraction(np.abs(A).max()) * Fraction(np.abs(B).max()) / 2**70
    assert abs(Fraction(tr.C[0, 0]) - 3) <= bound


def test_matches_numpy_on_normal_data():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((40, 64))
    B = rng.standard_normal((64, 30))
    ref = A @ B
    C = ozaki_gemm(A, B, 16)
    assert np.abs(C - ref).max() <= 1e-13 * np.abs(A).max() * np.abs(B).max() * 64


def test_rejects_bad_input():
    A = np.ones((2, 2))
    with pytest.raises(ValueError):
        ozaki_gemm(A, A, 1)
    with pytest.raises(ValueError):
        ozaki_gemm(A, A, 21)
    with pytest.raises(TypeError):
        ozaki_gemm(A.astype(np.float32), A, 4)
    with pytest.raises(ValueError):
        ozaki_gemm(np.ones((2, 3)), A, 4)


def test_math_fma_available():
    assert hasattr(math, "fma")
