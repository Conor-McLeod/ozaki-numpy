"""Inverse scaling: CRT reconstruction of A_core @ B_core, then undo the shifts.

CUDA counterpart: par_gemmul8/include/ozaki/inverse_scaling.hpp
(inverse_scaling -> inverse_scaling_launch -> inverse_scaling_device, with
accumulator_double / accumulator_double2).

For each entry of C, with residues c_i = C_mid[i] and X = (A_core @ B_core):

    S    = sum_i qPi_i * c_i           X = S (mod M)   (crt_tables.py)
    quot = rint(S / M)
    X    = S - M * quot                the representative of S in [-M/2, M/2]
    C    = X * 2^(shiftA[row] + shiftB[col])

S is up to ~N * 128 * M, more than 53 bits. So S itself is rounded, and X
comes out with an absolute error of a few ulps of S, i.e. roughly
N * 128 * M * 2^-53. That error is tiny *relative to M*. It only matters
compared with the error phase D has already made by truncating A and B, which
is about sqrt(M) (one unit in A_core times a B_core of size ~sqrt(M)):

    N = 6:   CRT error ~2^5,  truncation error ~2^24  -> harmless
    N = 20 in plain doubles:  CRT error ~2^112, truncation ~2^77 -> would dominate

That's why, above 6 moduli, the constants are split into hi + lo doubles:

  * N <= 6 moduli: every qPi_i and M fit in a double exactly.
      S   = fma(qPi_i, c_i, S) over i          (accumulator_double)
      X   = fma(P, quot, S)                    with P = -M
  * N > 6: qPi_i and M are split into hi + lo doubles (crt_tables.py:
    qPi_double2, P_double2). S_hi is accumulated *exactly*, and the rounding
    moves into the much smaller S_lo sum.
      S_hi = fma(qPi_hi_i, c_i, S_hi); S_lo = fma(qPi_lo_i, c_i, S_lo)
      quot = rint(invP * S_hi)
      X    = fma(P_lo, quot, fma(P_hi, quot, S_hi) + S_lo)

numpy has no vectorised fma, and a*b + c rounds twice, which could
change the last bit. Python 3.13's math.fma rounds once, like the GPU.
np.frompyfunc applies it per element: slow, but it gives the same doubles as
the CUDA kernel.

Not reproduced: alpha/beta (C = alpha*AB + beta*C), float32 output, and the
"device pointer alpha" kernel variant. This only computes C = A @ B.
"""

import math

import numpy as np

from .crt_tables import (
    P_IS_DOUBLE,
    P_double,
    P_double2,
    check_num_moduli,
    invP,
    qPi_double,
    qPi_double2,
)

_fma_ufunc = np.frompyfunc(math.fma, 3, 1)


def fma(a, b, c):
    """Elementwise a*b + c with a single rounding (CUDA's fma)."""
    return _fma_ufunc(a, b, c).astype(np.float64)


def crt_reconstruct(C_mid, num_moduli):
    """X = A_core @ B_core, rebuilt from its residues C_mid, as float64
    (inverse_scaling_device, before the final scalbn)."""
    check_num_moduli(num_moduli)
    c = C_mid.astype(np.float64)

    if num_moduli <= P_IS_DOUBLE:
        # accumulator_double: TP = double.
        S = np.zeros(c.shape[1:])
        for i, q in enumerate(qPi_double(num_moduli)):
            S = fma(q, c[i], S)
        quot = np.rint(invP(num_moduli) * S)
        return fma(P_double(num_moduli), quot, S)

    # accumulator_double2: TP = double2.
    S_hi = np.zeros(c.shape[1:])
    S_lo = np.zeros(c.shape[1:])
    for i, (q_hi, q_lo) in enumerate(qPi_double2(num_moduli)):
        S_hi = fma(q_hi, c[i], S_hi)
        S_lo = fma(q_lo, c[i], S_lo)
    quot = np.rint(invP(num_moduli) * S_hi)
    P_hi, P_lo = P_double2(num_moduli)
    return fma(P_lo, quot, fma(P_hi, quot, S_hi) + S_lo)


def inverse_scaling(C_mid, shiftA, shiftB, num_moduli):
    """C = crt_reconstruct(C_mid) * 2^(shiftA[i] + shiftB[j]).

    CUDA: typed_scalbn(CRT_ans, shiftA[row] + shiftB[col]).
    """
    X = crt_reconstruct(C_mid, num_moduli)
    shift = shiftA[:, None].astype(np.int32) + shiftB[None, :].astype(np.int32)
    return np.ldexp(X, shift)
