"""Inverse scaling: CRT reconstruction of A' @ B', then undo the scaling.

Paper: Section 3 steps 3-4 (eqs. 4-5), Section 4.3 (eq. 8), Algorithm 1
lines 8-12. CUDA counterpart: par_gemmul8/include/ozaki/inverse_scaling.hpp
(inverse_scaling -> inverse_scaling_launch -> inverse_scaling_device, with
accumulator_double / accumulator_double2).

For each entry, with residues U_i = C_mid[i] (paper U_i, but signed here; see
moduli.py) and CRT weights w_i = 𝒫/p_i * q_i (crt_tables.py):

    C'   = sum_i w_i * U_i                  C' = A'B' (mod 𝒫)          eq. 4
    Q    = round(C' / 𝒫)                                              eq. 8
    C''  = C' - 𝒫 * Q = rmod(C', 𝒫)         = A'B' exactly, as |A'B'| < 𝒫/2  eq. 5
    C    = diag(mu^-1) C'' diag(nu^-1)      = C'' * 2^(shiftA[i] + shiftB[j])

C' is up to ~N * 128 * 𝒫, more than 53 bits, so it can't be held exactly in a
double. It doesn't need to be. Only C'' (about the size of A'B') has to come
out accurately, and an fma does the subtraction with a single rounding. The
rounding of C' itself gives C'' an absolute error of about N * 128 * 𝒫 * 2^-53.
That is tiny *relative to 𝒫*. It only matters compared with the error phase D
already made by truncating A and B, which is about sqrt(𝒫) (one unit in A'
times a B' entry of size ~sqrt(𝒫)):

    N = 6:   CRT error ~2^5,  truncation error ~2^24  -> harmless
    N = 20 in plain doubles:  CRT error ~2^112, truncation ~2^77 -> would dominate

That's why, above 6 moduli, the constants are split in two (paper eq. 6):
w_i = s_i1 + s_i2 and 𝒫 = 𝒫_1 + 𝒫_2.

  * N <= 6: every w_i and 𝒫 fit in one double exactly (accumulator_double).
      C'  = fma(w_i, U_i, C') over i
      Q   = rint(𝒫_inv * C')
      C'' = fma(-𝒫, Q, C')
  * N > 6 (accumulator_double2; Algorithm 1 lines 8-11):
      C'(1) = sum_i s_i1 * U_i     exact: the s_i1 share a common power of two
      C'(2) = sum_i s_i2 * U_i     the small, rounded part
      Q     = rint(𝒫_inv * C'(1))
      C''   = fma(-𝒫_2, Q, fma(-𝒫_1, Q, C'(1)) + C'(2))

par_gemmul8's tables store -𝒫 as `P`, so its code reads fma(P.y, quot, ...).
Its names for C'(1), C'(2), Q, C'' are C64f.x, C64f.y, quot, CRT_ans.

numpy has no vectorised fma, and a*b + c rounds twice, which could
change the last bit. Python 3.13's math.fma rounds once, like the GPU.
np.frompyfunc applies it per element: slow, but it gives the same doubles as
the CUDA kernel.

Not reproduced: alpha/beta (C = alpha*AB + beta*C), float32 output (SGEMM
emulation), and the "device pointer alpha" kernel variant. This only
computes C = A @ B.
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
    """C'' = A' @ B', rebuilt from its residues C_mid, as float64
    (inverse_scaling_device up to CRT_ans, before the final scalbn)."""
    check_num_moduli(num_moduli)
    U = C_mid.astype(np.float64)
    P_inv = invP(num_moduli)

    if num_moduli <= P_IS_DOUBLE:
        # accumulator_double: TP = double.
        C1 = np.zeros(U.shape[1:])
        for i, w in enumerate(qPi_double(num_moduli)):
            C1 = fma(w, U[i], C1)
        Q = np.rint(P_inv * C1)
        minus_P = P_double(num_moduli)
        return fma(minus_P, Q, C1)

    # accumulator_double2: TP = double2.
    C1 = np.zeros(U.shape[1:])  # C'(1)
    C2 = np.zeros(U.shape[1:])  # C'(2)
    for i, (s1, s2) in enumerate(qPi_double2(num_moduli)):
        C1 = fma(s1, U[i], C1)
        C2 = fma(s2, U[i], C2)
    Q = np.rint(P_inv * C1)
    minus_P1, minus_P2 = P_double2(num_moduli)
    return fma(minus_P2, Q, fma(minus_P1, Q, C1) + C2)


def inverse_scaling(C_mid, shiftA, shiftB, num_moduli):
    """C = diag(mu^-1) C'' diag(nu^-1) = C'' * 2^(shiftA[i] + shiftB[j])
    (Algorithm 1 line 12; CUDA: typed_scalbn(CRT_ans, shiftA[row] +
    shiftB[col]))."""
    C_crt = crt_reconstruct(C_mid, num_moduli)
    shift = shiftA[:, None].astype(np.int32) + shiftB[None, :].astype(np.int32)
    return np.ldexp(C_crt, shift)
