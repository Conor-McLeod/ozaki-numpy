"""Phase E and the moduli GEMM loop: products of big integers as int8 GEMMs.

Paper: Section 3 step 3 (eq. 4), Algorithm 1 lines 4-7, Sections 4.2-4.3.
CUDA counterparts:
    par_gemmul8/include/ozaki/moduli_expand.hpp    moduli_expand (phase E)
    include/ozaki/mod.hpp                          wrapping, mod_small/middle/large
    include/ozaki/moduli_gemm_loop.hpp             moduli_gemm_loop
    include/ozaki/conv_hi2mid.hpp                  conv_hi2mid

A' @ B' has entries up to 𝒫/2, far too big to compute with an int8 GEMM
directly. But arithmetic mod p is compatible with products and sums:

    (A' @ B') mod p  ==  (rmod(A', p) @ rmod(B', p)) mod p

and each rmod(A', p_i) fits in an int8. So for every modulus p_i (paper
symbols, then par_gemmul8's):

    A'_i = rmod(A', p_i)        A_lo[i]    int8    phase E (line 4)
    B'_i = rmod(B', p_i)        B_lo[i]    int8    phase E (line 5)
    C'_i = A'_i @ B'_i          C_hi       int32   one int8 GEMM (line 6)
    U_i  = mod(C'_i, p_i)       C_mid[i]   int8    conv_hi2mid (line 7)

C_mid holds the residues of the exact product A' @ B', one int8 per modulus.
Inverse scaling glues them back together with the CRT.

One difference from the paper: it reduces C'_i with the *unsigned* mod to
U_i in [0, p_i) (a UINT8). par_gemmul8 uses the *signed* rmod into
[-p_i/2, p_i/2] (an int8). Both are valid inputs to the CRT sum, but the
doubles summed in inverse scaling differ. We follow par_gemmul8.
"""

import numpy as np

from .crt_tables import MODULI, check_num_moduli
from .scaling import int8_gemm


def rmod(x, p):
    """The paper's rmod(x, p) = x - p * round(x / p): x mod p as the
    representative in [-p/2, p/2], stored as int8 (par_gemmul8's
    mod.hpp: mod_small / mod_middle / mod_large, followed by wrapping<IDX>).

    `x` may be an integer array or an integer-valued float64 array. np.fmod is
    exact for both. It returns a result with the sign of x, in (-p, p). One
    wrapping step then moves it into [-p/2, p/2], as mod.hpp's wrapping does.

    For odd p that range is [-(p-1)/2, (p-1)/2], at most +-127, so it fits in an
    int8. For p = 256, x = 128 (mod 256) sits exactly halfway, and the result
    is +128 or -128. +128 doesn't fit, and the int8 cast wraps it to -128, as
    CUDA's static_cast<int8_t> does. Paper Section 4.1: "not an issue because
    128 = -128 mod 256".

    How the remainder is computed differs, but not the remainder itself:
      * paper Section 4.2: fma with a precomputed float reciprocal of p_i
      * par_gemmul8: Barrett reduction, a - p*mulhi(a, 2^32/p). Past 2^63 it
        splits a double into mantissa * 2^exp and reduces each part with
        table lookups (make_mant_exp, mod_pow2).
      * here: np.fmod.
    """
    r = np.fmod(x, p)
    half = p // 2
    r = np.where(r > half, r - p, r)
    r = np.where(r < -half, r + p, r)
    return r.astype(np.int64).astype(np.int8)


def moduli_expand(A_core, B_core, num_moduli):
    """Phase E: A'_i = rmod(A', p_i), B'_i = rmod(B', p_i)
    (par_gemmul8: A_lo, B_lo).

    CUDA: moduli_expand.hpp: moduli_expand, with ModUnroll writing one int8
    plane per modulus. Here the planes are stacked along a new leading axis:
    A_lo has shape (num_moduli, m, k), B_lo has shape (num_moduli, k, n).
    """
    check_num_moduli(num_moduli)
    moduli = MODULI[:num_moduli]
    A_lo = np.stack([rmod(A_core, p) for p in moduli])
    B_lo = np.stack([rmod(B_core, p) for p in moduli])
    return A_lo, B_lo


def conv_hi2mid(C_hi, mod_idx):
    """C_mid[i] = rmod(C'_i, p_i) (conv_hi2mid.hpp: conv_hi2mid_kernel<IDX>).
    The paper's U_i = mod(C'_i, p_i) is the same residue, taken unsigned."""
    return rmod(C_hi, MODULI[mod_idx])


def moduli_gemm_loop(A_lo, B_lo):
    """One int8 GEMM per modulus, reduced back to int8 (Algorithm 1 lines 6-7;
    moduli_gemm_loop.hpp: moduli_gemm_loop, default non-fused path).

    Returns C_mid with shape (num_moduli, m, n).

    The optional CUTLASS path in par_gemmul8 (OZAKI_FUSED=1) does the mod
    inside the GEMM epilogue, so C_hi is never written to memory. The
    arithmetic is the same.
    """
    num_moduli = A_lo.shape[0]
    C_mid = []
    for i in range(num_moduli):
        C_hi = int8_gemm(A_lo[i], B_lo[i])
        C_mid.append(conv_hi2mid(C_hi, i))
    return np.stack(C_mid)
