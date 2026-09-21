"""Phase E and the moduli GEMM loop: products of big integers as int8 GEMMs.

CUDA counterparts:
    par_gemmul8/include/ozaki/moduli_expand.hpp    moduli_expand (phase E)
    include/ozaki/mod.hpp                          wrapping, mod_small/middle/large
    include/ozaki/moduli_gemm_loop.hpp             moduli_gemm_loop
    include/ozaki/conv_hi2mid.hpp                  conv_hi2mid

A_core @ B_core has entries up to M/2, far too big to compute with an int8
GEMM directly. But arithmetic mod p is compatible with products and sums:

    (A_core @ B_core) mod p  ==  ((A_core mod p) @ (B_core mod p)) mod p

and each "A_core mod p" fits in an int8. So for every modulus p_i:

    A_lo[i] = A_core mod p_i                  (int8)      phase E
    B_lo[i] = B_core mod p_i                  (int8)      phase E
    C_hi    = A_lo[i] @ B_lo[i]               (int32)     one int8 GEMM
    C_mid[i] = C_hi mod p_i                   (int8)      conv_hi2mid

C_mid holds the residues of the exact product A_core @ B_core, one int8 per
modulus. Inverse scaling glues them back together with the CRT.
"""

import numpy as np

from .crt_tables import MODULI, check_num_moduli
from .scaling import int8_gemm


def symmetric_mod(x, p):
    """x mod p, as the representative in [-p/2, p/2], stored as int8
    (mod.hpp: mod_small / mod_middle / mod_large, followed by wrapping<IDX>).

    `x` may be an integer array or an integer-valued float64 array. np.fmod is
    exact for both. It returns a result with the sign of x, in (-p, p). One
    wrapping step then moves it into [-p/2, p/2], as mod.hpp's wrapping does.

    For odd p that range is [-(p-1)/2, (p-1)/2], at most +-127, so it fits in an
    int8. For p = 256 the range includes +128, which doesn't. The final int8 cast
    wraps 128 to -128, which is also what CUDA's static_cast<int8_t> does. -128
    is congruent to 128 mod 256, so the residue is still right.

    CUDA computes the remainder with a precomputed reciprocal (Barrett
    reduction, a - p*mulhi(a, 2^32/p)) and, for values past 2^63, splits a
    double into mantissa * 2^exp and reduces each part with table lookups
    (make_mant_exp, mod_pow2). Those are faster ways of getting the same
    remainder.
    """
    r = np.fmod(x, p)
    half = p // 2
    r = np.where(r > half, r - p, r)
    r = np.where(r < -half, r + p, r)
    return r.astype(np.int64).astype(np.int8)


def moduli_expand(A_core, B_core, num_moduli):
    """Phase E: A_lo[i] = A_core mod p_i, B_lo[i] = B_core mod p_i.

    CUDA: moduli_expand.hpp: moduli_expand, with ModUnroll writing one int8
    plane per modulus. Here the planes are stacked along a new leading axis:
    A_lo has shape (num_moduli, m, k), B_lo has shape (num_moduli, k, n).
    """
    check_num_moduli(num_moduli)
    moduli = MODULI[:num_moduli]
    A_lo = np.stack([symmetric_mod(A_core, p) for p in moduli])
    B_lo = np.stack([symmetric_mod(B_core, p) for p in moduli])
    return A_lo, B_lo


def conv_hi2mid(C_hi, mod_idx):
    """C_mid[i] = C_hi mod p_i (conv_hi2mid.hpp: conv_hi2mid_kernel<IDX>)."""
    return symmetric_mod(C_hi, MODULI[mod_idx])


def moduli_gemm_loop(A_lo, B_lo):
    """One int8 GEMM per modulus, reduced back to int8
    (moduli_gemm_loop.hpp: moduli_gemm_loop, default non-fused path).

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
