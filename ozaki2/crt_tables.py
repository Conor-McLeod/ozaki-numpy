"""The moduli and the Chinese Remainder Theorem (CRT) constants.

CUDA counterpart: par_gemmul8/include/ozaki/crt_table.hpp and
include/ozaki/crt_table_int8_data.hpp.

par_gemmul8 (and GEMMul8, where the tables come from) stores every constant as
hex floats. Here they are *derived* from the list of moduli with exact Python
integers, so you can see what each one means. tests/test_crt_tables.py checks
that the derived values are bit-for-bit the vendored ones.

Notation, for a run with `num_moduli` = N:

    p_0, ..., p_{N-1}   the moduli (MODULI[:N])
    M = p_0 * ... * p_{N-1}
    M_i = M / p_i
    q_i = M_i^{-1} mod p_i       (the inverse exists because the p_i are
                                  pairwise coprime)
    qPi_i = q_i * M_i            the CRT weight: qPi_i = 1 (mod p_i) and
                                  qPi_i = 0 (mod p_j) for every j != i

so that for any integer X with |X| < M/2 and residues r_i = X mod p_i

    X = sum_i qPi_i * r_i  -  M * round(sum_i qPi_i * r_i / M).
"""

import math

import numpy as np

# The moduli, in the order par_gemmul8 uses them (detail::modulus<IDX>).
# The CUDA comment calls them "prime moduli", but 255 = 3*5*17,
# 253 = 11*23, 247 = 13*19 and 217 = 7*31 are not prime. The CRT only needs the
# moduli to be *pairwise coprime*, and they are (tested). They are chosen as
# the largest such numbers <= 256, so that a residue in [-p/2, p/2] fits into
# an int8 and each modulus carries as close to 8 bits of information as
# possible.
MODULI = (
    256, 255, 253, 251, 247, 241, 239, 233, 229, 227,
    223, 217, 211, 199, 197, 193, 191, 181, 179, 173,
)  # fmt: skip

# ozaki/limits.hpp: min_moduli, max_moduli_double.
MIN_MODULI = 2
MAX_MODULI = 20

# ozaki/limits.hpp: ModuliThreshold::p_is_double. Up to this many moduli, M
# (at most ~2^48) and every qPi_i are exact in one double, so inverse scaling
# works in plain doubles. Above it, M and qPi_i need more than 53 bits, and it
# switches to a two-double ("double-double") representation. See
# inverse_scaling.py for why the CRT sum's own rounding is acceptable.
P_IS_DOUBLE = 6


def check_num_moduli(num_moduli):
    if not MIN_MODULI <= num_moduli <= MAX_MODULI:
        raise ValueError(
            f"num_moduli must be in [{MIN_MODULI}, {MAX_MODULI}], got {num_moduli}"
        )


def modulus_product(num_moduli):
    """M = p_0 * ... * p_{N-1}, as an exact Python int."""
    check_num_moduli(num_moduli)
    return math.prod(MODULI[:num_moduli])


def crt_weights(num_moduli):
    """The exact CRT weights qPi_i = q_i * M_i (Python ints)."""
    M = modulus_product(num_moduli)
    weights = []
    for p in MODULI[:num_moduli]:
        M_i = M // p
        # pow(x, -1, p) is the modular inverse of x mod p.
        q_i = pow(M_i, -1, p)
        weights.append(q_i * M_i)
    return tuple(weights)


# --- the constants inverse scaling uses ---------------------------------------


def P_double(num_moduli):
    """detail::P[N-2].x. Note the sign: par_gemmul8 stores P = -M, so that
    "subtract M * quot" can be written as the single fma(P, quot, S)."""
    return float(-modulus_product(num_moduli))


def P_double2(num_moduli):
    """detail::P[N-2] as (hi, lo): -M = hi + lo exactly, hi = nearest double."""
    minus_M = -modulus_product(num_moduli)
    hi = float(minus_M)
    lo = float(minus_M - int(hi))
    return hi, lo


def invP(num_moduli):
    """detail::invP[N-2]. The CUDA comment says "1/P", but the stored value
    is +1/M (positive), which is what the code relies on:
    quot = rint(invP * S) = round(S / M)."""
    # int / int in Python is correctly rounded true division.
    return 1 / modulus_product(num_moduli)


def qPi_double(num_moduli):
    """detail::qPi_1[N-2]: each CRT weight rounded to one double. Exact for
    N <= P_IS_DOUBLE, which is the only case inverse scaling uses it for."""
    return tuple(float(w) for w in crt_weights(num_moduli))


# How many low bits of qPi_i are moved out of the "hi" word in qPi_double2,
# per num_moduli (7..20), read off the vendored detail::qPi_2 table.
#
# Why split at all: inverse scaling accumulates
#     S_hi = sum_i fma(qPi_hi_i, C_mid_i, S_hi)
# If every qPi_hi_i is a multiple of 2^s and the whole sum fits in 53 bits above
# 2^s, then every product and partial sum is an exactly representable double,
# so S_hi is *exact*. The rounding error is pushed into the small S_lo sum.
#
# The exactness condition only needs s >= (the minimum). GEMMul8 picked
# s 1-3 bits above that minimum, and its generator isn't published, so the split
# points are kept as data. tests/test_crt_tables.py checks the exactness
# condition for them.
QPI_DOUBLE2_SPLIT_BITS = {
    7: 13, 8: 21, 9: 30, 10: 38, 11: 46, 12: 53, 13: 61,
    14: 69, 15: 76, 16: 84, 17: 91, 18: 99, 19: 107, 20: 115,
}  # fmt: skip


def qPi_double2(num_moduli):
    """detail::qPi_2[N-7]: each CRT weight as (hi, lo). hi = qPi_i with its low
    s bits cleared (exact), lo = the remainder qPi_i - hi, rounded to a double
    (exact up to N = 12; from N = 13 the remainder needs more than 53 bits)."""
    if num_moduli <= P_IS_DOUBLE:
        raise ValueError("qPi_double2 is only defined for num_moduli > 6")
    s = QPI_DOUBLE2_SPLIT_BITS[num_moduli]
    pairs = []
    for w in crt_weights(num_moduli):
        hi = (w >> s) << s
        pairs.append((float(hi), float(w - hi)))
    return tuple(pairs)


# --- the constant scaling uses -------------------------------------------------

# log2P<2> in the vendored table is 0x1.dfd1ecp+2 = 7.4971876..., but the
# documented formula below gives 7.4971657 (47 float32 ulps smaller). It's
# the only entry that disagrees. To keep the shifts identical to
# par_gemmul8, the vendored value is used for N = 2.
_LOG2P_VENDORED_N2 = float.fromhex("0x1.dfd1ecp+2")


def log2P(num_moduli):
    """detail::log2P<N> = round-down-to-float32(log2(M - 1)/2 - 0.5).

    Returned as a Python float holding a float32 value. It is
    log2(sqrt((M-1)/2)): the bit budget of *one* factor. Scaling picks shifts
    so that each entry of A_core @ B_core is at most 2^log2P * 2^log2P
    = (M-1)/2 < M/2. That's the range the CRT can reconstruct
    (see scaling.compute_shift).
    """
    check_num_moduli(num_moduli)
    if num_moduli == 2:
        return _LOG2P_VENDORED_N2
    exact = math.log2(modulus_product(num_moduli) - 1) / 2 - 0.5
    f = np.float32(exact)
    if float(f) > exact:  # round *down*: step one float32 ulp toward -inf
        f = np.nextafter(f, np.float32(-np.inf))
    return float(f)
