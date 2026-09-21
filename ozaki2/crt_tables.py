"""The moduli and the Chinese Remainder Theorem (CRT) constants.

Paper: Uchino, Ozaki, Imamura, "High-Performance and Power-Efficient Emulation
of Matrix Multiplication using INT8 Matrix Engines" (SC Workshops '25),
Theorem 1 and Section 4.1. CUDA counterpart: par_gemmul8/include/ozaki/
crt_table.hpp and include/ozaki/crt_table_int8_data.hpp.

par_gemmul8 (and GEMMul8, where the tables come from) stores every constant as
hex floats. Here they are *derived* from the list of moduli with exact Python
integers, so you can see what each one means. tests/test_crt_tables.py checks
that the derived values are bit-for-bit the vendored ones.

Notation (the paper's), for N = num_moduli:

    p_1, ..., p_N        the moduli, pairwise coprime (MODULI[:N])
    𝒫 = p_1 * ... * p_N   their product (moduli_product)
    q_i                  the inverse of 𝒫/p_i mod p_i: (𝒫/p_i) * q_i = 1 (mod p_i)
    𝒫/p_i * q_i          the CRT weight of modulus i (crt_weights; `qPi` in
                         par_gemmul8). It is 1 mod p_i and 0 mod every other p_j.

Theorem 1 (CRT): if x = y_i (mod p_i) for every i, then

    x = sum_i 𝒫/p_i * q_i * y_i   (mod 𝒫),

and if |x| < 𝒫/2 this pins x down uniquely:
x = S - 𝒫 * round(S / 𝒫) with S = sum_i 𝒫/p_i * q_i * y_i.

(The paper numbers the moduli from 1. Python and par_gemmul8 index from 0, so
MODULI[0] is the paper's p_1.)

SIGN WARNING: the paper's 𝒫 is the positive product. par_gemmul8's table `P`
holds -𝒫 (as double / double-double), so that the "- 𝒫 * Q" of Algorithm 1
line 11 becomes a plain fma(P, quot, ...). In this module the product is
`moduli_product`, and `P_double` / `P_double2` are par_gemmul8's negated `P`.
"""

import math

import numpy as np

# The moduli, in the order par_gemmul8 uses them (detail::modulus<IDX>). Paper
# Section 4.1: taken from {256, 255, 253, 251, ...}, pairwise coprime integers
# <= 256, so that rmod(A', p_i) lies in [-128, 127] and fits in an int8. For
# p = 256 the residue can be +128, and casting to int8 wraps it to -128, which
# is harmless because 128 = -128 (mod 256).
#
# par_gemmul8's comment calls them "prime moduli", but 255 = 3*5*17,
# 253 = 11*23, 247 = 13*19 and 217 = 7*31 are not prime. The CRT only needs
# them to be *pairwise coprime* (tested).
MODULI = (
    256, 255, 253, 251, 247, 241, 239, 233, 229, 227,
    223, 217, 211, 199, 197, 193, 191, 181, 179, 173,
)  # fmt: skip

# ozaki/limits.hpp: min_moduli, max_moduli_double. Paper: "N <= 20 is
# sufficient for DGEMM emulation".
MIN_MODULI = 2
MAX_MODULI = 20

# ozaki/limits.hpp: ModuliThreshold::p_is_double. Up to this many moduli, 𝒫
# (at most ~2^48) and every CRT weight are exact in one double, so inverse
# scaling works in plain doubles. Above it, they need more than 53 bits, and
# it switches to the two-double representation of paper eq. 6 (s_i1, s_i2) and
# 𝒫 = 𝒫_1 + 𝒫_2. (The paper stores 𝒫 as two doubles for every N, with
# 𝒫_2 = s_i2 = 0 when one double suffices; par_gemmul8 picks a code path
# instead.) See inverse_scaling.py for why the CRT sum's own rounding is fine.
P_IS_DOUBLE = 6


def check_num_moduli(num_moduli):
    if not MIN_MODULI <= num_moduli <= MAX_MODULI:
        raise ValueError(
            f"num_moduli must be in [{MIN_MODULI}, {MAX_MODULI}], got {num_moduli}"
        )


def moduli_product(num_moduli):
    """The paper's 𝒫 = p_1 * ... * p_N, as an exact (positive) Python int."""
    check_num_moduli(num_moduli)
    return math.prod(MODULI[:num_moduli])


def crt_weights(num_moduli):
    """The exact CRT weights 𝒫/p_i * q_i (Python ints). par_gemmul8 calls
    them qPi."""
    P = moduli_product(num_moduli)
    weights = []
    for p in MODULI[:num_moduli]:
        # pow(x, -1, p) is the modular inverse of x mod p: this is q_i.
        q = pow(P // p, -1, p)
        weights.append(P // p * q)
    return tuple(weights)


# --- the constants inverse scaling uses ---------------------------------------


def P_double(num_moduli):
    """par_gemmul8's detail::P[N-2].x = -double(𝒫) (note the sign)."""
    return float(-moduli_product(num_moduli))


def P_double2(num_moduli):
    """par_gemmul8's detail::P[N-2] = (-𝒫_1, -𝒫_2), where in the paper
    𝒫_1 = double(𝒫) and 𝒫_2 = double(𝒫 - 𝒫_1)."""
    P = moduli_product(num_moduli)
    P1 = float(P)
    P2 = float(P - int(P1))
    return -P1, -P2


def invP(num_moduli):
    """The paper's 𝒫_inv = double(1/𝒫) (par_gemmul8's detail::invP[N-2]).

    It is positive: par_gemmul8's "invP = 1/P" comment means the paper's 𝒫,
    not its own negated `P` table. Used as Q = round(𝒫_inv * C'(1)) (eq. 8).
    """
    # int / int in Python is correctly rounded true division.
    return 1 / moduli_product(num_moduli)


def qPi_double(num_moduli):
    """par_gemmul8's detail::qPi_1[N-2]: each CRT weight rounded to one double
    (the paper's s_i1 with s_i2 = 0). Exact for N <= P_IS_DOUBLE, which is
    the only case inverse scaling uses it for."""
    return tuple(float(w) for w in crt_weights(num_moduli))


def qPi_double2_split_bits_paper(num_moduli):
    """How many low bits of each CRT weight go into s_i2, by paper eq. 6.

    s_i1 keeps the upper
        beta_i = 53 - 8 - ceil(log2 N) + floor(log2 w_i) - floor(log2 max_j w_j)
    bits of w_i = 𝒫/p_i * q_i. The w_i-dependent terms cancel: every weight
    is cut at the same bit position, beta_max bits below the top bit of the
    largest weight. Why: C'(1) = sum_i s_i1 * U_i then adds N numbers of up
    to (largest weight) * 2^8, and all of them are multiples of that cut, so
    they fit in 53 bits and the sum is exact (see inverse_scaling.py).
    """
    w_max = max(crt_weights(num_moduli))
    beta_max = 53 - 8 - math.ceil(math.log2(num_moduli))
    return w_max.bit_length() - beta_max


# The split positions actually used in par_gemmul8's detail::qPi_2 table, per
# N (7..20). They are 1 bit below the paper's formula (2 bits for N = 17, 18),
# i.e. the shipped hi words keep 1-2 more bits than eq. 6 says. The exactness
# argument still holds for them (tested), with less headroom than eq. 6 gives.
# Kept as data so the tables match par_gemmul8 bit for bit.
QPI_DOUBLE2_SPLIT_BITS = {
    7: 13, 8: 21, 9: 30, 10: 38, 11: 46, 12: 53, 13: 61,
    14: 69, 15: 76, 16: 84, 17: 91, 18: 99, 19: 107, 20: 115,
}  # fmt: skip


def qPi_double2(num_moduli):
    """par_gemmul8's detail::qPi_2[N-7]: each CRT weight as (s_i1, s_i2)
    (paper eq. 6). s_i1 = the weight with its low bits cleared (exact),
    s_i2 = double(the rest). s_i2 is exact up to N = 12. From N = 13 the
    remainder needs more than 53 bits, and s_i2 is its rounding."""
    if num_moduli <= P_IS_DOUBLE:
        raise ValueError("qPi_double2 is only defined for num_moduli > 6")
    s = QPI_DOUBLE2_SPLIT_BITS[num_moduli]
    pairs = []
    for w in crt_weights(num_moduli):
        s1 = (w >> s) << s
        pairs.append((float(s1), float(w - s1)))
    return tuple(pairs)


# --- the constant scaling uses -------------------------------------------------

# par_gemmul8's log2P<2> is 0x1.dfd1ecp+2 = 7.4971876..., but the formula below
# gives 7.4971657 (47 float32 ulps smaller). It's the only entry that
# disagrees. To keep the shifts identical to par_gemmul8, the vendored value
# is used for N = 2.
_LOG2P_VENDORED_N2 = float.fromhex("0x1.dfd1ecp+2")


def log2P(num_moduli):
    """The paper's 𝒫'_accu = single(log2(𝒫 - 1)/2 - 0.5), rounded *down* to
    float32 (par_gemmul8's detail::log2P<N>).

    Returned as a Python float holding a float32 value. It is
    log2(sqrt((𝒫-1)/2)): the bit budget of *one* factor. Scaling picks
    shifts so that 2 * sum_h |a'_ih| |b'_hj| < 𝒫 (paper eq. 3), i.e. each
    factor gets about half the bits of 𝒫/2 (see scaling.compute_shift).
    """
    check_num_moduli(num_moduli)
    if num_moduli == 2:
        return _LOG2P_VENDORED_N2
    exact = math.log2(moduli_product(num_moduli) - 1) / 2 - 0.5
    f = np.float32(exact)
    if float(f) > exact:  # round *down*: step one float32 ulp toward -inf
        f = np.nextafter(f, np.float32(-np.inf))
    return float(f)
