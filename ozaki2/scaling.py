"""Scaling, phases A-D: turn floating-point A and B into integer matrices.

Paper: Section 3 step 2 and Section 4.2 "accurate mode" (Algorithm 1
lines 1-3). CUDA counterparts:
    par_gemmul8/src/seq/seq_scaling.cu          seq::scaling (the orchestrator)
    include/ozaki/scaling_accu.hpp              phases A and C
    include/ozaki/scaling_trunc.hpp             upper_bound_int8, TruncScalbn
    include/ozaki/scaling_core.hpp              phase D
    include/ozaki/find_max.hpp                  the amax reductions

Goal (paper step 2): find scale vectors mu (one power of two per row of A) and
nu (one per column of B), and truncate

    A' = trunc(diag(mu) @ A)        par_gemmul8: A_core
    B' = trunc(B @ diag(nu))        par_gemmul8: B_core

so that A' and B' are integer matrices satisfying (paper eq. 3)

    2 * sum_h |a'_ih| |b'_hj| < 𝒫     for all i, j,

where 𝒫 is the product of the moduli (crt_tables.py). Then every entry of
A' @ B' has |x| < 𝒫/2, the range in which the CRT pins x down uniquely.
Within that bound, the scales are chosen to keep as many bits of A and B as
possible.

par_gemmul8 stores exponents, not the powers of two. After phase C,
mu_i = 2^-shiftA[i] and nu_j = 2^-shiftB[j].

Accurate mode finds the scales in two passes (the phase letters are from
par_gemmul8's seq_scaling.hpp):

  A. mu'_i = 2^(5 - floor(log2 max_h |a_ih|)) (nu'_j likewise), so the row's
     largest element lands in [32, 64). Round up at that scale:
     Abar = ceil(diag(mu') |A|). That's a small nonnegative int8 matrix
     (par_gemmul8: A_bound; mu'_i = 2^shiftA0[i]).
  B. Cbar = Abar @ Bbar: one int8 GEMM that bounds diag(mu')|A||B|diag(nu')
     entrywise, without any floating-point GEMM (par_gemmul8: C_hi).
  C. From the largest Cbar entry in row i / column j, work out how many
     more bits row i of A and column j of B can afford: mu, nu.
  D. Apply the final scales and truncate: A', B'.

The paper's alternative "fast mode" skips phases A-B and bounds the sum with
Cauchy-Schwarz instead (eq. 7). par_gemmul8 only implements accurate mode.

Differences from the CUDA code, none of which change the result:
  * Layout: numpy's natural row-major (m, k) / (k, n) arrays. The CUDA code is
    column-major with leading dimensions padded to multiples of 256, which is
    why it has separate "rowwise" and "colwise" kernels.
  * Only C = A @ B (no op_A/op_B transposes, no alpha/beta).
  * The bit-level float manipulation (trunc_scalbn_*) is written as
    np.ldexp + np.ceil/np.trunc, which gives the same integers.
  * The CUDA code stores A_core as int32, int64 or double depending on
    num_moduli (TruncScalbn; the core values reach ~2^78 at 20 moduli). Here
    A_core is always an integer-valued float64, which holds these values
    exactly (see trunc_core).
"""

import numpy as np

from .crt_tables import check_num_moduli, log2P

# ozaki/types.hpp: max_ufp. Phase A scales each row so its largest element is
# in [2^5, 2^6) (the unit in the first place, "ufp", becomes 2^5).
MAX_UFP = 5

# The float32 constant in compute_shift: -0x1.0000060000000p-1F, which is
# -(0.5 + 3*2^-23). See compute_shift for why it isn't exactly -0.5.
_MINUS_HALF_PLUS = float.fromhex("-0x1.000006p-1")


def ilogb(x):
    """floor(log2(|x|)) as an int, with ilogb(0) = 0 (math.hpp: typed_ilogb).

    np.frexp writes x = mant * 2^exp with mant in [0.5, 1), so
    floor(log2(|x|)) = exp - 1. It also handles subnormals.
    """
    _, exp = np.frexp(x)
    return np.where(x == 0, 0, exp - 1)


# --- Phase A ------------------------------------------------------------------


def upper_bound_int8(x, shift):
    """ceil(|x| * 2^shift) as int8 (scaling_trunc.hpp: upper_bound_int8 /
    trunc_scalbn_8i). `shift` broadcasts against x (per row or column).

    The CUDA code builds the result from the exponent and mantissa bits:
    `floor + has_frac` is a ceiling, and any nonzero value too small to reach
    the integer bits gives 1. np.ldexp can underflow to 0 when a row mixes
    very large and very small values (e.g. 1e300 and 1e-300), so we also force
    "nonzero -> at least 1". That keeps it an upper bound.
    """
    bound = np.ceil(np.ldexp(np.abs(x), shift))
    bound = np.where(x != 0, np.maximum(bound, 1.0), 0.0)
    # |x| <= amax < 2^(ilogb(amax)+1), so |x| * 2^(5 - ilogb(amax)) < 2^6 = 64,
    # and the bound is at most 64. It fits in an int8.
    return bound.astype(np.int8)


def extract_bound(A, B):
    """Phase A: the first-pass shifts and the int8 upper bounds.

    CUDA: scaling_accu.hpp: extract_bound_launch, with
    initial_shift_rowwise_kernel + extract_bound_rowwise_kernel for A and
    extract_bound_colwise_kernel for B.

    Returns (A_bound, shiftA, B_bound, shiftB). At this point shiftA/B are
    *exponents to multiply by*: A_bound = ceil(|A| * 2^shiftA[:, None]).
    """
    # Row-wise absolute maxima of A, column-wise of B (find_max.hpp:
    # find_amax_tile / find_amax).
    amaxA = np.abs(A).max(axis=1)
    amaxB = np.abs(B).max(axis=0)

    # Choose shift so that amax * 2^shift lands in [2^5, 2^6).
    shiftA = (MAX_UFP - ilogb(amaxA)).astype(np.int16)
    shiftB = (MAX_UFP - ilogb(amaxB)).astype(np.int16)

    A_bound = upper_bound_int8(A, shiftA[:, None].astype(np.int32))
    B_bound = upper_bound_int8(B, shiftB[None, :].astype(np.int32))
    return A_bound, shiftA, B_bound, shiftB


# --- Phase B ------------------------------------------------------------------


def int8_gemm(A_i8, B_i8):
    """int8 x int8 -> int32 GEMM (matmult.hpp: gemm_low_prec_i8x1, which calls
    cublasGemmEx with CUDA_R_8I inputs and CUBLAS_COMPUTE_32I).

    numpy has no int8 GEMM with int32 accumulation, so we widen the inputs to
    int32 first. numpy's integer matmul wraps on overflow, as the int32
    accumulator in cuBLAS does, so the results match bit for bit. Paper
    Section 4.3: for k <= 2^17 the only possible overflow is an entry of
    exactly 2^31 (k products of -128 * -128 in the p = 256 plane), which
    wraps to -2^31. Both are 0 mod 256, so the residue is still right.
    ozaki_gemm rejects k > 2^17.
    """
    return np.matmul(A_i8.astype(np.int32), B_i8.astype(np.int32))


def bound_gemm(A_bound, B_bound):
    """Phase B: Cbar = Abar @ Bbar (par_gemmul8: C_hi = A_bound @ B_bound,
    written inline in seq::scaling).

    Every entry of Abar and Bbar is >= the magnitude of the scaled element it
    came from, so Cbar_ij >= mu'_i * sum_h |a_ih| |b_hj| * nu'_j.
    """
    return int8_gemm(A_bound, B_bound)


# --- Phase C ------------------------------------------------------------------


def compute_shift(amax, num_moduli):
    """How many extra bits a row/column can afford, given the largest entry
    `amax` (int32) of its row/column of Cbar (scaling_accu.hpp:
    compute_shift<num_moduli>).

    Paper, accurate mode:
        mu_i = mu'_i * 2^floor(𝒫'_accu - 0.51 * log2(max_h Cbar_ih))
    CUDA, in float32:
        log2amax = __log2f(float(amax))
        return floor_rd( fma_rd(-0x1.000006p-1, log2amax, log2P) )

    Idea: Cbar says that row i of A times any column of B produces at most
    2^log2(amax), in the phase-A units. There are 𝒫'_accu = log2P ~
    log2(𝒫)/2 - 0.5 bits of budget *per factor*. The row's result spends
    log2(amax) of them in total, so each factor gets back
    log2P - log2(amax)/2 more bits.

    The coefficient: the paper uses 0.51. par_gemmul8 (following GEMMul8's
    code) uses -0x1.000006p-1 = -(0.5 + 3*2^-23), which gives larger shifts
    (more bits kept) than the paper would. We follow the code. Either way it is
    slightly more than 0.5 and everything rounds down. That biases the result
    toward a *smaller* shift, so an error in __log2f (a fast hardware
    approximation, not correctly rounded) can't push it over the safe limit.

    Here: np.log2 on float32 stands in for __log2f. The two can differ in
    the last bit, and so, very rarely, the resulting shift could differ by one.
    The fma itself is reproduced exactly: the product of two float32 values
    is exact in float64, and so is the sum with log2P (it needs about 50
    bits). Taking floor of that exact value gives the same integer as CUDA's
    round-down fma followed by round-down conversion.

    amax == 0 (the whole row of C is exactly zero) makes CUDA compute
    floor(+inf), which saturates. Whatever the shift is, that row's product is
    exactly zero, so we just return 0 there.
    """
    amax = np.asarray(amax)
    with np.errstate(divide="ignore"):
        log2amax = np.log2(amax.astype(np.float32))  # float32 result
    exact = log2P(num_moduli) + _MINUS_HALF_PLUS * log2amax.astype(np.float64)
    with np.errstate(invalid="ignore"):
        shift = np.floor(exact)
    return np.where(amax == 0, 0, shift).astype(np.int32)


def refine_shifts(C_hi, shiftA, shiftB, num_moduli):
    """Phase C: final shifts (scaling_accu.hpp: refine_shift_rowwise_kernel
    for A, refine_shift_colwise_kernel for B).

    SIGN CONVENTION CHANGE. On input, shiftA/B are the exponents of the
    paper's mu', nu' (multiply by 2^shift). On output they are *negated*:
    shiftA_out = -(shiftA_in + extra), i.e. mu_i = 2^-shiftA[i]. From here on

        A' = trunc(A * 2^-shiftA)   and   A ~= A' * 2^shiftA,

    so shiftA is the exponent of mu^-1: the one that undoes the scaling in
    step 4, C = diag(mu^-1) C'' diag(nu^-1).
    """
    extraA = compute_shift(C_hi.max(axis=1), num_moduli)  # row max, per row of A
    extraB = compute_shift(C_hi.max(axis=0), num_moduli)  # col max, per col of B
    shiftA = (-(shiftA.astype(np.int32) + extraA)).astype(np.int16)
    shiftB = (-(shiftB.astype(np.int32) + extraB)).astype(np.int16)
    return shiftA, shiftB


# --- Phase D ------------------------------------------------------------------


def trunc_core(A, shiftA, B, shiftB):
    """Phase D: A' = trunc(diag(mu) A), B' = trunc(B diag(nu)) (Algorithm 1
    lines 2-3; par_gemmul8: A_core, B_core from scaling_core.hpp:
    trunc_core_launch, with TruncScalbn from scaling_trunc.hpp).

    Truncation (round toward zero) never increases a magnitude, so the phase C
    bound (eq. 3) still holds for the truncated matrices.

    Representation: np.ldexp multiplies by a power of two exactly (a
    subnormal result is < 1 and truncates to 0 anyway), and np.trunc of a
    double is exact, so every A_core entry is an exact integer stored in a
    float64. CUDA picks the narrowest type that holds these integers
    (TruncScalbn):
        num_moduli <= 7   int32   (trunc_scalbn_to_i32)
        num_moduli <= 15  int64   (trunc_scalbn_to_i64)
        otherwise         double  (trunc_scalbn_to_fp; |A_core| > 2^63)
    It does that because integer mod is cheaper than float mod in phase E.
    Here one float64 path covers all cases, because np.fmod is exact.
    """
    A_core = np.trunc(np.ldexp(A, -shiftA[:, None].astype(np.int32)))
    B_core = np.trunc(np.ldexp(B, -shiftB[None, :].astype(np.int32)))
    return A_core, B_core


# --- Phases A-D ---------------------------------------------------------------


def scaling(A, B, num_moduli):
    """seq::scaling (src/seq/seq_scaling.cu): phases A-D.

    Returns a dict with the intermediates (paper symbol in brackets):
      shiftA0, shiftB0   phase A exponents: mu'_i = 2^shiftA0[i]   [mu', nu']
      A_bound, B_bound   phase A int8 upper bounds                 [Abar, Bbar]
      C_hi_bound         phase B int32 bound product               [Cbar]
      shiftA, shiftB     phase C exponents: mu_i = 2^-shiftA[i]    [mu, nu]
      A_core, B_core     phase D integer matrices (float64 storage) [A', B']
    """
    check_num_moduli(num_moduli)
    A_bound, shiftA0, B_bound, shiftB0 = extract_bound(A, B)
    C_hi_bound = bound_gemm(A_bound, B_bound)
    shiftA, shiftB = refine_shifts(C_hi_bound, shiftA0, shiftB0, num_moduli)
    A_core, B_core = trunc_core(A, shiftA, B, shiftB)
    return dict(
        shiftA0=shiftA0,
        shiftB0=shiftB0,
        A_bound=A_bound,
        B_bound=B_bound,
        C_hi_bound=C_hi_bound,
        shiftA=shiftA,
        shiftB=shiftB,
        A_core=A_core,
        B_core=B_core,
    )
