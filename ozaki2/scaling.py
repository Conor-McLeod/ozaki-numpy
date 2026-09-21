"""Scaling, phases A-D: turn floating-point A and B into integer matrices.

CUDA counterparts:
    par_gemmul8/src/seq/seq_scaling.cu          seq::scaling (the orchestrator)
    include/ozaki/scaling_accu.hpp              phases A and C
    include/ozaki/scaling_trunc.hpp             upper_bound_int8, TruncScalbn
    include/ozaki/scaling_core.hpp              phase D
    include/ozaki/find_max.hpp                  the amax reductions

Goal: find a power-of-two scale per row of A and per column of B,

    A_core = trunc(A * 2^-shiftA[i])      (row i)
    B_core = trunc(B * 2^-shiftB[j])      (column j)

such that A_core and B_core are integers and every entry of A_core @ B_core
is guaranteed to satisfy |X| < M/2 (M = product of the moduli). That bound is
what lets the CRT reconstruct X exactly from its residues later. Within that
bound, the shifts are chosen to keep as many bits of A and B as possible.

The "accurate" scaling in par_gemmul8 finds the shifts in two passes:

  A. Pick a first, rough shift per row/column so that the row's largest
     element lands in [32, 64). Round |A| *up* to integers at that scale to get
     a small int8 upper bound, A_bound.
  B. C_hi = A_bound @ B_bound: one int8 GEMM that gives an upper bound on
     |A| @ |B| (in those rough scaled units) without any floating-point GEMM.
  C. From the largest C_hi entry in row i / column j, work out how many
     more bits row i of A and column j of B can afford, and refine the shifts.
  D. Apply the final shifts and truncate: A_core, B_core.

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
    int32 first. The result is identical as long as the int32 sum doesn't
    overflow: |entry| <= 128*128*k < 2^31 needs k < 2^17. ozaki_gemm checks it.
    """
    return np.matmul(A_i8.astype(np.int32), B_i8.astype(np.int32))


def bound_gemm(A_bound, B_bound):
    """Phase B: C_hi = A_bound @ B_bound.

    Every entry of A_bound and B_bound is >= the magnitude of the scaled
    element it came from, so C_hi[i, j] >= sum_l |A[i,l]| |B[l,j]| (scaled by
    2^(shiftA[i] + shiftB[j])). C_hi bounds the magnitude of every term that
    can appear in the product.
    """
    return int8_gemm(A_bound, B_bound)


# --- Phase C ------------------------------------------------------------------


def compute_shift(amax, num_moduli):
    """How many extra bits a row/column can afford, given the largest entry
    `amax` (int32) of its row/column of C_hi (scaling_accu.hpp:
    compute_shift<num_moduli>).

    CUDA, in float32:
        log2amax = __log2f(float(amax))
        return floor_rd( fma_rd(-0x1.000006p-1, log2amax, log2P) )

    Idea: C_hi says that row i of A times column j of B produces at most
    2^log2(amax), in the rough units. There are log2P ~ log2(M)/2 - 0.5 bits of
    budget *per factor*. The row's result spends log2(amax) of them in total,
    so each factor gets back log2P - log2(amax)/2 more bits.

    Why -(0.5 + 3*2^-23) rather than -0.5: __log2f is a fast hardware
    approximation, not correctly rounded. Making the coefficient slightly
    bigger in magnitude than 0.5 and rounding everything down biases the
    result toward a *smaller* shift, so an approximation error can't push it
    over the safe limit.

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

    SIGN CONVENTION CHANGE. On input, shiftA/B are exponents to multiply by
    (phase A). On output they are *negated*: shiftA_out = -(shiftA_in + extra).
    From here on

        A_core = trunc(A * 2^-shiftA)   and   A ~= A_core * 2^shiftA,

    so shiftA is "the exponent that undoes the scaling", which is what
    inverse scaling needs.
    """
    extraA = compute_shift(C_hi.max(axis=1), num_moduli)  # row max, per row of A
    extraB = compute_shift(C_hi.max(axis=0), num_moduli)  # col max, per col of B
    shiftA = (-(shiftA.astype(np.int32) + extraA)).astype(np.int16)
    shiftB = (-(shiftB.astype(np.int32) + extraB)).astype(np.int16)
    return shiftA, shiftB


# --- Phase D ------------------------------------------------------------------


def trunc_core(A, shiftA, B, shiftB):
    """Phase D: A_core = trunc(A * 2^-shiftA), B_core = trunc(B * 2^-shiftB)
    (scaling_core.hpp: trunc_core_launch, with TruncScalbn from
    scaling_trunc.hpp).

    Truncation (round toward zero) never increases a magnitude, so the phase C
    bound still holds for the truncated matrices.

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

    Returns a dict with the intermediates:
      shiftA0, shiftB0   phase A shifts (exponents to multiply by)
      A_bound, B_bound   phase A int8 upper bounds
      C_hi_bound         phase B int32 bound product
      shiftA, shiftB     phase C final shifts (exponents that undo scaling)
      A_core, B_core     phase D integer matrices (float64 storage)
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
