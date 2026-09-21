"""The whole pipeline: C = A @ B via Ozaki Scheme II.

CUDA counterpart: par_gemmul8/src/seq/seq_ozaki_gemm.cu (seq::ozaki_gemm).

    scaling            phases A-D   ozaki2/scaling.py
    moduli_expand      phase E      ozaki2/moduli.py
    moduli_gemm_loop                ozaki2/moduli.py
    inverse_scaling                 ozaki2/inverse_scaling.py

Not reproduced (none of it changes the arithmetic): the workspace (the
preallocated buffers A_lo, B_lo, C_hi, C_mid, shiftA, shiftB, A_core, B_core),
cuBLAS handle setup, per-stage timing, debug printing, transposes and
alpha/beta.
"""

from dataclasses import dataclass

import numpy as np

from .crt_tables import check_num_moduli
from .inverse_scaling import inverse_scaling
from .moduli import moduli_expand, moduli_gemm_loop
from .scaling import scaling

# int8_gemm accumulates up to k products of magnitude <= 128*128 in int32.
_MAX_K = (2**31 - 1) // (128 * 128)


@dataclass
class OzakiTrace:
    """Every intermediate of one ozaki_gemm call, named as in par_gemmul8."""

    num_moduli: int
    # phase A
    shiftA0: np.ndarray  # (m,)  int16, exponent to multiply by
    shiftB0: np.ndarray  # (n,)
    A_bound: np.ndarray  # (m, k) int8
    B_bound: np.ndarray  # (k, n) int8
    # phase B
    C_hi_bound: np.ndarray  # (m, n) int32
    # phase C
    shiftA: np.ndarray  # (m,) int16, exponent that undoes the scaling
    shiftB: np.ndarray  # (n,)
    # phase D
    A_core: np.ndarray  # (m, k) integer-valued float64
    B_core: np.ndarray  # (k, n)
    # phase E
    A_lo: np.ndarray  # (num_moduli, m, k) int8
    B_lo: np.ndarray  # (num_moduli, k, n) int8
    # moduli GEMM loop
    C_mid: np.ndarray  # (num_moduli, m, n) int8
    # inverse scaling
    C: np.ndarray  # (m, n) float64


def ozaki_gemm(A, B, num_moduli, trace=False):
    """C = A @ B for float64 A (m, k) and B (k, n), using num_moduli int8 GEMMs.

    More moduli means a larger M, more bits kept in A_core/B_core, and a more
    accurate C: roughly 4 bits per modulus per factor. Around 14+ moduli it is
    about as accurate as a float64 GEMM.

    With trace=True, returns an OzakiTrace holding every intermediate instead
    of just C.
    """
    check_num_moduli(num_moduli)
    A = np.asarray(A)
    B = np.asarray(B)
    if A.dtype != np.float64 or B.dtype != np.float64:
        raise TypeError("ozaki_gemm takes float64 A and B")
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError(f"shapes {A.shape} and {B.shape} can't be multiplied")
    if A.shape[1] > _MAX_K:
        raise ValueError(f"k = {A.shape[1]} would overflow the int32 GEMM")
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        raise ValueError("A and B must be finite")

    s = scaling(A, B, num_moduli)
    A_lo, B_lo = moduli_expand(s["A_core"], s["B_core"], num_moduli)
    C_mid = moduli_gemm_loop(A_lo, B_lo)
    C = inverse_scaling(C_mid, s["shiftA"], s["shiftB"], num_moduli)

    if not trace:
        return C
    return OzakiTrace(
        num_moduli=num_moduli, A_lo=A_lo, B_lo=B_lo, C_mid=C_mid, C=C, **s
    )
