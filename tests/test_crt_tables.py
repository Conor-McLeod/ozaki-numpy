import math

import pytest

from ozaki2 import crt_tables as t

from . import vendored_crt_tables as v

ALL_N = range(t.MIN_MODULI, t.MAX_MODULI + 1)
DOUBLE2_N = range(t.P_IS_DOUBLE + 1, t.MAX_MODULI + 1)


def fx(s):
    return float.fromhex(s)


def test_moduli_pairwise_coprime():
    for i, p in enumerate(t.MODULI):
        for q in t.MODULI[:i]:
            assert math.gcd(p, q) == 1


@pytest.mark.parametrize("n", ALL_N)
def test_crt_weights_are_the_crt_basis(n):
    for i, w in enumerate(t.crt_weights(n)):
        for j, p in enumerate(t.MODULI[:n]):
            assert w % p == (1 if i == j else 0)


@pytest.mark.parametrize("n", ALL_N)
def test_P_matches_vendored(n):
    hi, lo = v.P[n - 2]
    assert t.P_double2(n) == (fx(hi), fx(lo))
    assert t.P_double(n) == fx(hi)


@pytest.mark.parametrize("n", ALL_N)
def test_invP_matches_vendored(n):
    assert t.invP(n) == fx(v.invP[n - 2])


@pytest.mark.parametrize("n", ALL_N)
def test_qPi_double_matches_vendored(n):
    assert list(t.qPi_double(n)) == [fx(x) for x in v.qPi_1[n - 2]]


@pytest.mark.parametrize("n", DOUBLE2_N)
def test_qPi_double2_matches_vendored(n):
    expected = [(fx(a), fx(b)) for a, b in v.qPi_2[n - 7]]
    assert list(t.qPi_double2(n)) == expected


@pytest.mark.parametrize("n", DOUBLE2_N)
def test_qPi_double2_split_properties(n):
    """s_i1 is the weight with its low s bits cleared, s_i2 is the rest
    (rounded to a double for N >= 13), and C'(1) = sum_i s_i1 * U_i can't
    round (paper eq. 6)."""
    s = t.QPI_DOUBLE2_SPLIT_BITS[n]
    worst = 0
    for w, (hi, lo), p in zip(t.crt_weights(n), t.qPi_double2(n), t.MODULI):
        assert lo == float(w - int(hi))
        assert int(hi) % 2**s == 0
        worst += int(hi) * (p // 2 + 1)  # |C_mid_i| <= p/2 (128 for p = 256)
    # All partial sums of hi_i * c_i are multiples of 2^s below 2^(s+53).
    assert worst < 2 ** (s + 53)


@pytest.mark.parametrize("n", DOUBLE2_N)
def test_vendored_split_keeps_1_to_2_more_bits_than_paper_eq6(n):
    """par_gemmul8's qPi_2 table splits 1-2 bits lower than paper eq. 6."""
    diff = t.qPi_double2_split_bits_paper(n) - t.QPI_DOUBLE2_SPLIT_BITS[n]
    assert diff == (2 if n in (17, 18) else 1)


@pytest.mark.parametrize("n", ALL_N)
def test_log2P_matches_vendored(n):
    assert t.log2P(n) == fx(v.log2P[n])
