import numpy as np

# Different sizes for M, N and K so the shapes show which dimension is which.
# A is M x K, B is K x N, C is M x N.
M, N, K = 4, 6, 8
A = np.arange(M * K).reshape(M, K)
B = np.arange(K * N).reshape(K, N)


# ---------------------------------------------------------------------------
# 1. K-strips on one machine
#
# K is the dimension that gets summed over: C[i, j] = sum over k of
# A[i, k] * B[k, j]. So we can cut the sum in two. Cut A's columns and B's rows
# at the same k, multiply the matching halves, and add the results.
# ---------------------------------------------------------------------------

half_k = K // 2

# strip 0 is the first half of K, strip 1 the second half
A_s0 = A[:, :half_k]  # M x K/2, A's left half
A_s1 = A[:, half_k:]  # M x K/2, A's right half
B_s0 = B[:half_k, :]  # K/2 x N, B's top half
B_s1 = B[half_k:, :]  # K/2 x N, B's bottom half

# A_s0 only ever multiplies B_s0, and A_s1 only ever multiplies B_s1.
# Each product is already the full M x N size of C, just a partial sum of it.
C_strips = (A_s0 @ B_s0) + (A_s1 @ B_s1)

print("K-strips on one machine:", np.array_equal(C_strips, A @ B))  # True


# ---------------------------------------------------------------------------
# 2. The same thing on a 2x2 grid of ranks, the way ref_par_ozaki does it
#
# Now M and N are also cut in half, so C has four tiles and each rank owns one.
# A rank's grid position is (prow, pcol), with prow = rank % 2 and
# pcol = rank // 2:
#
#            pcol 0   pcol 1
#   prow 0   rank 0   rank 2
#   prow 1   rank 1   rank 3
# ---------------------------------------------------------------------------

half_m = M // 2
half_n = N // 2


def rows(p):
    # the rows of A / C that grid row p covers
    return slice(p * half_m, (p + 1) * half_m)


def cols(p):
    # the columns of B / C that grid column p covers
    return slice(p * half_n, (p + 1) * half_n)


def strip(s):
    # the K range of strip s
    return slice(s * half_k, (s + 1) * half_k)


# Every block is named by (grid row or column, strip):
#   A_block[(prow, s)] is A's rows for prow, K-strip s   (half_m x half_k)
#   B_block[(s, pcol)] is B's K-strip s, columns for pcol (half_k x half_n)
A_block = {(p, s): A[rows(p), strip(s)] for p in range(2) for s in range(2)}
B_block = {(s, p): B[strip(s), cols(p)] for s in range(2) for p in range(2)}

# The tile rank (prow, pcol) owns is the sum over both strips:
#   C[prow, pcol] = A[prow, s0] @ B[s0, pcol] + A[prow, s1] @ B[s1, pcol]


def starting_layout(prow, pcol):
    # What the scatter hands each rank: the A and B blocks sitting at its own
    # grid position. Because A's second index is the strip, the A block is from
    # strip pcol. Because B's first index is the strip, the B block is from
    # strip prow.
    return {
        "A": {(prow, pcol): A_block[(prow, pcol)]},
        "B": {(prow, pcol): B_block[(prow, pcol)]},
    }


def run_grid(layout, label):
    print(f"\n{label}")
    held = {}
    for rank in range(4):
        prow, pcol = rank % 2, rank // 2
        held[rank] = layout(prow, pcol)

    # Before any communication: which strips can each rank start right away?
    # A strip is ready only if the rank holds BOTH its A block and its B block.
    for rank in range(4):
        prow, pcol = rank % 2, rank // 2
        ready = [
            s
            for s in range(2)
            if (prow, s) in held[rank]["A"] and (s, pcol) in held[rank]["B"]
        ]
        # Print the strip as s0 / s1 so it's clear which index is the strip:
        # it's the second index of an A block and the first of a B block.
        a_names = [f"A[{p},s{s}]" for (p, s) in held[rank]["A"]]
        b_names = [f"B[s{s},{p}]" for (s, p) in held[rank]["B"]]
        can_start = ", ".join(f"strip {s}" for s in ready) or "nothing, must wait"
        print(
            f"  rank {rank} (prow {prow}, pcol {pcol}) holds "
            f"{' '.join(a_names + b_names)}, can start: {can_start}"
        )

    # The exchange. A rank's missing A block is in its own grid row, so A is
    # swapped with the row peer (same prow, other pcol). Its missing B block is
    # in its own grid column, so B is swapped with the column peer (same pcol,
    # other prow). Take a snapshot first: both swaps happen at once, so each
    # rank sends what it held before the swap.
    before = {r: {"A": dict(h["A"]), "B": dict(h["B"])} for r, h in held.items()}
    for rank in range(4):
        prow, pcol = rank % 2, rank // 2
        row_peer = prow + 2 * (1 - pcol)
        col_peer = (1 - prow) + 2 * pcol
        held[rank]["A"].update(before[row_peer]["A"])
        held[rank]["B"].update(before[col_peer]["B"])

    # Each rank now has all four blocks it needs, and adds up both strips.
    C_tiles = {}
    for rank in range(4):
        prow, pcol = rank % 2, rank // 2
        C_tiles[(prow, pcol)] = sum(
            held[rank]["A"][(prow, s)] @ held[rank]["B"][(s, pcol)]
            for s in range(2)
        )

    C_grid = np.block(
        [[C_tiles[(0, 0)], C_tiles[(0, 1)]], [C_tiles[(1, 0)], C_tiles[(1, 1)]]]
    )
    print("  matches A @ B:", np.array_equal(C_grid, A @ B))  # True


run_grid(starting_layout, "2x2 grid, ref_par_ozaki's layout")
# Only ranks 0 and 3 (prow == pcol, the diagonal) can start before the exchange.
# Ranks 1 and 2 hold half of each strip, so they sit idle until a swap lands.


# ---------------------------------------------------------------------------
# 3. A skewed starting layout (Cannon's algorithm), where every rank can start
#
# Give rank (prow, pcol) the A and B blocks of strip (prow + pcol) % 2. Then
# every rank starts with one complete strip. The missing blocks are still in the
# same row peer and column peer, so the exchange code above doesn't change.
# ---------------------------------------------------------------------------


def skewed_layout(prow, pcol):
    s = (prow + pcol) % 2
    return {
        "A": {(prow, s): A_block[(prow, s)]},
        "B": {(s, pcol): B_block[(s, pcol)]},
    }


run_grid(skewed_layout, "2x2 grid, skewed layout")
