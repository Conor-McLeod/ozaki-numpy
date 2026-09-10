import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import math

    return math, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Scaling
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First we generate our random input matrices $A$ and $B$, for which we want to compute $AB = C$.

    The values are both positive and negative and range across magnitudes to make it interesting.
    """)
    return


@app.cell
def _(np):
    rng = np.random.default_rng(2003)

    A = 10.0 ** rng.uniform(-5, 5, size=(3, 3)) * rng.choice(
        [-1, 1], size=(3, 3)
    )
    B = 10.0 ** rng.uniform(-5, 5, size=(3, 3)) * rng.choice(
        [-1, 1], size=(3, 3)
    )

    print(A)
    print(B)
    return (A,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We need to form the scaling matrices. These will scale A and B so that they fit within the int8 range.

    The scaling matrices are diagonal matrices, because we know that applying a diagonal matrix to another matrix is the same as taking the value of the $k^{th}$ diagonal and multiplying it by each entry in the $k^{th}$ row. For example:

    $$
    \begin{bmatrix}
    a & 0 & 0 \\
    0 & b & 0 \\
    0 & 0 & c
    \end{bmatrix}
    \begin{bmatrix}
    5 & 2 & 2 \\
    8 & 4 & 7 \\
    1 & 0 & 9
    \end{bmatrix}
    =
    \begin{bmatrix}
    5a & 2a & 2a \\
    8b & 4b & 7b \\
    c & 0 & 9c
    \end{bmatrix}
    $$

    To do this, we only need to find the absolute max of each row (for A) and each column (for B). We figure out what multiplier we have to apply to get it into the INT8 range, and apply this to the whole row.
    """)
    return


@app.cell
def _(A, np):
    # find row amaxes
    row_amaxes = np.abs(A).max(axis=1)
    print(row_amaxes)
    what = np.amax(A, axis=1)
    print(what)
    scaling_matrix = np.diag(row_amaxes)
    print(scaling_matrix)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We want to scale a find the largest integer
    """)
    return


@app.cell
def _(math):
    math.log2(16)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's say `a` is -320. Too big to fit into INT8.
    Since `a < 0`, `limit = 128`

    `limit / abs(a)`  =  128 / 320 = 0.4

    This just tells us that if we multiply `a` by 0.4 it will exactly be in the range. But our scaling doesn't just apply any multiple, the multiplier has to be a power of 2.

    `math.log2()` tells us the power we would have to raise 2 to to get what we pass it.

    So `log2(limit / abs(a))` tells us the power we have to raise 2 to to get 0.4, sucn that `2 ** log2(limit / abs(a))` just gives us 0.4 right back. That value is -1.3219280948873622.

    0.4 is not a power of 2, (and most of the time `limit / abs(a)` won't be) so the result is a decimal power, but we need an integer power. So we floor the result. `math.floor` rounds to the *next lowest integer*, not to 0. So `math.floor(-1.3219280948873622)` returns `-2`, not `-1`.

    Finally, if we multiply our original `a = -320` by `2 ** -2`, we get -80, which is within the INT8 range. That is, we divided it by 16 to fit it into the INT8 range.

    If we had done `k + 1 = -1`, then we only half it and get -160, which still has to big of an absolute value and doesn't fit in the range.

    If we had done `k - 1 = -3`, then we scale it by $\frac{1}{8}$ and get $-40$, which is in the range, but we don't want to scale numbers more than we have to, since the following truncation step is more severe the smaller the numbers are.
    """)
    return


@app.cell
def _(math):
    a = -320
    print(128 / abs(a))
    print(math.log2(128 / abs(a)))
    print(2 ** math.log2(128 / abs(a)))
    k = math.floor(math.log2(128 / abs(a)))
    print(k)

    print(a * 2**k)

    print(a * (2 ** (k + 1)))
    print(a * (2 ** (k - 1)))
    return (a,)


@app.cell
def _(a, math):
    def max_shift(row_max):
        if row_max == 0:
            return None
        # Encodes the [-128, 127] range.
        limit = 127 if a > 0 else 128

        k = math.floor(math.log2(128 / abs(row_max)))

        # log2 could (rarely) be a hair off if a is very close to a power of two,
        # so these two while loops nudge it if that is the case.

        # "if a * 2^(k+1) is still under the limit (we scaled down too agressively)
        # then update k to k+1"
        # again this is defensive and should rarely be the case.
        # mostly a * 2^(k+1) is over the limit; 2^k is already the largest multiplier
        # that still puts a within the INT8 range
        while abs(row_max) * 2.0 ** (k + 1) <= limit:
            k += 1

        # "if a * 2^(k+1) is above under the limit, then update k to k+1""
        # In this case because of precision details our k could in fact cause a to still be a hair outside 
        # of the range. This correct for these edge cases.
        while abs(row_max) * 2.0**k > limit:
            k -= 1

        return k

    return (max_shift,)


@app.cell
def _(A, max_shift, np):
    def get_scaling_matrix(matrix):

        row_amaxes = np.abs(matrix).max(axis=1)

        powers = []
        for amax in row_amaxes:
            powers.append(max_shift(amax))

        diag_elements = 2.0 ** np.array(powers)
    
        return(np.diag(diag_elements))

    scale_1 = get_scaling_matrix(A)

    scaled_A = scale_1 @ A
    print(A)
    print(scaled_A)
    return


if __name__ == "__main__":
    app.run()
