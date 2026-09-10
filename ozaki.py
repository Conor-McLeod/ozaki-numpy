import numpy as np


def get_row_scaling_matrix(matrix):
    # returns an ndarray where the the i^th element is the max of the absolute values of
    # the i^th row of the matrix.
    row_amaxes = np.abs(matrix).max(axis=1)

    # np.abs(matrix) returns the matrix where all values are now positive.
    # .argmax(axis=1) then finds, for each row, the column index of the largest
    # value in that row. So argmax_idx is an ndarray where the i^th element is
    # the column index of the largest absolute value in the i^th row of the
    # matrix.
    argmax_idx = np.abs(matrix).argmax(axis=1)

    # matrix.shape[0] is the number of rows of matrix.
    # np.arange(matrix.shape[0]) generates the row indices given that size.
    # matrix[np.arange(matrix.shape[0]), argmax_idx] pairs each row index of
    # matrix with the column index containing the largest absolute value in that
    # row, then accesses each of those elements. So argmax_vals is an ndarray
    # where the i^th element is the actual entry in row i of the matrix whose
    # absolute value is the largest in the row. argmax_idx is basically
    # row_amaxes but the elements are the original signed values instead of the
    # absolute values.
    argmax_vals = matrix[np.arange(matrix.shape[0]), argmax_idx]

    # The i^th value in limits is 128 if the i^th value in argmax_vals is
    # negative, and 127 if the i^th value in argmax_vals is positive.
    limits = np.where(argmax_vals < 0, 128.0, 127.0)

    # np.where(row_amaxes == 0, 1.0, row_amaxes) returns a new matrix where
    # entries with value 0.0 in row_amaxes are changed to 1.0, and all other
    # entries are left the same. This prevents a divide by zero error from ever
    # occurring in the next part.
    # limits / np.where(row_amaxes == 0, 1.0, row_amaxes) divides each entry in
    # limits (either 128 or 127 depending on whether the amax for that row
    # was originally negative or positive) by each row amax.
    # The resulting raw scalars go through np.log2 to get the power needed to
    # raise 2 to to get each of those values.
    # Those are then floored (rounded down to nearest integer, not towards 0) to
    # produce the ndarray k, which holds the values of k such that taking each
    # i^th amax and multiplying it by 2.0 ** k scales it into INT8 range.
    k = np.floor(np.log2(limits / np.where(row_amaxes == 0, 1.0, row_amaxes)))

    # The overflow mask records the indexes of row_amaxes which even after
    # scaling are still above their respective limit, which could happen very
    # rarely when the max is very near a power of 2.
    overflow_mask = (row_amaxes * (2.0**k)) > limits
    # These indexes have their k decremented so that they are now within the
    # range. Now we can be certain that every multiplier will bring it's row
    # amax within the range.
    k[overflow_mask] -= 1

    # Turn the powers into actual multipliers.
    multipliers = 2.0**k

    return np.diag(multipliers)


def get_column_scaling_matrix(matrix):
    # Column scaling is just row scaling applied to the transpose: the
    # amax/argmax/limits logic is identical, just per-column instead of
    # per-row. The resulting diagonal matrix is applied on the right
    # (matrix @ D) so that the k^th diagonal entry multiplies the k^th
    # column instead of the k^th row.
    return get_row_scaling_matrix(matrix.T)


def main():
    rng = np.random.default_rng(2003)

    A = 10.0 ** rng.uniform(-5, 5, size=(3, 3)) * rng.choice([-1, 1], size=(3, 3))
    B = 10.0 ** rng.uniform(-5, 5, size=(3, 3)) * rng.choice([-1, 1], size=(3, 3))

    print(f"Matrix A: \n {A}")
    print(f"Matrix B: \n {B}")

    row_scale_A = get_scaling_matrix(A)
    col_scale_B = get_column_scaling_matrix(B)

    scaled_A = row_scale_A @ A
    scaled_B = B @ col_scale_B

    print(f"Row-scaled A: \n {scaled_A}")
    print(f"Column-scaled B: \n {scaled_B}")


if __name__ == "__main__":
    main()
