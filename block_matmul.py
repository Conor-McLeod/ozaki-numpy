import numpy as np

A = np.arange(16).reshape(4, 4)
B = np.arange(16).reshape(4, 4)
D = np.arange(15).reshape(5, 3)


print(A)
print(B)


# naive implementation to begin with
def split_into_quarters(matrix, as_tuple=True):
    rows, cols = matrix.shape
    if rows % 2 != 0 or cols % 2 != 0:
        raise ValueError("Matrix dimensions must be divisible by 2")

    blocks = []
    # i is row index, j is column index
    for i in range(2):
        for j in range(2):
            # floor division operator // gives us an int. Just regular division
            # always returns a float

            # block height computed from the row count // 2
            block_height = rows // 2
            # block width compute from the column count // 2
            block_width = cols // 2

            # when we do matrix[], the first range is the row range and the
            # second range is the column range. E.g. A[0:2, 0:3] selects rows 0
            # and 1 and columns 0,1,2.
            block = matrix[
                i * block_height : i * block_height + block_height,
                j * block_width : j * block_width + block_width,
            ]
            blocks.append(block)

    if as_tuple:
        return tuple(blocks)
    
    return np.block([blocks[0:2], blocks[2:4]])


# [A00 A01]
# [A10 A11]
A00, A01, A10, A11 = split_into_quarters(A)
B00, B01, B10, B11 = split_into_quarters(B)

C00 = (A00 @ B00) + (A01 @ B10)
C01 = (A00 @ B01) + (A01 @ B11)
C10 = (A10 @ B00) + (A11 @ B10)
C11 = (A10 @ B01) + (A11 @ B11)

A_blocked

# Reassemble
C_blocked = np.block([[C00, C01], [C10, C11]])

# Compare to direct multiplication
C_direct = A @ B

print(np.array_equal(C_blocked, C_direct))  # True

try:
    split_into_quarters(D)
except ValueError as e:
    print(e)
