import numpy as np

A = np.arange(16).reshape(4, 4)
B = np.arange(16).reshape(4, 4)
D = np.arange(15).reshape(5, 3)


print(A)
print(B)

print(A.shape)

# Split into blocks of size 2x2 each
A00 = A[0:2, 0:2]
print(A00)


# naive implementation to begin with
def split_into_quarters(matrix):
    if matrix.shape[0] % 2 != 0 or matrix.shape[1] % 2 != 0:
        raise ValueError("Matrix dimensions must be divisible by 2")

    blocks = []
    # i is row index, j is column index
    for i in range(2):
        for j in range(2):
            # floor division operator // gives us an int. Just regular division
            # always returns a float
            block_width = matrix.shape[0] // 2
            block_height = matrix.shape[1] // 2

            block = matrix[
                i * block_width : i * block_width + block_width,
                j * block_height : j * block_height + block_height,
            ]
            blocks.append(block)

    return tuple(blocks)


A00, A01, A10, A11 = split_into_quarters(A)
print(A00)
print(A01)
