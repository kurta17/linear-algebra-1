import numpy as np

def gauss_elimination_2(A, B):
    A = A.astype(float)
    B = B.astype(float)
    m, n = A.shape
    
    for i in range(min(m, n) - 1, -1, -1):
        if A[i, i] != 0:
            B[i] /= A[i, i]
            A[i, i:] /= A[i, i]
            for j in range(i):
                factor = A[j, i]
                A[j, i:] -= factor * A[i, i:]
                B[j] -= factor * B[i]
    
    return A, B