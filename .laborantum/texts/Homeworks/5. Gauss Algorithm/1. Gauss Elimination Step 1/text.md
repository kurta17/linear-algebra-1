import numpy as np

def gauss_elimination_1(A, B, permutations=True):
    A = A.astype(float)  # Convert to float for numerical stability
    B = B.astype(float)
    m, n = A.shape
    P = np.eye(m)  # Initialize permutation matrix as identity matrix
    
    for i in range(min(m, n)):
        # Find the row with the maximum absolute value in the current column
        max_row = np.argmax(np.abs(A[i:m, i])) + i
        
        # If the maximum absolute value in the current column is zero, skip this column
        if np.abs(A[max_row, i]) < 1e-10:
            continue
        
        # Perform row swaps in A, B, and P to maximize stability
        if i != max_row:
            A[[i, max_row]] = A[[max_row, i]]
            B[[i, max_row]] = B[[max_row, i]]
            P[[i, max_row]] = P[[max_row, i]]
        
        # Eliminate elements below the pivot
        for j in range(i + 1, m):
            if A[j, i] != 0:
                factor = A[j, i] / A[i, i]
                A[j, i:] -= factor * A[i, i:]
                B[j] -= factor * B[i]
    
    return P, A, B
