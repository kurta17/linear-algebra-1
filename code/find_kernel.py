import numpy as np

def gauss_elimination_1(A, B, permutations=True):
    A = A.astype(float)
    B = B.astype(float)
    P = np.eye(A.shape[0])

    n = A.shape[0]

    for k in range(n):
        if permutations:
            max_row_index = np.argmax(np.abs(A[k:, k])) + k
            if max_row_index != k:
                A[[k, max_row_index]] = A[[max_row_index, k]]
                B[[k, max_row_index]] = B[[max_row_index, k]]
                P[[k, max_row_index]] = P[[max_row_index, k]]

        if A[k, k] == 0:
            continue

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            B[i] -= factor * B[k]

        diag_factor = A[k, k]
        if diag_factor != 0:
            A[k] /= diag_factor
            B[k] /= diag_factor

    return P, A, B

def gauss_elimination_2(A, B):
    A = A.astype(float)
    B = B.astype(float)
    n = A.shape[0]

    for i in range(n-1, -1, -1):
        if A[i, i] == 0:
            continue 
        for j in range(i-1, -1, -1):
            factor = A[j, i] / A[i, i]
            A[j, :] -= factor * A[i, :]
            B[j] -= factor * B[i]
    
    return A, B

def find_kernel(A):
    A = A.astype(float)
    n, m = A.shape

    P, U, _ = gauss_elimination_1(A, np.eye(n), permutations=True)
    
    U, _ = gauss_elimination_2(U, np.eye(n))

    pivot_columns = []
    for j in range(m):
        if any(np.isclose(U[i, j], 1) for i in range(n)) and all(U[i, j] == 0 or np.isclose(U[i, j], 1) for i in range(n)):
            pivot_columns.append(j)

    free_vars = [j for j in range(m) if j not in pivot_columns]

    null_space_basis = []
    for free_var in free_vars:
        basis_vector = np.zeros(m)
        basis_vector[free_var] = 1
        for i in range(len(pivot_columns)):
            row_index = np.where(np.isclose(U[:, pivot_columns[i]], 1))[0][0]
            basis_vector[pivot_columns[i]] = -U[row_index, free_var]
        null_space_basis.append(basis_vector)
    
    return np.array(null_space_basis).T