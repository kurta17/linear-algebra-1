import numpy as np

def qr_decomposition(A):
    A = A.astype(float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for k in range(n):
        u = A[:, k].copy()
        for j in range(k):
            R[j, k] = np.dot(Q[:, j], A[:, k])
            u -= R[j, k] * Q[:, j]
        
        R[k, k] = np.linalg.norm(u)
        Q[:, k] = u / R[k, k]
    
    return Q, R

