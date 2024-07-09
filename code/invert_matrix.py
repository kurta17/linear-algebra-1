from gauss_elimination_1 import gauss_elimination_1
from gauss_elimination_2 import gauss_elimination_2

import numpy as np

def invert_matrix(A):
    m, n = A.shape
    if m != n:
        raise ValueError("Input matrix must be square")
    
    I = np.eye(m)
    
    P, U, y = gauss_elimination_1(A, I)
    inv_A = gauss_elimination_2(U, y)
    
    if np.allclose(np.diag(U), 0):
        return None
    
    return inv_A