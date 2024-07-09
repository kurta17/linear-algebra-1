from gauss_elimination_1 import gauss_elimination_1
from gauss_elimination_2 import gauss_elimination_2

import numpy as np

def find_kernel(A):
    A = np.array(A)
    m, n = A.shape
    B = np.zeros((m, n))
    
    P, A_stage1, B_stage1 = gauss_elimination_1(A.copy(), B)
    
    pivots = []
    for i in range(min(m, n)):
        if np.any(A_stage1[i, :] != 0):
            pivots.append(np.argmax(A_stage1[i, :] != 0))
    
    free_vars = [i for i in range(n) if i not in pivots]
    
    null_space_basis = []
    for free_var in free_vars:
        basis_vector = np.zeros(n)
        basis_vector[free_var] = 1
        for row, pivot in enumerate(pivots):
            if pivot < free_var:
                basis_vector[pivot] = -A_stage1[row, free_var]
        null_space_basis.append(basis_vector)
    
    return np.array(null_space_basis).T