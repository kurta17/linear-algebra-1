import numpy as np

def solve_sle(A):
  """
  Solves a system of linear equations using Gaussian elimination.

  Args:
    A: A numpy array representing the augmented matrix of the system.

  Returns:
    A dictionary containing the following keys:
      'partial': A numpy array representing a particular solution of the system.
      'null_space': A list of numpy arrays representing the basis vectors of the null space.
  """

  # Gaussian elimination
  m, n = A.shape
  for i in range(m):
    if abs(A[i, i]) < 1e-7:
      # Handle cases where the pivot element is very small
      continue
    for j in range(i+1, m):
      factor = A[j, i] / A[i, i]
      A[j, :] -= factor * A[i, :]

  # Back substitution
  partial = np.zeros(n)
  for i in range(m-1, -1, -1):
    if abs(A[i, i]) < 1e-7:
      continue
    partial[i] = (A[i, n-1] - np.sum(A[i, i+1:] * partial[i+1:])) / A[i, i]

  # Find null space basis vectors
  null_space = []
  for col in range(n-m, n):
    if abs(A[0, col]) < 1e-7:
      continue
    basis = np.zeros(n)
    basis[col] = 1
    for i in range(m):
      if A[i, col] != 0:
        basis[i] = -A[i, col] / A[0, col]
    null_space.append(basis.reshape(-1, 1))

  return {'partial': partial.reshape(-1, 1), 'null_space': null_space}

# Coefficient matrix
A = np.array([[1, 1, 1], [3, 0, 3]]).astype(float)


# Solve the system
solution = solve_sle(A.copy())

# Print the answer in the specified format
print("task_1:", solution)
