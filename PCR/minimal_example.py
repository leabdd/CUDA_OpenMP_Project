from pcr import *

# Tridiagonale Matrix A definieren
A = np.array([[4, 3, 0, 0], [8, 4, 9, 0], [0, 2, 8, 4], [0, 0, 4, 8]]).astype(float)

# Rechte Seite y definieren
y = np.array([1, 2, 3, 4])

# PCR-Solver initialisieren und lösen
x = solve(A, y)
print("x = ", x)

# Test
print("y_test = ", A.dot(x))


if np.isclose(x, np.linalg.solve(A, y), 1e-10).all():
    print(
        "The PCR and Numpy Linalg solvers are equal within a tolerance of at least 1E-10."
    )
