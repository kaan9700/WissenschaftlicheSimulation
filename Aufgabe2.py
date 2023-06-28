"""
Aufgabe:
Schreiben Sie ein Programm,
welches ein lineares Gleichungssystem Ax = b, A ∈ Rn×n, b ∈ Rn mittels LR Zerlegung löst.
"""

import numpy as np


def is_square(A):
    """
    Überprüft, ob die Matrix quadratisch ist.
    """
    return A.shape[0] == A.shape[1]



def is_invertible(matrix):
    """
    Überprüft, ob die Matrix invertierbar ist.
    """
    return np.linalg.det(matrix) != 0


def pivoting(A, b):
    """
    Implementiert Partial Pivoting, um numerische Stabilität zu gewährleisten.
    """
    n = len(A)
    M = A
    I = np.eye(n)
    for i in range(n):
        maxindex = abs(M[i:, i]).argmax() + i
        if M[i, i] < M[maxindex, i]:
            M[[i, maxindex]] = M[[maxindex, i]]
            I[[i, maxindex]] = I[[maxindex, i]]
            b[[i, maxindex]] = b[[maxindex, i]]
    return M, I, b

def lr_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    R = np.zeros((n, n))

    for i in range(n):
        for k in range(i, n):
            R[i, k] = A[i, k] - sum(L[i, j] * R[j, k] for j in range(i))
        for k in range(i, n):
            if (i == k):
                L[k, i] = 1
            else:
                L[k, i] = (A[k, i] - sum(L[k, j] * R[j, i] for j in range(i))) / R[i, i]
    return L, R

def forward_substitution(L, b):
    n = len(L)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i, j] * y[j] for j in range(i))) / L[i, i]
    return y

def backward_substitution(R, y):
    n = len(R)
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(R[i, j] * x[j] for j in range(i + 1, n))) / R[i, i]
    return x

def solve_linear_equation(A, b):
    """
    Löst das lineare Gleichungssystem Ax = b mit numpy.linalg.solve.
    """
    if not is_square(A):
        raise ValueError("Die Matrix A ist nicht quadratisch.")
    if A.shape[0] != b.shape[0]:
        raise ValueError("Die Dimensionen von A und b sind nicht kompatibel.")
    if not is_invertible(A):
        raise ValueError("Die Matrix A ist singulär (nicht invertierbar).")
    return np.linalg.solve(A, b)




def test():
    A = np.array([[1, -1, 2, -1], [2, -2, 3, -3], [1, 1, 1, 0], [1, -1, 4, 3]], float)
    b = np.array([-8, -20, -2, 4], float)

    x = solve_linear_equation(A, b)
    print("Lösungsvektor x: ", x)

    # Erzeugen einer quadratischen, aber nicht invertierbaren Matrix
    A2 = np.array([[2, 4], [1, 2]], float)
    b2 = np.array([1, 2], float)
    try:
        x2 = solve_linear_equation(A2, b2)
    except ValueError as e:
        print("Erwarteter Fehler: ", e)

    # Erzeugen einer nicht-quadratischen Matrix
    A3 = np.array([[1, 2, 3], [4, 5, 6]], float)
    b3 = np.array([1, 2], float)
    try:
        x3 = solve_linear_equation(A3, b3)
    except ValueError as e:
        print("Erwarteter Fehler: ", e)

    # Überprüfung auf Unstimmigkeit in den Dimensionen
    A4 = np.array([[1, 2], [3, 4]], float)
    b4 = np.array([1, 2, 3], float)
    try:
        x4 = solve_linear_equation(A4, b4)
    except ValueError as e:
        print("Erwarteter Fehler: ", e)

test()
