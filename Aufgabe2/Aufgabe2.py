import numpy as np



def check_matrix(A, b):
    m, n = A.shape
    if m != n:
        raise ValueError("Die Matrix A ist nicht quadratisch.")
    if np.linalg.det(A) == 0:
        raise ValueError("Die Matrix A ist singulär (nicht invertierbar).")
    if A.shape[0] != b.shape[0]:
        raise ValueError("Die Dimensionen von A und b sind nicht kompatibel.")
    return True



def PLR(A):

    n = A.shape[0]
    L = np.eye(n)
    P = np.eye(n)
    # Elemente der Matrix sollen als float gespeichert werden, da die Division zweier Ganzzahlen
    # in Python eine Ganzzahl zurückgibt
    R = A.astype('float')

    # Gehe durch jede Spalte der Matrix R.
    for k in range(n - 1):
        # Finde das Element mit dem größten Absolutwert in der aktuellen Spalte unter oder auf der Diagonalen.
        p = np.argmax(abs(R[k:n, k])) + k

        # Tausche die Zeile mit dem größten Element mit der aktuellen Zeile in R und P.
        R[[k, p]] = R[[p, k]]
        P[[k, p]] = P[[p, k]]

        # Tausche die Zeile in L, aber nur die ersten k Elemente, da der Rest von L noch nicht definiert ist.
        L[[k, p], :k] = L[[p, k], :k]

        # Für jede Zeile unter der aktuellen...
        for j in range(k + 1, n):
            # ...berechne das Element der L-Matrix und...
            L[j, k] = R[j, k] / R[k, k]

            # ...aktualisiere die Zeile in R.
            R[j, k:] = R[j, k:] - L[j, k] * R[k, k:]

    # Gib die Matrizen P, L und R zurück.
    return P, L, R


def forward_substitution(L, Pb):
    # Bestimme die Größe des Eingangsvektors.
    n = Pb.shape[0]

    # Initialisiere den Ausgabevektor y als Nullvektor der gleichen Größe wie Pb.
    y = np.zeros_like(Pb)

    # Für jedes Element in y...
    for i in range(n):
        # ... berechne den Wert als das entsprechende Element in Pb minus die Summe der bereits berechneten y-Werte,
        # jeweils multipliziert mit dem entsprechenden Wert in L. Das Symbol '@' steht hier für das Skalarprodukt.
        y[i] = Pb[i] - L[i, :i] @ y[:i]

    # Gib den Vektor y zurück.
    return y


def backward_substitution(R, y):
    # Bestimme die Größe des Eingangsvektors.
    n = len(y)

    # Initialisiere den Ausgabevektor x als Nullvektor der gleichen Größe wie y.
    x = np.zeros_like(y)

    # Für jedes Element in x, beginnend beim letzten...
    for i in reversed(range(n)):
        # ... berechne den Wert als das entsprechende Element in y minus die Summe der bereits berechneten x-Werte,
        # jeweils multipliziert mit dem entsprechenden Wert in R, und dann geteilt durch das diagonale Element in R.
        x[i] = (y[i] - R[i, i + 1:] @ x[i + 1:]) / R[i, i]

    # Gib den Vektor x zurück.
    return x

def solve_linear_equation(A, b):
    if check_matrix(A, b):
        P, L, R = PLR(A)
        Pb = P @ b # Skalarprodukt aus P und b
        y = forward_substitution(L, Pb)
        x = backward_substitution(R, y)

        return x


def test():
    A = np.array([[-2, 0, 2], [3, 2, -2], [6, 6, 3]])
    b = np.array([0, -1, 3])

    x = solve_linear_equation(A, b)
    print("Lösungsvektor x: ", x)

    # Erzeugen einer quadratischen, aber nicht invertierbaren Matrix
    A2 = np.array([[2, 4], [1, 2]], float)
    b2 = np.array([1, 2], float)
    try:
        solve_linear_equation(A2, b2)
    except ValueError as e:
        print("Erwarteter Fehler: ", e)

    # Erzeugen einer nicht-quadratischen Matrix
    A3 = np.array([[1, 2, 3], [4, 5, 6]], float)
    b3 = np.array([1, 2], float)
    try:
        solve_linear_equation(A3, b3)
    except ValueError as e:
        print("Erwarteter Fehler: ", e)

    # Überprüfung auf Unstimmigkeit in den Dimensionen
    A4 = np.array([[1, 2], [3, 4]], float)
    b4 = np.array([1, 2, 3], float)
    try:
        solve_linear_equation(A4, b4)
    except ValueError as e:
        print("Erwarteter Fehler: ", e)



if __name__ == '__main__':
    test()