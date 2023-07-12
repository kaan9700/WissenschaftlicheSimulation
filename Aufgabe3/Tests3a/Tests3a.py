from Aufgabe3.Aufgabe3a import bisection
import numpy as np
def test_bisection():
    # Test 1: Funktion mit einer Nullstelle im gegebenen Intervall
    f1, a1, b1, n1, e1 = "x^2 - 4", 0, 5, 1000, 1e-5
    root1, iterations1 = bisection(f1, a1, b1, n1, e1)
    assert np.isclose(root1, 2, atol=1e-5), f"Test 1 fehlgeschlagen: Erwartet 2, bekommen {root1}"

    # Test 2: Funktion mit einer Nullstelle im gegebenen Intervall
    f2, a2, b2, n2, e2 = "x^3 - 27", 1, 5, 1000, 1e-5
    root2, iterations2 = bisection(f2, a2, b2, n2, e2)
    assert np.isclose(root2, 3, atol=1e-5), f"Test 2 fehlgeschlagen: Erwartet 3, bekommen {root2}"

    # Test 3: Funktion mit einer Nullstelle im gegebenen Intervall
    f3, a3, b3, n3, e3 = "x^2 + 2*x - 8", 0, 5, 1000, 1e-5
    root3, iterations3 = bisection(f3, a3, b3, n3, e3)
    assert np.isclose(root3, 2, atol=1e-5), f"Test 3 fehlgeschlagen: Erwartet 2, bekommen {root3}"

    # Test 4: Funktion mit einer Nullstelle im gegebenen Intervall
    f4, a4, b4, n4, e4 = "x^3 - x^2 - x - 1", 1, 2, 1000, 1e-5
    root4, iterations4 = bisection(f4, a4, b4, n4, e4)
    assert np.isclose(root4, 1.83929, atol=1e-5), f"Test 4 fehlgeschlagen: Erwartet 1.83929, bekommen {root4}"

    print("Alle Tests bestanden.")

test_bisection()
