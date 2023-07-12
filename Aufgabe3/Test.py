import sympy as sp
from sympy import Interval
from sympy.calculus.util import continuous_domain

x = sp.symbols('x')


def is_continuous(f, a, b):
    return continuous_domain(f, x, Interval(a, b)).is_Interval


def is_twice_differentiable(f, a, b):
    x = sp.symbols('x')
    first_derivative = sp.diff(f, x)
    second_derivative = sp.diff(first_derivative, x)
    return is_continuous(first_derivative, a, b) and is_continuous(second_derivative, a, b)


def tests():
    x = sp.symbols('x')

    # Test für eine kontinuierliche, aber nicht zweimal differenzierbare Funktion
    f1 = sp.Abs(x)
    assert is_continuous(f1, -1, 1) == True
    assert is_twice_differentiable(f1, -1, 1) == False

    # Test für eine zweimal differenzierbare Funktion
    f2 = sp.sin(x)
    assert is_continuous(f2, -sp.pi, sp.pi) == True
    assert is_twice_differentiable(f2, -sp.pi, sp.pi) == True

    f3 = sp.sqrt(x)
    assert is_continuous(f3, 1, 2) == True
    assert is_twice_differentiable(f3, 0, 2) == True

    f4 = 1/x
    assert is_continuous(f4, -1, 1) == False
    assert is_twice_differentiable(f4, -1, 1) == False

    print("Alle Tests bestanden!")


tests()

f = sp.cos(x)-x

print(is_continuous(f, 1, 2))
