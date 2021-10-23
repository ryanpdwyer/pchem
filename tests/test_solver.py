import pchem
import sympy as sm
from pytest import approx


def test_solve():
    P, V, R, T = sm.symbols('P V R T',
                positive=True)
    subs=dict(P=1, V=1, R=1)
    soln = pchem.solve(P*V - R*T, T, subs)
    assert soln == approx(1.0)