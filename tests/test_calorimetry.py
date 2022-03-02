from pchem import calorimetry as cal
from pytest import approx
from numpy import testing
from scipy import optimize, linalg
import numpy as np

# Is it better to do...
# dq = dq_heat + dq_resistor + dq_rxns...
# dS_rxns
# dS_universe = -dq_heat/T_surr + -dq_rxns/T_surr + dS_rxns
# max dS_universe...

def test_melting():
    # s0 = cal.State(s = {0: 0.0, 10: 1.0}, T=273.15)
    H2O = cal.Substance("H2O(l)", state='l', H0=-285.826e3, S0=69.96, cP=4.18*18.02, molarMass=18.02,
                        density=1.0)

    dH_ice = -6.01e3
    T_ice  = 273.15
    ice = cal.phaseChange(dH_ice, T_ice, H2O, cP=2.108*H2O.molarMass,
                    name="ice", state='s', density=0.91)

    chemicals = {0: H2O, 10: ice}

    rxns = [{0: 1, 10: -1}]

    s0 = cal.State3(273.15, chemicals=chemicals, rxns=rxns, V=1.0)

    s0.set_state({0: 0, 10: 1.0})

    A, b = cal.arrayRxns(s0.state, s0.rxns)
    Ag = np.c_[A, np.zeros(len(s0.state))] # Add zeros for temp?
    Tsurr = 298.15
    k = 0.1
    x = np.array([-1e-3,1.0])
    print(Ag@x+b)
    dH_step = -(s0.T - Tsurr)*k # 0.1 J

    def HconstaintFunc(x):
        state = Ag@x+b
        T = s0.T + x[-1]
        return s0.H(x=state, T=T) - s0.H(x=b)

    Hconstraint = optimize.NonlinearConstraint(HconstaintFunc, lb=dH_step, ub=dH_step)
    bounds = optimize.LinearConstraint(Ag, lb=-b, ub=np.inf) # Reaction constraint...

    def Sfunc(x):
        state = Ag@x+b
        T = s0.T + x[-1]
        return -s0.S(x=state, T=T)
    res = optimize.minimize(Sfunc, x0=[0.0, 0.0], constraints=[bounds, Hconstraint])
        #method='SLSQP')
    x_opt = res.x
    print(res)

    s_new = Ag @ x_opt + b
    T_new = s0.T + x_opt[-1]

    print(s0.H(x=s_new, T=T_new) - s0.H())

    print(s_new)

    assert s_new[0] == approx(dH_step/6010.0, rel=0.0005)
    assert T_new == approx(273.15)



def test_rxn():
    H2O = cal.Substance("H2O(l)", state='l', H0=-285.826e3, S0=69.96,
                        cP=4.18*18.02, molarMass=18.02, density=1.0)
    Hplus = cal.Substance("H+(aq)", state="aq", H0=0, S0=0, cP=0, molarMass=1.01, density=0)
    OH = cal.Substance("OH-(aq)", state="aq", H0=-230.02e3, S0=-10.9, molarMass=17.01, cP=-148.5, density=0)

    # H2Ogas = cal.Substance("H2O(g)", state="g", H0=-241.83e3, S0=188.84, cP=1.864*18.02, molarMass=18.02)



    chemicals = {0: H2O, 1: Hplus, 2: OH}

    rxns = [{0: -1, 1: 1, 2: 1}]

    s0 = cal.State3(T=298.15, chemicals=chemicals, rxns=rxns, V=1.0)

    s0.set_state({0: 1000/18.02-1.0, 1: 1.0, 2: 1.0})

    A, b = cal.arrayRxns(s0.state, s0.rxns)
    print(f"{A=}\n{b=}")
    assert 0 == 1
    # Ag = np.c_[A, np.zeros(len(s0.state))] # Add zeros for temp?
    # Tsurr = 298.15
    # k = 0.1
    # dH_step = -(s0.T - Tsurr)*k # 0.1 J

    # def HconstaintFunc(x):
    #     state = Ag@x+b
    #     T = s0.T + x[-1]
    #     return s0.H(x=state, T=T) - s0.H(x=b)

    # Hconstraint = optimize.NonlinearConstraint(HconstaintFunc, lb=dH_step, ub=dH_step)
    # bounds = optimize.LinearConstraint(Ag, lb=-b, ub=np.inf) # Reaction constraint...

    # def Sfunc(x):
    #     state = Ag@x+b
    #     T = s0.T + x[-1]
    #     return -s0.S(x=state, T=T)
    # res = optimize.minimize(Sfunc, x0=[0.0, 0.0], constraints=[bounds, Hconstraint])
    #     #method='SLSQP')
    # x_opt = res.x
    # print(res)

    # s_new = Ag @ x_opt + b
    # T_new = s0.T + x_opt[-1]

    # print(s0.H(x=s_new, T=T_new) - s0.H())

    # print("Properties:\n")
    # cP=s0.get_prop('cP')
    # conc = s0.conc()
    # print(f"{cP=}")
    # print(f"{conc=}")
    # b_in = np.array(list(s0.state.values()))
    # print(f"{b_in=}")
    # cPtot = cP @ b_in
    # H_out = s0.get_prop_conc('Hbar')
    # print(f"{cPtot=}")
    # print(f"{H_out=}")
    # print(list(s0.state.keys()))
    # print(f"{s_new=}")
    # print(f"{T_new=}")

    # assert 0==1


def test_basic_rxn_equilibrium():
    P = cal.Substance("P(aq)", state="aq", H0=0, S0=0, cP=0, molarMass=100.0, density=1.0)
    D = cal.Substance("D(aq)", state="aq", H0=0, S0=0, cP=0, molarMass=100.0, density=1.0)
    PD = cal.Substance("PD(aq)", state="aq", H0=0, S0=50.0, cP=0, molarMass=100.0, density=1.0)

    chemicals = {0: P, 1: D, 2: PD}
    rxns = [{0: -1, 1: -1, 2: 1}]

    s0 = cal.State3(T=298.15, chemicals=chemicals, rxns=rxns, V=1.0)

    s0.set_state({0: 1.0, 1: 1.0, 2: 0.0})
    

    res, s_new= s0._solve_constT(T=298.15)

    print(f"{res=}\n{s_new=}")

    assert 0 == 1
    




def test_rxn_rescaled():
    H2O = cal.Substance("H2O(l)", state='l', H0=-285.826e3, S0=69.96,
                        cP=4.18*18.02, molarMass=18.02, density=1.0)
    Hplus = cal.Substance("H+(aq)", state="aq", H0=0, S0=0, cP=0, molarMass=1.01, density=0)
    OH = cal.Substance("OH-(aq)", state="aq", H0=-230.02e3, S0=-10.9, molarMass=17.01, cP=-148.5, density=0)

    chemicals = {0: H2O, 1: Hplus, 2: OH}

    rxns = [{0: -1, 1: 1, 2: 1}]

    s0 = cal.State3(T=298.15, chemicals=chemicals, rxns=rxns, V=1.0)

    s0.set_state({0: 1000/18.02-1.0, 1: 1.0, 2: 1.0})

    res, s_new, T_new = s0._solve(dH=0)
    print(s0.conc())
    # This is good
    # Unfortunately, 
    print(f"{res=}\n{s_new=}\n{T_new=}")

    s0.set_state(dict(zip(s0.state.keys(), s_new)))
    s0.T = T_new

    res2, s_new2, T_new2 = s0._solve(dH=0)

    # res2, s_new2 = s0._solve_constT(T=298.15)

    print("Second optimization:")

    print(f"{res2=}\n{s_new2=}")



    assert 0==1




def test_basic_properties():
    H2O = cal.Substance("H2O(l)", state='l', H0=-285.826e3, S0=69.96,
                        cP=4.18*18.02, molarMass=18.02, density=1.0)
    Hplus = cal.Substance("H+(aq)", state="aq", H0=0, S0=0, cP=0, molarMass=1.01, density=0)
    OH = cal.Substance("OH-(aq)", state="aq", H0=-230.02, S0=-10.9, molarMass=17.01, cP=-148.5, density=0)

    # H2Ogas = cal.Substance("H2O(g)", state="g", H0=-241.83e3, S0=188.84, cP=1.864*18.02, molarMass=18.02)



    chemicals = {0: H2O, 1: Hplus, 2: OH}

    rxns = [{0: -1, 1: 1, 2: 1}]

    s0 = cal.State2(s = {0: 1.0, 1: 0.0, 2: 0.0}, T=273.15, chemicals=chemicals, rxns=rxns)



    assert s0.mass() == approx(18.02)
    assert s0.volume() == approx(18.02e-3)
    testing.assert_allclose([1, 0,0], s0.conc())

    # moles = 10.0/abs(dH_ice)
    # assert s1.s[0] == approx(moles, rel=1e-4) # Slightly larger error here
    # assert s1.T == approx(T_ice)

def test_state3():
    H2O = cal.Substance("H2O(l)", state='l', H0=-285.826e3, S0=69.96,
                        cP=4.18*18.02, molarMass=18.02, density=1.0)
    Hplus = cal.Substance("H+(aq)", state="aq", H0=0, S0=0, cP=0, molarMass=1.01, density=0)
    OH = cal.Substance("OH-(aq)", state="aq", H0=-230.02, S0=-10.9, molarMass=17.01, cP=-148.5, density=0)

    # H2Ogas = cal.Substance("H2O(g)", state="g", H0=-241.83e3, S0=188.84, cP=1.864*18.02, molarMass=18.02)



    chemicals = {0: H2O, 1: Hplus, 2: OH}

    rxns = [{0: -1, 1: 1, 2: 1}]

    s0 = cal.State3(T=298.15, chemicals=chemicals, rxns=rxns, V=1.0)

    s0.set_state({0: 1000/18.02, 1: 0, 2: 0})

    testing.assert_array_equal(np.array([0, 1,2]), s0.x_keys)
    testing.assert_array_equal(np.array([1000/18.02, 0, 0]), s0.x)
    testing.assert_array_equal(np.array([18.02, 1.01, 17.01]), s0.get_prop('molarMass'))
    assert s0.mass() == approx(1000.0)
    assert s0.G() == approx((H2O.H0 - H2O.S0*298.15)*1000/18.02)


    A, b = cal.arrayRxns(s0.state, s0.rxns)
    # Ag = np.c_[A, np.zeros(len(s0.state))]
    bounds = optimize.LinearConstraint(A, lb=-b, ub=np.inf)

    fX = lambda x: s0.G(b + A @ x , 298.15)
    res = optimize.minimize(fX, x0=[1e-7], constraints=bounds)
    print(res)
    


if __name__ == '__main__':
    test_melting()