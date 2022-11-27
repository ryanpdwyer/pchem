from dataclasses import dataclass
from typing import Iterable, Optional

from scipy import optimize, linalg

import numpy as np
import statsmodels.api as sm
# System state: moles of each component...
# System state: temperature T, pressure P, etc (assuming we stay in local equilibrium)

# chemicals = {0: dict(name="H2O(s)", H0=-285.826, T0=298.15, S0=69.95), 1: dict(name="H2O(l)", S0=0, T0=273.15)}

R = 8.3145
R_Lbar = R/100.0

@dataclass
class Substance:
    name: str
    H0: float
    S0: float
    cP: float # Molar heat capacity
    molarMass: float #
    state: str
    density: float
    T0: float = 298.15
    conc0: float = 1

    def Hbar(self, T, conc=1):
        return self.H0 + (T-self.T0) * self.cP

    def S0bar(self, T, conc=1):
        return self.S0 + self.cP * np.log(T/self.T0)
    
    def Sbar(self, T, conc=1):
        return self.S0bar(T) - R * np.log(conc/self.conc0)
    
    def G0bar(self, T, conc=1):
        return self.Hbar(T) - T*self.S0bar(T)

    def Gbar(self, T, conc):
        return self.Hbar(T) - T*self.Sbar(T, conc)



@dataclass
class Substance2:
    name: str
    H0: float
    S0: float
    cP: float # Molar heat capacity
    molarMass: float #
    state: str
    density: float
    atoms: dict
    T0: float = 298.15
    conc0: float = 1

    def Hbar(self, T, conc=1):
        return self.H0 + (T-self.T0) * self.cP

    def S0bar(self, T, conc=1):
        return self.S0 + self.cP * np.log(T/self.T0)
    
    def Sbar(self, T, conc=1):
        return self.S0bar(T) - R * np.log(conc/self.conc0)
    
    def G0bar(self, T, conc=1):
        return self.Hbar(T) - T*self.S0bar(T)

    def Gbar(self, T, conc):
        return self.Hbar(T) - T*self.Sbar(T, conc)




def phaseChange(deltaH, T0, sub, **kwargs):
    Hbar = sub.Hbar(T0)
    S0 = sub.S0bar(T0)
    deltaH = deltaH
    Hnew = Hbar + deltaH
    Snew = S0 + deltaH/T0
    cPnew = kwargs.get('cP', sub.cP)
    kwargs.pop('cP')
    return type(sub)(T0=T0, H0=Hnew, S0=Snew, cP=cPnew,
                    molarMass=sub.molarMass, **kwargs)



# def Gsys(state, T, P):
#     concs = {0: 1, 10: 1}
#     Gbar = {key: chemicals[key].Gbar(T, concs[key]) for key in state}
#     return np.array(list(Gbar.values()))

# def Hsys(state, T, P):
#     concs = {0: 1, 10: 1}
#     Hbar = {key: chemicals[key].Hbar(T) for key in state}
#     return np.array(list(Hbar.values()))


# def SsysFull(state, T, P):
#     concs = {0: 1, 10: 1}
#     S0bar = {key: chemicals[key].S0bar(T) for key in state}
#     return np.array(list(S0bar.values()))


# def cPsys(state, T, P):
#     concs = {0: 1, 10: 1}
#     cP = {key: chemicals[key].cP for key in state}
#     return np.array(list(cP.values()))

# If I have a state and a list of reactions, I can minimize the free energy...
def arrayRxns(state, rxns):
    A = np.array([[rxn.get(key, 0) for rxn in rxns] for key in state])
    return A, np.array([val for val in state.values()])



def heatFuncFull(Hmolar, cPmolar, Ag, b, x):
    s = Ag @ x + b
    dHrxn = Hmolar.T @ Ag @ x
    dHTemp = cPmolar.T @ s * x[-1]
    return dHrxn + dHTemp



@dataclass
class State:
    s : Iterable
    T : float


# Switch the format to something a little easier...
# C

@dataclass
class State2:
    s: Iterable
    T: float
    chemicals: dict
    rxns: Iterable[dict]
    V : Optional[float] = None

    def mass(self):
        return sum(self.chemicals[key].molarMass * val for key, val in self.s.items())


    def volume(self):
        if self.V:
            return V
        else:
            return sum(self.chemicals[key].molarMass * val / self.chemicals[key].density for key, val in self.s.items() if self.chemicals[key].state in ['s', 'l'])/1000.0

    def conc(self, state=None):
        concs = []
        V = self.volume()
        state = self.s if state is None else state
        for key, val in self.s.items():
            c = self.chemicals[key]
            if c.state in ['s', 'l']:
                concs.append(1)
            elif c.state == 'aq':
                concs.append(val/V)
            elif c.state == 'g':
                concs.append(val*R_Lbar*self.T/V)
            else:
                raise ValueError("State must be aq, s, l, g.")
        
        return np.array(concs)
    
    def Hvec(self, T=None):
        state = self.s
        T = self.T if T is None else T
        Hbar = {key: self.chemicals[key].Hbar(T) for key in state}
        return np.array(list(Hbar.values()))
    
    def Gsys(self, state=None, T=None):
        state = self.s if state is None else state
        T = self.T if T is None else T
        total = 0.0
        for (key, val), c in zip(self.state.items(), self.conc()):
            total += self.chemicals[key].Hbar(T)*val - T*val*self.chemicals()



@dataclass
class State3:
    T: float
    chemicals: dict
    rxns: Iterable[dict]
    V : Optional[float] = None
    x : Optional[Iterable[float]] = None
    x_keys : Optional[Iterable[int]] = None
    


    def set_state(self, state: dict):
        self.x_keys = np.array(list(state.keys()))
        self.chem_vals = list(self.chemicals[key] for key in self.x_keys)
        # Sort things so that x is in order of phase...
        states = [x.state+x.name for x in self.chem_vals]
        sort_inds = np.argsort(states)
        self.x_keys = self.x_keys[sort_inds]
        self.chem_vals = list(self.chemicals[key] for key in self.x_keys)

        states = [x.state for x in self.chem_vals]

        self.N_aq = sum(1 for x in states if x == 'aq')



        self.x = np.array([state[key] for key in self.x_keys])
        self.molarMass = self.get_prop('molarMass')

        all_atoms = set()
        for x in self.chem_vals:
            all_atoms.update(x.atoms.keys())

        self.all_atoms = tuple(sorted(all_atoms))

        self.N_atoms = len(all_atoms)
        self.N_chem = len(self.chem_vals)
        self.N_cond = self.N_chem - self.N_aq
        A = np.zeros(shape=(self.N_atoms, self.N_chem))
        for i, atom in enumerate(self.all_atoms):
            for j, chem in enumerate(self.chem_vals):
                A[i, j] = chem.atoms.get(atom, 0)
        

        self.A = A
        self.n_moles = self.A @ self.x 

    
    @property
    def state(self):
        return dict(zip(self.x_keys, self.x))

    def get_prop(self, prop):
        return np.array(list(getattr(chem, prop) for chem in self.chem_vals))
    
    def get_prop_conc(self, prop, x=None, T=None):
        x = self.x if x is None else x
        T = self.T if T is None else T
        conc = self.conc(x)
        conc = np.array([c if c > 1e-30 else 1e-30 for c in conc])
        return np.array(list(getattr(chem, prop)(T=T, conc=c) for chem, c in zip(self.chem_vals, conc)))
    
    def _solve(self, dH):
        A, b = arrayRxns(self.state, self.rxns)
        Ag = np.c_[A, np.zeros(len(self.state))] # Add zeros for temp?
        T0 = self.T

        def HconstaintFunc(x):
            state = Ag@x+b
            T = T0 + x[-1]
            return self.H(x=state, T=T) - self.H(x=b, T=T0)

        Hconstraint = optimize.NonlinearConstraint(HconstaintFunc, lb=dH, ub=dH)
        bounds = optimize.LinearConstraint(Ag, lb=-b, ub=np.inf) # Reaction constraint...

        def Sfunc(x):
            state = Ag@x+b
            T = T0 + x[-1]
            return -self.S(x=state, T=T) # Negative entropy...
        
        print(Ag.shape)
        res = optimize.minimize(Sfunc, x0=np.zeros(Ag.shape[1]), constraints=[bounds, Hconstraint])
            #method='SLSQP')

        s_new = Ag @ res.x + b
        T_new = T0 + res.x[-1]
        return res, s_new, T_new

    def mu(self, x=None, T=None):
        x = self.x if x is None else x
        T = x[-1] if T is None and len(x)>self.N_chem else T
        T = self.T if T is None else T
        print(T)
        x_conc = x[:self.N_chem]
        RT = T*8.3145
        Gbar = self.get_prop_conc('Gbar', x=x_conc, T=T)
    

        return sm.WLS(Gbar, self.A.T, weights=x_conc).fit().params

    def fH(self, x, H):
        T0 = self.T
        A = self.A
        y = np.zeros_like(x)
        x_conc = x[:self.N_chem]
        T = x[-1]
        RT = 8.3145*T
        V = self.volume(x_conc)
        Gbar = self.get_prop_conc('Gbar', x_conc, T=T) / RT
        H0 = self.get_prop_conc('Hbar') @ self.x
        Hbar = self.get_prop_conc('Hbar', x_conc, T=T)
        mus = x[self.N_chem:-1]
        y[:self.N_chem] = Gbar + A.T @ mus # Needed a transpose here... going from chemical potential of atoms to species
        # If solid or liquid species, multiply by moles to eliminate
        # any species that are not in the system.
        y[:self.N_aq] = y[:self.N_aq] * V # Aqueous species only matter if V > 0...
        y[self.N_aq:self.N_chem] = y[self.N_aq:self.N_chem] * x[self.N_aq:self.N_chem]
        y[self.N_chem:-1] = self.n_moles - A @ x_conc
        y[-1] = (H - (Hbar @ x_conc - H0)) / RT
        return y

    def f(self, x):
        T = self.T
        A = self.A
        y = np.zeros_like(x)
        x_conc = x[:self.N_chem]
        RT = 8.3145*T
        Gbar = self.get_prop_conc('Gbar', x_conc) / RT
        mus = x[self.N_chem:]
        y[:self.N_chem] = Gbar + A.T @ mus # Needed a transpose here... going from chemical potential of atoms to species
        # If solid or liquid species, multiply by moles to eliminate
        # any species that are not in the system.
        # y[self.N_aq:self.N_chem] = y[self.N_aq:self.N_chem] * x[self.N_aq:self.N_chem]
        y[self.N_chem:] = self.n_moles - A @ x_conc
        return y

    def fi(self, x):
        n_moles = x[:self.N_chem]
        n_moles_aq = n_moles[:self.N_aq]
        N_total = self.N_chem+self.N_atoms
        T = self.T
        A = self.A
        y = np.zeros(N_total)
        RT = 8.3145*T
        A = np.zeros(shape=(N_total, N_total))
        A_aq_block = np.eye(self.N_aq, self.N_aq) # Identity matrix for aq species
        A[:self.N_aq, :self.N_aq] = A_aq_block
        A[:self.N_chem, self.N_chem:] = -self.A.T # Zeros for the s,l species
        A_lower_left = self.A[:, :self.N_aq] @ np.diag(n_moles_aq)
        A[self.N_chem:, :self.N_aq] = A_lower_left
        A[self.N_chem:, self.N_aq:self.N_chem] = self.A[:, self.N_aq:]
        neg_G_o_RT = -self.get_prop_conc('Gbar', x=n_moles,T=T)/RT
        y[:self.N_chem] = neg_G_o_RT # Output variables here...
        y[self.N_chem:] = self.f(x)[self.N_chem:]
        out = linalg.lstsq(A, y)
        dx = out[0]
        x_new = np.r_[n_moles_aq*np.exp(dx[:self.N_aq]),
                    n_moles[self.N_aq:] + dx[self.N_aq:self.N_chem], 
                    dx[self.N_chem:]
        ]
        return x_new

    
    def fHi(self, x, dH):
        n_moles = x[:self.N_chem]
        n_moles_aq = n_moles[:self.N_aq]
        N_total = self.N_chem+self.N_atoms+1
        T = self.T
        cPtotal = self.get_prop('cP') @ n_moles
        y = np.zeros(N_total)
        RT = 8.3145*T
        A = np.zeros(shape=(N_total, N_total))
        A_aq_block = np.eye(self.N_aq, self.N_aq) # Identity matrix for aq species
        A[:self.N_aq, :self.N_aq] = A_aq_block
        A[:self.N_chem, self.N_chem:-1] = -self.A.T # Zeros for the s,l species
        
        neg_H_over_RT = -self.get_prop_conc('Hbar', x=n_moles,T=T) / (RT)

        A[:self.N_chem, -1] = neg_H_over_RT
        
        A_lower_left = self.A[:, :self.N_aq] @ np.diag(n_moles_aq)
        A[self.N_chem:-1, :self.N_aq] = A_lower_left
        A[self.N_chem:-1, self.N_aq:self.N_chem] = self.A[:, self.N_aq:]

        A[-1, :self.N_aq] = -neg_H_over_RT[:self.N_aq] * n_moles_aq
        A[-1, self.N_aq: self.N_chem] = -neg_H_over_RT[self.N_aq: self.N_chem]

        A[-1, -1] = cPtotal / 8.3145
        
        neg_G_o_RT = -self.get_prop_conc('Gbar', x=n_moles,T=T)/RT
        
        y[:self.N_chem] = neg_G_o_RT # Output variables here...

        y[self.N_chem:] = self.fH(x, dH)[self.N_chem:] # True for dT too? Probably not...
        out = linalg.lstsq(A, y)
        dx = out[0]
        x_new = np.r_[n_moles_aq*np.exp(dx[:self.N_aq]),
                    n_moles[self.N_aq:] + dx[self.N_aq:self.N_chem], 
                    dx[self.N_chem:-1],
                    x[-1]*np.exp(dx[-1]) # Temperature variable...
        ]
        return x_new
    
    def _transform(self, x):
        x_new = x.copy()
        x_new[:self.N_aq] = np.log(x_new[:self.N_aq]) # Log transform conc. variables
        return x_new
    
    def _solve_iterative(self, x=None, T=None, max_iters=100, stop_tol=1e-8):
        """NASA Gordon 1994 model """
        x_moles = self.x if x is None else x[:self.N_chem]
        T = self.T if T is None else T
        self.T = T
        x_moles_initial = np.where(x_moles < 1e-10, 1e-10, x_moles)
        x0 = np.r_[x_moles_initial, np.zeros(self.N_atoms)]
        x_guess = x0
        delta = []
        for i in range(max_iters):
            x_new = self.fi(x_guess)
            resid = self._transform(x_new) - self._transform(x_guess)
            max_change = max(abs(resid[:self.N_chem]))
            delta.append(max_change)
            x_guess = x_new
            if max_change < stop_tol:
                break

        return x_new, delta


        
    
    def _solve_constT(self, T=None):
        T = self.T if T is None else T


        def f(x):
            A = self.A
            y = np.zeros_like(x)
            x_conc = x[:self.N_chem]
            RT = 8.3145*T
            Gbar = self.get_prop_conc('Gbar', x_conc) / RT
            mus = x[self.N_chem:]
            y[:self.N_chem] = Gbar + A.T @ mus # Needed a transpose here... going from chemical potential of atoms to species
            y[self.N_chem:] = self.n_moles - A @ x_conc
            return y
        
        
        x0 = np.zeros(self.N_chem+self.N_atoms)
        x0[:self.N_chem] = self.x # Current state

        return optimize.fsolve(f, x0=x0, full_output=1)


    def G(self, x=None, T=None):
        x = self.x if x is None else x
        T = self.T if T is None else T
        Gmolar = self.get_prop_conc('Gbar', x, T)
        m = x>0
        return np.dot(Gmolar[m], x[m])
    
    def S(self, x=None, T=None):
        x = self.x if x is None else x
        T = self.T if T is None else T
        Smolar = self.get_prop_conc('Sbar', x, T)
        m = x>0
        return np.dot(Smolar[m], x[m])

    def _solve_no_rxns(self, dH):
        A = self.A
        b = self.x # Current state...
        Ag = np.c_[A, np.zeros(self.N_atoms)] # Add zeros for temp?
        T0 = self.T

        Hbar0 = self.get_prop_conc('Hbar', x=b, T=T0) @ b

        cP = self.get_prop('cP')

        def HconstaintFunc(x):
            T = T0 + x[-1]
            Hbar = self.get_prop_conc('Hbar', x=x[:-1], T=T)
            return Hbar @ x[:-1] - Hbar0

        def HconstaintFunc_jac(x):
            T = T0 + x[-1]
            Hbar = self.get_prop_conc('Hbar', x=x[:-1], T=T)
            return np.r_[Hbar, cP@x[:-1]].reshape(1, -1)

        Hconstraint = optimize.NonlinearConstraint(HconstaintFunc, lb=dH, ub=dH, jac=HconstaintFunc_jac)
        bounds = optimize.LinearConstraint(Ag, lb=self.n_moles, ub=self.n_moles) # Reaction constraint...



        lower_bound = np.zeros(len(b)+1)
        lower_bound[-1] = -np.inf
        bounds_variables = optimize.Bounds(lower_bound, np.inf, keep_feasible=True)


        def Sfunc(x):
            T = T0 + x[-1]
            cP_total = cP @ x[:-1]
            Sbar = self.get_prop_conc('Sbar', x=x[:-1], T=T)
            m = x[:-1] > 0.0
            S = Sbar[m] @ x[:-1][m]
            return -S, np.r_[-Sbar, -cP_total/T]
        
        print(Ag.shape)
        res = optimize.minimize(Sfunc, x0=np.r_[self.x, 0.0], 
        constraints=[bounds, Hconstraint],
        bounds=bounds_variables,
        jac=True)

        s_new = res.x[:-1]
        T_new = T0 + res.x[-1]
        return res, s_new, T_new

    def H(self, x=None, T=None):
        x = self.x if x is None else x
        T = self.T if T is None else T
        Hmolar = self.get_prop_conc('Hbar', x, T)
        m = x>0
        return np.dot(Hmolar[m], x[m])

    def mass(self, x=None):
        x = self.x if x is None else x
        return np.dot(self.molarMass, x)


    def volume(self, x=None):
        if self.V:
            return self.V
        else:
            x = self.x if x is None else x
            return sum(chem.molarMass * mol / chem.density for chem, mol in zip(self.chem_vals, x) if chem.state in ['l']) / 1000.0

    def conc(self, x=None):
        concs = []
        x = self.x if x is None else x
        V = self.volume(x)
        for chem, mol in zip(self.chem_vals, x):
            if chem.state in ['s', 'l']:
                concs.append(1)
            elif chem.state == 'aq':
                concs.append(mol/V)
            elif chem.state == 'g':
                concs.append(mol*R_Lbar*self.T/V) # Assuming ideal gas
            else:
                raise ValueError("State must be aq, s, l, g.")
        
        return np.array(concs)
    
    def Hvec(self, T=None):
        state = self.s
        T = self.T if T is None else T
        Hbar = {key: self.chemicals[key].Hbar(T) for key in state}
        return np.array(list(Hbar.values()))
    
    def Gsys(self, state=None, T=None):
        state = self.s if state is None else state
        T = self.T if T is None else T
        total = 0.0
        for (key, val), c in zip(self.state.items(), self.conc()):
            total += self.chemicals[key].Hbar(T)*val - T*val*self.chemicals()
    
    # def cPsys(self, state=None, T=None):
    #     state = self.s if state is None else state
    #     T = self.T if T is None else T
    #     total = 0.0
    #     for (key, val), c in zip(self.state.items(), self.conc()):
    #         total += self.chemicals[key].cP*val # Moles



def getState(keys, new_val):
    return dict(zip(keys, new_val))


def newState(state: State3, dH, chemicals, rxns):
    """Calculate a new state using dH"""
    s = state.s
    s_initial = s
    T = state.T
    state_keys = list(s.keys())
    cPsys = state.cPsys()
    Hmolar = state.Hvec()

    A, b = arrayRxns(s, rxns)
    Ag = np.c_[A, np.zeros(len(s))]
    bounds = optimize.LinearConstraint(Ag, lb=-b, ub=np.inf)
    
    def Ssys(Ag, b, x, T):
        s_vec = Ag @ x + b
        s = getState(state_keys, s_vec)
        Tnew = T + x[-1]
        return SsysFull(s, Tnew, 1) @ s_vec # Moles of each...

    
    def Hsys2(Ag, b, x, T):
        s_vec =Ag @ x + b
        s = getState(state_keys, s_vec)
        Tnew = T + x[-1]
        return Hsys(s, Tnew, 1) @ s_vec - Hsys(s_initial, T, 1) @ b

    heatFunc = lambda x: Hsys2(Ag, b, x, T)
    func = lambda x: -Ssys(Ag, b, x, T)
    Hconstraint = optimize.NonlinearConstraint(heatFunc, lb=dH, ub=dH)
    res = optimize.minimize(func, x0=[0.0, 0.0], constraints=(bounds, Hconstraint))
    
    s_new = b + Ag @ res.x
    return State(s=dict(zip(state_keys, s_new)), T=state.T+res.x[-1])


