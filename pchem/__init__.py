

from copy import copy
import functools
import operator

import sympy as sm
import numpy as np
import pandas as pd

try:
    import CoolProp.CoolProp as CP
except:
    pass


class Solve:
    def __init__(self):
        """Solve equation for the given variable; if given, a dictionary of subs
        (substitutions) can be given. This is useful if you want to solve numerically
        rather than symbolically. 
        
        Parameters:
        equation : the sympy equation to solve
        variable : the sympy variable to solve for
        subs : the dictionary of substitutions
        unwrap : if there is only one solution, return it directly rather than returning a list.
        
        Returns:
        The solution (if one solution and unwrap=True), or a list of all possible solutions.
        
        Examples: 
        >>> solve(a*x**2 + b*x + c, x)
            [(-b + sqrt(-4*a*c + b**2))/(2*a), -(b + sqrt(-4*a*c + b**2))/(2*a)]
            
        """
        self.context = {}

    def __call__(self, equation, variable, subs=None, unwrap=True):
        if subs is not None:
            subs_out = {}
            for key, val in subs.items():
                if isinstance(key, str):
                    if key not in self.context:
                        key_out = [x for x in equation.atoms() if hasattr(x, 'name') and x.name == key][0]
                        self.context[key] = key_out
                    else:
                        key_out = self.context[key]
                else:
                    key_out = key

                subs_out[key_out] = val
            
            subs_out.pop(variable, None)
            out = sm.solve(equation.subs(subs_out), variable)
        else:
            out = sm.solve(equation, variable)
        
        if unwrap and len(out) == 1:
            out = out[0]

        return out

solve = Solve() # Instantiate the class...

def getprop(gas, prop, P=1, T=298.15):
    """For a gas,"""
    P_Pa = P * 1e5 # bar to Pa
    if prop == 'Vmolar':
        return 1000/CP.PropsSI('Dmolar', 'P', P_Pa, 'T', T, gas) # L/mol
    else:
        return CP.PropsSI(prop, 'P', P_Pa, 'T', T, gas)

def getprop_dens(gas, prop, density, T=298.15):
    """For a gas,"""
    if prop == 'Vmolar':
        return 1000/CP.PropsSI('Dmolar', 'D', density, 'T', T, gas) # L/mol
    else:
        return CP.PropsSI(prop, 'D', density, 'T', T, gas)



    
def getPressure(gas, T=300, Vbar=22.4):
    Z_prev = 1
    Pguess = (0.083145*T)/Vbar * Z_prev
    Z_guess = getprop(gas, 'Z', Pguess, T)
    while abs(Z_guess-Z_prev) > 0.001:
        Z_prev=Z_guess
        Pguess = Pguess*Z_guess
        Z_guess = getprop(gas, 'Z', Pguess, T)
        
    return Pguess*Z_guess


# From StackOverflow
def _flatten(a):
    return functools.reduce(operator.iconcat, a, [])

def getprop_df(gas, prop, P, T):
    P = np.array(P).reshape(-1)
    T = np.array(T).reshape(-1)
    
    df = pd.DataFrame(_flatten([[{ 'P': Px, 'T':Tx, prop: getprop(gas, prop, P=Px, T=Tx)} for Px in P] for Tx in T]),
                      )

    df['P_str'] = [str(x) for x in df['P'].values]
    df['T_str'] = [str(x) for x in df['T'].values]
    return df

def getprops_df(gas, props, P, T):
    P = np.array(P).reshape(-1)
    T = np.array(T).reshape(-1)
    dicts = []
    for Px in P:
        for Tx in T:
            d = dict(T=Tx, P=Px)
            for prop in props:
                d[prop] = getprop(gas, prop, P=Px, T=Tx)
            dicts.append(d)
    
    df = pd.DataFrame(dicts)
    df['P_str'] = [str(x) for x in df['P'].values]
    df['T_str'] = [str(x) for x in df['T'].values]
    return df

# def solve(equation, variable, subs=None, unwrap=True):
    # """Solve equation for the given variable; if given, a dictionary of subs
    # (substitutions) can be given. This is useful if you want to solve numerically
    # rather than symbolically. 
    
    # Parameters:
    # equation : the sympy equation to solve
    # variable : the sympy variable to solve for
    # subs : the dictionary of substitutions
    # unwrap : if there is only one solution, return it directly rather than returning a list.
    
    # Returns:
    # The solution (if one solution and unwrap=True), or a list of all possible solutions.
    
    # Examples: 
    # >>> solve(a*x**2 + b*x + c, x)
    #     [(-b + sqrt(-4*a*c + b**2))/(2*a), -(b + sqrt(-4*a*c + b**2))/(2*a)]
        
    # """
#     if subs is not None:
#         context = {}
#         for symbol in equation.atoms():
#             if hasattr(symbol, 'name'):
#                 context[symbol.name] = symbol
#         subs_out = {}
#         for key, val in subs.items():
#             if isinstance(key, str):
#                 key_out = context[key]
#             else:
#                 key_out = key
#             subs_out[key_out] = val

#         subs_out.pop(variable, None)
#         out = sm.solve(equation.subs(subs_out), variable)
#     else:
#         out = sm.solve(equation, variable)
#     if unwrap and len(out) == 1:
#         out = out[0]
#     return out