

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
    def __init__(self, display=False):
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
        self.display = display

    def __call__(self, equation, variable, subs=None, unwrap=True, display=None):
        if display is None:
            display = self.display
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
            if display:
                display(equation)
                display_str = [f"{key} = {val}" for key, val in subs_out.items()]
                display(" ".join(display_str))
            out = sm.solve(equation.subs(subs_out), variable)
        else:
            out = sm.solve(equation, variable)
        
        if unwrap and len(out) == 1:
            out = out[0]
        
        if display:
            display(f"Soln: {variable} =")
            display(out)
        return out

solve = Solve() # Instantiate the class...

def getprop(gas, prop, P=None, T=None, **kwargs):
    """For a gas, get a thermodynamic property at a given pressure and temperature.
    
    Parameters
    ----------
    gas: string
        the gas name (e.g. 'Air', 'Ar', 'CO2')
    prop: string
        the thermodynamic property ('Smolar', 'Umolar', 'Vmolar', etc)
    P: float
        the pressure in bar
    T: float
        the temperature in K

    Returns
    -------
    float
        The value of the property under the given conditions.
    """
    if len(kwargs) == 0:
        if P is None:
            P = 1
        if T is None:
            T = 298.15
        P_Pa = P * 1e5 # bar to Pa
        if prop == 'Vmolar':
            return 1000/CP.PropsSI('Dmolar', 'P', P_Pa, 'T', T, gas) # L/mol
        else:
            return CP.PropsSI(prop, 'P', P_Pa, 'T', T, gas)
    else:
        args = []
        if P is not None:
            args.append('P')
            args.append(P*1e5)
        if T is not None:
            args.append('T')
            args.append(T)
        for key, val in kwargs.items():
            args.append(key)
            args.append(val)
        
        args.append(gas)
        
        if prop == 'Vmolar':
            return 1000/CP.PropsSI('Dmolar', *args) # L/mol
        else:
            return CP.PropsSI(prop, *args)



def getprop_dens(gas, prop, density, T=298.15):
    """For a gas,"""
    if prop == 'Vmolar':
        return 1000/CP.PropsSI('Dmolar', 'D', density, 'T', T, gas) # L/mol
    else:
        return CP.PropsSI(prop, 'D', density, 'T', T, gas)



    
def getPressure(gas, T=300, Vbar=22.4, threshold=1e-4):
    """Get the pressure of a gas at a given temperature and molar volume."""
    Z_prev = 1
    Pguess = (0.083145*T)/Vbar * Z_prev
    Z_guess = getprop(gas, 'Z', Pguess, T)
    while abs(Z_guess-Z_prev) > threshold:
        Z_prev=Z_guess
        Pguess = Pguess*Z_guess
        Z_guess = getprop(gas, 'Z', Pguess, T)
        
    return Pguess*Z_guess


# From StackOverflow
def _flatten(a):
    return functools.reduce(operator.iconcat, a, [])


def getprop_df(gas, prop, P, T):
    """Get a gas property as a dataframe.
    
    Parameters
    ----------
    gas: string
        the gas name (e.g. 'Air', 'Ar', 'CO2')
    prop: string
        the thermodynamic property ('Smolar', 'Umolar', 'Vmolar', etc)
    P: array-like
        the pressure in bar
    T: array-like
        the temperature in K
    
    Returns
    -------
    pandas.DataFrame
        The dataframe with the given property initial conditions and properties.
    """
    P = np.array(P).reshape(-1)
    T = np.array(T).reshape(-1)
    
    df = pd.DataFrame(_flatten([[{ 'P': Px, 'T':Tx, prop: getprop(gas, prop, P=Px, T=Tx)} for Px in P] for Tx in T]),
                      )

    df['P_str'] = [str(x) for x in df['P'].values]
    df['T_str'] = [str(x) for x in df['T'].values]
    return df

def getprops_df(gas, props, P, T):
    """Get a gas property as a dataframe.
    
    Parameters
    ----------
    gas: string
        the gas name (e.g. 'Air', 'Ar', 'CO2')
    props: array-like
        a list of properties to return, e.g. ['Smolar', 'Umolar', 'Vmolar']
    P: array-like
        the pressure in bar
    T: array-like
        the temperature in K
    
    Returns
    -------
    pandas.DataFrame
        The dataframe with the given property initial conditions and properties.
    """
    P = np.array(P).reshape(-1) # Make sure it's a 1D array
    T = np.array(T).reshape(-1) # Make sure it's a 1D array
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