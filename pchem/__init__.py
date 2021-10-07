import sympy as sm
from copy import copy

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