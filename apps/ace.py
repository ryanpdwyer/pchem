import streamlit as st
from streamlit_ace import st_ace
import types

import sympy as sm
from sympy.abc import *
from pchem import solve
# See 
# https://discuss.streamlit.io/t/take-code-input-from-user/6413/2?u=ryanpdwyer

# import random, string
# import importlib
# import os




# def import_code(code, name):
#     # create blank module
#     module = types.ModuleType(name)
#     # populate the module with code
#     exec(code, module.__dict__)
#     return module

def run():
    # data = st.file_uploader("Upload data files:", accept_multiple_files=True)
    st.markdown("## Sympy Shell")
    # Spawn a new Ace editor
    content = st_ace(language='python',
    value=
"""
# All single letter variables are defined

gas_law = P*V - n * R * T

subs = dict(
    P=0.2,
    V=2.0,
    n=0.1,
    T=298,
    R=0.08314
)

print(solve(gas_law, T))
""")

    # m = import_code(content, "aceExampleCode")
    # Display editor's content as you type
    exec(content, globals(), dict(print=st.write))

    # strategy_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8)) 
    # with open(strategy_name+'.py', 'w') as the_file:
    #     the_file.write(content)
    # TestStrategy = getattr(importlib.import_module(strategy_name), 'TestStrategy')

    # # do stuff
    # if os.path.exists(strategy_name+'.py'):
    #   os.remove(strategy_name+'.py')
    # else:
    #   print("The file does not exist")

    # I should probably persist the code in some way?
    # streamlit generates code, user tweaks for their application?

if __name__ == '__main__':
    run()