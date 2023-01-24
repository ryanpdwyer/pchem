import streamlit as st
import numpy as np
import pandas as pd


st.markdown("## An electron's journey")
st.markdown("""
Your goal is to get the electron to the lowest energy possible state.
""")

electron_description = {"O": {"1s": dict(n=2,E=-50), "2s": dict(n=2, E=-30), "2p": dict(n=4, E=-10)}}

Natoms = {"O": 2}

def make_atoms(Natoms):
    atoms = []
    for key, val in Natoms.items():
        atoms.extend([key]*val)
    return atoms

def make_electrons(electron_descript):
    electron_dict = {}
    for atom, val in electron_description.items():
        electrons = []
        for key, val in val.items():
            electrons.extend([key]*val['n'])
        electron_dict[atom] = electrons
    return electron_dict

def random_electron(atom):
    electrons = electron_dict[atom]
    return np.random.choice(electrons)

atoms = make_atoms(Natoms)
st.write(atoms)

electron_dict = make_electrons(electron_description)
st.write(electron_description)
st.write(electron_dict)

my_atom = st.selectbox("Choose an atom", list(electron_description.keys()))

st.button("Choose an electron!")

st.write(my_atom)

st.write(random_electron(my_atom))