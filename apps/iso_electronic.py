import pandas as pd
import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st
from collections import defaultdict
from string import Template
from base import *

def iso_electronic():

    H = dict(Z=1,e1s=1)
    df = pd.read_csv("elements.csv")
    element_symbols = dict(df.loc[:, ['AtomicNumber', 'Symbol']].values)
    symbol_Z = dict(df.loc[:, ['Symbol', 'AtomicNumber']].values)

    st.title("Isoelectronic ions")
    st.markdown("<br/>", unsafe_allow_html=True)

    charges = dict(H=-1, He=0, Li=1, Be=2, N=-3, O=-2, F=-1, Ne=0, Na=1, Mg=2,
                                        P=-3, S=-2, Cl=-1, Ar=0, K=1, Ca=2)
    symbols = list(symbol_Z.keys())
    cols = st.columns(8)

    def make_periodic_table():
        el_fired = []

        for col, label in zip(cols, [1, 2, 13, 14, 15, 16, 17, 18]):
            col.markdown(label)


        el_fired.append(cols[0].button('H-'))
        for col in cols[1:-1]:
            col.markdown('<span style="color:white; font-size: 22px;">XX</span>', unsafe_allow_html=True)

        el_fired.append(cols[-1].button('He'))
        el_fired.append(cols[0].button('Li⁺'))
        el_fired.append(cols[1].button('Be²⁺'))

        el_fired.append(cols[-4].button('N³⁻'))
        el_fired.append(cols[-3].button('O²⁻'))
        el_fired.append(cols[-2].button('F⁻'))
        el_fired.append(cols[-1].button('Ne'))
        el_fired.append(cols[0].button('Na⁺'))
        el_fired.append(cols[1].button('Mg²⁺'))

        el_fired.append(cols[-4].button('P³⁻'))
        el_fired.append(cols[-3].button('S²⁻'))
        el_fired.append(cols[-2].button('Cl⁻'))
        el_fired.append(cols[-1].button('Ar'))
        el_fired.append(cols[0].button('K⁺'))
        el_fired.append(cols[1].button('Ca²⁺'))

        return el_fired

    el_fired = make_periodic_table()

    for i, button in enumerate(el_fired):
        if button:
            element = list(charges.keys())[i]
            with open('el-2.json', 'w') as f:
                json.dump(dict(el=element), f)

    with open('el-2.json', 'r') as f:
        element = json.load(f)['el']
        charge = charges[element]


    protons = symbol_Z[element]
    e_config = get_e_config(protons, charge)
    shells = n_in_shell(e_config)


    show_e_config = st.checkbox('Show e⁻ configuration')
    if show_e_config:
        st.markdown(write_e_config(protons, e_config, element_symbols))
    else:
        st.markdown(write_ion(protons, e_config, element_symbols))


    fig, ax = plt.subplots(figsize=(4,4))
    lims = 1.4
    ax.set_xlim(-lims,lims)
    ax.set_ylim(-lims,lims)
    ax.set_axis_off()
    circles = []
    circles.append(plt.Circle((0, 0), 0.1, color='g'))


    for shell, e_in_shell in shells.items():
        n_lower = sum(val for key, val in shells.items() if key < shell)
        Z_shell = Zeff(protons, n_lower, e_in_shell)
        n_electron = 0
        if e_in_shell > 0:
            circles.append(plt.Circle((0, 0), shell_r(shell, Z_shell), fc='none', linewidth=0.7, ec='0'))
        subshells = {key[1]: val for key, val in e_config.items() if str(shell) in key}
        arrs = np.concatenate([np.full(val, key) for key, val in subshells.items()])
        for i, elec in enumerate(arrs):
            draw_electron(ax, circles, shell, i, Z_shell, elec)

    for circle in circles:
        ax.add_artist(circle)

    for orbital, color in colors.items():
        ax.plot(0,0, 'o', color=color, label=orbital)
    ax.text(0,0,'+'+str(protons), ha='center', va='center', color='1', fontsize=8)
    ax.legend()
    st.write(fig)