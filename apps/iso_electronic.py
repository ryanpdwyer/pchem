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
    if 'el_iso' not in st.session_state:
        st.session_state.el_iso = 'F'
    
    H = dict(Z=1,e1s=1)
    df = elements_df
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
            st.session_state.el_iso = element


    element = st.session_state.el_iso
    charge = charges[element]


    protons = symbol_Z[element]
    e_config = get_e_config(protons, charge)
    e_config_no_zeros = {key: val for key, val in e_config.items() if val != 0}
    shells = n_in_shell(e_config)


    col_e_config, col_Zeff, col_key = st.columns(3)
    show_e_config = col_e_config.checkbox('Show e⁻ configuration')
    show_Zeff = col_Zeff.checkbox('Show Zeff')
    show_key = col_key.checkbox('Show subshell key')


    empty = st.empty()


    if show_e_config:
        initial_e_config_text = write_e_config(protons, e_config, element_symbols)
    else:
        initial_e_config_text = write_ion(protons, e_config, element_symbols)


    fig, ax = plt.subplots(figsize=(4,4))
    for i, (orbital, color) in enumerate(colors.items()):
        if i <= 2:
            ax.plot(0,0, '.', markersize=8, color=color, label=orbital, zorder=-10-i)
    
    lims = 1.4
    ax.set_xlim(-lims,lims)
    ax.set_ylim(-lims,lims)
    ax.set_axis_off()
    circles = []
    circles.append(plt.Circle((0, 0), 0.1, color='g'))

    Zeff_dict = {}

    for shell, e_in_shell in shells.items():
        n_lower = sum(val for key, val in shells.items() if key < shell)
        Z_shell = Zeff(protons, n_lower, e_in_shell)
        Z_shell_simple = protons - n_lower - (e_in_shell-1)/2.0
        Zeff_dict[shell] = Z_shell_simple

        n_electron = 0
        if e_in_shell > 0:
            circles.append(plt.Circle((0, 0), shell_r(shell, Z_shell), fc='none', linewidth=0.7, ec='0'))
        subshells = {key[1]: val for key, val in e_config.items() if str(shell) in key}
        arrs = np.concatenate([np.full(val, key) for key, val in subshells.items()])
        for i, elec in enumerate(arrs):
            draw_electron(ax, circles, shell, i, Z_shell, elec)

    for circle in circles:
        ax.add_artist(circle)

    ax.text(0,0,'+'+str(protons), ha='center', va='center', color='1', fontsize=8)
    if show_key:
        ax.legend()

    st.write(fig)

    if show_Zeff:
        n_max = max(int(x[0]) for x in e_config_no_zeros.keys())
        Z_eff_str = f" &nbsp; &nbsp; with Z<sub>eff</sub>(n={n_max}) = {Zeff_dict[n_max]:.1f}"
    else:
        Z_eff_str = ''
    
    empty.markdown(initial_e_config_text+Z_eff_str, unsafe_allow_html=True)
