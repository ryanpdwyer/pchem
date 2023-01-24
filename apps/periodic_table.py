import pandas as pd
import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st
from collections import defaultdict
from string import Template
from base import *

def periodic_table():

    H = dict(Z=1,e1s=1)
    df = elements_df
    element_symbols = dict(df.loc[:, ['AtomicNumber', 'Symbol']].values)
    symbol_Z = dict(df.loc[:, ['Symbol', 'AtomicNumber']].values)


    periodic_table = st.checkbox("Select on periodic table?", True)
    if periodic_table:
        show_d_orbitals = st.checkbox("Show d-orbitals?")
        el_fired = []
        symbols = list(symbol_Z.keys())
        if show_d_orbitals:
            cols = st.columns(18)
            labels = range(1,19)
            for col, label in zip(cols, labels):
                col.markdown(label)

            el_fired.append(cols[0].button('H'))
            for col in cols[1:-1]:
                col.markdown('<span style="color:white; font-size: 22px;">XX</span>', unsafe_allow_html=True)
        
            el_fired.append(cols[-1].button('He'))

            main_group =  [cols[i] for i in [0, 1,-6,-5, -4,-3,-2,-1]]
            transition_metals = cols[2:12]
            for col, el in zip(main_group, symbols[2:10]):
                el_fired.append(col.button(el))
            
            for col, el in zip(main_group, symbols[10:18]):
                el_fired.append(col.button(el))
            
            for col in transition_metals:
                col.markdown('<span style="color:white; font-size: 22px;">XX</span>', unsafe_allow_html=True)
                col.markdown('<span style="color:white; font-size: 22px;">XX</span>', unsafe_allow_html=True)

            for col, el in zip(cols, symbols[18:36]):
                el_fired.append(col.button(el))

        else:
            cols = st.columns(8)

            for col, label in zip(cols, [1, 2, 13, 14, 15, 16, 17, 18]):
                col.markdown(label)

    
            el_fired.append(cols[0].button('H'))
            for col in cols[1:-1]:
                col.markdown('<span style="color:white; font-size: 22px;">XX</span>', unsafe_allow_html=True)
            
            el_fired.append(cols[-1].button('He'))
            for col, el in zip(cols, symbols[2:10]):
                el_fired.append(col.button(el))
            
            for col, el in zip(cols, symbols[10:18]):
                el_fired.append(col.button(el))


        for i, button in enumerate(el_fired):
            if button:
                element = symbols[i]
                with open('el.json', 'w') as f:
                    json.dump(dict(el=element), f)
        
        with open('el.json', 'r') as f:
            element = json.load(f)['el']
    else:
        element = st.selectbox('Element', options=list(element_symbols.values()))


    protons = symbol_Z[element]
    charge = st.slider('Charge', -3, 3, 0, 1)
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
        ax.plot(0,0, 'o', color=color, label=orbital, zorder=-10)
    
    ax.text(0,0,'+'+str(protons), ha='center', va='center', color='1', fontsize=8)
    ax.legend()
    st.write(fig)