import pandas as pd
import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st
from collections import defaultdict
from string import Template
from base import *
import plotly.express as px
# Import special functions 
import scipy.special as spe

def psi_r(r,n=1,l=0, Z=1):
    r = np.array(r)*Z
    coeff = np.sqrt((2.0/n)**3 * spe.factorial(n-l-1) /(2.0*n*spe.factorial(n+l)))
    
    laguerre = spe.assoc_laguerre(2.0*r/n,n-l-1,2*l+1)
    
    return coeff * np.exp(-r/n) * (2.0*r/n)**l * laguerre

def psi_r2(r, n, l, Z):
    return psi_r(r, n, l, Z)**2

angular_momentum_dict = dict(s=0, p=1, d=2, f=3)

def subshell_electron_counts(n):
    d = dict()
    if n <= 2:
        d['s'] = n
    elif n<=10:
        d['s'] = 2
        d['p'] = n-2
    elif n<=18:
        d['s'] = 2
        d['p'] = 6
        d['d'] = n-8
    elif n<=32:
        d['s'] = 2
        d['p'] = 6
        d['d'] = 10
        d['f'] = n-18
    else:
        raise ValueError("n must be less than 32")
    return d

def run():
    if 'el_random_electron' not in st.session_state:
        st.session_state.el_random_electron = 'C'

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
                st.session_state.el_random_electron = symbols[i]
        
        element = st.session_state.el_random_electron
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


    # fig, ax = plt.subplots()

    df = pd.DataFrame()
    df2 = pd.DataFrame()
    # fig2, ax2 = plt.subplots()
    
    c = st.container()
    samps_per_e = st.slider("Samples per electron:", min_value=10, max_value=300,
                    value=100)
                    
    for shell, e_in_shell in shells.items():
        n_lower = sum(val for key, val in shells.items() if key < shell)
        Z_shell = Zeff(protons, n_lower, e_in_shell)
        r_max = 1+5*(shell**2/Z_shell)
        r = np.linspace(0, r_max, int(r_max*100)+100)
        subshells = subshell_electron_counts(e_in_shell)
        for subshell, e_in_subshell in subshells.items():
            l_subshell = angular_momentum_dict[subshell]
            p_rad = psi_r2(r, shell, l_subshell, Z_shell)*r**2
            p_rad = p_rad / sum(p_rad)
            cdf = np.cumsum(p_rad)
            mask = cdf < 0.999
            rand = np.random.rand(samps_per_e*e_in_shell)
            r_values = np.interp(rand, cdf, r)
            dr = np.diff(r).mean()
            theta_vals = np.random.rand(samps_per_e*e_in_shell)*2*np.pi
            df = df.append(pd.DataFrame(dict(r=r[mask], p=p_rad[mask]/dr, n=shell, l=l_subshell, subshell=f"{shell}{subshell}")))
            df2 = df2.append(pd.DataFrame(dict(x=np.cos(theta_vals)*r_values, y=np.sin(theta_vals)*r_values, r=np.round(r_values, 3), n=shell, l=l_subshell, subshell=f"{shell}{subshell}")))
    
    fig = px.line(df, x='r', y='p', color='subshell', line_group='subshell', hover_name='subshell')
    show_radial = c.checkbox('Show radial probability')
    if show_radial:
        c.plotly_chart(fig)
    fig2 = px.scatter(df2, x='x', y='y', color='subshell', hover_name='subshell', hover_data=['r'], opacity=0.5, width=700, height=700)
    fig2.update_yaxes(scaleanchor = "x", scaleratio = 1)
    fig2.update_traces(marker={'size': 5})
    # Find the 0.90 percentile of the radial distribution for each subshell using a pandas groupby
    # and then plot a circle with that radius
    df2['r'] = df2['r'].astype(float)
    df3 = df2.groupby('subshell')['r'].quantile(0.90).reset_index().rename(columns={'r': 'r90'})
    st.write(df3)
    c.plotly_chart(fig2)
        


if __name__ == '__main__':
    run()