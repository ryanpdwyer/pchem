

import streamlit as st
from scipy.integrate import quad
from scipy import linalg
import sympy as sm
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pchem import nintegrate, vectorize
x = sm.Symbol('x')

hbar = 1.0
m=1.0
L = 10.0

def psi_pib(x, n):
    return np.sqrt(2/L)*np.sin(n*np.pi*x/L)

def T(psi):
    """The kinetic energy operator: -ħ²/(2m) d²ψ/dx²"""
    return -hbar**2/(2*m) * sm.diff(psi, x, x)

def V(x):
    return sm.Piecewise((0.5*(x-L/4)*(x-3*L/4), abs(x - L/2) < L/4), (0, True)) 

def avg_E(psi, V):
    return nintegrate(psi*H(psi, V), (x, 0, L)) / nintegrate(psi*psi, (x, 0, L))

def operator(psi, Op, V=None):
    return nintegrate(psi*Op(psi), (x, 0, L)) / nintegrate(psi*psi, (x, 0, L))

def H(psi, V):
    return T(psi) + V(x)*psi

def gauss(x, mu, sigma):
    return sm.exp(-(x-mu)**2/(2*sigma**2)) * 1/sm.sqrt(sigma*sm.sqrt(sm.pi))

x_vals = np.linspace(0, L, 1001)



def run():
    st.title("Variational Method")
    st.markdown("Here's an example showing the variational method applied to a particle with a little bump inside a box. The particle's wavefunction is a Gaussian with variable width $\sigma$: $\phi(x) = A e^{-\\frac{(x-\\mu)^2}{ 2\sigma ^2}}$, with $\mu=\\frac{L}{2} =5a_0$. What is your best guess for the ground state wavefunction and energy?" )
    with st.sidebar:
        sigma = st.slider('sigma', 0.25, 2.0, value=0.25)
    if 'sigma' not in st.session_state:
        st.session_state.sigma = []
    if 'energies' not in st.session_state:
        st.session_state.energies = []
    if 'energiesT' not in st.session_state:
        st.session_state.energiesT = []
    
    psi = gauss(x, L/2, sigma)

    E1 = avg_E(psi, V)
    T1 = operator(psi, T)

    if sigma not in st.session_state.sigma:
        st.session_state.energies.append(E1)
        st.session_state.sigma.append(sigma)
        st.session_state.energiesT.append(T1)


    V_vals = vectorize(V(x),x, x_vals)
    psi_vals = vectorize(psi, x, x_vals)
    fig, axes = plt.subplots(nrows=2, sharex=True)
    ax0 = axes[0]
    ax0.set_xlim(0, L)
    ax0.set_ylabel("P.E. $V$ (hartree)")
    ax0.axhline(E1, color='0.2', linewidth=0.7, linestyle='-', label="$\\langle E\\rangle$")
    ax0.axhline(E1-T1, color='0.2', linewidth=0.7, linestyle='--', label="$\\langle V\\rangle$")
    ax0.plot(x_vals, V_vals, label="$V(x)$")
    # Set line color to be the same as the fill color
    ax0.fill_between(x_vals, psi_vals**2 * V_vals, 0, alpha=0.5, label='$\psi^2 V(x)$', color='C0')
    ax0.legend(loc='upper right')
    ax = axes[1]
    ax.plot(x_vals, psi_vals)
    ax.set_xlabel("x")
    ax.set_ylabel("$\psi$")

    st.pyplot(fig)
    sigma_array = np.array(st.session_state.sigma)
    sortinds = np.argsort(sigma_array)
    sigma_array = sigma_array[sortinds]
    Tarray = np.array(st.session_state.energiesT)[sortinds]
    Earray = np.array(st.session_state.energies)[sortinds]
    Varray = Earray-Tarray
    df = pd.DataFrame({'sigma': sigma_array, 'Energy': Earray, 'T': Tarray, 'type': 'Total'})
    df2 = pd.DataFrame({'sigma': sigma_array, 'Energy': Varray, 'T': Tarray, 'type': 'Potential'})
    df = df.append(df2)

    # annotations
    f2 = px.line(df, x='sigma', y='Energy', color='type', symbol='type', hover_data='T')
    # Add scatter plot markers to points at the same location
    st.plotly_chart(f2)
    


if __name__ == '__main__':
    run()