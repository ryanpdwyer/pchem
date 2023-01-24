

import streamlit as st
from scipy.integrate import quad
from scipy import linalg
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

L = 10.0

def psi_pib(x, n):
    return np.sqrt(2/L)*np.sin(n*np.pi*x/L)

def V(x):
    return np.where(abs(x - L/2) < L/4, 0.5*(x-L/4)*(x-3*L/4), 0)

def T(n):
    hbar = 1.0
    m = 1.0
    return hbar**2*n**2*np.pi**2/(2*m*L**2)

def Tmat(N):
    return np.diag([T(n) for n in range(1, N+1)])

def V_el(i, j):
    return quad(lambda x: psi_pib(x, i)*V(x)*psi_pib(x, j), 0, L)[0]



def C_el(i, j):
    return quad(lambda x: psi_pib(x, i)*psi_pib(x, j), 0, L)[0]


def Hmat_element(i, j):
    if i == j:
        return T(i) + V_el(i, j)
    else:
        return V_el(i, j)

def Hmat(N):
    return np.array([[Hmat_element(i, j) for i in range(1, N+1)] for j in range(1, N+1)])



x = np.linspace(0, L, 1001)



def run():
    st.title("Linear Variational Method")
    with st.sidebar:
        N = st.slider("N", 1, 15, 1)
    if 'ns' not in st.session_state:
        st.session_state.ns = []
    if 'energies' not in st.session_state:
        st.session_state.energies = []
    if 'energiesT' not in st.session_state:
        st.session_state.energiesT = []
    H3 = Hmat(N)

    
    psi_n = np.array([psi_pib(x,i) for i in range(1, N+1)])

    eigs, vecs = linalg.eigh(H3)
    cn = vecs[:,0]/np.sign(vecs[0,0])

    if N not in st.session_state.ns:
        st.session_state.energies.append(eigs[0])
        st.session_state.ns.append(N)
        st.session_state.energiesT.append(cn@Tmat(N)@cn)

    psi = cn@psi_n
    V_current = eigs[0] - cn@Tmat(N)@cn
    fig, axes = plt.subplots(nrows=2, sharex=True)
    ax0 = axes[0]
    ax0.set_xlim(0, L)
    ax0.set_ylabel("P.E. $V$ (hartree)")
    ax0.axhline(eigs[0], color='0.2', linewidth=0.7, linestyle='-', label="$\\langle E\\rangle$")
    ax0.axhline(V_current, color='0.2', linewidth=0.7, linestyle='--', label="$\\langle V\\rangle$")
    ax0.plot(x, V(x), label="$V(x)$")
    # Set line color to be the same as the fill color
    ax0.fill_between(x, psi**2 * V(x), 0, alpha=0.5, label='$\psi^2 V(x)$', color='C0')
    ax0.legend()
    ax = axes[1]
    ax.plot(x, psi)
    ax.set_xlabel("x")
    ax.set_ylabel("$\psi$")

    st.pyplot(fig)
    Narray = np.array(st.session_state.ns)
    sortinds = np.argsort(Narray)
    Narray = Narray[sortinds]
    Tarray = np.array(st.session_state.energiesT)[sortinds]
    Earray = np.array(st.session_state.energies)[sortinds]
    Varray = Earray-Tarray
    df = pd.DataFrame({'N': Narray, 'Energy': Earray, 'T': Tarray, 'type': 'Total'})
    df2 = pd.DataFrame({'N': Narray, 'Energy': Varray, 'T': Tarray, 'type': 'Potential'})
    df = df.append(df2)

    f2 = px.line(df, x='N', y='Energy', color='type', symbol='type', hover_data='T')
    # Add scatter plot markers to points at the same location
    st.plotly_chart(f2)

    # Coefficient vector
    st.write(cn)
    


if __name__ == '__main__':
    run()