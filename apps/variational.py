

import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

L = 10.0

def psi_pib(x, n):
    return np.sqrt(2/L)*np.sin(n*np.pi*x/L)

def V(x):
    return np.where(abs(x - L/2) < L/4, 0.5*(x-L/4)*(x-3*L/4), 0)



def run():
    c1 = st.slider("c1", -1.0, 1.0, value=0.9)
    c2 = st.slider("c2", -1.0, 1.0, value=0.0)
    c3 = st.slider("c3", -1.0, 1.0, value=0.0)
    cn = np.array([c1, c2, c3])
    m = cn@cn
    x = np.linspace(0, L, 1001)
    psi_n = np.array([psi_pib(x,i) for i in range(1, 4)])
    psi = cn@psi_n/np.sqrt(m)
    fig, axes = plt.subplots(nrows=2)
    ax0 = axes[0]
    ax0.plot(x, V(x))
    ax = axes[1]
    ax.plot(x, psi)
    ax.set_xlabel("x")
    ax.set_ylabel("$\psi$")
    st.pyplot(fig)
    


if __name__ == '__main__':
    run()