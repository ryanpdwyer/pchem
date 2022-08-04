
import copy
import time
from os import write

import matplotlib as mpl
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle
from numpy.lib.function_base import interp
from numpy.lib.npyio import save
from scipy import interpolate
import CoolProp.CoolProp as CP

from util import write_excel


def arrow(angle):
    angle = np.radians(angle)
    length = 0.1
    return patches.FancyArrow(0.5, -0.12, np.cos(angle)*length, np.sin(angle)*length)

def rect(bl, tr, **kwargs):
    h = tr[1] - bl[1]
    w = tr[0] - bl[0]
    return patches.Rectangle(bl, w, h, **kwargs)


def draw(current, container):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.axis('off')
    
    water_color = "#d7f1fa"
    # rect((0.4,-0.13),(0.6,0.01), fc="1", ec="0", linewidth=1)
    air = patches.Rectangle((0.1,0.1), 0.8, 0.8, linewidth=1, ec="0",
            fc='1')
    
    waterCu = patches.Rectangle((0.1,0.1), 0.8, 0.8, linewidth=3, ec="#b87333",
            fc='1')

    container_dict= {"Dewar": [patches.Rectangle((0.05,0.05), 0.9, 0.9, linewidth=1, ec="0",
            fc='1'), air],
            "Styrofoam": [patches.Rectangle((0.05,0.05), 0.9, 0.9, linewidth=1, ec="0",
            fc='0.9'), air],
            "Cu in Air": [patches.Rectangle((0.05,0.05), 0.9, 0.9, linewidth=1, ec="0",
            fc='1'), waterCu],
            "Cu in Water": [patches.Rectangle((0.05,0.05), 0.9, 0.9, linewidth=1, ec="0",
            fc=water_color), waterCu]

    }

    shapes = [
    rect((0.4,-0.13),(0.6,0.01), fc="1", ec="0", linewidth=1),
    *container_dict[container],
    patches.Rectangle((0.36,0.15), 0.64-0.36, 0.08, fc="#835828", linewidth=0.5, ec="0"),
    arrow(150-current*(120)/5),
    ]

    for shape in shapes:
        ax.add_patch(shape)
    
    ax.add_line(lines.Line2D([0.40, 0.35, 0.35, 0.358], [-0.05, -0.05, 0.18, 0.18], color="0", linewidth=0.75))
    ax.add_line(lines.Line2D([0.60, 0.65, 0.65, 0.642], [-0.05, -0.05, 0.18, 0.18], color="0", linewidth=0.75))
    ax.text(0.43, -0.185, "Current", fontdict=dict(size=8))
    ax.text(0.47, 0.17, "1 Ω", fontdict=dict(size=8, color='1'))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 1)
    # ax.set_aspect('equal')
    
    return fig, ax

Volume = 0.25

def Pressure_bar(m_gas, T_Celsius):
    return CP.PropsSI('P','D', m_gas/Volume, 'T', T_Celsius+273.15, 'SulfurHexafluoride')/1e5

def simulate(Tsys, Tsurr, current, work, container, m_gas):
    mMass = 0.1460554192*1000
    n_gas = m_gas / mMass
    c = container
    work += current**2 * dt
    density = m_gas/Volume 
    cp = m_gas/1000 * CP.PropsSI('Cvmass', 'D', density, 'T', Tsys+273.15, 'SulfurHexafluoride')
    print(cp)

    Tsys = Tsys + (current**2 *dt - c*(Tsys-Tsurr)*dt)/cp
    return work, Tsys


dt = 2.0


def run():
    data_default = dict(t=[0], Tsys=[20.0], work=[0], m=[3.0])

    st.markdown("""# Thermodynamics of a Mystery Gas

This interactive lets you explore thermodynamics by loading a cylinder (fixed volume $V=$ 0.250 L) with a mass of an unknown gas.
The sliders let you 

- Control the temperature of the system ($T_\\text{sys}$) and surroundings ($T_\\text{surr}$).
- Do work on the system by controlling the current (0 to 2.5 A) through a 1 Ω resistor.
- Choose the walls of the container; note that the dewar has perfectly adiabatic walls.
""")

    if 'running' not in st.session_state:
        st.session_state.running = False
    
    containers = {"Dewar": 0, "Styrofoam": 0.03, "Cu in Air": 0.3, "Cu in Water": 10}

    containers_list = list(containers.keys())

    container_index_dict = {name: i for i, name in enumerate(containers_list)}

    if 'container' not in st.session_state:
        st.session_state.container = "Dewar"

    if 'thermGasData' not in st.session_state:
        st.session_state.thermGasData = copy.deepcopy(data_default)
    

    m_gas = st.sidebar.slider("Mass (g) of gas: ", value=float(st.session_state.thermGasData["m"][-1]), max_value=20.0, min_value=0.05, step=0.05)
    Tsys = st.sidebar.slider("System temperature (°C)", value=float(st.session_state.thermGasData["Tsys"][-1]), max_value=100.0, min_value=0.0, step=0.1)
    Tsurr = st.sidebar.slider("Surroundings temperature (°C)", value=20.0, max_value=100.0, min_value=0.0, step=0.1)
    current = st.sidebar.slider("Current (A)", value=0.0, min_value=0.0, max_value=2.5, step=0.01)
    container = st.sidebar.selectbox("System walls:", containers_list)
    
    st.session_state.container = containers[container]

    button_text = "Pause" if st.session_state.running else "Run"
    start_stop_sim = st.sidebar.button(f"{button_text} simulation")

    if start_stop_sim:
        st.session_state.running = not st.session_state.running
        if st.session_state.running: # Reset temperature...
            st.session_state.thermGasData["Tsys"][-1] = Tsys
        
        st.experimental_rerun()
    
    reset_simulation = st.sidebar.button(f"Reset simulation")

    if reset_simulation:
        st.session_state.running = False
        st.session_state.thermGasData = copy.copy(data_default)
        st.session_state.container = "Dewar"

        st.experimental_rerun()

    if st.session_state.running:
        st.markdown("### Simulation state: running")
    else:
        st.markdown("### Simulation state: paused")

    work = st.session_state.thermGasData["work"][-1]
    Tsys = st.session_state.thermGasData["Tsys"][-1]
    Psys = Pressure_bar(m_gas, Tsys)



    st.markdown(f"""## Properties
$T_{{\\text{{sys}}}}$ = {Tsys:.2f} °C,  &nbsp; &nbsp;$T_{{\\text{{surr}}}}$ = {Tsurr:.2f} °C

Total work $w$ = {work:.2f} J, &nbsp; &nbsp; Walls: {container}

Pressure $P$ = {Psys:.3f} bar
    """, unsafe_allow_html=True
        )

    fig, ax = draw(current, container)
    st.pyplot(fig)

    show_data = st.checkbox(label="Show data")

    save_excel_button = False
    if show_data:
        df = pd.DataFrame(st.session_state.thermGasData)
        df.rename(columns={"work": "work (J)", "Tsys": "Tsys (°C)", "t": "Time (s)", 'm': 'Mass (g)'}, inplace=True)
        pressures = [Pressure_bar(m, T) for m, T in zip(df['Mass (g)'].values, df["Tsys (°C)"].values)]
        df['P (bar)'] = pressures
        st.write(df)

        filename = st.text_input("Filename:", value="CHE341-gas-data")
        save_excel_button = st.button("Save to Excel")
    
        if save_excel_button:
            write_excel(df, filename)

    # Needs to be at the bottom
    if st.session_state.running:
        work, Tsys = simulate(Tsys, Tsurr, current, work, st.session_state.container, m_gas)
        st.session_state.thermGasData["work"].append(work)
        st.session_state.thermGasData["Tsys"].append(Tsys)
        st.session_state.thermGasData['m'].append(m_gas)
        st.session_state.thermGasData['t'].append(st.session_state.thermGasData['t'][-1]+dt)
        time.sleep(0.5)
        st.experimental_rerun()




if __name__ == '__main__':
    run()
