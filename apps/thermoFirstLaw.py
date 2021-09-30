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

from util import write_excel

# {{InterpretationBox[
# TooltipBox[
# GraphicsBox[{{GrayLevel[0], 
# RectangleBox[{0, 0}]}, {GrayLevel[0], 
# RectangleBox[{1, -1}]}, 
# {RGBColor[0.87`, 0.94`, 1], 
# RectangleBox[{0, -1}, {2, 1}]}}, 
# AspectRatio -> 1, Frame -> True, 
# FrameStyle -> RGBColor[0.5800000000000001`, 0.6266666666666667`, 0.6666666666666666`], 
# FrameTicks -> None, PlotRangePadding -> None, 
# ImageSize -> Dynamic[{Automatic, Times[1.35`, Times[CurrentValue["FontCapHeight"], 
# Power[AbsoluteCurrentValue[Magnification], -1]]]}]], "RGBColor[0.87, 0.94, 1]"], 
# RGBColor[0.87`, 0.94`, 1], Editable -> False, Selectable -> False]
# ,Rectangle[$CellContext`bl$pt,$CellContext`tr$pt]},
# Text[total work [J]:,{0.28\[VeryThinSpace]+$CellContext`txt,-0.16+$CellContext`txt}],
# Text[T [\[Degree]C]:,{0.36\[VeryThinSpace]+$CellContext`txt,-0.25+$CellContext`txt}],
# Text[$CellContext`work,{0.6\[VeryThinSpace]+$CellContext`txt,-0.16+$CellContext`txt}],
# Text[$CellContext`Tcurr,{0.6\[VeryThinSpace]+$CellContext`txt,-0.25+$CellContext`txt}],
# $CellContext`dict[$CellContext`c],
# {FaceForm[InterpretationBox[TooltipBox[GraphicsBox[{{GrayLevel[0], RectangleBox[{0, 0}]}, 
# {GrayLevel[0], RectangleBox[{1, -1}]},
# {GrayLevel[0, 0], RectangleBox[{0, -1}, {2, 1}]}},
# AspectRatio -> 1, Frame -> True, FrameStyle -> GrayLevel[0`, 0`],
# FrameTicks -> None, PlotRangePadding -> None,
# ImageSize -> Dynamic[{Automatic, Times[1.35`, Times[CurrentValue["FontCapHeight"],
# Power[AbsoluteCurrentValue[Magnification], -1]]]}]], "GrayLevel[0, 0]"],
# GrayLevel[0, 0], Editable -> False, Selectable -> False]],
# EdgeForm[InterpretationBox[TooltipBox[GraphicsBox[{{GrayLevel[0], RectangleBox[{0, 0}]},
# {GrayLevel[0], RectangleBox[{1, -1}]}, {GrayLevel[0], RectangleBox[{0, -1}, {2, 1}]}}, AspectRatio -> 1, Frame -> True, FrameStyle -> GrayLevel[0`], 
# FrameTicks -> None, PlotRangePadding -> None, ImageSize -> Dynamic[{Automatic, Times[1.35`, Times[CurrentValue["FontCapHeight"],
# Power[AbsoluteCurrentValue[Magnification], -1]]]}]], "GrayLevel[0]"], GrayLevel[0], Editable -> False, Selectable -> False]],
# Rectangle[{0.4,-0.13},{0.6,0.01}]},
# Arrow[{{0.5,-0.12},{0.5\[VeryThinSpace]+0.12 Cos[FractionBox["1", "180"] (135-FractionBox[RowBox[{"90", " ", "\:f74e"}], "$CellContext`Imax"]) \[Pi]],
# -0.12+0.12 Sin[FractionBox["1", "180"] (135-FractionBox[RowBox[{"90", " ", "\:f74e"}], "$CellContext`Imax"]) \[Pi]]}}],
# {FaceForm[{InterpretationBox[TooltipBox[GraphicsBox[{{GrayLevel[0], RectangleBox[{0, 0}]}, 
# {GrayLevel[0], RectangleBox[{1, -1}]}, {GrayLevel[0.9`], RectangleBox[{0, -1}, {2, 1}]}}, 
# AspectRatio -> 1, Frame -> True, FrameStyle -> GrayLevel[0.6000000000000001`], FrameTicks -> None, 
# PlotRangePadding -> None, ImageSize -> Dynamic[{Automatic, Times[1.35`, Times[CurrentValue["FontCapHeight"], Power[AbsoluteCurrentValue[Magnification], -1]]]}]], "GrayLevel[0.9]"],
# GrayLevel[0.9`], Editable -> False, Selectable -> False],Opacity[0.8]}],EdgeForm[InterpretationBox[TooltipBox[GraphicsBox[{{GrayLevel[0], RectangleBox[{0, 0}]}, 
# {GrayLevel[0], RectangleBox[{1, -1}]}, {GrayLevel[0], RectangleBox[{0, -1}, {2, 1}]}}, AspectRatio -> 1, Frame -> True, FrameStyle -> GrayLevel[0`], FrameTicks -> None,
#  PlotRangePadding -> None, ImageSize -> Dynamic[{Automatic, Times[1.35`, Times[CurrentValue["FontCapHeight"], Power[AbsoluteCurrentValue[Magnification], -1]]]}]], "GrayLevel[0]"], 
# GrayLevel[0], Editable -> False, Selectable -> False]],Rectangle[{0.46\[VeryThinSpace]-$CellContext`th,0.4},{0.54\[VeryThinSpace]-$CellContext`th,1.2}]},
# {$CellContext`dark$red,Rectangle[{0.47\[VeryThinSpace]-$CellContext`th,0.4},{0.53\[VeryThinSpace]-$CellContext`th,0.4\[VeryThinSpace]+$CellContext`dy (-14+$CellContext`Tcurr)}]},
# Table[Line[{{0.475\[VeryThinSpace]-$CellContext`th,0.4\[VeryThinSpace]+$CellContext`dy+$CellContext`i $CellContext`dy},{0.525\[VeryThinSpace]-$CellContext`th,0.4\[VeryThinSpace]+$CellContext`dy+$CellContext`i $CellContext`dy}}],{$CellContext`i,0,$CellContext`npts}],Line[{{0.4,-0.06},{0.3,-0.06},{0.3,0.19},{0.7,0.19},{0.7,-0.06},{0.6,-0.06}}],{InterpretationBox[TooltipBox[GraphicsBox[{{GrayLevel[0], RectangleBox[{0, 0}]}, {GrayLevel[0], RectangleBox[{1, -1}]}, {RGBColor[0.7333333333333333`, 0.6`, 0.4666666666666667`], RectangleBox[{0, -1}, {2, 1}]}}, AspectRatio -> 1, Frame -> True, FrameStyle -> RGBColor[0.4888888888888889`, 0.4`, 0.3111111111111111`], FrameTicks -> None, PlotRangePadding -> None, ImageSize -> Dynamic[{Automatic, Times[1.35`, Times[CurrentValue["FontCapHeight"], Power[AbsoluteCurrentValue[Magnification], -1]]]}]], "RGBColor[0.7333333333333333, 0.6, 0.4666666666666667]"], RGBColor[0.7333333333333333`, 0.6`, 0.4666666666666667`], Editable -> False, Selectable -> False],Rectangle[{0.36,0.15},{0.64,0.23}]},Text[15,{0.43\[VeryThinSpace]-$CellContext`th,0.395\[VeryThinSpace]+$CellContext`dy}],Text[20,{0.43\[VeryThinSpace]-$CellContext`th,0.395\[VeryThinSpace]+6 $CellContext`dy}],Text[25,{0.43\[VeryThinSpace]-$CellContext`th,0.395\[VeryThinSpace]+11 $CellContext`dy}]}

T = np.arange(0, 101, dtype=float)
cP = np.array([
4.217,
4.213,
4.21,
4.207,
4.205,
4.202,
4.2,
4.198,
4.196,
4.194,
4.192,
4.191,
4.189,
4.188,
4.187,
4.186,
4.185,
4.184,
4.183,
4.182,
4.182,
4.181,
4.181,
4.18,
4.18,
4.18,
4.179,
4.179,
4.179,
4.179,
4.178,
4.178,
4.178,
4.178,
4.178,
4.178,
4.178,
4.178,
4.178,
4.179,
4.179,
4.179,
4.179,
4.179,
4.179,
4.18,
4.18,
4.18,
4.18,
4.181,
4.181,
4.181,
4.182,
4.182,
4.182,
4.183,
4.183,
4.183,
4.184,
4.184,
4.185,
4.185,
4.186,
4.186,
4.187,
4.187,
4.188,
4.188,
4.189,
4.189,
4.19,
4.19,
4.191,
4.192,
4.192,
4.193,
4.194,
4.194,
4.195,
4.196,
4.196,
4.197,
4.198,
4.199,
4.2,
4.2,
4.201,
4.202,
4.203,
4.204,
4.205,
4.206,
4.207,
4.208,
4.209,
4.21,
4.211,
4.212,
4.213,
4.214,
4.216
])

cP_water = interpolate.interp1d(T, cP)

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
    water = patches.Rectangle((0.1,0.1), 0.8, 0.8, linewidth=1, ec="0",
            fc='#d7f1fa')
    
    waterCu = patches.Rectangle((0.1,0.1), 0.8, 0.8, linewidth=3, ec="#b87333",
            fc='#d7f1fa')

    container_dict= {"Dewar": [patches.Rectangle((0.05,0.05), 0.9, 0.9, linewidth=1, ec="0",
            fc='1'), water],
            "Styrofoam": [patches.Rectangle((0.05,0.05), 0.9, 0.9, linewidth=1, ec="0",
            fc='0.9'), water],
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

def simulate(Tsys, Tsurr, current, work, container):
    c = container
    work += current**2 * dt
    cp = 18.01 * cP_water(Tsys)
    Tsys = Tsys + (current**2 *dt - c*(Tsys-Tsurr)*dt)/cp
    return work, Tsys


dt = 2.0

def run():
    data_default = dict(t=[0], Tsys=[20.0],work=[0])

    st.markdown("""# First Law of Thermodynamics

This interactive lets you explore the first law of thermodynamics for a system
consisting of 1 mol of water at a constant pressure of 1 atm. The sliders let you 

- Control the temperature of the system ($T_\\text{sys}$) and surroundings ($T_\\text{surr}$).
- Do work on the system by controlling the current (0 to 5 A) through a 1 Ω resistor.
- Choose the walls of the container; note that the dewar has perfectly adiabatic walls.
""")

    if 'running' not in st.session_state:
        st.session_state.running = False
    
    containers = {"Dewar": 0, "Styrofoam": 0.03, "Cu in Air": 0.3, "Cu in Water": 10}

    containers_list = list(containers.keys())

    container_index_dict = {name: i for i, name in enumerate(containers_list)}

    if 'container' not in st.session_state:
        st.session_state.container = "Dewar"

    if 'data' not in st.session_state:
        st.session_state.data = copy.copy(data_default)
    

    Tsys = st.sidebar.slider("System temperature (°C)", value=float(st.session_state.data["Tsys"][-1]), max_value=100.0, min_value=0.0, step=0.1)
    Tsurr = st.sidebar.slider("Surroundings temperature (°C)", value=20.0, max_value=100.0, min_value=0.0, step=0.1)
    current = st.sidebar.slider("Current (A)", value=0.0, min_value=0.0, max_value=5.0, step=0.01)
    container = st.sidebar.selectbox("System walls:", containers_list)
    
    st.session_state.container = containers[container]

    button_text = "Pause" if st.session_state.running else "Run"
    start_stop_sim = st.sidebar.button(f"{button_text} simulation")

    if start_stop_sim:
        st.session_state.running = not st.session_state.running
        if st.session_state.running: # Reset temperature...
            st.session_state.data["Tsys"][-1] = Tsys
        
        st.experimental_rerun()
    
    reset_simulation = st.sidebar.button(f"Reset simulation")

    if reset_simulation:
        st.session_state.running = False
        st.session_state.data = copy.copy(data_default)
        st.session_state.container = "Dewar"

        st.experimental_rerun()

    if st.session_state.running:
        st.markdown("### Simulation state: running")
    else:
        st.markdown("### Simulation state: paused")

    work = st.session_state.data["work"][-1]
    Tsys = st.session_state.data["Tsys"][-1]



    st.markdown(f"""## Properties
$T_{{\\text{{sys}}}}$ = {Tsys:.2f} °C,  &nbsp; &nbsp;$T_{{\\text{{surr}}}}$ = {Tsurr:.2f} °C

Total work $w$ = {work:.2f} J, &nbsp; &nbsp; Walls: {container}
    """, unsafe_allow_html=True
        )

    fig, ax = draw(current, container)
    st.pyplot(fig)

    show_data = st.checkbox(label="Show data")

    save_excel_button = False
    if show_data:
        df = pd.DataFrame(st.session_state.data)
        df.rename(columns={"work": "work (J)", "Tsys": "Tsys (°C)", "t": "Time (s)"}, inplace=True)
        st.write(df)

        filename = st.text_input("Filename:", value="CHE341-1stLaw-data")
        save_excel_button = st.button("Save to Excel")
    
        if save_excel_button:
            write_excel(df, filename)

    # Needs to be at the bottom
    if st.session_state.running:
        work, Tsys = simulate(Tsys, Tsurr, current, work, st.session_state.container)
        st.session_state.data["work"].append(work)
        st.session_state.data["Tsys"].append(Tsys)
        st.session_state.data['t'].append(st.session_state.data['t'][-1]+dt)
        time.sleep(0.5)
        st.experimental_rerun()




if __name__ == '__main__':
    run()
