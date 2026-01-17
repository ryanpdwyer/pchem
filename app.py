"""
PChem - Chemistry and FYS Educational Tools

Multi-page Streamlit application using st.navigation for organized navigation
with separate URLs for each app.
"""
import streamlit as st

st.set_page_config(
    page_title="Chemistry and FYS Tools",
    page_icon="⚗️",
    layout="wide",
)

# Home page
home_page = [
    st.Page("pages/home.py", title="Home", icon="🏠", url_path="home", default=True),
]

# Define pages by category
kinetics_pages = [
    st.Page("pages/kinetics/arrhenius.py", title="CHE 120 Kinetics", icon="⚗️", url_path="kinetics"),
    st.Page("pages/kinetics/ocean_optics.py", title="Ocean Optics Kinetics", icon="🔬", url_path="ocean-optics"),
]

thermodynamics_pages = [
    st.Page("pages/thermodynamics/first_law.py", title="1st Law", icon="🔥", url_path="first-law"),
    st.Page("pages/thermodynamics/calorimeter.py", title="2nd Law Calorimeter", icon="🌡️", url_path="calorimeter"),
    st.Page("pages/thermodynamics/mystery_gas.py", title="Mystery Gas", icon="💨", url_path="mystery-gas"),
    st.Page("pages/thermodynamics/equilibrium.py", title="Where is Equilibrium?", icon="⚖️", url_path="equilibrium"),
    st.Page("pages/thermodynamics/compressibility.py", title="Compressibility Factor", icon="📊", url_path="compressibility"),
]

quantum_pages = [
    st.Page("pages/quantum/variational_gaussian.py", title="Variational Gaussian", icon="🌊", url_path="variational-gaussian"),
    st.Page("pages/quantum/variational_linear.py", title="Linear Variational", icon="📐", url_path="variational-linear"),
    st.Page("pages/quantum/zeff.py", title="Zeff", icon="⚛️", url_path="zeff"),
    st.Page("pages/quantum/electron_viz.py", title="Electron Visualization", icon="✨", url_path="electron-viz"),
]

data_analysis_pages = [
    st.Page("pages/data_analysis/combine_uvvis.py", title="Combine UV-Vis", icon="📈", url_path="combine-uvvis"),
    st.Page("pages/data_analysis/combine_raman.py", title="Combine Raman", icon="📉", url_path="combine-raman"),
    st.Page("pages/data_analysis/combine_echem.py", title="Combine Electrochem", icon="🔋", url_path="combine-echem"),
    st.Page("pages/data_analysis/echem_peaks.py", title="Peak Picking", icon="📍", url_path="echem-peaks"),
    st.Page("pages/data_analysis/plot_excel.py", title="Plot Excel", icon="📊", url_path="plot-excel"),
    st.Page("pages/data_analysis/plot_solartron.py", title="Plot Solartron", icon="📉", url_path="plot-solartron"),
]

electrochemistry_pages = [
    st.Page("pages/electrochemistry/impedance.py", title="Impedance", icon="⚡", url_path="impedance"),
]

ai_tools_pages = [
    st.Page("pages/ai_tools/image_classifier.py", title="Image Classifier", icon="🖼️", url_path="image-classifier"),
    st.Page("pages/ai_tools/image_regression.py", title="Image Regression", icon="📸", url_path="image-regression"),
    st.Page("pages/ai_tools/chatgpt_compare.py", title="GPT-3 vs ChatGPT", icon="🤖", url_path="gpt-compare"),
    st.Page("pages/ai_tools/neural_network.py", title="Neural Network Game", icon="🧠", url_path="neural-network"),
    st.Page("pages/ai_tools/mir_puzzle.py", title="MIR Puzzle", icon="🧩", url_path="mir-puzzle"),
]

utilities_pages = [
    st.Page("pages/utilities/sympy_shell.py", title="Sympy Shell", icon="🔢", url_path="sympy-shell"),
]

# Navigation with sections
pg = st.navigation({
    "": home_page,
    "Kinetics": kinetics_pages,
    "Thermodynamics": thermodynamics_pages,
    "Quantum Chemistry": quantum_pages,
    "Data Analysis": data_analysis_pages,
    "Electrochemistry": electrochemistry_pages,
    "AI Tools": ai_tools_pages,
    "Utilities": utilities_pages,
})

# Common header for all pages
st.sidebar.title("Chemistry & FYS Tools")

# Run selected page
pg.run()
