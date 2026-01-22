"""
PChem - Chemistry and FYS Educational Tools
"""
import streamlit as st

st.set_page_config(
    page_title="Chemistry & FYS Tools",
    page_icon="flask",
    layout="wide",
)

# All pages - flat list, no sidebar sections
all_pages = [
    st.Page("pages/home.py", title="Home", url_path="home", default=True),
    # CHE 120
    st.Page("pages/kinetics/arrhenius.py", title="Kinetics", url_path="kinetics"),
    st.Page("pages/quantum/zeff.py", title="Zeff", url_path="zeff"),
    st.Page("pages/quantum/electron_viz.py", title="Electron Visualization", url_path="electron-viz"),
    # Kinetics
    st.Page("pages/kinetics/ocean_optics.py", title="Ocean Optics Kinetics", url_path="ocean-optics"),
    # Thermodynamics
    st.Page("pages/thermodynamics/first_law.py", title="1st Law", url_path="first-law"),
    st.Page("pages/thermodynamics/calorimeter.py", title="2nd Law Calorimeter", url_path="calorimeter"),
    st.Page("pages/thermodynamics/mystery_gas.py", title="Mystery Gas", url_path="mystery-gas"),
    st.Page("pages/thermodynamics/equilibrium.py", title="Where is Equilibrium?", url_path="equilibrium"),
    st.Page("pages/thermodynamics/compressibility.py", title="Compressibility Factor", url_path="compressibility"),
    # Quantum
    st.Page("pages/quantum/variational_gaussian.py", title="Variational Gaussian", url_path="variational-gaussian"),
    st.Page("pages/quantum/variational_linear.py", title="Linear Variational", url_path="variational-linear"),
    # Data Analysis
    st.Page("pages/data_analysis/combine_uvvis.py", title="Combine UV-Vis", url_path="combine-uvvis"),
    st.Page("pages/data_analysis/combine_raman.py", title="Combine Raman", url_path="combine-raman"),
    st.Page("pages/data_analysis/combine_echem.py", title="Combine Electrochem", url_path="combine-echem"),
    st.Page("pages/data_analysis/echem_peaks.py", title="Peak Picking", url_path="echem-peaks"),
    st.Page("pages/data_analysis/plot_excel.py", title="Plot Excel", url_path="plot-excel"),
    st.Page("pages/data_analysis/plot_solartron.py", title="Plot Solartron", url_path="plot-solartron"),
    # Electrochemistry
    st.Page("pages/electrochemistry/impedance.py", title="Impedance", url_path="impedance"),
    # AI Tools
    st.Page("pages/ai_tools/image_classifier.py", title="Image Classifier", url_path="image-classifier"),
    st.Page("pages/ai_tools/image_regression.py", title="Image Regression", url_path="image-regression"),
    st.Page("pages/ai_tools/chatgpt_compare.py", title="GPT-3 vs ChatGPT", url_path="gpt-compare"),
    st.Page("pages/ai_tools/neural_network.py", title="Neural Network Game", url_path="neural-network"),
    st.Page("pages/ai_tools/mir_puzzle.py", title="MIR Puzzle", url_path="mir-puzzle"),
    # Utilities
    st.Page("pages/utilities/sympy_shell.py", title="Sympy Shell", url_path="sympy-shell"),
]

# Navigation without sidebar (position="hidden")
pg = st.navigation(all_pages, position="hidden")

# Run selected page
pg.run()
