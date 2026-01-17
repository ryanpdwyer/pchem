"""Home - Chemistry and FYS Tools Overview"""
import streamlit as st

st.title("Chemistry and FYS Educational Tools")

st.markdown("""
Welcome! This application contains interactive tools for chemistry and physics courses.
Use the sidebar to navigate to specific tools, or browse the categories below.
""")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ⚗️ Kinetics")
    st.markdown("""
    - [CHE 120 Kinetics](/kinetics) - Arrhenius equation visualization
    - [Ocean Optics Kinetics](/ocean-optics) - UV-Vis kinetics analysis
    """)

    st.markdown("### 🔥 Thermodynamics")
    st.markdown("""
    - [1st Law](/first-law) - First law simulation
    - [2nd Law Calorimeter](/calorimeter) - Calorimetry simulation
    - [Mystery Gas](/mystery-gas) - Gas thermodynamics
    - [Where is Equilibrium?](/equilibrium) - Entropy visualization
    - [Compressibility Factor](/compressibility) - Van der Waals equation
    """)

    st.markdown("### ⚛️ Quantum Chemistry")
    st.markdown("""
    - [Variational Gaussian](/variational-gaussian) - Gaussian trial functions
    - [Linear Variational](/variational-linear) - Linear combination methods
    - [Zeff](/zeff) - Effective nuclear charge
    - [Electron Visualization](/electron-viz) - Electron density visualization
    """)

with col2:
    st.markdown("### 📈 Data Analysis")
    st.markdown("""
    - [Combine UV-Vis](/combine-uvvis) - Combine UV-Vis spectra
    - [Combine Raman](/combine-raman) - Combine Raman spectra
    - [Combine Electrochem](/combine-echem) - Combine electrochemistry data
    - [Peak Picking](/echem-peaks) - Electrochemical peak analysis
    - [Plot Excel](/plot-excel) - Plot Excel absorbance data
    - [Plot Solartron](/plot-solartron) - Plot Solartron data
    """)

    st.markdown("### ⚡ Electrochemistry")
    st.markdown("""
    - [Impedance](/impedance) - EIS fitting and analysis
    """)

    st.markdown("### 🤖 AI Tools")
    st.markdown("""
    - [Image Classifier](/image-classifier) - GPT-4 image classification
    - [Image Regression](/image-regression) - GPT-4 image regression
    - [GPT-3 vs ChatGPT](/gpt-compare) - Compare GPT models
    - [Neural Network Game](/neural-network) - Interactive neural net
    - [MIR Puzzle](/mir-puzzle) - Method of initial rates
    """)

    st.markdown("### 🔢 Utilities")
    st.markdown("""
    - [Sympy Shell](/sympy-shell) - Symbolic math calculator
    """)
