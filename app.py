"""
PChem - Chemistry and FYS Educational Tools

Multi-page Streamlit application using st.navigation for organized navigation
with separate URLs for each app.
"""
import streamlit as st

st.set_page_config(
    page_title="Chemistry & FYS Tools",
    page_icon="flask",
    layout="wide",
)

# Clean, minimal CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;500;600&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons+Outlined');

    /* Global typography */
    html, body, [class*="css"] {
        font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        border-right: 1px solid #e9ecef;
    }

    section[data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.1rem;
        font-weight: 600;
        color: #343a40;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        padding: 0.5rem 0;
        border-bottom: 2px solid #dee2e6;
        margin-bottom: 1rem;
    }

    /* Navigation section headers */
    section[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"] {
        font-size: 0.7rem;
        font-weight: 500;
        color: #6c757d;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 1.5rem;
    }

    /* Navigation links */
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {
        font-size: 0.9rem;
        color: #495057;
        padding: 0.4rem 0.75rem;
        border-radius: 4px;
        transition: background 0.15s ease;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {
        background: #e9ecef;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {
        background: #e7f1ff;
        color: #0d6efd;
        font-weight: 500;
    }

    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Headers */
    h1 {
        font-weight: 600;
        color: #212529;
        letter-spacing: -0.01em;
    }

    h2, h3 {
        font-weight: 500;
        color: #343a40;
    }

    /* Subtle dividers */
    hr {
        border: none;
        border-top: 1px solid #e9ecef;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Home page
home_page = [
    st.Page("pages/home.py", title="Home", url_path="home", default=True),
]

# Define pages by category - clean titles, no icons
che120_pages = [
    st.Page("pages/kinetics/arrhenius.py", title="Kinetics", url_path="kinetics"),
    st.Page("pages/quantum/zeff.py", title="Zeff", url_path="zeff"),
    st.Page("pages/quantum/electron_viz.py", title="Electron Visualization", url_path="electron-viz"),
]

kinetics_pages = [
    st.Page("pages/kinetics/ocean_optics.py", title="Ocean Optics Kinetics", url_path="ocean-optics"),
]

thermodynamics_pages = [
    st.Page("pages/thermodynamics/first_law.py", title="1st Law", url_path="first-law"),
    st.Page("pages/thermodynamics/calorimeter.py", title="2nd Law Calorimeter", url_path="calorimeter"),
    st.Page("pages/thermodynamics/mystery_gas.py", title="Mystery Gas", url_path="mystery-gas"),
    st.Page("pages/thermodynamics/equilibrium.py", title="Where is Equilibrium?", url_path="equilibrium"),
    st.Page("pages/thermodynamics/compressibility.py", title="Compressibility Factor", url_path="compressibility"),
]

quantum_pages = [
    st.Page("pages/quantum/variational_gaussian.py", title="Variational Gaussian", url_path="variational-gaussian"),
    st.Page("pages/quantum/variational_linear.py", title="Linear Variational", url_path="variational-linear"),
]

data_analysis_pages = [
    st.Page("pages/data_analysis/combine_uvvis.py", title="Combine UV-Vis", url_path="combine-uvvis"),
    st.Page("pages/data_analysis/combine_raman.py", title="Combine Raman", url_path="combine-raman"),
    st.Page("pages/data_analysis/combine_echem.py", title="Combine Electrochem", url_path="combine-echem"),
    st.Page("pages/data_analysis/echem_peaks.py", title="Peak Picking", url_path="echem-peaks"),
    st.Page("pages/data_analysis/plot_excel.py", title="Plot Excel", url_path="plot-excel"),
    st.Page("pages/data_analysis/plot_solartron.py", title="Plot Solartron", url_path="plot-solartron"),
]

electrochemistry_pages = [
    st.Page("pages/electrochemistry/impedance.py", title="Impedance", url_path="impedance"),
]

ai_tools_pages = [
    st.Page("pages/ai_tools/image_classifier.py", title="Image Classifier", url_path="image-classifier"),
    st.Page("pages/ai_tools/image_regression.py", title="Image Regression", url_path="image-regression"),
    st.Page("pages/ai_tools/chatgpt_compare.py", title="GPT-3 vs ChatGPT", url_path="gpt-compare"),
    st.Page("pages/ai_tools/neural_network.py", title="Neural Network Game", url_path="neural-network"),
    st.Page("pages/ai_tools/mir_puzzle.py", title="MIR Puzzle", url_path="mir-puzzle"),
]

utilities_pages = [
    st.Page("pages/utilities/sympy_shell.py", title="Sympy Shell", url_path="sympy-shell"),
]

# Navigation with sections
pg = st.navigation({
    "": home_page,
    "CHE 120": che120_pages,
    "Kinetics": kinetics_pages,
    "Thermodynamics": thermodynamics_pages,
    "Quantum Chemistry": quantum_pages,
    "Data Analysis": data_analysis_pages,
    "Electrochemistry": electrochemistry_pages,
    "AI Tools": ai_tools_pages,
    "Utilities": utilities_pages,
})

# Sidebar title
st.sidebar.title("Chemistry & FYS Tools")

# Run selected page
pg.run()
