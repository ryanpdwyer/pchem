"""Home - Chemistry and FYS Tools Overview"""
import streamlit as st

# Custom CSS for home page cards
st.markdown("""
<style>
    .tool-section {
        margin-bottom: 2rem;
    }
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e9ecef;
    }
    .section-header .material-icons-outlined {
        font-size: 1.25rem;
        color: #6c757d;
    }
    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #495057;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin: 0;
    }
    .tool-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    .tool-list li {
        padding: 0.4rem 0;
    }
    .tool-list a {
        color: #495057;
        text-decoration: none;
        font-size: 0.95rem;
        transition: color 0.15s ease;
    }
    .tool-list a:hover {
        color: #0d6efd;
    }
    .tool-desc {
        color: #868e96;
        font-size: 0.85rem;
        margin-left: 0.25rem;
    }
    .welcome-text {
        color: #6c757d;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 2rem;
    }
</style>
<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Outlined" rel="stylesheet">
""", unsafe_allow_html=True)

st.title("Chemistry & FYS Tools")

st.markdown("""
<p class="welcome-text">
Interactive tools for chemistry and FYS courses. Select a tool from the sidebar or browse the categories below.
</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="tool-section">
        <div class="section-header">
            <span class="material-icons-outlined">school</span>
            <h3 class="section-title">CHE 120</h3>
        </div>
        <ul class="tool-list">
            <li><a href="/kinetics">Kinetics</a> <span class="tool-desc">Arrhenius equation</span></li>
            <li><a href="/zeff">Zeff</a> <span class="tool-desc">Effective nuclear charge</span></li>
            <li><a href="/electron-viz">Electron Visualization</a> <span class="tool-desc">Density visualization</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tool-section">
        <div class="section-header">
            <span class="material-icons-outlined">science</span>
            <h3 class="section-title">Kinetics</h3>
        </div>
        <ul class="tool-list">
            <li><a href="/ocean-optics">Ocean Optics Kinetics</a> <span class="tool-desc">UV-Vis analysis</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tool-section">
        <div class="section-header">
            <span class="material-icons-outlined">local_fire_department</span>
            <h3 class="section-title">Thermodynamics</h3>
        </div>
        <ul class="tool-list">
            <li><a href="/first-law">1st Law</a> <span class="tool-desc">Energy conservation</span></li>
            <li><a href="/calorimeter">2nd Law Calorimeter</a> <span class="tool-desc">Heat measurement</span></li>
            <li><a href="/mystery-gas">Mystery Gas</a> <span class="tool-desc">Gas identification</span></li>
            <li><a href="/equilibrium">Where is Equilibrium?</a> <span class="tool-desc">Entropy visualization</span></li>
            <li><a href="/compressibility">Compressibility Factor</a> <span class="tool-desc">Van der Waals</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tool-section">
        <div class="section-header">
            <span class="material-icons-outlined">blur_on</span>
            <h3 class="section-title">Quantum Chemistry</h3>
        </div>
        <ul class="tool-list">
            <li><a href="/variational-gaussian">Variational Gaussian</a> <span class="tool-desc">Trial functions</span></li>
            <li><a href="/variational-linear">Linear Variational</a> <span class="tool-desc">Linear combinations</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="tool-section">
        <div class="section-header">
            <span class="material-icons-outlined">insert_chart_outlined</span>
            <h3 class="section-title">Data Analysis</h3>
        </div>
        <ul class="tool-list">
            <li><a href="/combine-uvvis">Combine UV-Vis</a> <span class="tool-desc">Merge spectra</span></li>
            <li><a href="/combine-raman">Combine Raman</a> <span class="tool-desc">Merge Raman data</span></li>
            <li><a href="/combine-echem">Combine Electrochem</a> <span class="tool-desc">Merge electrochemistry</span></li>
            <li><a href="/echem-peaks">Peak Picking</a> <span class="tool-desc">Peak analysis</span></li>
            <li><a href="/plot-excel">Plot Excel</a> <span class="tool-desc">Absorbance data</span></li>
            <li><a href="/plot-solartron">Plot Solartron</a> <span class="tool-desc">Solartron data</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tool-section">
        <div class="section-header">
            <span class="material-icons-outlined">bolt</span>
            <h3 class="section-title">Electrochemistry</h3>
        </div>
        <ul class="tool-list">
            <li><a href="/impedance">Impedance</a> <span class="tool-desc">EIS fitting & analysis</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tool-section">
        <div class="section-header">
            <span class="material-icons-outlined">psychology</span>
            <h3 class="section-title">AI Tools</h3>
        </div>
        <ul class="tool-list">
            <li><a href="/image-classifier">Image Classifier</a> <span class="tool-desc">GPT-4 classification</span></li>
            <li><a href="/image-regression">Image Regression</a> <span class="tool-desc">GPT-4 regression</span></li>
            <li><a href="/gpt-compare">GPT-3 vs ChatGPT</a> <span class="tool-desc">Model comparison</span></li>
            <li><a href="/neural-network">Neural Network Game</a> <span class="tool-desc">Interactive learning</span></li>
            <li><a href="/mir-puzzle">MIR Puzzle</a> <span class="tool-desc">Initial rates method</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tool-section">
        <div class="section-header">
            <span class="material-icons-outlined">calculate</span>
            <h3 class="section-title">Utilities</h3>
        </div>
        <ul class="tool-list">
            <li><a href="/sympy-shell">Sympy Shell</a> <span class="tool-desc">Symbolic math</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
