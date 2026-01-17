# Streamlit Modernization Plan

## Executive Summary

Modernize the PChem Streamlit application from a monolithic single-page app with dropdown navigation to a modern multi-page architecture with:
- **Separate URLs for each app** (e.g., `/kinetics`, `/impedance`)
- **Organized sidebar navigation** with categories
- **Modern Streamlit APIs** (upgrade from 1.5.0 to 1.53+)
- **Updated dependencies** across the board

## Current State Analysis

### Architecture Issues
1. **Single entry point** (`pchemapp.py`) with dropdown selectbox navigation
2. **No URL routing** - all apps share the same URL
3. **Outdated Streamlit version** (1.5.0 from Jan 2022)
4. **Deprecated APIs**:
   - `@st.cache` → should be `@st.cache_data` or `@st.cache_resource`
   - Internal `Server` and `get_script_run_ctx` usage (unstable internal APIs)
5. **Outdated dependencies** (numpy 1.22, openai 0.14.0, etc.)

### Current App Inventory (25 apps)

| App Name | Module | Category |
|----------|--------|----------|
| CHE 120 Kinetics | arrhen.py | Kinetics |
| Kinetics - Ocean Optics | kinetics.py | Kinetics |
| 1st Law of Thermodynamics | thermoFirstLaw.py | Thermodynamics |
| 2nd Law Calorimeter | thermoCalorimeter.py | Thermodynamics |
| Mystery Gas | thermoGas.py | Thermodynamics |
| Where is Equilibrium? | entropySplit.py | Thermodynamics |
| Compressibility Factor | compressibility.py | Thermodynamics |
| Variational Gaussian | variational_gaussian.py | Quantum |
| Linear Variational | variational.py | Quantum |
| Zeff | zeff.py | Quantum |
| Electron Visualization | random_electrons.py | Quantum |
| Combine UV-Vis Data | combineCSV.py | Data Analysis |
| Combine Raman Data | combineRaman.py | Data Analysis |
| Combine CSV Electrochem | combineCSVElectrochem.py | Data Analysis |
| Electrochemistry Peak Picking | combineEChemZip.py | Data Analysis |
| Plot Excel Data | plotExcel.py | Data Analysis |
| Plot Solartron Data | solartronData.py | Data Analysis |
| Impedance | impedance.py | Electrochemistry |
| GPT-4o-mini Image Classifier | imagemodel.py | AI Tools |
| GPT-4o-mini Image Regression | imagemodel_regression.py | AI Tools |
| GPT-3 vs ChatGPT | openai_chat.py | AI Tools |
| Neural Network Game | neuralnetwork.py | AI Tools |
| MIR Puzzle | mirpuzzle.py | AI Tools |
| Sympy Shell | ace.py | Utilities |

---

## Proposed Architecture

### New Directory Structure

```
pchem/
├── app.py                      # NEW: Main entrypoint with st.navigation
├── pages/                      # NEW: Organized page modules
│   ├── __init__.py
│   ├── kinetics/
│   │   ├── __init__.py
│   │   ├── arrhenius.py       # CHE 120 Kinetics
│   │   └── ocean_optics.py    # Ocean Optics Kinetics
│   ├── thermodynamics/
│   │   ├── __init__.py
│   │   ├── first_law.py
│   │   ├── calorimeter.py
│   │   ├── mystery_gas.py
│   │   ├── equilibrium.py
│   │   └── compressibility.py
│   ├── quantum/
│   │   ├── __init__.py
│   │   ├── variational_gaussian.py
│   │   ├── variational_linear.py
│   │   ├── zeff.py
│   │   └── electron_viz.py
│   ├── data_analysis/
│   │   ├── __init__.py
│   │   ├── combine_uvvis.py
│   │   ├── combine_raman.py
│   │   ├── combine_echem.py
│   │   ├── echem_peaks.py
│   │   ├── plot_excel.py
│   │   └── plot_solartron.py
│   ├── electrochemistry/
│   │   ├── __init__.py
│   │   └── impedance.py
│   └── ai_tools/
│       ├── __init__.py
│       ├── image_classifier.py
│       ├── image_regression.py
│       ├── chatgpt_compare.py
│       ├── neural_network.py
│       └── mir_puzzle.py
├── apps/                       # LEGACY: Keep for backward compatibility
│   └── ...                     # Existing modules (can be imported)
├── shared/                     # NEW: Shared utilities
│   ├── __init__.py
│   ├── utils.py               # Refactored from apps/util.py
│   ├── chemistry.py           # Refactored from apps/base.py
│   └── config.py              # App configuration
└── ...
```

### New Entrypoint (app.py)

```python
import streamlit as st

# Define pages with st.Page
kinetics_pages = [
    st.Page("pages/kinetics/arrhenius.py", title="CHE 120 Kinetics", icon="⚗️"),
    st.Page("pages/kinetics/ocean_optics.py", title="Ocean Optics", icon="🔬"),
]

thermo_pages = [
    st.Page("pages/thermodynamics/first_law.py", title="1st Law", icon="🔥"),
    st.Page("pages/thermodynamics/calorimeter.py", title="Calorimeter", icon="🌡️"),
    st.Page("pages/thermodynamics/mystery_gas.py", title="Mystery Gas", icon="💨"),
    st.Page("pages/thermodynamics/equilibrium.py", title="Equilibrium", icon="⚖️"),
    st.Page("pages/thermodynamics/compressibility.py", title="Compressibility", icon="📊"),
]

quantum_pages = [
    st.Page("pages/quantum/variational_gaussian.py", title="Variational Gaussian", icon="🌊"),
    st.Page("pages/quantum/variational_linear.py", title="Linear Variational", icon="📐"),
    st.Page("pages/quantum/zeff.py", title="Zeff", icon="⚛️"),
    st.Page("pages/quantum/electron_viz.py", title="Electron Visualization", icon="✨"),
]

data_pages = [
    st.Page("pages/data_analysis/combine_uvvis.py", title="Combine UV-Vis", icon="📈"),
    st.Page("pages/data_analysis/combine_raman.py", title="Combine Raman", icon="📉"),
    st.Page("pages/data_analysis/combine_echem.py", title="Combine Electrochem", icon="🔋"),
    st.Page("pages/data_analysis/echem_peaks.py", title="Peak Picking", icon="📍"),
    st.Page("pages/data_analysis/plot_excel.py", title="Plot Excel", icon="📊"),
    st.Page("pages/data_analysis/plot_solartron.py", title="Plot Solartron", icon="📉"),
]

echem_pages = [
    st.Page("pages/electrochemistry/impedance.py", title="Impedance", icon="⚡"),
]

ai_pages = [
    st.Page("pages/ai_tools/image_classifier.py", title="Image Classifier", icon="🖼️"),
    st.Page("pages/ai_tools/image_regression.py", title="Image Regression", icon="📸"),
    st.Page("pages/ai_tools/chatgpt_compare.py", title="GPT-3 vs ChatGPT", icon="🤖"),
    st.Page("pages/ai_tools/neural_network.py", title="Neural Network Game", icon="🧠"),
    st.Page("pages/ai_tools/mir_puzzle.py", title="MIR Puzzle", icon="🧩"),
]

utility_pages = [
    st.Page("pages/utilities/sympy_shell.py", title="Sympy Shell", icon="🔢"),
]

# Navigation with sections
pg = st.navigation({
    "Kinetics": kinetics_pages,
    "Thermodynamics": thermo_pages,
    "Quantum Chemistry": quantum_pages,
    "Data Analysis": data_pages,
    "Electrochemistry": echem_pages,
    "AI Tools": ai_pages,
    "Utilities": utility_pages,
})

# Common header
st.logo("logo.png")  # Optional
st.title("Chemistry and FYS Tools")

# Run selected page
pg.run()
```

### URL Structure

With `st.navigation`, each page gets a unique URL:
- `/` or `/arrhenius` - CHE 120 Kinetics (default)
- `/ocean_optics` - Ocean Optics Kinetics
- `/first_law` - 1st Law of Thermodynamics
- `/calorimeter` - 2nd Law Calorimeter
- `/impedance` - Impedance Analysis
- etc.

---

## Implementation Phases

### Phase 1: Foundation (No Breaking Changes)

1. **Update dependencies** in requirements.txt:
   ```
   streamlit>=1.53.0
   numpy>=1.24.0
   pandas>=2.0.0
   matplotlib>=3.7.0
   plotly>=5.15.0
   scipy>=1.11.0
   sympy>=1.12
   openai>=1.0.0
   openpyxl>=3.1.0
   CoolProp>=6.6.0
   ```

2. **Update deprecated APIs** in existing modules:
   - Replace `@st.cache` with `@st.cache_data` or `@st.cache_resource`
   - Remove internal Server/session hacks (no longer needed)

3. **Test existing functionality** with new dependencies

### Phase 2: Create New Structure

1. **Create `pages/` directory** with category subdirectories
2. **Create new entrypoint** `app.py` with `st.navigation`
3. **Migrate each app module**:
   - Convert `run()` functions to standalone page scripts
   - Update imports to use new shared utilities path
   - Add page-specific titles and metadata

### Phase 3: Migrate Apps (One Category at a Time)

For each app:
1. Create new page file in appropriate `pages/` subdirectory
2. Refactor `run()` function to top-level code
3. Update any deprecated Streamlit APIs
4. Test individually
5. Add to navigation

**Migration Pattern:**
```python
# OLD: apps/arrhen.py
def run():
    st.header("CHE 120 Kinetics")
    # ... implementation

# NEW: pages/kinetics/arrhenius.py
import streamlit as st

st.header("CHE 120 Kinetics")
# ... implementation (same code, no wrapper function)
```

### Phase 4: Update Deployment

1. **Update Procfile**:
   ```
   web: sh setup.sh && streamlit run app.py
   ```

2. **Update nginx.conf** (if needed for new URL routing)

3. **Update setup.sh** for new Streamlit config options

4. **Test production deployment**

### Phase 5: Cleanup

1. Remove old `apps/pchemapp.py` entry point
2. Consolidate utilities into `shared/`
3. Update documentation
4. Update CI/CD if needed

---

## Dependency Upgrade Details

| Package | Current | Target | Notes |
|---------|---------|--------|-------|
| streamlit | 1.5.0 | >=1.53.0 | Major upgrade, new APIs |
| numpy | 1.22.0 | >=1.24.0 | Compatibility updates |
| pandas | 1.3.5 | >=2.0.0 | Major version, some API changes |
| matplotlib | 3.5.1 | >=3.7.0 | Minor updates |
| plotly | 5.5.0 | >=5.15.0 | Minor updates |
| scipy | 1.7.3 | >=1.11.0 | Minor updates |
| sympy | 1.9 | >=1.12 | Minor updates |
| openai | 0.14.0 | >=1.0.0 | **MAJOR**: Complete API rewrite |
| openpyxl | 3.0.9 | >=3.1.0 | Minor updates |
| CoolProp | 6.4.1 | >=6.6.0 | Minor updates |
| streamlit-ace | 0.1.1 | Check compatibility | May need alternative |

### OpenAI API Migration Notes

The openai package had a complete rewrite in v1.0.0. Key changes:
```python
# OLD (0.14.0)
import openai
openai.api_key = "..."
response = openai.Completion.create(...)
response = openai.ChatCompletion.create(...)

# NEW (1.0.0+)
from openai import OpenAI
client = OpenAI(api_key="...")
response = client.completions.create(...)
response = client.chat.completions.create(...)
```

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| OpenAI API breaking changes | High | Careful migration of AI tools |
| Streamlit internal API removal | High | Remove session hacks, use new APIs |
| Pandas 2.0 deprecations | Medium | Test data processing functions |
| User disruption | Low | Keep old URLs working via redirects |
| Deployment issues | Medium | Test in staging before production |

---

## Testing Strategy

1. **Unit tests**: Ensure `pchem/` package functions work
2. **Integration tests**: Test each page loads without errors
3. **Manual testing**: Verify each app's functionality
4. **Deployment testing**: Test on staging environment

---

## Success Criteria

- [ ] All 25 apps accessible via unique URLs
- [ ] Sidebar navigation with categories
- [ ] All deprecated APIs removed
- [ ] All tests passing
- [ ] Production deployment working
- [ ] Page load times acceptable

---

## References

- [Streamlit Multi-page Apps Documentation](https://docs.streamlit.io/develop/concepts/multipage-apps)
- [st.navigation API Reference](https://docs.streamlit.io/develop/api-reference/navigation/st.navigation)
- [st.Page API Reference](https://docs.streamlit.io/develop/api-reference/navigation/st.page)
- [Streamlit 2025 Release Notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2025)
- [OpenAI Python Library Migration Guide](https://github.com/openai/openai-python/discussions/742)
