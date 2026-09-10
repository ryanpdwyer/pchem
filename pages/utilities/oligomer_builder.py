"""Oligomer Builder - draw a monomer, build and optimize oligomers."""
import streamlit as st
import streamlit.components.v1 as components

st.page_link("pages/home.py", label="← Home")

from streamlit_ketcher import st_ketcher
import py3Dmol

from pchemapps import oligomer as olig

st.title("Oligomer Builder")
st.markdown(
    """
Draw a monomer with **two `*` atoms** marking where the chain continues
(in the editor: pick the atom tool, click an atom position, and type `*`).
The first `*` is the **head**, the second is the **tail**; the app links
*tail → head* to build the chain and caps the ends with H.
You can also paste a SMILES string instead of drawing.
"""
)

PRESETS = {
    "3-hexylthiophene (P3HT)": "CCCCCCc1cc(*)sc1*",
    "Thiophene": "*c1ccc(*)s1",
    "EDOT (PEDOT)": "*c1sc(*)c2OCCOc12",
    "Phenylene (p)": "*c1ccc(*)cc1",
    "Vinylene": "*/C=C/*",
    "Fluorene (9,9-dioctyl)": "CCCCCCCCC1(CCCCCCCC)c2cc(*)ccc2-c2ccc(*)cc21",
    "Aniline (PANI, reduced)": "*Nc1ccc(*)cc1",
    "Pyrrole": "*c1ccc(*)[nH]1",
    "Custom (blank)": "",
}

# ---------------------------------------------------------------- monomers
n_mono = st.radio("Number of different monomers", [1, 2, 3], horizontal=True)
letters = "ABC"[:n_mono]

monomers: dict[str, olig.Monomer] = {}
for L in letters:
    with st.expander(f"Monomer {L}", expanded=True):
        preset = st.selectbox(
            "Start from", list(PRESETS), key=f"preset_{L}",
            index=0 if L == "A" else 1,
        )
        typed = st.text_input(
            "SMILES (edit here or draw below)", PRESETS[preset], key=f"smi_{L}_{preset}"
        )
        drawn = st_ketcher(typed, height=420, key=f"ketcher_{L}_{preset}")
        smiles = drawn if drawn else typed
        swap = st.checkbox("Swap head and tail", key=f"swap_{L}",
                           help="Only matters for copolymers or flipped (lowercase) units.")
        try:
            mono = olig.parse_monomer(smiles)
            if swap:
                mono = olig.Monomer(mono.mol, mono.tail, mono.head)
            monomers[L] = mono
            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(olig.monomer_svg(mono))
            with c2:
                st.code(mono.smiles, language=None)
                st.caption("Highlighted atoms are the head/tail attachment points.")
        except ValueError as e:
            st.error(str(e))

# ---------------------------------------------------------------- sequence
st.subheader("Chain")
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    pattern = st.text_input(
        "Repeat pattern",
        "A",
        help="One letter per unit. Uppercase = as drawn (tail→head, regioregular). "
        "Lowercase = flipped unit, e.g. 'Aa' gives head-to-head / tail-to-tail. "
        "'AB' alternates monomers A and B.",
    )
with c2:
    repeats = st.number_input("Repeats", 1, 40, 6)
with c3:
    n_conf = st.number_input("Conformers to try", 1, 20, 1,
                             help="Lowest-energy conformer is shown.")

seq = None
try:
    seq = olig.expand_sequence(pattern, int(repeats))
    st.caption(f"Sequence: `{seq}` ({len(seq)} units)")
except ValueError as e:
    st.error(str(e))

if len(monomers) < n_mono or seq is None:
    st.stop()


@st.cache_data(show_spinner=False)
def _build(mono_smiles: tuple, seq: str):
    mol = olig.build_oligomer(dict(mono_smiles), seq)
    return olig.Chem.MolToSmiles(mol)


@st.cache_data(show_spinner=False)
def _optimize(smiles: str, n_conf: int, planar: bool):
    mol = olig.Chem.MolFromSmiles(smiles)
    mol3d, energies = olig.embed_and_optimize(mol, n_conformers=n_conf, planar=planar)
    return olig.Chem.MolToMolBlock(mol3d), energies


try:
    mono_smiles = tuple((L, m.smiles) for L, m in monomers.items())
    oligo_smiles = _build(mono_smiles, seq)
except (ValueError, RuntimeError) as e:
    st.error(str(e))
    st.stop()

oligo = olig.Chem.MolFromSmiles(oligo_smiles)
info = olig.summary(oligo)
st.image(olig.oligomer_svg(oligo, width=1000, height=260), use_container_width=True)
m1, m2, m3 = st.columns(3)
m1.metric("Formula", info["formula"])
m2.metric("Molar mass (g/mol)", f"{info['MW']:.1f}")
m3.metric("Atoms (with H)", info["atoms (with H)"])
st.code(oligo_smiles, language=None)

# ---------------------------------------------------------------- 3D
st.subheader("3D structure (ETKDG + MMFF94)")
if info["atoms (with H)"] > 800:
    st.warning("That's a big molecule; optimization may take a minute or more.")

planar = st.checkbox(
    "Hold backbone planar (anti, 180° between rings)", True,
    help="Restrains every inter-ring dihedral at 180° during the MMFF94 optimization, "
    "giving an extended conjugated chain (a good DFT starting point). "
    "Uncheck for a fully relaxed MMFF94 structure; note MMFF94 twists "
    "alkyl-substituted rings strongly out of plane.",
)
if st.button("Build and optimize 3D structure", type="primary"):
    with st.spinner("Embedding and optimizing..."):
        try:
            molblock, energies = _optimize(oligo_smiles, int(n_conf), planar)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
    st.session_state["olig_result"] = (oligo_smiles, molblock, energies)

res = st.session_state.get("olig_result")
if res and res[0] == oligo_smiles:
    _, molblock, energies = res
    mol3d = olig.Chem.MolFromMolBlock(molblock, removeHs=False)
    st.caption(
        f"MMFF94 energy of best conformer: {energies[0]:.1f} kcal/mol"
        + (f" (tried {len(energies)}; range {energies[0]:.1f} – {energies[-1]:.1f})"
           if len(energies) > 1 else "")
    )
    style = st.radio("Style", ["stick", "ball and stick", "spacefill"], horizontal=True)
    view = py3Dmol.view(width=900, height=450)
    view.addModel(molblock, "mol")
    if style == "stick":
        view.setStyle({"stick": {}})
    elif style == "ball and stick":
        view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}})
    else:
        view.setStyle({"sphere": {}})
    view.setBackgroundColor("white")
    view.zoomTo()
    components.html(view._make_html(), height=470, width=920)

    name = f"oligomer_{len(seq)}mer"
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download .xyz", olig.to_xyz(mol3d), f"{name}.xyz")
    d2.download_button("Download .mol (SDF)", molblock, f"{name}.mol")
    d3.download_button("Download .pdb", olig.to_pdb(mol3d), f"{name}.pdb")
elif res:
    st.info("Monomer or sequence changed — click the button to rebuild the 3D structure.")
