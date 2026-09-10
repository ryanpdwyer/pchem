# Handoff: Oligomer Builder page (pchem / js.munano.org)

## Goal
A Streamlit page where a student (Jack) draws a monomer in a free in-browser editor,
and the app builds an n-mer oligomer with RDKit, MMFF94-optimizes it, shows it in 3D,
and offers .xyz/.mol/.pdb downloads. Lives at js.munano.org/oligomer alongside the
other pages in this repo (`app.py` uses `st.navigation` + `st.Page`).

## Done (uncommitted, all in ~/programming/pchem)
- `pchemapps/oligomer.py` – core, no Streamlit:
  - `parse_monomer(smiles)` – monomer SMILES must contain exactly two `*` atoms.
    First `*` (by atom index, or by map number if `[*:1]`/`[*:2]`) = head, second = tail.
  - `expand_sequence(pattern, repeats)` – e.g. `"A"`,6 → `"AAAAAA"`. Uppercase = as drawn,
    lowercase = flipped unit (`"Aa"` → head-to-head/tail-to-tail). `"AB"` = copolymer.
  - `build_oligomer({"A": smiles, ...}, seq)` – joins tail(i)→head(i+1), caps ends with H.
  - `inter_ring_bonds`, `set_backbone_anti` – find single bonds joining two rings and set the
    heaviest-neighbour dihedral (S–C–C–S for thiophene) to 180°.
  - `embed_and_optimize(mol, n_conformers, seed, max_iters, planar, planar_angle)` –
    ETKDGv3 (falls back to `useRandomCoords=True`, needed for ≥8-mers), then MMFF94.
    With `planar=True` each inter-ring dihedral is set to 180° and held with
    `MMFFAddTorsionConstraint(±5°, k=100)` during minimization; reported energy is the
    unrestrained MMFF energy at the final geometry. Conformers returned sorted by energy.
  - `summary`, `monomer_svg` (head/tail labelled + highlighted), `oligomer_svg`, `to_xyz/molblock/pdb`.
- `pages/utilities/oligomer_builder.py` – the UI: presets (P3HT, thiophene, EDOT, phenylene,
  vinylene, fluorene, aniline, pyrrole), 1–3 monomers each with a `streamlit-ketcher` editor
  + SMILES text box + "swap head/tail" checkbox, repeat pattern / repeats / conformers,
  2D depiction, formula/MW/atom count, "Hold backbone planar" checkbox, build button,
  py3Dmol viewer via `components.html(view._make_html())`, download buttons.
  Build and optimize are `st.cache_data`-cached on (SMILES, n_conf, planar).
- Registered: `app.py` (`url_path="oligomer"`), `pages/home.py` (Utilities list),
  `requirements.txt` (+ `rdkit>=2024.3`, `streamlit-ketcher>=0.0.1`, `py3Dmol>=2.0`).

## Verified
- P3HT 3/6/8/16-mers build with correct 2,5′ head-to-tail connectivity; `Aa` gives HH/TT.
- Ketcher exports `*` atoms as `*` in SMILES (it displays them as "A"); the round trip into
  `parse_monomer` works. Note Ketcher reorders atoms in its SMILES export, which can swap
  head/tail relative to what was typed – the labelled 2D picture + swap checkbox cover this.
- Full page tested in Chrome on a local venv (python3.14, rdkit 2026.03, streamlit 1.63):
  editor → 2D → 3D viewer → downloads all work. 8-mer with 3 conformers ≈ 6 s, 16-mer ≈ 13 s.
- MMFF94 relaxes hexyl-substituted bithiophene to ~65–115° twists (≈11 kcal/mol below
  planar), so the planar restraint is on by default; unsubstituted thiophene stays planar anyway.

## Open problems (in priority order)
1. **Segfault for non-ring monomers.** `build_oligomer({'A': '*/C=C/*', 'B': '*c1ccc(*)cc1'}, 'ABABAB')`
   builds fine (`C=Cc1ccc(C=Cc2ccc(C=Cc3ccccc3)cc2)cc1`) but `Chem.AddHs` + `AllChem.EmbedMolecule`
   on the result segfaults (RDKit 2026.03.6 / py3.14). Reproduce with
   `python -X faulthandler`; crash is at the `EmbedMolecule` line, inside rdkit. Suspect the
   RWMol built by `InsertMol` + `RemoveAtom` leaves stale bond-stereo atom refs (the `/` `\`
   bonds pointed at the removed dummies). Likely fix: in `build_oligomer`, before removing
   dummies, for every bond with stereo whose stereo atoms include a dummy, either re-point
   the stereo atom at the new neighbour (join) and convert E/Z → CIS/TRANS via
   `bond.SetStereoAtoms` + `bond.SetStereo(STEREOCIS/TRANS)`, or set `STEREONONE` (end caps).
   Cheap workaround to test first: round-trip through SMILES
   (`Chem.MolFromSmiles(Chem.MolToSmiles(mol))`) at the end of `build_oligomer` – the page
   already does this implicitly (it passes SMILES between the cached functions), so the crash
   may only affect direct library use. Confirm which, then also make sure E/Z of vinylene
   survives (currently it is dropped: output has no `/`).
2. Monomers with bracketed attachment atoms (`*[C@H](C)...`) get an explicit H added on end
   capping – tested once, works; no test file exists yet. Add `tests/test_oligomer.py`
   covering: HT vs HH connectivity, copolymer `AB`, map-number ordering, capping, embedding
   of a 12-mer, planar restraint dihedrals ≈ 180.
3. Deploy: server must `pip install -r requirements.txt` (rdkit wheel, ketcher, py3Dmol);
   py3Dmol pulls 3Dmol.js from a CDN at view time, so the page needs outbound internet.
   Ketcher is bundled in the streamlit-ketcher wheel (no CDN).
4. Nice-to-haves: URL query params to share a built oligomer; optional xTB/GFN-FF step if
   `xtb` is on the server; larger 3D viewer (currently 900×450, left-aligned); show
   per-unit dihedrals table after optimization.

## Local test setup used
```
python3 -m venv venv && ./venv/bin/pip install streamlit rdkit streamlit-ketcher py3Dmol
cd ~/programming/pchem && ./venv/bin/streamlit run app.py --server.port 8599
# open http://localhost:8599/oligomer
```
The rest of `requirements.txt` isn't needed to run this one page.
