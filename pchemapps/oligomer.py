"""Build oligomers from monomers with two ``*`` attachment points.

A monomer is any SMILES containing exactly two dummy atoms (``*``).  The first
dummy (lowest atom index, or map number 1 if ``[*:1]``/``[*:2]`` are used) is
the *head*, the second is the *tail*.  Oligomers are built by joining the tail
of unit i to the head of unit i+1.

Sequence syntax: a string of letters, one per unit.  Uppercase letters use the
monomer as drawn; lowercase letters flip the unit (head and tail swapped), so
``"Aa"`` gives a head-to-head / tail-to-tail dimer.

Example (regioregular P3HT trimer)::

    seq = expand_sequence("A", 3)
    mol = build_oligomer({"A": "CCCCCCc1cc(*)sc1*"}, seq)
    mol3d, energies = embed_and_optimize(mol)
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


@dataclass
class Monomer:
    mol: Chem.Mol
    head: int  # index of the head dummy atom
    tail: int  # index of the tail dummy atom

    @property
    def smiles(self) -> str:
        return Chem.MolToSmiles(self.mol)


def parse_monomer(smiles: str) -> Monomer:
    """Parse a monomer SMILES and identify its head and tail dummy atoms."""
    smiles = (smiles or "").strip()
    if not smiles:
        raise ValueError("Empty monomer.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        raise ValueError(
            f"Monomer needs exactly two attachment points (* atoms); found {len(dummies)}."
        )
    for d in dummies:
        if d.GetDegree() != 1:
            raise ValueError("Each * attachment point must be bonded to exactly one atom.")
    # Order by atom-map number if both are set, otherwise by atom index.
    maps = [d.GetAtomMapNum() for d in dummies]
    if all(m > 0 for m in maps) and maps[0] != maps[1]:
        dummies.sort(key=lambda a: a.GetAtomMapNum())
    else:
        dummies.sort(key=lambda a: a.GetIdx())
    for d in dummies:
        d.SetAtomMapNum(0)
    return Monomer(mol, dummies[0].GetIdx(), dummies[1].GetIdx())


def expand_sequence(pattern: str, repeats: int) -> str:
    """Repeat a sequence pattern, ignoring whitespace."""
    pattern = "".join(pattern.split())
    if not pattern:
        raise ValueError("Sequence is empty.")
    if not pattern.isalpha():
        raise ValueError("Sequence may only contain letters (A, B, C, a, b, c ...).")
    if repeats < 1:
        raise ValueError("Repeats must be at least 1.")
    return pattern * repeats


def build_oligomer(monomers: dict[str, str | Monomer], sequence: str) -> Chem.Mol:
    """Join monomers according to *sequence* and return the capped oligomer.

    ``monomers`` maps uppercase letters to SMILES strings (or Monomer objects).
    Free ends are capped with hydrogen.
    """
    parsed: dict[str, Monomer] = {}
    for key, value in monomers.items():
        k = key.upper()
        parsed[k] = value if isinstance(value, Monomer) else parse_monomer(value)

    if not sequence:
        raise ValueError("Sequence is empty.")

    units = []
    for ch in sequence:
        k = ch.upper()
        if k not in parsed:
            raise ValueError(f"Sequence letter '{ch}' has no monomer defined.")
        units.append((parsed[k], ch.islower()))

    rw = Chem.RWMol()
    prev_tail = None  # index of the current open tail dummy in rw
    dummies_to_remove: list[int] = []
    # Replacement atom for each removed dummy.  RDKit represents alkene E/Z
    # stereo using the atom indices on either side of the double bond.  Those
    # indices must be redirected before removing an attachment-point dummy;
    # otherwise the bond can retain STEREOE/STEREOZ with an empty stereo-atom
    # list, which crashes ETKDG in some RDKit releases.
    dummy_replacements: dict[int, int | None] = {}

    for i, (mono, flipped) in enumerate(units):
        offset = rw.GetNumAtoms()
        rw.InsertMol(mono.mol)
        head = offset + (mono.tail if flipped else mono.head)
        tail = offset + (mono.head if flipped else mono.tail)
        for a in range(offset, rw.GetNumAtoms()):
            rw.GetAtomWithIdx(a).SetIntProp("unit", i)
        if prev_tail is not None:
            a_prev = rw.GetAtomWithIdx(prev_tail).GetNeighbors()[0].GetIdx()
            a_head = rw.GetAtomWithIdx(head).GetNeighbors()[0].GetIdx()
            bond_type = rw.GetBondBetweenAtoms(prev_tail, a_prev).GetBondType()
            rw.AddBond(a_prev, a_head, bond_type)
            dummy_replacements[prev_tail] = a_head
            dummy_replacements[head] = a_prev
            dummies_to_remove += [prev_tail, head]
        prev_tail = tail

    # Cap the two free ends with hydrogen (implicit H after dummy removal).
    first_head = units[0][0].tail if units[0][1] else units[0][0].head
    for end in (first_head, prev_tail):
        nb = rw.GetAtomWithIdx(end).GetNeighbors()[0]
        if nb.GetNoImplicit():
            nb.SetNumExplicitHs(nb.GetNumExplicitHs() + 1)
        dummy_replacements[end] = None
        dummies_to_remove.append(end)

    # Preserve E/Z geometry at joined attachment points.  Stereo at a capped
    # end is no longer defined (the former substituent becomes H), so clear it.
    for bond in rw.GetBonds():
        stereo_atoms = list(bond.GetStereoAtoms())
        if not stereo_atoms:
            continue
        redirected = [dummy_replacements.get(idx, idx) for idx in stereo_atoms]
        if any(idx is None for idx in redirected):
            bond.SetStereo(Chem.BondStereo.STEREONONE)
        elif redirected != stereo_atoms:
            bond.SetStereoAtoms(*redirected)

    for idx in sorted(set(dummies_to_remove), reverse=True):
        rw.RemoveAtom(idx)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    # Recreate slash/backslash directions for serialization from the updated
    # double-bond stereo atom references.
    Chem.SetDoubleBondNeighborDirections(mol)
    # Removing an attachment can also make a tetrahedral center achiral (for
    # example, capping ``*[C@H]`` gives it two hydrogens).
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def inter_ring_bonds(mol: Chem.Mol) -> list[tuple[int, int, int, int]]:
    """Dihedral atom quadruples (a-b-c-d) for single bonds joining two rings.

    b-c is the inter-ring bond; a and d are the heaviest ring neighbours on
    each side (S for thiophene, N for pyrrole), so a dihedral of 180 deg is the
    usual *anti* arrangement of a conjugated backbone.
    """
    ri = mol.GetRingInfo()
    quads = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        b, c = bond.GetBeginAtom(), bond.GetEndAtom()
        if not (ri.NumAtomRings(b.GetIdx()) and ri.NumAtomRings(c.GetIdx())):
            continue

        def pick(atom, other):
            nbs = [n for n in atom.GetNeighbors()
                   if n.GetIdx() != other.GetIdx() and n.GetAtomicNum() > 1]
            return max(nbs, key=lambda n: (n.GetAtomicNum(), -n.GetIdx())) if nbs else None

        a, d = pick(b, c), pick(c, b)
        if a is not None and d is not None:
            quads.append((a.GetIdx(), b.GetIdx(), c.GetIdx(), d.GetIdx()))
    return quads


def set_backbone_anti(mol: Chem.Mol, conf_id: int = 0, angle: float = 180.0) -> int:
    """Set every inter-ring dihedral to *angle* degrees (in place). Returns count."""
    from rdkit.Chem import rdMolTransforms
    conf = mol.GetConformer(conf_id)
    quads = inter_ring_bonds(mol)
    for a, b, c, d in quads:
        try:
            rdMolTransforms.SetDihedralDeg(conf, a, b, c, d, angle)
        except (ValueError, RuntimeError):
            pass
    return len(quads)


def embed_and_optimize(
    mol: Chem.Mol, n_conformers: int = 1, seed: int = 42, max_iters: int = 2000,
    planar: bool = False, planar_angle: float = 180.0,
) -> tuple[Chem.Mol, list[float]]:
    """Generate 3D conformers (ETKDG) and MMFF94-optimize them.

    If *planar*, every inter-ring dihedral is set to *planar_angle* (180 = anti,
    extended backbone) and held there with a torsion restraint during the
    optimization.  MMFF94 otherwise twists alkyl-substituted biaryls well
    away from planar, so this gives a conjugated starting structure.

    Returns a molecule (with explicit H) whose conformers are sorted by energy,
    and the list of MMFF energies in kcal/mol (restraint energy excluded).
    """
    molh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = list(AllChem.EmbedMultipleConfs(molh, numConfs=n_conformers, params=params))
    if len(cids) < n_conformers:
        # Default ETKDG often fails for long floppy chains; random coords is robust.
        params.useRandomCoords = True
        molh.RemoveAllConformers()
        cids = list(AllChem.EmbedMultipleConfs(molh, numConfs=n_conformers, params=params))
    if not cids:
        raise RuntimeError("3D embedding failed for this molecule.")

    props = AllChem.MMFFGetMoleculeProperties(molh)
    if props is None:
        raise RuntimeError("MMFF94 has no parameters for this molecule.")
    quads = inter_ring_bonds(molh) if planar else []
    energies = []
    for cid in cids:
        if planar:
            set_backbone_anti(molh, cid, planar_angle)
        ff = AllChem.MMFFGetMoleculeForceField(molh, props, confId=cid)
        for a, b, c, d in quads:
            ff.MMFFAddTorsionConstraint(a, b, c, d, False,
                                        planar_angle - 5, planar_angle + 5, 100.0)
        ff.Minimize(maxIts=max_iters)
        ff_free = AllChem.MMFFGetMoleculeForceField(molh, props, confId=cid)
        energies.append(ff_free.CalcEnergy())

    order = sorted(range(len(cids)), key=lambda i: energies[i])
    out = Chem.Mol(molh)
    out.RemoveAllConformers()
    for i in order:
        conf = Chem.Conformer(molh.GetConformer(cids[i]))
        out.AddConformer(conf, assignId=True)
    return out, [energies[i] for i in order]


def summary(mol: Chem.Mol) -> dict:
    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "MW": Descriptors.MolWt(mol),
        "heavy atoms": mol.GetNumHeavyAtoms(),
        "atoms (with H)": Chem.AddHs(mol).GetNumAtoms(),
        "SMILES": Chem.MolToSmiles(mol),
    }


def monomer_svg(mono: Monomer, width: int = 350, height: int = 220) -> str:
    """2D depiction of a monomer with head/tail attachment points labelled."""
    m = Chem.Mol(mono.mol)
    m.GetAtomWithIdx(mono.head).SetProp("_displayLabel", "head")
    m.GetAtomWithIdx(mono.tail).SetProp("_displayLabel", "tail")
    return _svg(m, width, height, highlight=[mono.head, mono.tail])


def oligomer_svg(mol: Chem.Mol, width: int = 900, height: int = 300) -> str:
    return _svg(mol, width, height)


def _svg(mol: Chem.Mol, width: int, height: int, highlight=None) -> str:
    m = Chem.Mol(mol)
    AllChem.Compute2DCoords(m)
    d = rdMolDraw2D.MolDraw2DSVG(width, height)
    d.drawOptions().clearBackground = False
    d.DrawMolecule(m, highlightAtoms=highlight or [])
    d.FinishDrawing()
    svg = d.GetDrawingText()
    return svg[svg.index("<svg"):]


def to_xyz(mol: Chem.Mol, conf_id: int = 0) -> str:
    return Chem.MolToXYZBlock(mol, confId=conf_id)


def to_molblock(mol: Chem.Mol, conf_id: int = 0) -> str:
    return Chem.MolToMolBlock(mol, confId=conf_id)


def to_pdb(mol: Chem.Mol, conf_id: int = 0) -> str:
    return Chem.MolToPDBBlock(mol, confId=conf_id)
