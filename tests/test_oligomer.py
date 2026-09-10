import math

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdMolTransforms

from pchemapps.oligomer import (
    build_oligomer,
    embed_and_optimize,
    inter_ring_bonds,
    parse_monomer,
    to_gaussian_input,
)


def _inter_unit_elements(mol):
    pairs = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if (
            begin.HasProp("unit")
            and end.HasProp("unit")
            and begin.GetIntProp("unit") != end.GetIntProp("unit")
        ):
            pairs.append(tuple(sorted((begin.GetSymbol(), end.GetSymbol()))))
    return pairs


def test_head_to_tail_and_flipped_head_to_head_connectivity():
    monomers = {"A": "[*:1]NCC[*:2]"}

    head_to_tail = build_oligomer(monomers, "AA")
    head_to_head = build_oligomer(monomers, "aA")

    assert _inter_unit_elements(head_to_tail) == [("C", "N")]
    assert _inter_unit_elements(head_to_head) == [("N", "N")]
    assert Chem.MolToSmiles(head_to_tail) == "CCNCCN"
    assert Chem.MolToSmiles(head_to_head) == "CCNNCC"


def test_ab_copolymer_uses_the_defined_monomers():
    mol = build_oligomer(
        {"A": "[*:1]CC[*:2]", "B": "[*:1]OC[*:2]"},
        "AB",
    )

    assert Chem.MolToSmiles(mol) == "CCOC"
    assert rdMolDescriptors.CalcMolFormula(mol) == "C3H8O"
    assert _inter_unit_elements(mol) == [("C", "O")]


def test_atom_map_numbers_determine_head_and_tail_order():
    monomer = parse_monomer("[*:2]CCN[*:1]")

    head_neighbor = monomer.mol.GetAtomWithIdx(monomer.head).GetNeighbors()[0]
    tail_neighbor = monomer.mol.GetAtomWithIdx(monomer.tail).GetNeighbors()[0]
    assert head_neighbor.GetSymbol() == "N"
    assert tail_neighbor.GetSymbol() == "C"


def test_bracketed_attachment_atom_is_hydrogen_capped():
    mol = build_oligomer({"A": "*[C@H](C)C*"}, "A")

    assert Chem.MolToSmiles(mol, isomericSmiles=True) == "CCC"
    assert rdMolDescriptors.CalcMolFormula(mol) == "C3H8"
    assert Chem.AddHs(mol).GetNumAtoms() == 11
    assert all(
        atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
        for atom in mol.GetAtoms()
    )


@pytest.mark.parametrize(
    ("vinylene", "expected_stereo"),
    [
        (r"*/C=C/*", Chem.BondStereo.STEREOE),
        (r"*/C=C\*", Chem.BondStereo.STEREOZ),
    ],
)
def test_vinylene_stereo_survives_joining_and_embeds(vinylene, expected_stereo):
    mol = build_oligomer(
        {"A": vinylene, "B": "*c1ccc(*)cc1"},
        "ABABAB",
    )
    stereos = [
        bond.GetStereo()
        for bond in mol.GetBonds()
        if bond.GetStereo() != Chem.BondStereo.STEREONONE
    ]

    # The terminal vinylene is capped to CH2 and is no longer stereogenic;
    # the two internal vinylene units retain their specified E/Z geometry.
    assert stereos == [expected_stereo, expected_stereo]
    mol3d, energies = embed_and_optimize(mol, max_iters=25)
    assert mol3d.GetNumConformers() == 1
    assert len(energies) == 1
    assert math.isfinite(energies[0])


def test_twelve_mer_embeds():
    mol = build_oligomer({"A": "*CC*"}, "A" * 12)

    assert rdMolDescriptors.CalcMolFormula(mol) == "C24H50"
    mol3d, energies = embed_and_optimize(mol, max_iters=25)
    assert mol3d.GetNumConformers() == 1
    assert len(energies) == 1
    assert math.isfinite(energies[0])


def test_planar_restraint_holds_inter_ring_dihedrals_near_anti():
    mol = build_oligomer({"A": "*c1ccc(*)s1"}, "AAAA")
    mol3d, _ = embed_and_optimize(mol, planar=True, max_iters=200)
    quads = inter_ring_bonds(mol3d)

    assert len(quads) == 3
    conf = mol3d.GetConformer()
    angles = [rdMolTransforms.GetDihedralDeg(conf, *quad) for quad in quads]
    assert all(abs(abs(angle) - 180.0) <= 5.0 for angle in angles)


def test_gaussian_input_uses_optimized_geometry():
    mol = build_oligomer({"A": "*CC*"}, "A")
    mol3d, _ = embed_and_optimize(mol, max_iters=25)

    text = to_gaussian_input(
        mol3d,
        route="HF/STO-3G SP",
        title="REST submission test",
        charge=-1,
        multiplicity=2,
    )

    assert text.startswith("# HF/STO-3G SP\n\nREST submission test\n\n-1 2\n")
    coordinate_lines = text.splitlines()[5:]
    assert len([line for line in coordinate_lines if line.strip()]) == mol3d.GetNumAtoms()
