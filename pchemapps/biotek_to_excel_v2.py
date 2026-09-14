#!/usr/bin/env python3
"""Combine several BioTek Gen5 ABTS runs into ONE student-friendly .xlsx.

v1 (``biotek_to_excel.py``) writes one workbook per .txt file with the kinetic
data laid out long-on-wavelength (734 and 950 as separate rows). This v2 builds
on v1's parser and instead produces a single, pivot-ready workbook for a whole
experiment:

  - Data_long : tidy master table, one row per (Experiment, Loc, Time_index),
                with A_734 / A_950 / dA as COLUMNS plus is_initial / is_final
                flags -- trivial to groupby / pivot.
  - Per_well  : one row per (Experiment, Loc) with dA_initial / dA_final /
                delta_dA (the c_50 input -- no curve fit baked in).
  - QC_initial: initial A_734 per well + per-sample mean/std, to confirm the
                starting ABTS is roughly constant across wells (the control).
  - dA_vs_time: chart-ready wide sheet (rows = time, columns = labeled wells)
                for a direct dA-vs-t overlay across concentrations.
  - Summary / PlateInfo / Layout for orientation.

Plate convention (from PlateInfo): each sample sits in one plate column; rows
A-F are a 2x serial dilution (top = highest conc), G = Control (0 mM), H =
Blank (EtOH). dA = A_734 - A_950 (950 nm is the scatter/reference correction).

Usage:
    biotek_to_excel_v2.py DIR [--plate-info PI.xlsx] [--out OUT.xlsx] [--all]

By default only files whose populated plate column matches a labeled column in
PlateInfo are combined; --all includes every .txt in DIR. Run under the env that
has pandas/openpyxl, e.g. ``micromamba run -n py314 python biotek_to_excel_v2.py``.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from pchemapps import biotek_to_excel as v1  # reuse the parser + helpers

ROWS = v1.ROWS
COLS = v1.COLS
WELLS = v1.WELLS

TIDY_COLS = [
    "Experiment", "Loc", "Letter", "Number", "Type", "Sample",
    "Conc_mM", "Conc_derived", "Notes",
    "Time_index", "Time_s", "Time_min",
    "A_734", "A_950", "dA", "is_initial", "is_final",
]


def tidy_cols(dilution: float) -> list[str]:
    """TIDY_COLS, with Conc_final_mM inserted when the assay dilutes the sample.

    PlateInfo's Conc_mM is what was *pipetted*; adding a small volume of it to
    the radical solution dilutes it in the well. Only the in-well concentration
    belongs on a c_50 x-axis, so surface it as its own column rather than
    silently rescaling the number the student typed.
    """
    if dilution == 1.0:
        return list(TIDY_COLS)
    i = TIDY_COLS.index("Conc_mM") + 1
    return TIDY_COLS[:i] + ["Conc_final_mM"] + TIDY_COLS[i:]


# ----------------------------- PlateInfo helpers -----------------------------


def derive_serial_dilution(plate_info: pd.DataFrame, factor: float = 2.0) -> pd.DataFrame:
    """Fill in Conc_mM for Sample wells down each plate column as a serial dilution.

    Within each plate column (Number), walk the Sample-type wells top-to-bottom
    (row A -> H). The topmost well that already has a Conc_mM is the anchor; each
    subsequent Sample well that is blank is filled as previous / factor. Already
    entered values are respected. Control / Blank wells are left untouched.

    Adds a boolean ``Conc_derived`` column marking the values we filled in.
    """
    pi = plate_info.copy()
    if "Conc_derived" not in pi.columns:
        pi["Conc_derived"] = False
    pi["Conc_mM"] = pd.to_numeric(pi["Conc_mM"], errors="coerce")

    letter_rank = {r: i for i, r in enumerate(ROWS)}
    pi["_rank"] = pi["Letter"].map(letter_rank)

    for col in sorted(pi["Number"].dropna().unique()):
        mask = (pi["Number"] == col) & (pi["Type"].astype(str).str.strip() == "Sample")
        sub = pi[mask].sort_values("_rank")
        prev = None
        for idx in sub.index:
            cur = pi.at[idx, "Conc_mM"]
            if pd.notna(cur):
                prev = float(cur)
                continue
            if prev is not None:
                prev = prev / factor
                pi.at[idx, "Conc_mM"] = prev
                pi.at[idx, "Conc_derived"] = True
    return pi.drop(columns="_rank")


def labeled_columns(plate_info: pd.DataFrame) -> set[int]:
    """Plate columns (Number) that have at least one labeled Sample well."""
    has_sample = plate_info["Sample"].apply(
        lambda s: pd.notna(s) and str(s).strip() != ""
    )
    return {
        int(n) for n in plate_info.loc[has_sample, "Number"].dropna().unique()
    }


def detect_data_column(pf: v1.ParsedFile) -> list[int]:
    """The plate column(s) this file's *sample* sits in.

    The owned column is identified by the **initial reads** (taken before the
    antioxidant, only on the sample column). The kinetic block is often a
    full-plate read -- every well gets a baseline reading -- so it cannot tell
    us which column held the sample. We therefore use initials when present and
    only fall back to the kinetic columns when there are no initial reads.
    """
    init_cols = {
        int(w[1:]) for grid in pf.init_reads.values()
        for w, v in grid.items() if v is not None and np.isfinite(v)
    }
    if init_cols:
        return sorted(init_cols)
    kin_cols: set[int] = set()
    for _times, A in pf.kinetics.values():
        if A.shape[0] == 0:
            continue
        for wi, ok in enumerate(np.isfinite(A).any(axis=0)):
            if ok:
                kin_cols.add(int(WELLS[wi][1:]))
    return sorted(kin_cols)


def has_initial_reads(pf: v1.ParsedFile) -> bool:
    return any(
        v is not None and np.isfinite(v)
        for grid in pf.init_reads.values() for v in grid.values()
    )


# ------------------------------- Tidy builder --------------------------------


def build_tidy(pf: v1.ParsedFile, plate_info: pd.DataFrame,
               experiment: str, keep_cols: list[int] | None = None,
               dilution: float = 1.0) -> pd.DataFrame:
    """One row per (Experiment, Loc, Time_index); A_734/A_950/dA as columns.

    Reuses v1.build_data_long (long-on-wavelength) and pivots wavelength to
    columns. dA = A_734 - A_950. The 734 reading's timestamp is adopted for the
    row (Gen5 reads 950 ~38 s later at the same Time_index).

    ``keep_cols`` restricts output to those plate columns (the sample's owned
    column). This matters because the kinetic block is usually a full-plate
    read -- without filtering, empty wells in other columns would be emitted and
    mislabeled by the master PlateInfo.
    """
    cols = tidy_cols(dilution)
    dl = v1.build_data_long(pf, plate_info)
    if dl.empty:
        return pd.DataFrame(columns=cols)
    if keep_cols is not None:
        dl = dl[dl["Number"].isin(keep_cols)]
        if dl.empty:
            return pd.DataFrame(columns=cols)

    dl = dl.copy()
    dl["Wavelength"] = pd.to_numeric(dl["Wavelength"], errors="coerce")

    # Absorbance by wavelength -> columns, keyed on (Time_index, Loc).
    wide = dl.pivot_table(
        index=["Time_index", "Loc"], columns="Wavelength", values="A",
        aggfunc="first",
    ).rename(columns={734.0: "A_734", 950.0: "A_950"})
    for col in ("A_734", "A_950"):
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide.reset_index()[["Time_index", "Loc", "A_734", "A_950"]]
    wide["dA"] = wide["A_734"] - wide["A_950"]

    # Representative time per Time_index: prefer the 734 reading.
    t_lookup = (
        dl.sort_values("Wavelength")
          .drop_duplicates("Time_index")[["Time_index", "Time_s", "Time_hms"]]
    )
    wide = wide.merge(t_lookup, on="Time_index", how="left")
    wide["Time_min"] = wide["Time_s"] / 60.0

    # Per-well metadata straight from the (dilution-filled) PlateInfo.
    meta_cols = ["Letter", "Number", "Type", "Sample", "Conc_mM", "Notes"]
    if "Conc_derived" in plate_info.columns:
        meta_cols.append("Conc_derived")
    meta = plate_info.set_index("Loc")[meta_cols]
    wide = wide.merge(meta, left_on="Loc", right_index=True, how="left")
    if "Conc_derived" not in wide.columns:
        wide["Conc_derived"] = False

    wide["Experiment"] = experiment
    if dilution != 1.0:
        wide["Conc_final_mM"] = (
            pd.to_numeric(wide["Conc_mM"], errors="coerce") / dilution
        )
    wide["is_initial"] = wide["Time_index"] == v1.INITIAL_TIME_INDEX

    # is_final = the last kinetic point (Time_index >= 0) for each well.
    kinetic = wide[wide["Time_index"] >= 0]
    last_idx = kinetic.groupby("Loc")["Time_index"].transform("max")
    wide["is_final"] = False
    wide.loc[kinetic.index, "is_final"] = (
        wide.loc[kinetic.index, "Time_index"] == last_idx
    )

    out = wide.reindex(columns=cols)
    return out.sort_values(["Number", "Letter", "Time_index"]).reset_index(drop=True)


# ------------------------------ Derived tables -------------------------------


def _first(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) else np.nan


def build_per_well(tidy: pd.DataFrame) -> pd.DataFrame:
    """One row per (Experiment, Loc): initial/final dA and their difference.

    Plain arithmetic only -- this is the input a student uses to estimate c_50
    (plot delta_dA or reduction vs Conc_mM); no curve fit is performed here.
    """
    rows = []
    for (exp, loc), g in tidy.groupby(["Experiment", "Loc"], sort=False):
        init = g[g["is_initial"]]
        final = g[g["is_final"]]
        dA_i = _first(init["dA"])
        dA_f = _first(final["dA"])
        delta = dA_i - dA_f if pd.notna(dA_i) and pd.notna(dA_f) else np.nan
        frac = (dA_f / dA_i) if pd.notna(dA_i) and dA_i != 0 and pd.notna(dA_f) else np.nan
        row = {
            "Experiment": exp, "Loc": loc,
            "Letter": _first(g["Letter"]), "Number": _first(g["Number"]),
            "Type": _first(g["Type"]), "Sample": _first(g["Sample"]),
            "Conc_mM": _first(g["Conc_mM"]),
        }
        if "Conc_final_mM" in tidy.columns:
            row["Conc_final_mM"] = _first(g["Conc_final_mM"])
        rows.append({
            **row,
            "A734_initial": _first(init["A_734"]),
            "dA_initial": dA_i, "dA_final": dA_f,
            "delta_dA": delta, "frac_remaining": frac,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Experiment", "Number", "Letter"]).reset_index(drop=True)


def build_qc_initial(tidy: pd.DataFrame) -> pd.DataFrame:
    """Initial-read QC: A_734 per well + per-sample mean/std and a deviation flag.

    Lets the student confirm the starting ABTS signal is constant across the
    Sample wells of each Experiment (the assay control).
    """
    init = tidy[tidy["is_initial"]].copy()
    if init.empty:
        return pd.DataFrame()
    init = init[[
        "Experiment", "Loc", "Letter", "Number", "Type", "Sample",
        "Conc_mM", "A_734", "A_950", "dA",
    ]].rename(columns={
        "A_734": "A_734_initial", "A_950": "A_950_initial", "dA": "dA_initial",
    })

    # Mean/std over the Sample wells of each Experiment.
    samp = init[init["Type"].astype(str).str.strip() == "Sample"]
    stats = samp.groupby("Experiment")["A_734_initial"].agg(
        sample_mean_A734="mean", sample_std_A734="std",
    ).reset_index()
    init = init.merge(stats, on="Experiment", how="left")
    init["pct_dev"] = (
        (init["A_734_initial"] - init["sample_mean_A734"])
        / init["sample_mean_A734"] * 100.0
    )
    init["flag_gt5pct"] = (
        (init["Type"].astype(str).str.strip() == "Sample")
        & (init["pct_dev"].abs() > 5.0)
    )
    return init.sort_values(["Experiment", "Number", "Letter"]).reset_index(drop=True)


def _well_label(row) -> str:
    bits = [str(row["Experiment"]), str(row["Loc"])]
    conc = row["Conc_mM"]
    if pd.notna(conc):
        bits.append(f"{conc:g} mM")
    return " | ".join(bits)


def build_dA_vs_time(tidy: pd.DataFrame) -> pd.DataFrame:
    """Wide sheet: rows = timepoints, columns = labeled wells, values = dA.

    Charting these columns directly gives the dA-vs-t overlay across
    concentrations without building a pivot.
    """
    if tidy.empty:
        return pd.DataFrame()
    t = tidy.copy()
    t["_label"] = t.apply(_well_label, axis=1)
    piv = t.pivot_table(index="Time_index", columns="_label", values="dA",
                        aggfunc="first")
    # Representative minutes per Time_index (medians; timepoints are nominal).
    tmin = t.groupby("Time_index")["Time_min"].median()
    piv.insert(0, "Time_min", tmin)
    piv = piv.reset_index().sort_values("Time_index")

    # Order well columns by Experiment, plate column, then row (dilution order).
    order_key = (
        t.drop_duplicates("_label")
         .assign(_n=lambda d: pd.to_numeric(d["Number"], errors="coerce"))
         .sort_values(["Experiment", "_n", "Letter"])["_label"].tolist()
    )
    fixed = ["Time_index", "Time_min"]
    cols = fixed + [c for c in order_key if c in piv.columns]
    return piv[cols]


# --------------------------------- Workbook ----------------------------------


def write_combined_workbook(tidy: pd.DataFrame, plate_info: pd.DataFrame,
                            meta: list[tuple[str, str]], out_path: str) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        pd.DataFrame(meta, columns=["Field", "Value"]).to_excel(
            xw, sheet_name="Summary", index=False)
        plate_info.to_excel(xw, sheet_name="PlateInfo", index=False)
        _write_layout(xw, plate_info)
        if not tidy.empty:
            tidy.to_excel(xw, sheet_name="Data_long", index=False)
            pw = build_per_well(tidy)
            if not pw.empty:
                pw.to_excel(xw, sheet_name="Per_well", index=False)
            qc = build_qc_initial(tidy)
            if not qc.empty:
                qc.to_excel(xw, sheet_name="QC_initial", index=False)
            dav = build_dA_vs_time(tidy)
            if not dav.empty:
                dav.to_excel(xw, sheet_name="dA_vs_time", index=False)
    v1._autosize(out_path)


def _write_layout(xw, plate_info: pd.DataFrame) -> None:
    pi = plate_info.set_index("Loc")
    layout_rows = []
    for r in ROWS:
        row = {"": r}
        for c in COLS:
            loc = f"{r}{c}"
            cell = ""
            if loc in pi.index:
                s = pi.loc[loc, "Sample"]
                conc = pi.loc[loc, "Conc_mM"]
                if pd.notna(s) and str(s).strip():
                    cell = str(s).strip()
                    if pd.notna(conc):
                        cell += f" ({conc:g} mM)"
            row[str(c)] = cell
        layout_rows.append(row)
    pd.DataFrame(layout_rows).to_excel(xw, sheet_name="Layout", index=False)


# --------------------------------- Combine -----------------------------------


def combine(txt_files: list[str], plate_info: pd.DataFrame,
            include_all: bool = False,
            dilution: float = 1.0) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Parse each .txt, build its tidy block, concatenate. Returns (tidy, meta)."""
    labeled = labeled_columns(plate_info)
    used, skipped = [], []
    blocks: list[pd.DataFrame] = []
    wavelengths: set[str] = set()
    n_timepoints = 0

    def _note_wls_and_tpts(pf: v1.ParsedFile) -> None:
        nonlocal n_timepoints
        wavelengths.update(pf.init_reads.keys())
        for _t, A in pf.kinetics.values():
            n_timepoints = max(n_timepoints, A.shape[0])

    for path in txt_files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            pf = v1.parse_file(path)
        except Exception as e:  # noqa: BLE001
            skipped.append((name, f"parse error: {e}"))
            continue
        data_cols = detect_data_column(pf)
        if not data_cols:
            skipped.append((name, "no readings found"))
            continue
        # No initials to pin the sample column, and the kinetic spans several
        # columns (full-plate read) -> we can't tell which column is the sample.
        if not has_initial_reads(pf) and len(data_cols) > 1:
            skipped.append((
                name,
                f"no initial reads; full-plate kinetic spans columns {data_cols} "
                "-- cannot identify the sample column",
            ))
            continue
        if not include_all and labeled and not (set(data_cols) & labeled):
            skipped.append((name, f"column(s) {data_cols} not labeled in PlateInfo"))
            continue
        block = build_tidy(pf, plate_info, name, keep_cols=data_cols,
                           dilution=dilution)
        if block.empty:
            skipped.append((name, "no tidy rows"))
            continue
        blocks.append(block)
        _note_wls_and_tpts(pf)
        used.append((name, f"plate column {data_cols}"))

    tidy = (pd.concat(blocks, ignore_index=True)
            if blocks else pd.DataFrame(columns=tidy_cols(dilution)))

    meta: list[tuple[str, str]] = [
        ("Workbook", "Combined BioTek ABTS runs (v2)"),
        ("Experiments combined", str(len(used))),
    ]
    for name, why in used:
        meta.append((f"  + {name}", why))
    for name, why in skipped:
        meta.append((f"  - {name} (skipped)", why))
    meta.append(("Wavelengths (nm)", ", ".join(sorted(wavelengths)) or "-"))
    meta.append(("Kinetic timepoints", str(n_timepoints)))
    meta.append(("dA", "A_734 - A_950 (950 nm reference correction)"))
    n_derived = int(plate_info.get("Conc_derived", pd.Series(dtype=bool)).sum())
    if n_derived:
        meta.append((
            "Conc_mM note",
            f"{n_derived} well(s) filled by 2x serial dilution down the column "
            "from the topmost entered value; the rest are as entered in PlateInfo",
        ))
    else:
        meta.append((
            "Conc_mM note", "as entered in PlateInfo (nothing derived)",
        ))
    if dilution != 1.0:
        meta.append((
            "Conc_final_mM",
            f"Conc_mM / {dilution:g} -- the in-well concentration after the "
            "antioxidant is added to the radical solution. Plot c_50 against "
            "THIS column, not Conc_mM.",
        ))
    return tidy, meta


# ----------------------------------- CLI -------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", help="directory of BioTek .txt exports")
    ap.add_argument("--plate-info", default=None, help="path to PlateInfo .xlsx")
    ap.add_argument("--out", "-o", default=None, help="output .xlsx (default: DIR/Combined.xlsx)")
    ap.add_argument("--all", action="store_true",
                    help="include every .txt, not just files matching a labeled column")
    ap.add_argument("--dilution-factor", type=float, default=1.0, metavar="F",
                    help="how much the assay dilutes the pipetted sample in the "
                         "well (e.g. 20 for 10 uL antioxidant into 190 uL "
                         "radical). Adds a Conc_final_mM = Conc_mM / F column")
    args = ap.parse_args(argv)

    if args.dilution_factor <= 0:
        print("error: --dilution-factor must be positive", file=sys.stderr)
        return 2

    if not os.path.isdir(args.dir):
        print(f"error: not a directory: {args.dir}", file=sys.stderr)
        return 2
    txt_files = sorted(
        os.path.join(args.dir, f) for f in os.listdir(args.dir)
        if f.lower().endswith(".txt")
    )
    if not txt_files:
        print(f"no .txt files in {args.dir}", file=sys.stderr)
        return 1

    pi_path = args.plate_info or v1.auto_find_plate_info(txt_files[0])
    if pi_path is None:
        print("error: no PlateInfo found; pass --plate-info", file=sys.stderr)
        return 2
    plate_info = v1.load_plate_info(pi_path)
    plate_info = derive_serial_dilution(plate_info)
    print(f"PlateInfo: {pi_path}", file=sys.stderr)

    tidy, meta = combine(txt_files, plate_info, include_all=args.all,
                         dilution=args.dilution_factor)
    if tidy.empty:
        print("error: no data combined (nothing matched)", file=sys.stderr)
        return 1

    out = args.out or os.path.join(args.dir, "Combined.xlsx")
    write_combined_workbook(tidy, plate_info, meta, out)
    n_exp = tidy["Experiment"].nunique()
    print(f"wrote {out} ({n_exp} experiments, {len(tidy)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
