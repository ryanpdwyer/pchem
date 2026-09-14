#!/usr/bin/env python3
"""Convert a BioTek Gen5 .txt export into a student-friendly .xlsx workbook.

Usage:
    biotek_to_excel.py FILE.txt [--out OUT.xlsx] [--plate-info PI.xlsx]
    biotek_to_excel.py --batch DIR [--plate-info PI.xlsx]

The script handles two kinds of Gen5 exports:

  - Full kinetic run (Procedure + Layout + Initial Reads at each wavelength
    + Kinetic Reads at each wavelength). Output workbook has Data_long,
    Abs_<wl>_wide per wavelength, and a dA_wide sheet when both 734 and 950
    are present (dA = A_734 - A_950).

  - Single-wavelength spectrum control (just one absorbance reading per well
    at a single wavelength). Output workbook has a Spectrum sheet.

If two BioTek runs need to be combined into one logical experiment (e.g.
initial reads in one file and the kinetic block in another), splice them
into one canonical .txt before running the script -- the script itself only
handles one file at a time.

PlateInfo.xlsx is a long table (one row per well, 96 rows) with columns:
    Loc, Type, Sample, Conc_mM, Notes, Letter, Number
matching the convention used previously with Demmi. The script joins each
well's metadata into the tidy data and uses Sample/Conc_mM to label charts.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter

ROWS = list("ABCDEFGH")
COLS = list(range(1, 13))
WELLS = [f"{r}{c}" for r in ROWS for c in COLS]

# Initial reads happen before the antioxidant is added. We tag them with
# Time_index = -1 in the long table. A nominal Time_s of -30 s matches the
# convention Demmi used in 2302 (roughly 30 s between addition and the first
# kinetic point); the absolute value isn't load-bearing.
INITIAL_TIME_INDEX = -1
INITIAL_TIME_S = -30.0

# The initial-read section header is the Gen5 protocol step name, typed by
# whoever built the protocol -- so its spelling varies. Every run through
# 2302/02-17 (and the 2026 runs, which inherited the protocol) says "intial";
# the 04-13 protocol fixed the typo. Both name the same step.
INITIAL_READ_PREFIXES = ("intial read", "initial read")


# ----------------------------- Parsing ---------------------------------------


def read_text(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("cp1252")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def split_sections(text: str) -> list[tuple[str, list[str]]]:
    """Split file into (header, body_lines) sections separated by blank lines."""
    sections: list[tuple[str, list[str]]] = []
    current: list[str] = []
    for raw in text.splitlines():
        ln = raw.rstrip("\t \r")
        if ln == "":
            if current:
                sections.append((current[0], current[1:]))
                current = []
            continue
        current.append(ln)
    if current:
        sections.append((current[0], current[1:]))
    return sections


def _is_na(s: str) -> bool:
    return s.strip().upper() in {"#N/A", "N/A", "NA"}


def parse_procedure(body: list[str]) -> list[tuple[str, str]]:
    out = []
    for ln in body:
        parts = ln.split("\t", 1)
        out.append((parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""))
    return out


def parse_8x12_grid(body: list[str]) -> dict[str, str]:
    """Body lines: column header '\\t1\\t2...\\t12' then rows A..H.
    Returns dict {well: raw_string}. '#N/A' and blanks become ''.
    """
    out: dict[str, str] = {w: "" for w in WELLS}
    if not body:
        return out
    for ln in body[1:]:  # skip column header
        cells = ln.split("\t")
        if not cells:
            continue
        row_label = cells[0].strip()
        if row_label not in ROWS:
            continue
        for c in COLS:
            if c >= len(cells):
                break
            v = cells[c].strip()
            if _is_na(v):
                v = ""
            out[f"{row_label}{c}"] = v
    return out


def _is_grid_body(body: list[str]) -> bool:
    """True if ``body`` is an 8x12 plate grid: a '\t1\t2...\t12' column header
    followed by rows labelled A..H (as opposed to a kinetic time table)."""
    if not body:
        return False
    hdr = [c.strip() for c in body[0].split("\t")]
    if hdr[:1] != [""] or hdr[1:3] != ["1", "2"]:
        return False
    return any(ln.split("\t")[0].strip() in ROWS for ln in body[1:])


def parse_grid_float(body: list[str]) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {}
    for w, s in parse_8x12_grid(body).items():
        if s == "":
            out[w] = None
        else:
            try:
                out[w] = float(s)
            except ValueError:
                out[w] = None
    return out


def parse_kinetic(body: list[str]) -> tuple[list[str], list[str], np.ndarray]:
    """First body line: 'Kinetic read\\tA1\\tA2...\\tH12'.
    Following lines: HH:MM:SS time + 96 absorbance values.
    Returns (times_hms, wells_order, A[ntimes, 96]).
    """
    if not body:
        return [], [], np.zeros((0, 96))
    header = body[0].split("\t")
    wells = header[1:97] if header[0].lower().startswith("kinetic") else WELLS
    times: list[str] = []
    rows: list[list[float]] = []
    for ln in body[1:]:
        cells = ln.split("\t")
        if not cells or not re.match(r"^\d+:\d+:\d+$", cells[0].strip()):
            continue
        times.append(cells[0].strip())
        vals: list[float] = []
        for c in cells[1:97]:
            c = c.strip()
            if c == "" or _is_na(c):
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(c))
                except ValueError:
                    vals.append(np.nan)
        while len(vals) < 96:
            vals.append(np.nan)
        rows.append(vals[:96])
    A = np.array(rows, dtype=float) if rows else np.zeros((0, 96))
    return times, wells, A


def parse_spectrum(body: list[str]) -> dict:
    """Spectrum-only file body:
        Wavelength 1 (300 nm)
        \\t<col>
        <row>\\t<value>\\tSpectrum Read#1
    """
    info: dict = {"wavelength_nm": None, "readings": []}
    if not body:
        return info
    m = re.search(r"\(([0-9.]+)\s*nm\)", body[0])
    if m:
        info["wavelength_nm"] = float(m.group(1))
    if len(body) < 2:
        return info
    header = body[1].split("\t")
    cols = [c.strip() for c in header[1:] if c.strip()]
    for ln in body[2:]:
        cells = ln.split("\t")
        if not cells:
            continue
        row_label = cells[0].strip()
        if row_label not in ROWS:
            continue
        for i, col in enumerate(cols, start=1):
            if i >= len(cells):
                break
            v = cells[i].strip()
            if v == "" or _is_na(v):
                continue
            try:
                A = float(v)
            except ValueError:
                continue
            info["readings"].append({
                "Loc": f"{row_label}{col}",
                "Letter": row_label,
                "Number": int(col),
                "A": A,
            })
    return info


def hms_to_seconds(hms: str) -> float:
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


@dataclass
class ParsedFile:
    path: str
    is_spectrum_only: bool = False
    procedure: list[tuple[str, str]] = field(default_factory=list)
    layout: dict[str, str] = field(default_factory=dict)
    well_ids: dict[str, str] = field(default_factory=dict)
    init_reads: dict[str, dict[str, Optional[float]]] = field(default_factory=dict)
    kinetics: dict[str, tuple[list[str], np.ndarray]] = field(default_factory=dict)
    spectrum: Optional[dict] = None


def parse_file(path: str) -> ParsedFile:
    text = read_text(path)
    sections = split_sections(text)
    pf = ParsedFile(path=path)
    saw_full_protocol = False

    # Track the order initial-read wavelengths appear so we can map bare
    # "Kinetic read" blocks (Demmi-era 2023 files have no '734,950:734'
    # banner) to wavelengths by position.
    init_wl_order: list[str] = []
    pending_kinetic_idx = 0

    def _store_kinetic(wl: str, kin_body: list[str]) -> None:
        times, wells_order, A = parse_kinetic(kin_body)
        if A.shape[0]:
            idx = [wells_order.index(w) if w in wells_order else -1
                   for w in WELLS]
            A_canon = np.full((A.shape[0], 96), np.nan)
            for k, j in enumerate(idx):
                if j >= 0:
                    A_canon[:, k] = A[:, j]
        else:
            A_canon = np.zeros((0, 96))
        pf.kinetics[wl] = (times, A_canon)

    i = 0
    while i < len(sections):
        header, body = sections[i]
        h = header.strip()
        h_lc = h.lower()
        if h == "Procedure Summary":
            pf.procedure = parse_procedure(body)
            saw_full_protocol = True
        elif h == "Layout":
            pf.layout = parse_8x12_grid(body)
        elif h == "Well IDs":
            for ln in body[1:]:
                parts = ln.split("\t")
                if parts and parts[0].strip():
                    pf.well_ids[parts[0].strip()] = (
                        parts[1].strip() if len(parts) > 1 else ""
                    )
        elif h_lc.startswith(INITIAL_READ_PREFIXES) and ":" in h:
            wl = h.split(":", 1)[1].strip()
            pf.init_reads[wl] = parse_grid_float(body)
            if wl not in init_wl_order:
                init_wl_order.append(wl)
        elif h == "Spectrum":
            pf.spectrum = parse_spectrum(body)
        elif h_lc.startswith("kinetic read"):
            # Demmi-era files: bare 'Kinetic read\tA1\t...' header with no
            # preceding wavelength banner. Map by position to the Nth initial
            # read wavelength, cycling if the file has duplicate protocol
            # blocks (Gen5 sometimes exports the same protocol twice in one
            # file -- later block overwrites earlier).
            if init_wl_order:
                wl = init_wl_order[pending_kinetic_idx % len(init_wl_order)]
            else:
                wl = f"k{pending_kinetic_idx}"
            pending_kinetic_idx += 1
            _store_kinetic(wl, [header] + body)
        elif re.match(r"^[^:]+:(\S+)\s*$", h) and _is_grid_body(body):
            # 2026-09 protocol ("ABTS assay.prt"): the initial read is exported
            # as 'Read 1:734' / 'Read 1:950' over an 8x12 grid -- same shape as
            # the old 'Intial Read:734' section, just a different step name.
            wl = h.split(":", 1)[1].strip()
            pf.init_reads[wl] = parse_grid_float(body)
            if wl not in init_wl_order:
                init_wl_order.append(wl)
        elif re.match(r"^[^:]+:(\S+)\s*$", h):
            # 2026-era files: '734,950:734' (or 'Read 2:734') banner followed
            # (after a blank line) by the 'Kinetic read\tA1\t...' table section.
            wl = h.split(":", 1)[1].strip()
            kin_body = body
            if (not kin_body or not kin_body[0].lower().startswith("kinetic")) \
                    and i + 1 < len(sections):
                next_header, next_body = sections[i + 1]
                if next_header.lower().startswith("kinetic"):
                    kin_body = [next_header] + next_body
                    i += 1
                    pending_kinetic_idx += 1  # consumed bare-kinetic too
            _store_kinetic(wl, kin_body)
        i += 1

    pf.is_spectrum_only = pf.spectrum is not None and not saw_full_protocol
    return pf


# ----------------------------- PlateInfo --------------------------------------


PLATEINFO_COLS = ["Loc", "Type", "Sample", "Conc_mM", "Notes", "Letter", "Number"]


def load_plate_info(path: str) -> pd.DataFrame:
    """Load PlateInfo.xlsx tolerant of leading blank rows/columns."""
    df = pd.read_excel(path, header=None, dtype=object)
    header_row = None
    for i, row in df.iterrows():
        vals = [str(v).strip() if v is not None else "" for v in row.tolist()]
        if "Loc" in vals and "Sample" in vals:
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"{path}: no header row with 'Loc' and 'Sample'")
    headers = [str(v).strip() if v is not None else "" for v in df.iloc[header_row].tolist()]
    body = df.iloc[header_row + 1:].copy()
    body.columns = headers
    body = body.loc[:, [c for c in body.columns if c]]
    for c in PLATEINFO_COLS:
        if c not in body.columns:
            body[c] = None
    body = body[PLATEINFO_COLS].reset_index(drop=True)
    body = body.dropna(subset=["Loc"]).copy()
    body["Loc"] = body["Loc"].astype(str).str.strip()
    body = body[body["Loc"].isin(WELLS)].reset_index(drop=True)
    return body


def make_blank_plate_info() -> pd.DataFrame:
    rows = []
    for r in ROWS:
        for c in COLS:
            rows.append({
                "Loc": f"{r}{c}", "Type": None, "Sample": None,
                "Conc_mM": None, "Notes": None, "Letter": r, "Number": c,
            })
    return pd.DataFrame(rows, columns=PLATEINFO_COLS)


def auto_find_plate_info(input_path: str) -> Optional[str]:
    d = os.path.dirname(os.path.abspath(input_path))
    for parent in (d, os.path.dirname(d)):
        for name in ("PlateInfo.xlsx", "plate_info.xlsx"):
            cand = os.path.join(parent, name)
            if os.path.exists(cand):
                return cand
    return None


# ------------------------------- Output --------------------------------------


def _well_label(loc: str, sample, conc) -> str:
    bits = [loc]
    if pd.notna(sample) and str(sample).strip():
        bits.append(str(sample).strip())
    if pd.notna(conc):
        bits.append(f"{conc} mM")
    return " | ".join(bits)


def build_data_long(pf: ParsedFile, plate_info: pd.DataFrame) -> pd.DataFrame:
    """Tidy long table: one row per (Time_index x Loc x Wavelength) reading.

    Initial reads get Time_index = -1, Time_s = INITIAL_TIME_S.
    Kinetic reads get Time_index = 0..N, Time_s = parsed from HH:MM:SS.
    """
    pi = plate_info.set_index("Loc")
    rows: list[dict] = []

    # Initial reads
    for wl, grid in pf.init_reads.items():
        wl_f = float(wl)
        for loc in WELLS:
            v = grid.get(loc)
            if v is None:
                continue
            md = {c: pi.loc[loc, c] if loc in pi.index else None
                  for c in ["Type", "Sample", "Conc_mM", "Notes"]}
            rows.append({
                "Time_index": INITIAL_TIME_INDEX,
                "Time_s": INITIAL_TIME_S,
                "Time_hms": "initial",
                "Loc": loc, "Letter": loc[0], "Number": int(loc[1:]),
                **md,
                "Wavelength": wl_f,
                "A": v,
            })

    # Kinetic reads
    for wl, (times_hms, A) in pf.kinetics.items():
        if A.shape[0] == 0:
            continue
        wl_f = float(wl)
        times_s = np.array([hms_to_seconds(t) for t in times_hms])
        for ti, t_hms in enumerate(times_hms):
            for wi, loc in enumerate(WELLS):
                a = A[ti, wi]
                if not np.isfinite(a):
                    continue
                md = {c: pi.loc[loc, c] if loc in pi.index else None
                      for c in ["Type", "Sample", "Conc_mM", "Notes"]}
                rows.append({
                    "Time_index": ti,
                    "Time_s": float(times_s[ti]),
                    "Time_hms": t_hms,
                    "Loc": loc, "Letter": loc[0], "Number": int(loc[1:]),
                    **md,
                    "Wavelength": wl_f,
                    "A": float(a),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        ["Wavelength", "Time_index", "Letter", "Number"]
    ).reset_index(drop=True)


def build_wide(data_long: pd.DataFrame, wavelength: float,
               labeled_wells: list[str], plate_info: pd.DataFrame,
               value_col: str = "A") -> pd.DataFrame:
    """Time-rows x labeled-wells. value_col can be 'A' or 'dA'."""
    if data_long.empty or "Wavelength" not in data_long.columns:
        return pd.DataFrame()
    pi = plate_info.set_index("Loc")
    sub = data_long[data_long["Wavelength"] == wavelength]
    if sub.empty:
        return pd.DataFrame()
    piv = sub.pivot_table(
        index=["Time_index", "Time_s", "Time_hms"],
        columns="Loc", values=value_col, aggfunc="first",
    ).reset_index().sort_values("Time_index")
    cols = ["Time_index", "Time_s", "Time_hms"]
    out_cols = cols + [w for w in labeled_wells if w in piv.columns]
    piv = piv[out_cols]
    # rename labeled columns to "Loc | Sample | Conc mM"
    new_names = {}
    for w in labeled_wells:
        if w in piv.columns:
            sample = pi.loc[w, "Sample"] if w in pi.index else None
            conc = pi.loc[w, "Conc_mM"] if w in pi.index else None
            new_names[w] = _well_label(w, sample, conc)
    return piv.rename(columns=new_names)


def add_dA(data_long: pd.DataFrame) -> pd.DataFrame:
    """For each (Time_index, Loc) compute dA = A(734) - A(950) and add as a
    pseudo-wavelength row with Wavelength = 'dA'.

    Gen5 reads the two wavelengths sequentially, so at a given Time_index the
    734 and 950 rows have different Time_hms / Time_s (~38 s apart). We key the
    subtraction on (Time_index, Loc) only and adopt the 734 timestamp for the
    resulting dA row.
    """
    if data_long.empty:
        return data_long
    wls = data_long["Wavelength"].dropna().unique().tolist()
    if 734.0 not in wls or 950.0 not in wls:
        return data_long

    piv = data_long.pivot_table(
        index=["Time_index", "Loc"], columns="Wavelength", values="A",
        aggfunc="first",
    )
    if 734.0 not in piv.columns or 950.0 not in piv.columns:
        return data_long
    dA_series = piv[734.0] - piv[950.0]
    dA_df = dA_series.reset_index().rename(columns={0: "A"})
    dA_df.columns = ["Time_index", "Loc", "A"]
    dA_df = dA_df.dropna(subset=["A"])

    # Pull Time_s / Time_hms from the 734 reading at each Time_index.
    t_lookup = (
        data_long[data_long["Wavelength"] == 734.0]
        .drop_duplicates("Time_index")[["Time_index", "Time_s", "Time_hms"]]
    )
    dA_df = dA_df.merge(t_lookup, on="Time_index", how="left")

    # Per-well metadata
    meta_cols = ["Loc", "Letter", "Number", "Type", "Sample", "Conc_mM", "Notes"]
    meta = data_long.drop_duplicates("Loc")[meta_cols]
    dA_df = dA_df.merge(meta, on="Loc", how="left")

    dA_df["Wavelength"] = "dA"
    for c in data_long.columns:
        if c not in dA_df.columns:
            dA_df[c] = None
    dA_df = dA_df[data_long.columns]
    return pd.concat([data_long, dA_df], ignore_index=True)


def write_workbook(pf: ParsedFile, plate_info: pd.DataFrame, out_path: str) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        # --- Summary ---
        summary = [("Source file", os.path.basename(pf.path))]
        if pf.is_spectrum_only and pf.spectrum:
            summary.append(("Type", "Spectrum control"))
            summary.append(("Wavelength (nm)", pf.spectrum.get("wavelength_nm")))
            summary.append(("Readings", len(pf.spectrum.get("readings", []))))
        else:
            summary.append(("Type", "Kinetic run"))
            summary.append(("Wavelengths (nm)", ", ".join(sorted(pf.init_reads.keys()))))
            for wl, (times, _) in pf.kinetics.items():
                if times:
                    summary.append((
                        f"Kinetic {wl} nm",
                        f"{len(times)} points, last @ {times[-1]} "
                        f"({hms_to_seconds(times[-1])/60:.2f} min)",
                    ))
        if pf.procedure:
            summary.append(("", ""))
            summary.append(("--- Procedure ---", ""))
            summary.extend(pf.procedure)
        pd.DataFrame(summary, columns=["Field", "Value"]).to_excel(
            xw, sheet_name="Summary", index=False,
        )

        # --- PlateInfo ---
        plate_info.to_excel(xw, sheet_name="PlateInfo", index=False)

        # --- Layout (visual 8x12 with Sample names) ---
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
                            cell += f" ({conc} mM)"
                row[str(c)] = cell
            layout_rows.append(row)
        pd.DataFrame(layout_rows).to_excel(xw, sheet_name="Layout", index=False)

        # --- Spectrum-only branch ---
        if pf.is_spectrum_only and pf.spectrum:
            srows = []
            for r in pf.spectrum["readings"]:
                loc = r["Loc"]
                md = {c: pi.loc[loc, c] if loc in pi.index else None
                      for c in ["Type", "Sample", "Conc_mM", "Notes"]}
                srows.append({
                    "Loc": loc, "Letter": r["Letter"], "Number": r["Number"],
                    **md,
                    "Wavelength_nm": pf.spectrum.get("wavelength_nm"),
                    "A": r["A"],
                })
            if srows:
                pd.DataFrame(srows).to_excel(xw, sheet_name="Spectrum", index=False)
            return  # spectrum file is done

        # --- Kinetic-run branch: tidy long + wide views ---
        data_long = build_data_long(pf, plate_info)
        data_long = add_dA(data_long)
        if not data_long.empty:
            data_long.to_excel(xw, sheet_name="Data_long", index=False)

        labeled_wells = [
            loc for loc in WELLS
            if loc in pi.index and pd.notna(pi.loc[loc, "Sample"])
            and str(pi.loc[loc, "Sample"]).strip()
        ]

        if not labeled_wells:
            return  # nothing labeled in PlateInfo -> no chart-ready sheets

        for wl in sorted(pf.init_reads.keys()):
            wide = build_wide(data_long, float(wl), labeled_wells, plate_info)
            if not wide.empty:
                wide.to_excel(xw, sheet_name=f"Abs_{wl}_wide", index=False)

        # dA wide (if both wavelengths present)
        if (not data_long.empty and "Wavelength" in data_long.columns
                and "dA" in data_long["Wavelength"].astype(str).unique()):
            dA_long = data_long[data_long["Wavelength"].astype(str) == "dA"]
            wide = build_wide(
                dA_long.assign(Wavelength=999.0), 999.0,
                labeled_wells, plate_info,
            )
            if not wide.empty:
                wide.to_excel(xw, sheet_name="dA_wide", index=False)

    _autosize(out_path)


def _autosize(path: str) -> None:
    import openpyxl
    wb = openpyxl.load_workbook(path)
    for ws in wb.worksheets:
        for col_idx, col in enumerate(ws.columns, start=1):
            try:
                width = max(
                    (len(str(c.value)) for c in col if c.value is not None),
                    default=0,
                )
            except Exception:
                width = 12
            ws.column_dimensions[get_column_letter(col_idx)].width = (
                min(max(width + 2, 8), 28)
            )
    wb.save(path)


# --------------------------------- CLI ---------------------------------------


def cmd_convert(args: argparse.Namespace) -> int:
    path = args.file
    if not os.path.exists(path):
        print(f"error: not found: {path}", file=sys.stderr)
        return 2

    pf = parse_file(path)
    pi_path = args.plate_info or auto_find_plate_info(path)
    if pi_path is None:
        print(
            "warning: no PlateInfo.xlsx found -- output will be unlabeled. "
            "Use --plate-info or place PlateInfo.xlsx next to the data.",
            file=sys.stderr,
        )
        plate_info = make_blank_plate_info()
    else:
        plate_info = load_plate_info(pi_path)
        print(f"PlateInfo: {pi_path}", file=sys.stderr)

    out = args.out or os.path.splitext(path)[0] + ".xlsx"
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    write_workbook(pf, plate_info, out)
    print(f"wrote {out}")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    d = args.dir
    if not os.path.isdir(d):
        print(f"error: not a directory: {d}", file=sys.stderr)
        return 2
    txt_files = sorted(
        os.path.join(d, f) for f in os.listdir(d)
        if f.lower().endswith(".txt")
    )
    if not txt_files:
        print(f"no .txt files in {d}", file=sys.stderr)
        return 1
    pi_path = args.plate_info or auto_find_plate_info(txt_files[0])
    if pi_path is None:
        print(
            "warning: no PlateInfo.xlsx found -- outputs will be unlabeled.",
            file=sys.stderr,
        )
        plate_info = make_blank_plate_info()
    else:
        plate_info = load_plate_info(pi_path)
        print(f"PlateInfo: {pi_path}", file=sys.stderr)

    out_dir = args.out or d
    os.makedirs(out_dir, exist_ok=True)
    for path in txt_files:
        try:
            pf = parse_file(path)
        except Exception as e:
            print(f"  SKIP {os.path.basename(path)}: {e}", file=sys.stderr)
            continue
        out = os.path.join(
            out_dir, os.path.splitext(os.path.basename(path))[0] + ".xlsx",
        )
        write_workbook(pf, plate_info, out)
        print(f"wrote {out}")
    return 0


def cmd_new_plateinfo(args: argparse.Namespace) -> int:
    make_blank_plate_info().to_excel(args.out, index=False)
    print(f"wrote blank PlateInfo template: {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")

    p_conv = sub.add_parser("convert", help="convert one .txt file to .xlsx")
    p_conv.add_argument("file", help="input .txt file")
    p_conv.add_argument("--out", "-o", default=None, help="output .xlsx (default: same name)")
    p_conv.add_argument("--plate-info", default=None, help="path to PlateInfo.xlsx")
    p_conv.set_defaults(func=cmd_convert)

    p_batch = sub.add_parser(
        "batch", help="convert every .txt in a directory to its own .xlsx",
    )
    p_batch.add_argument("dir", help="directory of .txt files")
    p_batch.add_argument("--out", "-o", default=None, help="output dir (default: same)")
    p_batch.add_argument("--plate-info", default=None, help="path to PlateInfo.xlsx")
    p_batch.set_defaults(func=cmd_batch)

    p_new = sub.add_parser(
        "new-plateinfo", help="write a blank PlateInfo.xlsx (96 rows, A1..H12)",
    )
    p_new.add_argument("--out", "-o", required=True, help="output .xlsx path")
    p_new.set_defaults(func=cmd_new_plateinfo)

    args = ap.parse_args(argv)
    if args.cmd is None:
        ap.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
