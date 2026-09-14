"""BioTek Gen5 ABTS plate-reader exports -> one student-friendly Excel workbook.

Streamlit front end for ``biotek_to_excel_v2`` (the same code that builds the
Combined.xlsx files in the research folders). Upload the ``.txt`` exports for a
run, describe the plate, download the workbook.
"""
from __future__ import annotations

import io
import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pchemapps import biotek_to_excel as v1
from pchemapps import biotek_to_excel_v2 as v2

# Validated default palette (dataviz skill): categorical slots + blue ramp.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
               "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]
NEUTRAL = "#8a8985"

ROLE_OPTIONS = ["Sample", "Control", "Blank (EtOH)", "Unknown", ""]

_HELP = """
**What this does.** Each Gen5 `.txt` export holds one sample in one plate
column: an initial 734/950 nm read of the ABTS•+ solution, then a 734/950 nm
kinetic read after the antioxidant is added. This page parses every file,
joins it to a plate map, and writes one workbook with:

- `Data_long` – tidy table, one row per well per timepoint, with `A_734`,
  `A_950`, `dA = A_734 - A_950`, and `is_initial` / `is_final` flags.
- `Per_well` – initial and final `dA`, their difference, and the fraction of
  radical remaining — the c₅₀ input.
- `QC_initial` – is the starting ABTS the same in every well of the column?
- `dA_vs_time` – chart-ready wide sheet, one column per labeled well.

**Plate map.** The sample column is found from the initial reads. By default
rows A–F are a 2× serial dilution (A = highest), G is the ABTS-only control,
H is the ethanol blank. Edit the table if your plate differs — the
`Conc_mM` column is whatever *you* say it is; only the top well of each column
needs a value and the rest are filled by the dilution factor.

**Concentrations.** Enter the concentration *in the well* (after the
antioxidant is mixed into the radical solution). If your plate map records the
pipetted stock instead, set the assay dilution factor (e.g. 20 for 10 µL into
190 µL) and the workbook adds a `Conc_final_mM` column.
"""


# ------------------------------------------------------------------ helpers


def _save_uploads(files) -> tuple[str, list[str]]:
    """Write uploads to a temp dir; parse_file wants real paths."""
    d = tempfile.mkdtemp(prefix="biotek_")
    paths = []
    for f in files:
        p = os.path.join(d, os.path.basename(f.name))
        with open(p, "wb") as fh:
            fh.write(f.getvalue())
        paths.append(p)
    return d, sorted(paths)


def _sample_name_from_filename(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    for pre in ("ABTS ", "abts "):
        if stem.startswith(pre):
            stem = stem[len(pre):]
    return stem.strip()


def _init_734(pf: v1.ParsedFile, loc: str) -> float:
    grid = pf.init_reads.get("734") or {}
    v = grid.get(loc)
    return float(v) if v is not None else np.nan


def default_plate_info(parsed: list[tuple[str, v1.ParsedFile, list[int]]],
                       n_sample_rows: int, control_row: str, blank_row: str,
                       top_conc: float) -> pd.DataFrame:
    """One row per well of every detected sample column, roles pre-filled."""
    pi = v1.make_blank_plate_info()
    sample_rows = v1.ROWS[:n_sample_rows]
    for name, pf, cols in parsed:
        for col in cols:
            for r in v1.ROWS:
                loc = f"{r}{col}"
                m = pi["Loc"] == loc
                pi.loc[m, "Sample"] = name
                if r == control_row:
                    pi.loc[m, ["Type", "Conc_mM"]] = ["Control", 0.0]
                elif r == blank_row:
                    pi.loc[m, "Type"] = "Blank (EtOH)"
                elif r in sample_rows:
                    pi.loc[m, "Type"] = "Sample"
                    if r == sample_rows[0]:
                        pi.loc[m, "Conc_mM"] = top_conc
                else:
                    pi.loc[m, "Type"] = ""
    return pi


def plate_warnings(pi: pd.DataFrame,
                   parsed: list[tuple[str, v1.ParsedFile, list[int]]]) -> list[str]:
    """Wells whose initial 734 nm read contradicts their assigned role."""
    out = []
    for name, pf, cols in parsed:
        for col in cols:
            for r in v1.ROWS:
                loc = f"{r}{col}"
                row = pi[pi["Loc"] == loc]
                if row.empty:
                    continue
                typ = str(row["Type"].iloc[0] or "").strip()
                a = _init_734(pf, loc)
                if np.isnan(a):
                    continue
                if typ.startswith("Blank") and a > 0.2:
                    out.append(f"{name} {loc} is labeled Blank but its initial "
                               f"A734 = {a:.3f} (ABTS present).")
                elif typ in ("Sample", "Control") and a < 0.2:
                    out.append(f"{name} {loc} is labeled {typ} but its initial "
                               f"A734 = {a:.3f} (looks empty / blank).")
    return out


def _workbook_bytes(tidy: pd.DataFrame, pi: pd.DataFrame, meta) -> bytes:
    fd, path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    try:
        v2.write_combined_workbook(tidy, pi, meta, path)
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(path)


# ------------------------------------------------------------------- charts


def _base_layout(fig: go.Figure, xtitle: str, ytitle: str, height=380) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=50, r=20, t=40, b=50),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="v", x=1.02, y=1, font=dict(size=11)),
        hovermode="closest",
    )
    fig.update_xaxes(title=xtitle, showgrid=True, gridcolor="rgba(128,128,128,0.18)",
                     zeroline=False, showline=False)
    fig.update_yaxes(title=ytitle, showgrid=True, gridcolor="rgba(128,128,128,0.18)",
                     zeroline=False, showline=False)
    return fig


def chart_dA_vs_time(tidy: pd.DataFrame, experiment: str) -> go.Figure:
    """dA vs time for one experiment; concentration on the blue ramp."""
    t = tidy[(tidy["Experiment"] == experiment) & (tidy["Time_index"] >= 0)]
    fig = go.Figure()
    conc_col = "Conc_final_mM" if "Conc_final_mM" in t.columns else "Conc_mM"
    wells = (t.drop_duplicates("Loc")
              .assign(_c=lambda d: pd.to_numeric(d[conc_col], errors="coerce"))
              .sort_values(["_c", "Letter"], ascending=[False, True]))
    sample_wells = wells[wells["Type"].astype(str).str.strip() == "Sample"]
    n = max(len(sample_wells), 1)
    for i, (_, w) in enumerate(wells.iterrows()):
        g = t[t["Loc"] == w["Loc"]].sort_values("Time_s")
        typ = str(w["Type"]).strip()
        if typ == "Sample":
            k = int(round((1 - i / max(n - 1, 1)) * (len(BLUE_RAMP) - 1)))
            k = min(max(k, 0), len(BLUE_RAMP) - 1)
            color, dash = BLUE_RAMP[k], "solid"  # highest conc = darkest
            label = f"{w['Loc']}  {w['_c']:g} mM" if pd.notna(w["_c"]) else w["Loc"]
        elif typ == "Control":
            color, dash, label = NEUTRAL, "dash", f"{w['Loc']}  control"
        else:
            color, dash, label = NEUTRAL, "dot", f"{w['Loc']}  {typ or 'unlabeled'}"
        fig.add_trace(go.Scatter(
            x=g["Time_min"], y=g["dA"], mode="lines+markers", name=label,
            line=dict(color=color, width=2, dash=dash),
            marker=dict(size=6, color=color),
            hovertemplate=f"{label}<br>t = %{{x:.2f}} min<br>dA = %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(title=dict(text=experiment, font=dict(size=14)))
    return _base_layout(fig, "Time (min)", "dA = A₇₃₄ − A₉₅₀")


def chart_frac_vs_conc(per_well: pd.DataFrame) -> go.Figure:
    """Fraction of radical remaining vs concentration, one series per sample."""
    conc_col = "Conc_final_mM" if "Conc_final_mM" in per_well.columns else "Conc_mM"
    s = per_well[per_well["Type"].astype(str).str.strip() == "Sample"].copy()
    s["_c"] = pd.to_numeric(s[conc_col], errors="coerce")
    s = s[s["_c"] > 0]
    fig = go.Figure()
    for i, (name, g) in enumerate(s.groupby("Sample", sort=False)):
        g = g.sort_values("_c")
        color = CATEGORICAL[i % len(CATEGORICAL)]
        fig.add_trace(go.Scatter(
            x=g["_c"], y=g["frac_remaining"], mode="lines+markers", name=str(name),
            line=dict(color=color, width=2), marker=dict(size=8, color=color),
            hovertemplate=f"{name}<br>%{{x:g}} mM<br>fraction remaining = %{{y:.3f}}<extra></extra>",
        ))
    fig.add_hline(y=0.5, line=dict(color=NEUTRAL, width=1, dash="dot"),
                  annotation_text="c₅₀", annotation_position="right")
    fig = _base_layout(fig, f"{conc_col.replace('_', ' ')} (log)", "Fraction of dA remaining")
    fig.update_xaxes(type="log")
    fig.update_yaxes(range=[-0.05, 1.1])
    return fig


# --------------------------------------------------------------------- page


def run() -> None:
    st.header("BioTek ABTS Plate Reader → Excel")
    with st.expander("How this works", expanded=False):
        st.markdown(_HELP)

    files = st.file_uploader(
        "Gen5 .txt exports (one per sample; select all files from the run)",
        type=["txt"], accept_multiple_files=True,
    )
    if not files:
        st.info("Upload the .txt files exported from Gen5 to begin.")
        return

    _, paths = _save_uploads(files)
    parsed: list[tuple[str, v1.ParsedFile, list[int]]] = []
    problems: list[str] = []
    for p in paths:
        try:
            pf = v1.parse_file(p)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{os.path.basename(p)}: could not parse ({e})")
            continue
        cols = v2.detect_data_column(pf)
        if not cols:
            problems.append(f"{os.path.basename(p)}: no readings found")
            continue
        if not v2.has_initial_reads(pf) and len(cols) > 1:
            problems.append(
                f"{os.path.basename(p)}: no initial reads and the kinetic read "
                f"spans columns {cols}; cannot tell which column is the sample"
            )
            continue
        parsed.append((_sample_name_from_filename(p), pf, cols))
    for msg in problems:
        st.error(msg)
    if not parsed:
        return

    # ---- file summary
    rows = []
    for name, pf, cols in parsed:
        n_t = max((A.shape[0] for _t, A in pf.kinetics.values()), default=0)
        rows.append({
            "File": os.path.basename(pf.path), "Sample (edit below)": name,
            "Plate column": ", ".join(str(c) for c in cols),
            "Initial reads": "yes" if v2.has_initial_reads(pf) else "no",
            "Kinetic timepoints": n_t,
            "Wavelengths": ", ".join(sorted(pf.kinetics.keys())),
        })
    st.subheader("1. Files")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ---- plate map
    st.subheader("2. Plate map")
    pi_file = st.file_uploader(
        "Optional: upload a PlateInfo.xlsx (columns Loc, Type, Sample, Conc_mM, "
        "Notes, Letter, Number) instead of filling the table in",
        type=["xlsx"], accept_multiple_files=False, key="plateinfo",
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_sample_rows = st.number_input("Sample rows (from A)", 1, 8, 6,
                                        help="A–F = 6 rows; A–G = 7 rows")
    with c2:
        top_conc = st.number_input("Top (row A) conc, mM", min_value=0.0,
                                   value=0.5, step=0.05, format="%.4f")
    with c3:
        serial = st.number_input("Serial dilution factor", min_value=1.0,
                                 value=2.0, step=0.5,
                                 help="Each row down the column is divided by this")
    with c4:
        assay_dil = st.number_input(
            "Assay dilution factor", min_value=1.0, value=1.0, step=1.0,
            help="1 if Conc_mM is already the in-well concentration. 20 if it "
                 "is the pipetted stock and 10 µL goes into 190 µL of radical.",
        )
    rc1, rc2 = st.columns(2)
    with rc1:
        control_row = st.selectbox("Control row (ABTS, no antioxidant)",
                                   list(v1.ROWS) + ["none"], index=6)
    with rc2:
        blank_row = st.selectbox("Blank row (EtOH, no ABTS)",
                                 list(v1.ROWS) + ["none"], index=7)

    if pi_file is not None:
        try:
            pi_full = v1.load_plate_info(io.BytesIO(pi_file.getvalue()))
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read PlateInfo: {e}")
            return
        st.caption("Using the uploaded PlateInfo. Edit it here if needed.")
    else:
        pi_full = default_plate_info(parsed, int(n_sample_rows), control_row,
                                     blank_row, float(top_conc))

    used_cols = sorted({c for _n, _pf, cols in parsed for c in cols})
    editable = (pi_full[pi_full["Number"].isin(used_cols)]
                .sort_values(["Number", "Letter"]).copy())
    editable["Conc_mM"] = pd.to_numeric(editable["Conc_mM"], errors="coerce")
    editable["A734_initial"] = [
        next((_init_734(pf, loc) for _n, pf, cols in parsed
              if int(loc[1:]) in cols), np.nan)
        for loc in editable["Loc"]
    ]
    show_cols = ["Loc", "Sample", "Type", "Conc_mM", "Notes", "A734_initial"]
    st.caption("Only the top well of each column needs a concentration; blank "
               "Sample wells below it are filled by the serial dilution factor. "
               "`A734_initial` is read from the file and is not editable.")
    edited = st.data_editor(
        editable[show_cols].reset_index(drop=True),
        hide_index=True, use_container_width=True, num_rows="fixed",
        column_config={
            "Loc": st.column_config.TextColumn(disabled=True),
            "Type": st.column_config.SelectboxColumn(options=ROLE_OPTIONS),
            "Conc_mM": st.column_config.NumberColumn(format="%.5g"),
            "A734_initial": st.column_config.NumberColumn(format="%.3f", disabled=True),
        },
        key="plate_editor",
    )
    # Fold edits back into the full 96-well frame.
    pi = pi_full.copy()
    ed = edited.set_index("Loc")
    for loc in ed.index:
        m = pi["Loc"] == loc
        for c in ("Sample", "Type", "Conc_mM", "Notes"):
            pi.loc[m, c] = ed.at[loc, c]
    pi["Conc_mM"] = pd.to_numeric(pi["Conc_mM"], errors="coerce")
    pi = v2.derive_serial_dilution(pi, factor=float(serial))

    for msg in plate_warnings(pi, parsed):
        st.warning(msg)

    # ---- build
    st.subheader("3. Workbook")
    out_name = st.text_input("Workbook file name", value="Combined.xlsx")
    if not out_name.lower().endswith(".xlsx"):
        out_name += ".xlsx"
    tidy, meta = v2.combine(paths, pi, include_all=True,
                            dilution=float(assay_dil))
    if tidy.empty:
        st.error("Nothing to combine; check the plate map.")
        return
    # Experiment names come from file names; swap in the editable sample names.
    exp_name = {os.path.splitext(os.path.basename(pf.path))[0]: name
                for name, pf, _c in parsed}
    tidy["Experiment"] = tidy["Experiment"].map(lambda e: exp_name.get(e, e))
    renamed = []
    for k, val in meta:
        for e, nm in exp_name.items():
            if e in k:
                k = k.replace(e, nm)
                break
        renamed.append((k, val))
    meta = renamed

    st.download_button(
        "Download workbook", data=_workbook_bytes(tidy, pi, meta),
        file_name=out_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

    per_well = v2.build_per_well(tidy)
    qc = v2.build_qc_initial(tidy)

    # ---- results
    st.subheader("4. Results")
    if not qc.empty:
        flagged = qc[qc["flag_gt5pct"]]
        stats = (qc.groupby("Experiment")[["sample_mean_A734", "sample_std_A734"]]
                   .first().reset_index())
        stats.columns = ["Sample", "Initial A734 mean", "Initial A734 std"]
        st.markdown("**Initial ABTS check** (Sample wells only; flags if a well "
                    "is > 5 % from its column mean)")
        st.dataframe(stats.round(4), hide_index=True, use_container_width=True)
        if not flagged.empty:
            st.warning("Wells more than 5 % from the column mean: "
                       + ", ".join(f"{r.Experiment} {r.Loc}" for r in flagged.itertuples()))

    st.markdown("**Fraction of radical remaining vs concentration**")
    st.plotly_chart(chart_frac_vs_conc(per_well), use_container_width=True)

    st.markdown("**dA vs time**")
    exps = list(dict.fromkeys(tidy["Experiment"]))
    for e in exps:
        st.plotly_chart(chart_dA_vs_time(tidy, e), use_container_width=True)

    with st.expander("Per_well table"):
        st.dataframe(per_well.round(4), hide_index=True, use_container_width=True)
