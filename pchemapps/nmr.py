"""NMR peak picking for Anasazi EFT-90 (NUTS binary) files.

Reader follows dwyer-research/2609/nuts/l6/l6nmr.py (read_nuts); processing is plain
numpy/scipy so the app does not need nmrglue.
"""
import io
import struct

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import signal
from scipy.optimize import minimize

from pchemapps.combineRaman import (add_peak_lines_plotly, edit_peaks, show_plotly,
                                    style_axes_plotly, trace_color)

DATA_OFFSET = 4104  # 1024-byte header + parameter block


def read_nuts(raw: bytes) -> dict:
    """FIDs and acquisition parameters from an Anasazi NUTS binary file."""
    if struct.unpack('<i', raw[:4])[0] != 0x04030201:
        raise ValueError('not a NUTS file (byte-order marker missing)')
    np1 = struct.unpack('<i', raw[0x180:0x184])[0]
    np2 = struct.unpack('<i', raw[0x220:0x224])[0] or 1
    sw, sf, off = struct.unpack('<3f', raw[0x1c0:0x1cc])
    data = np.frombuffer(raw[DATA_OFFSET:], '<f4')
    if data.size != np1 * np2 * 2:
        raise ValueError(f'data size mismatch: {data.size} vs {np1}x{np2}x2')
    data = data.reshape(np2, np1, 2)
    fid = (data[..., 0] + 1j * data[..., 1]).astype(complex)
    date = raw[0x4b0:0x4d0].split(b'\0')[0].decode(errors='replace')
    return dict(fid=fid, sw=float(sw), sf=float(sf), offset=float(off), date=date)


def fft_spectrum(fid: np.ndarray, sw: float, lb: float) -> np.ndarray:
    """Exponential line broadening (lb in Hz), first-point correction, FFT.
    Index 0 is downfield (highest ppm)."""
    t = np.arange(fid.size) / sw
    f = fid * np.exp(-np.pi * lb * t)
    f[0] = f[0] / 2
    return np.fft.fftshift(np.fft.fft(f))


def hz_axis(n: int, sw: float) -> np.ndarray:
    """Hz from the carrier for each point, downfield first."""
    return (n / 2 - np.arange(n)) * sw / n


def apply_phase(spec, p0, p1):
    """p0, p1 in degrees; p1 is the total first-order change across the spectrum."""
    x = np.linspace(-0.5, 0.5, spec.size)
    return (spec * np.exp(1j * np.radians(p0 + p1 * x))).real


def autophase(spec: np.ndarray):
    """Zero- and first-order phase that maximize positive area and penalize negative dips.
    Same cost as the H1_1D notebook. Returns (p0, p1) in degrees."""
    x = np.linspace(-0.5, 0.5, spec.size)
    a1 = np.abs(spec).sum()
    a2 = (np.abs(spec) ** 2).sum()

    def cost(p):
        r = (spec * np.exp(1j * (p[0] + p[1] * x))).real
        return -r.sum() / a1 + 5 * (np.clip(r, None, 0) ** 2).sum() / a2

    starts = [[a, b] for a in np.linspace(-np.pi, np.pi, 8, endpoint=False)
              for b in (-2 * np.pi, 0, 2 * np.pi)]
    best = min((minimize(cost, s, method='Nelder-Mead') for s in starts), key=lambda m: m.fun)
    p0, p1 = np.degrees(best.x)
    return float((p0 + 180) % 360 - 180), float(p1)


def peak_index(y, ppm, lo, hi):
    """Index of the tallest point of y between lo and hi ppm (as in L6 l6nmr.peak_index)."""
    idx = np.where((ppm >= min(lo, hi)) & (ppm <= max(lo, hi)))[0]
    if idx.size == 0:
        return None
    return idx[np.argmax(y[idx])]


@st.cache_data(show_spinner=False)
def process(raw: bytes, lb: float, row: str, win=None):
    """Read, FFT, optionally align and sum FIDs, auto-phase. Cached on the file bytes and settings.

    row: FID index as a string, or 'sum'. When summing with a search window win=(lo, hi) on the
    unreferenced ppm axis, every FID is first shifted by whole points so its tallest magnitude
    peak in win lines up with the first FID's (the L6 drift removal, l6nmr.align).
    """
    r = read_nuts(raw)
    n = r['fid'].shape[1]
    hz = hz_axis(n, r['sw'])
    raw_ppm = (hz + r['offset']) / r['sf']
    rows = range(r['fid'].shape[0]) if row == 'sum' else [int(row)]
    specs = [fft_spectrum(r['fid'][i], r['sw'], lb) for i in rows]
    drift_hz = []
    if row == 'sum' and win is not None:
        target = peak_index(np.abs(specs[0]), raw_ppm, *win)
        if target is not None:
            for k, sp in enumerate(specs):
                shift = target - peak_index(np.abs(sp), raw_ppm, *win)
                specs[k] = np.roll(sp, shift)
                drift_hz.append(shift * r['sw'] / n)
    spec = np.sum(specs, axis=0)
    p0, p1 = autophase(spec)
    return dict(spec=spec, hz=hz, raw_ppm=raw_ppm, sw=r['sw'], sf=r['sf'], offset=r['offset'],
                date=r['date'], n_fid=r['fid'].shape[0], p0=p0, p1=p1, drift_hz=drift_hz)


def pick_peaks(ppm, y, lo, hi, min_prom_frac, max_peaks):
    """Peaks between lo and hi ppm by prominence, as a fraction of the tallest point in that window."""
    m = (ppm >= lo) & (ppm <= hi)
    idx_win = np.where(m)[0]
    yy = y[m]
    if yy.size < 3:
        return pd.DataFrame({'δ (ppm)': [], 'Intensity': [], 'Prominence': []})
    scale = yy.max() - np.median(yy)
    idx, props = signal.find_peaks(yy, prominence=min_prom_frac * scale)
    order = np.argsort(props['prominences'])[::-1][:max_peaks]
    idx = idx_win[idx[order]]
    return pd.DataFrame({'δ (ppm)': ppm[idx], 'Intensity': y[idx],
                         'Prominence': props['prominences'][order] / scale}).reset_index(drop=True)


def run():
    st.header('NMR Peak Picking (Anasazi EFT-90)')
    st.markdown(
        'Upload one or more raw NUTS files from the EFT-90 (e.g. `caffeine1_howard_H1_1D`; no file '
        'extension). Each file is Fourier transformed and auto-phased; then reference the ppm axis '
        'and pick peaks.')

    files = st.file_uploader('NUTS files', accept_multiple_files=True)
    if not files:
        return

    c1, c2 = st.columns(2)
    lb = c1.number_input('Line broadening (Hz)', 0.0, 10.0, 0.5, 0.1)
    normalize = c2.checkbox('Normalize each spectrum to its tallest peak', value=True)

    st.subheader('Reference (peak registration)')
    st.caption(
        'The magnet has no lock, so every spectrum drifts. Pick a lone peak that is in every file '
        '(HOD in D₂O 4.79, CHCl₃ 7.26, TMS 0). In each file the tallest point inside the search window '
        '(on the unreferenced axis, checked with the plot below) is set to that δ. For multi-FID files '
        'that are summed, each FID is first shifted onto that peak. Same method as L6 (`reference` + `align`).')
    c1, c2, c3, c4 = st.columns(4)
    register = c1.checkbox('Register to a peak', value=True)
    ref_ppm = c2.number_input('Peak δ (ppm)', value=4.79, step=0.01, format='%.3f', disabled=not register)
    win_hi = c3.number_input('Search window, high (unreferenced ppm)', value=6.5, step=0.1, disabled=not register)
    win_lo = c4.number_input('Search window, low', value=5.0, step=0.1, disabled=not register)
    win = (min(win_lo, win_hi), max(win_lo, win_hi)) if register else None

    # --- read + process ---------------------------------------------------------------
    loaded = []
    for f in files:
        raw = f.getvalue()
        try:
            n_fid = read_nuts(raw)['fid'].shape[0]
        except Exception as e:  # noqa: BLE001
            st.error(f'{f.name}: {e}')
            continue
        row = '0'
        if n_fid > 1:
            choice = st.selectbox(f'{f.name} has {n_fid} FIDs (T1 / kinetics). Which one?',
                                  ['Sum of all'] + [str(i + 1) for i in range(n_fid)], index=n_fid,
                                  key=f'row_{f.name}')
            row = 'sum' if choice == 'Sum of all' else str(int(choice) - 1)
        with st.spinner(f'Processing {f.name}...'):
            loaded.append((f.name, process(raw, lb, row, win)))
    if not loaded:
        return

    # --- per-file settings: label, phase tweak, reference ------------------------------
    st.subheader('Phase and reference')
    st.caption(
        'Phase is automatic; add corrections in degrees if a peak still looks dispersive. '
        'Without a reference the ppm axis uses the instrument offset and is only approximate.')
    st.caption('To reference one file by a different peak, fill in **Observed δ** (what the plot shows) '
               'and **True δ** in its row; that overrides the registration above for that file.')
    settings = pd.DataFrame({
        'File': [n for n, _ in loaded],
        'Label': [n.replace('_H1_1D', '') for n, _ in loaded],
        'p0 tweak (°)': [0.0] * len(loaded),
        'p1 tweak (°)': [0.0] * len(loaded),
        'Observed δ (ppm)': [None] * len(loaded),
        'True δ (ppm)': [None] * len(loaded),
    })
    settings = st.data_editor(
        settings, hide_index=True, key='nmr_settings', disabled=['File'],
        column_config={'Observed δ (ppm)': st.column_config.NumberColumn(format='%.3f'),
                       'True δ (ppm)': st.column_config.NumberColumn(format='%.3f')})

    spectra, reg = [], []
    for (name, s), (_, opt) in zip(loaded, settings.iterrows()):
        y = apply_phase(s['spec'], s['p0'] + (opt['p0 tweak (°)'] or 0), s['p1'] + (opt['p1 tweak (°)'] or 0))
        if normalize:
            y = y / np.max(y)
        ppm = s['raw_ppm'].copy()
        obs, true = opt['Observed δ (ppm)'], opt['True δ (ppm)']
        found = None
        if pd.notna(obs) and pd.notna(true):
            ppm = ppm + (true - obs)
            how = 'manual'
        elif win is not None:
            i = peak_index(np.abs(s['spec']), s['raw_ppm'], *win)
            if i is None:
                how = 'no points in search window'
            else:
                found = s['raw_ppm'][i]
                ppm = ppm + (ref_ppm - found)
                how = 'registered'
        else:
            how = 'instrument offset'
        reg.append({'File': name, 'Reference': how,
                    'Peak found at (unreferenced ppm)': None if found is None else round(found, 3),
                    'Shift applied (Hz)': round((ppm[0] - s['raw_ppm'][0]) * s['sf'], 1),
                    'FID drift range (Hz)': round(np.ptp(s['drift_hz']), 1) if s['drift_hz'] else None})
        spectra.append(dict(name=name, label=str(opt['Label'] or name), ppm=ppm, y=y, s=s))

    if win is not None:
        with st.expander('Check the registration peak', expanded=False):
            st.caption('Unreferenced magnitude spectra; the shaded band is the search window and the marker is '
                       'the peak each file was registered to. Move the window if it caught the wrong peak.')
            chk = go.Figure()
            for i, ((name, s), r) in enumerate(zip(loaded, reg)):
                mag = np.abs(s['spec']) / np.abs(s['spec']).max()
                chk.add_trace(go.Scatter(x=s['raw_ppm'], y=mag, mode='lines', name=name,
                                         line=dict(color=trace_color(i), width=1)))
                if r['Peak found at (unreferenced ppm)'] is not None:
                    j = np.argmin(abs(s['raw_ppm'] - r['Peak found at (unreferenced ppm)']))
                    chk.add_trace(go.Scatter(x=[s['raw_ppm'][j]], y=[mag[j]], mode='markers', showlegend=False,
                                             marker=dict(color=trace_color(i), size=9, symbol='triangle-down')))
            chk.add_vrect(x0=win[0], x1=win[1], fillcolor='gray', opacity=0.15, line_width=0)
            style_axes_plotly(chk)
            chk.update_xaxes(title='unreferenced δ (ppm)', autorange='reversed')
            chk.update_layout(height=350, hovermode='closest')
            show_plotly(chk)
            st.dataframe(pd.DataFrame(reg), hide_index=True)

    # --- peak picking -----------------------------------------------------------------
    st.subheader('Peaks')
    all_ppm = np.concatenate([d['ppm'] for d in spectra])
    c1, c2, c3, c4 = st.columns(4)
    hi = c1.number_input('Plot / pick from (ppm, high)', value=float(np.ceil(min(12.0, all_ppm.max()))))
    lo = c2.number_input('to (ppm, low)', value=float(np.floor(max(-1.0, all_ppm.min()))))
    min_prom = c3.slider('Min. prominence (% of tallest)', 1, 50, 5) / 100
    max_peaks = c4.number_input('Max peaks per file', 1, 50, 12)
    lo, hi = min(lo, hi), max(lo, hi)
    st.caption('Tick **Show** to label a peak on the plot; edit labels; add or delete rows.')

    picked = []
    for i, d in enumerate(spectra):
        cand = pick_peaks(d['ppm'], d['y'], lo, hi, min_prom, int(max_peaks))
        shown = edit_peaks(cand, 'δ (ppm)', 'Intensity', d['label'], key=f'peaks_{d["name"]}',
                           n_show=len(cand))
        shown.insert(0, 'File', d['label'])
        picked.append(shown)
    peaks = pd.concat(picked, ignore_index=True) if picked else pd.DataFrame()

    # --- plot -------------------------------------------------------------------------
    offset = st.slider('Vertical offset between spectra', 0.0, 2.0, 0.0 if len(spectra) == 1 else 1.1, 0.05)
    fig = go.Figure()
    for i, d in enumerate(spectra):
        m = (d['ppm'] >= lo) & (d['ppm'] <= hi)
        fig.add_trace(go.Scatter(x=d['ppm'][m], y=d['y'][m] + i * offset, mode='lines', name=d['label'],
                                 line=dict(color=trace_color(i), width=1)))
        p = picked[i]
        if len(p):
            add_peak_lines_plotly(fig, p['δ (ppm)'], p['Label'], trace_color(i))
    style_axes_plotly(fig)
    fig.update_xaxes(title='δ (ppm)', autorange='reversed')
    fig.update_yaxes(title='Intensity' + (' (normalized)' if normalize else ''))
    fig.update_layout(height=550, hovermode='closest')
    show_plotly(fig)

    with st.expander('Acquisition parameters'):
        st.dataframe(pd.DataFrame([{
            'File': d['name'], 'Date': d['s']['date'], 'Spectrometer (MHz)': round(d['s']['sf'], 3),
            'Sweep width (Hz)': round(d['s']['sw'], 1), 'Points': d['s']['spec'].size,
            'FIDs in file': d['s']['n_fid'], 'Auto p0 (°)': round(d['s']['p0'], 1),
            'Auto p1 (°)': round(d['s']['p1'], 1)} for d in spectra]), hide_index=True)

    # --- downloads --------------------------------------------------------------------
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as xw:
        if len(peaks):
            peaks.drop(columns=['Show'], errors='ignore').to_excel(xw, sheet_name='Peaks', index=False)
        for d in spectra:
            sheet = ''.join(ch for ch in d['label'] if ch not in '[]:*?/\\')[:31] or 'spectrum'
            pd.DataFrame({'δ (ppm)': d['ppm'], 'Hz from carrier': d['s']['hz'], 'Intensity': d['y']}).to_excel(
                xw, sheet_name=sheet, index=False)
    st.download_button('Download Excel (peaks + spectra)', buf.getvalue(), file_name='nmr_peaks.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    if len(peaks):
        st.download_button('Download peaks (CSV)', peaks.drop(columns=['Show'], errors='ignore').to_csv(index=False),
                           file_name='nmr_peaks.csv', mime='text/csv')
