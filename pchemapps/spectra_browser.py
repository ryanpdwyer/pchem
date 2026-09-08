"""Run locally: streamlit run pchemapps/spectra_browser.py --server.address 127.0.0.1"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pchem.spectra import (read_raman, load_annotations, save_annotations, relink_annotations,
                           StaleAnnotations, reference_key, resolve_reference, pick_peaks,
                           subtract_reference)

PEAK_COLUMNS = ['wavenumber', 'intensity', 'label']


def valid_peaks(table):
    """Rows of a peak table with finite numeric coordinates, for plotting."""
    table = pd.DataFrame(table, columns=PEAK_COLUMNS)
    for column in ('wavenumber', 'intensity'):
        table[column] = pd.to_numeric(table[column], errors='coerce')
    table = table[np.isfinite(table[['wavenumber', 'intensity']]).all(axis=1)].copy()
    table['label'] = table.label.fillna('').astype(str)
    return table


def run():
    st.set_page_config(page_title='Spectra notebook', layout='wide')
    st.title('Spectra notebook')
    folder = Path(st.sidebar.text_input('Spectra folder', value=str(Path.cwd())).strip()).expanduser()
    recursive = st.sidebar.checkbox('Include dated subfolders', value=True)
    st.sidebar.button('Refresh files')
    st.sidebar.caption('Raw files stay unchanged. Save names, notes and peak labels beside each spectrum.')
    if not folder.is_dir():
        st.info('Enter a folder on the computer running this app.')
        return
    try:
        paths = sorted((p for p in (folder.rglob('*') if recursive else folder.iterdir())
                        if p.is_file() and p.suffix.lower() == '.csv'),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError as error:
        st.error(str(error))
        return
    spectra, notes, stale, problems = {}, {}, {}, []
    for path in paths:
        try:
            spectra[path] = read_raman(path)
        except (ValueError, OSError, UnicodeError, pd.errors.ParserError) as error:
            problems.append(f'{path.name}: {error}')
            continue
        try:
            notes[path] = load_annotations(path)
        except StaleAnnotations as error:
            problems.append(f'{path.name}: {error}')
            notes[path], stale[path] = {}, error.data
        except (ValueError, OSError) as error:
            problems.append(f'{path.name}: {error}')
            notes[path] = {}
    if problems:
        with st.expander(f'{len(problems)} file notices'):
            st.write('\n\n'.join(problems))
    if not spectra:
        st.info('No supported Enlighten CSV spectra found. Export one spectrum per file, in columns, including Wavenumber and Processed.')
        return
    def display(path):
        label = notes[path].get('sample_name', '')
        return f'{label + " — " if label else ""}{path.relative_to(folder)}'
    path = st.sidebar.selectbox('Active spectrum', list(spectra), format_func=display)
    df, saved = spectra[path], notes[path]
    key = str(path.resolve())
    st.caption(str(path))
    if path in stale:
        old = stale[path]
        st.warning('This raw file changed after its notes were saved (for example, it was re-exported). '
                   'The saved notes are shown below for reference, and saving is blocked until you confirm '
                   'that they belong to the current file.')
        with st.expander('Saved notes for the previous version of this file', expanded=True):
            st.json({k: old.get(k) for k in ('sample_name', 'solvent', 'concentration', 'student',
                                             'notes', 'peaks', 'subtraction', 'updated_utc') if k in old})
        if st.button('Re-link these notes to the current file'):
            try:
                relink_annotations(path)
                st.rerun()
            except (ValueError, OSError) as error:
                st.error(str(error))
    metadata = df.attrs['enlighten']
    st.write({k: metadata[k] for k in ('Timestamp', 'Integration Time', 'Scan Averaging', 'Laser Power', 'Laser Power %') if k in metadata})
    with st.expander('All acquisition settings'):
        st.json(metadata)
    with st.form('notes-' + key):
        name = st.text_input('Sample name', value=saved.get('sample_name', ''))
        a, b, c = st.columns(3)
        solvent = a.text_input('Solvent', value=saved.get('solvent', ''))
        concentration = b.text_input('Concentration (include units)', value=saved.get('concentration', ''))
        student = c.text_input('Student / pair', value=saved.get('student', ''))
        text = st.text_area('Experimental notes', value=saved.get('notes', ''))
        if st.form_submit_button('Save sample information'):
            try:
                save_annotations(path, dict(sample_name=name, solvent=solvent,
                    concentration=concentration, student=student, notes=text))
                st.success('Sample information saved beside the raw spectrum.')
            except (ValueError, OSError) as error:
                st.error(str(error))
    lo, hi = float(df.Wavenumber.min()), float(df.Wavenumber.max())
    a, b = st.columns(2)
    start = a.number_input('Plot minimum (cm⁻¹)', value=lo, key='lo'+key)
    end = b.number_input('Plot maximum (cm⁻¹)', value=hi, key='hi'+key)
    if start >= end:
        st.error('Minimum must be less than maximum.')
        return
    others = st.multiselect('Overlay spectra', [p for p in spectra if p != path], format_func=display)
    plot_area = st.container()  # Filled after the peak table so unsaved peaks show on the plot.
    with st.expander('Pick and label peaks'):
        visible = df[df.Wavenumber.between(start, end)]
        prominence = st.number_input('Minimum prominence (counts)', min_value=0., value=float(np.ptp(df.Processed)*.03), key='prom'+key)
        separation = st.number_input('Minimum separation (cm⁻¹)', min_value=0., value=8., key='sep'+key)
        peaks_key = 'peaks'+key
        version_key = 'peak_version'+key
        if peaks_key not in st.session_state:
            st.session_state[peaks_key] = pd.DataFrame(saved.get('peaks', []), columns=PEAK_COLUMNS)
        if st.button('Detect peaks in plotted range'):
            st.session_state[peaks_key] = pick_peaks(visible.Wavenumber, visible.Processed, prominence, separation)
            st.session_state[version_key] = st.session_state.get(version_key, 0) + 1
        edited = st.data_editor(st.session_state[peaks_key], num_rows='dynamic',
            key='editor'+key+str(st.session_state.get(version_key, 0)))
        st.caption('Edit labels or add/remove rows. Positions and intensities describe the original spectrum. '
                   'The plot shows this table; peaks are stored only after saving.')
        if st.button('Save peak annotations'):
            try:
                values = edited.copy()
                for column in ('wavenumber', 'intensity'):
                    values[column] = pd.to_numeric(values[column], errors='raise')
                if not np.isfinite(values[['wavenumber', 'intensity']]).all().all():
                    raise ValueError('Peak coordinates must be finite numbers.')
                if not values.wavenumber.between(lo, hi).all():
                    raise ValueError('Peak positions must lie within the spectrum.')
                values['label'] = values.label.fillna('').astype(str)
                save_annotations(path, {'peaks': values.to_dict('records')})
                st.session_state[peaks_key] = values
                st.rerun()
            except (ValueError, OSError) as error:
                st.error(str(error))
        st.download_button('Download peak table', edited.to_csv(index=False), file_name=path.stem+'.peaks.csv')
    with plot_area:
        fig = go.Figure()
        fig.add_scatter(x=df.Wavenumber, y=df.Processed, name=name or path.name, mode='lines')
        for other in others:
            frame = spectra[other]
            fig.add_scatter(x=frame.Wavenumber, y=frame.Processed, name=display(other), mode='lines')
        for peak in valid_peaks(edited).itertuples():
            fig.add_annotation(x=peak.wavenumber, y=peak.intensity,
                               text=peak.label or f'{peak.wavenumber:.1f}', showarrow=True)
        fig.update_layout(xaxis_title='Raman shift (cm⁻¹)', yaxis_title='Processed signal (counts)',
                          xaxis_range=[start, end], height=470)
        st.plotly_chart(fig, width='stretch')
        # Callable: the standalone HTML (with embedded plotly.js) is built only when clicked.
        st.download_button('Download interactive plot', lambda: fig.to_html().encode(),
                           file_name=path.stem+'.plot.html')
        st.caption('Overlays use original Processed values. Different exposure times or laser powers can change heights; no automatic normalization is applied.')
    with st.expander('Subtract a reference'):
        candidates = [p for p in spectra if p != path]
        if not candidates:
            st.info('Add a second spectrum to use as a reference.')
        else:
            recipe = saved.get('subtraction', {})
            saved_reference = resolve_reference(path, recipe['reference']) if 'reference' in recipe else None
            default_reference = next((i for i, p in enumerate(candidates) if p.resolve() == saved_reference), 0)
            reference = st.selectbox('Reference spectrum', candidates, index=default_reference, format_func=display, key='ref'+key)
            scale = st.number_input('Reference multiplier', min_value=0., value=float(recipe.get('scale', 1.)), step=.05, key='scale'+key)
            st.caption('Difference = active spectrum − multiplier × reference. Check acquisition settings and matching background treatment. Interpolation is limited to shared coverage.')
            try:
                result = subtract_reference(df, spectra[reference], scale)
                subfig = go.Figure()
                for column in ('Processed', 'Scaled reference', 'Difference'):
                    subfig.add_scatter(x=result.Wavenumber, y=result[column], name=column)
                subfig.update_layout(xaxis_title='Raman shift (cm⁻¹)', yaxis_title='Counts', xaxis_range=[start,end])
                st.plotly_chart(subfig, width='stretch')
                export = result.assign(source_file=path.name, reference_file=str(reference.relative_to(folder)), reference_scale=scale)
                st.download_button('Download subtraction CSV', export.to_csv(index=False), file_name=path.stem+'.subtracted.csv')
                if st.button('Save subtraction recipe with sample'):
                    save_annotations(path, {'subtraction': {'reference': reference_key(path, reference), 'scale': scale}})
                    st.success('Recipe saved. Raw data are unchanged.')
            except (ValueError, OSError) as error:
                st.error(str(error))
    # Reload saved notes so downloads include edits from this run.
    summary = []
    for source in spectra:
        try:
            item = load_annotations(source)
        except (ValueError, OSError):
            item = {}
        summary.append({'file': str(source.relative_to(folder)), **{k:item.get(k,'') for k in
            ('sample_name','student','solvent','concentration','notes','updated_utc')}})
    st.download_button('Download sample log CSV', pd.DataFrame(summary).to_csv(index=False), file_name='sample-log.csv')


if __name__ == '__main__':
    run()
