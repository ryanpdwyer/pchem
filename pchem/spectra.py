"""Local Enlighten spectra and nondestructive per-file annotations."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

SCHEMA = 'pchem.spectrum.v1'


class StaleAnnotations(ValueError):
    """A sidecar exists, but the raw spectrum changed since it was saved."""

    def __init__(self, message, data):
        super().__init__(message)
        self.data = data


def read_raman(path):
    """Read a column-oriented Enlighten CSV, retaining original columns/settings.

    No normalization, smoothing, baseline or extra dark subtraction is applied.
    """
    path = Path(path)
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    metadata = {}
    for i, line in enumerate(lines):
        fields = next(csv.reader([line]))
        if 'Wavenumber' in fields and 'Processed' in fields:
            break
        if len(fields) >= 2 and fields[0]:
            metadata[fields[0]] = fields[1]
    else:
        raise ValueError('Expected a column-oriented Enlighten CSV with Wavenumber and Processed columns.')
    df = pd.read_csv(io.StringIO('\n'.join(lines[i:])))
    for column in ('Wavenumber', 'Processed'):
        df[column] = pd.to_numeric(df[column], errors='raise')
    if len(df) < 2 or not np.isfinite(df[['Wavenumber', 'Processed']]).all().all():
        raise ValueError('Spectrum must contain at least two finite data points.')
    df = df.sort_values('Wavenumber').reset_index(drop=True)
    if df.Wavenumber.duplicated().any():
        raise ValueError('Duplicate wavenumbers are not supported.')
    df.attrs['enlighten'] = metadata
    df.attrs['source'] = str(path.resolve())
    return df


def sidecar_path(path):
    path = Path(path)
    return path.with_name(path.name + '.annotations.json')


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_sidecar(path):
    """Read the companion JSON without checking it against the raw file."""
    sidecar = sidecar_path(path)
    if not sidecar.exists():
        return {}
    data = json.loads(sidecar.read_text(encoding='utf-8'))
    if data.get('schema') != SCHEMA:
        raise ValueError('Unrecognized annotation format; existing sidecar was not changed.')
    return data


def load_annotations(path):
    data = read_sidecar(path)
    if data and data.get('sha256') != fingerprint(path):
        raise StaleAnnotations('Raw spectrum changed since annotations were saved; '
                               'existing notes were not changed.', data)
    return data


def _write_sidecar(path, data):
    """Atomically replace the companion JSON file; never modify the source spectrum."""
    path = Path(path)
    data = {**data, 'schema': SCHEMA, 'source_file': path.name, 'sha256': fingerprint(path),
            'updated_utc': datetime.now(timezone.utc).isoformat()}
    target = sidecar_path(path)
    fd, temporary = tempfile.mkstemp(prefix='.pchem-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return data


def save_annotations(path, annotations):
    """Merge annotations into the companion JSON file for a spectrum.

    Refuses to overwrite a malformed sidecar or one saved for a different
    version of the raw file (see relink_annotations).
    """
    old = load_annotations(path)
    return _write_sidecar(path, {**old, **annotations})


def relink_annotations(path):
    """Accept the current raw file as the source of an existing sidecar.

    The previous fingerprint is kept in ``previous_sha256`` so the change is
    recorded rather than silently forgotten.
    """
    data = read_sidecar(path)
    if not data:
        raise ValueError('No annotations exist for this spectrum.')
    if data.get('sha256') == fingerprint(path):
        return data
    history = list(data.get('previous_sha256', [])) + [data.get('sha256')]
    return _write_sidecar(path, {**data, 'previous_sha256': history})


def reference_key(path, reference):
    """Path of a reference spectrum relative to the annotated spectrum's folder.

    Relative paths survive copying or moving the whole data folder. Falls
    back to an absolute path when no relative path exists (e.g. Windows drives).
    """
    reference = Path(reference).resolve()
    try:
        return os.path.relpath(reference, Path(path).resolve().parent)
    except ValueError:
        return str(reference)


def resolve_reference(path, key):
    """Inverse of reference_key; absolute keys from older sidecars still work."""
    return (Path(path).resolve().parent / key).resolve()


def subtract_reference(sample, reference, scale=1.0):
    """Sample - scale*reference on sample points in their shared range only."""
    if not np.isfinite(scale) or scale < 0:
        raise ValueError('Reference scale must be finite and nonnegative.')
    lo = max(sample.Wavenumber.min(), reference.Wavenumber.min())
    hi = min(sample.Wavenumber.max(), reference.Wavenumber.max())
    out = sample.loc[sample.Wavenumber.between(lo, hi), ['Wavenumber', 'Processed']].copy()
    if len(out) < 2:
        raise ValueError('Spectra do not have sufficient overlapping wavenumber coverage.')
    out['Scaled reference'] = scale * np.interp(out.Wavenumber, reference.Wavenumber, reference.Processed)
    out['Difference'] = out.Processed - out['Scaled reference']
    return out


def pick_peaks(x, y, prominence, separation):
    """Pick maxima using absolute prominence and separation in cm^-1."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3:
        return pd.DataFrame(columns=['wavenumber', 'intensity', 'label'])
    candidates, _ = find_peaks(y, prominence=prominence)
    kept = []
    for index in sorted(candidates, key=lambda j: y[j], reverse=True):
        if all(abs(x[index] - x[j]) >= separation for j in kept):
            kept.append(index)
    kept.sort()
    return pd.DataFrame({'wavenumber': x[kept], 'intensity': y[kept], 'label': [''] * len(kept)})
