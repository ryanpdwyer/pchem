import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pchem.spectra import (read_raman, save_annotations, load_annotations, relink_annotations, StaleAnnotations,
                           sidecar_path, reference_key, resolve_reference, subtract_reference, pick_peaks)

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/raman/archived_dce_candidate.csv'


def test_real_enlighten():
    df = read_raman(EXAMPLE)
    assert len(df) == 1024
    assert df.attrs['enlighten']['Integration Time'] == '70'
    assert df.attrs['enlighten']['Scan Averaging'] == '20'
    band = df[df.Wavenumber.between(740, 770)]
    assert band.loc[band.Processed.idxmax(), 'Wavenumber'] == pytest.approx(755.09)
    vanilla = pd.read_csv(EXAMPLE, skiprows=34)
    np.testing.assert_array_equal(df.Processed, vanilla.Processed)


def test_sidecar_roundtrip_preserves_raw_and_other_fields(tmp_path):
    raw = tmp_path / 'sample.csv'
    raw.write_bytes(EXAMPLE.read_bytes())
    original = raw.read_bytes()
    save_annotations(raw, {'sample_name':'DCE, hexane', 'notes':'First\nsecond', 'peaks':[]})
    save_annotations(raw, {'peaks':[{'wavenumber':655.,'intensity':10.,'label':'gauche'}]})
    data = load_annotations(raw)
    assert data['notes'] == 'First\nsecond'
    assert data['sample_name'] == 'DCE, hexane'
    assert data['peaks'][0]['label'] == 'gauche'
    assert raw.read_bytes() == original
    assert sidecar_path(raw).is_file()


def test_corrupt_or_changed_data_does_not_overwrite_notes(tmp_path):
    raw = tmp_path/'sample.csv'
    raw.write_text('original')
    save_annotations(raw, {'notes':'retain'})
    before = sidecar_path(raw).read_bytes()
    raw.write_text('changed')
    with pytest.raises(StaleAnnotations, match='changed') as info:
        save_annotations(raw, {'notes':'replace'})
    assert info.value.data['notes'] == 'retain'  # Stale notes stay readable for display/relinking.
    assert sidecar_path(raw).read_bytes() == before
    sidecar_path(raw).write_text('invalid json')
    with pytest.raises(ValueError):
        save_annotations(raw, {'notes':'replace'})
    assert sidecar_path(raw).read_text() == 'invalid json'


def test_subtraction_interpolates_only_common_domain_and_preserves_inputs():
    sample = pd.DataFrame({'Wavenumber':[0.,1.,2.,3.,4.], 'Processed':[1.,3.,5.,7.,9.]})
    reference = pd.DataFrame({'Wavenumber':[1.,3.], 'Processed':[1.,3.]})
    before = sample.copy()
    result = subtract_reference(sample,reference,2)
    np.testing.assert_allclose(result.Difference,[1,1,1])
    assert result.Wavenumber.tolist() == [1,2,3]
    pd.testing.assert_frame_equal(sample,before)
    with pytest.raises(ValueError):
        subtract_reference(sample,reference.assign(Wavenumber=[10.,11.]))


def test_peak_distance_is_wavenumber_not_index():
    result = pick_peaks([0,1,2,3,4,5,6],[0,5,0,4,0,3,0],1,3)
    assert result.wavenumber.tolist() == [1,5]


def test_rejects_other_csv(tmp_path):
    f=tmp_path/'notes.csv'; f.write_text('sample,notes\na,b\n')
    with pytest.raises(ValueError,match='Enlighten'):
        read_raman(f)


def test_relink_accepts_changed_raw_and_records_old_fingerprint(tmp_path):
    raw = tmp_path/'sample.csv'
    raw.write_text('original')
    first = save_annotations(raw, {'notes':'keep me', 'peaks':[{'wavenumber':1.,'intensity':2.,'label':'x'}]})
    raw.write_text('re-exported')
    with pytest.raises(StaleAnnotations):
        load_annotations(raw)
    data = relink_annotations(raw)
    assert data['notes'] == 'keep me' and data['peaks'][0]['label'] == 'x'
    assert data['previous_sha256'] == [first['sha256']]
    assert load_annotations(raw)['sha256'] != first['sha256']
    save_annotations(raw, {'notes':'after relink'})  # Saving works again.
    assert load_annotations(raw)['notes'] == 'after relink'
    assert relink_annotations(raw)['previous_sha256'] == [first['sha256']]  # No-op when already current.
    with pytest.raises(ValueError, match='No annotations'):
        relink_annotations(tmp_path/'unannotated.csv')


def test_reference_key_is_relative_and_survives_moving_the_folder(tmp_path):
    (tmp_path/'a'/'2026-09-08').mkdir(parents=True)
    sample = tmp_path/'a'/'2026-09-08'/'sample.csv'
    reference = tmp_path/'a'/'ref.csv'
    key = reference_key(sample, reference)
    assert not Path(key).is_absolute()
    assert resolve_reference(sample, key) == reference.resolve()
    moved_sample = tmp_path/'b'/'2026-09-08'/'sample.csv'
    assert resolve_reference(moved_sample, key) == (tmp_path/'b'/'ref.csv').resolve()
    absolute = str(reference.resolve())  # Sidecars written by the first version.
    assert resolve_reference(sample, absolute) == reference.resolve()
