# Local spectra notebook

Runs on the instrument computer and reads its Enlighten output folder, including dated subfolders. Plot and overlay spectra, name samples, record solvent/concentration/student/notes, pick and label peaks, and subtract scaled reference spectra. Integration stays in the student Python notebook.

## Run

From this checkout, with Python 3.10 or later:

```bash
python -m pip install -r spectra-browser-requirements.txt
python -m streamlit run pchemapps/spectra_browser.py --server.address 127.0.0.1
```

Open the localhost URL printed by Streamlit.

### Windows instrument computer (micromamba)

Everything installs per-user; no administrator rights are needed. Internet is needed only during these steps.

1. **micromamba**, if not already installed. In PowerShell:

   ```powershell
   Invoke-Expression ((Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1 -UseBasicParsing).Content)
   ```

   Accept the defaults, then close and reopen PowerShell so `micromamba` is on the path.
2. **The app.** Download the branch as a zip from GitHub and unzip it, e.g. to `C:\Users\<lab>\spectra-browser`. Git is not required.
3. **Python and packages.** From that folder:

   ```powershell
   .\scripts\spectra-browser.ps1 -Install
   ```

   This creates the `py314` environment (Python 3.14 from conda-forge) if it does not exist, installs the packages from `spectra-browser-requirements.txt` with pip, and starts the app. If PowerShell refuses to run scripts, use `Spectra Browser.bat -Install` instead, or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
4. **Daily use.** Double-click `Spectra Browser.bat`, or run `.\scripts\spectra-browser.ps1`. A browser tab opens automatically. Close the PowerShell window to stop the app.

Use `-Env <name>` (or `$env:SPECTRA_ENV`) for a different environment name and `-Python 3.12` to create it with another version. The folder box defaults to `Documents\EnlightenSpectra`, Enlighten's save location with dated subfolders, when it exists. To update the app, replace the folder with a new zip and rerun with `-Install`; annotations live beside the spectra, not in the app folder.

## Student workflow

1. Choose a spectrum (newest first), enter a useful sample name, solvent, concentration with units, student/pair, and experimental notes. Click **Save sample information** before switching files.
2. Set the plot range and select overlays. Pan/zoom and hover for coordinates; download the interactive plot if useful.
3. Detect peaks in the selected range, edit their labels or add/remove rows, then **Save peak annotations**. Detection uses original data, absolute prominence, and separation in cm⁻¹; no smoothing is applied. The plot annotates whatever is in the peak table, saved or not.
4. Select a reference and multiplier to preview sample minus reference. Download the derived CSV and optionally save the recipe with the active sample. No extrapolation outside shared coverage is allowed.
5. Download the sample log CSV for a spreadsheet summary.

## Files and persistence

For `enlighten-example.csv`, notes are saved as `enlighten-example.csv.annotations.json` in the same directory. Each companion includes the source filename, SHA-256 fingerprint, timestamps, sample fields, saved peaks, and optionally a subtraction recipe. Preserve both files when copying data. Raw CSV files are never renamed or edited. Saves use an atomic temporary-file replacement. A malformed sidecar blocks overwriting it. If the raw file changes after notes were saved (for example, it was re-exported), the app shows the saved notes read-only and blocks saving until you click **Re-link these notes to the current file**; the old fingerprint is kept in `previous_sha256`. A saved subtraction recipe stores the reference as a path relative to the annotated spectrum, so a copied or moved data folder keeps working. One operator/browser session per folder is recommended; simultaneous edits to the same field are not merged.

Names are display labels, not filename changes. The CSV sample-log download is a report; the per-spectrum JSON files are the persistent source of annotations. Downloads go to the browser's configured download location; subtraction never overwrites source spectra.

## Supported data and scaling

First version supports individual, column-oriented Enlighten CSV files containing `Wavenumber` and `Processed`. Other CSVs are skipped with a notice; IR vendor formats and row/session exports are not yet supported. Metadata are preserved as strings in `df.attrs['enlighten']`. No normalization, additional dark subtraction, baseline subtraction, or smoothing is silently applied. `Processed` may already include processing enabled during acquisition. Matching exposure, laser settings, and background treatment matter for overlays/subtraction. Scan averaging is not an extra exposure-time multiplier. Band ratios within one spectrum do not require exposure normalization.

## Plain pandas teaching example

`examples/raman/vanilla_dce.ipynb` uses `pd.read_csv(..., skiprows=34)` on the bundled archived file, then plots and interpolates it. Students inspect the first lines and count the metadata rows once. The reader method belongs to pandas (`pd.read_csv`), not a DataFrame (`df.read_csv`). Header lengths can change with Enlighten versions/export settings, so update the skip count for newly collected files.

The bundled 2022 example is the *suspected* DCE spectrum originally labeled 2-chloropropane. It is included to exercise file loading and plotting, not as a certified chemical reference. Replace it with the newly collected, correctly identified spectrum for class. Original measurement ID: 20220601-154022-781259-WP-01098; 70 ms, 20 averages.

For automated folder browsing, `pchem.spectra.read_raman` finds the header instead of hard-coding the count. Existing pchem upload apps are unchanged. This local filesystem app is deliberately not added to the public app dispatcher.
