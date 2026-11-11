# Combine datacard and shape viewer

A local Streamlit app for browsing Combine TH1 templates without installing ROOT.
Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required. Dependencies are locked
in `uv.lock`; no JavaScript build, database, or running Combine environment is needed.

## Run

Run directly from GitHub, without cloning the repository or managing an environment:

```bash
uvx --from git+https://github.com/tcuisset/datacard-viewer.git view-card --card something.txt
```

All card options below also work with `view-card`, without an extra `--` separator.
Relative card paths resolve from your current directory. `uvx` downloads the app and
dependencies into its managed cache; it does not install them into your project.
Use `view-card --help` after the `uvx --from ...` prefix to list options.
To refresh a cached Git checkout, add `--refresh` to `uvx`.

From a local checkout (`uv run` installs the package into its project environment):

```bash
uv run --locked --no-dev view-card --card something.txt
```

Open http://localhost:8501. Use **Open / Reload** in the sidebar to change cards,
file mappings, mass, or substitution keywords. With no `--card`, the app opens a
blank path selector. Paths identify files on the machine running Python.

The combined example references an absent `output/` subdirectory. Map that prefix
to the datacard directory explicitly:

```bash
uv run --locked --no-dev view-card --card dPhijj_datacards/combined_allfs_dPhijj_v8.0.txt --path-map 'output/='
```

Mappings use `prefix=replacement`; the first matching prefix wins. Relative results
resolve against the card's directory. For example `--path-map 'output/=/data/shapes/'`
uses an absolute replacement directory. Multiple `--path-map` options are allowed.
Use `--mass 125` for `$MASS` and repeat `--keyword NAME=value` for extra substitutions.
The UI accepts the same mappings/keywords, one per line. Built-in keywords cannot
be overridden. Input files are never rewritten.

To compare the original combined card with an optimizer output, pass the second
card and its own shape-file mappings. The original card uses `output/` references;
the optimized card points to `optimized.root` beside it:

```bash
uv run --locked --no-dev view-card \
  --card dPhijj_datacards/combined_allfs_dPhijj_v8.0.txt \
  --path-map 'output/=' \
  --compare-card dPhijj_datacards/optimized.txt
```

The command also works with historical `optimized.txt` cards or output from another
optimizer. Choose **Compare** after opening both cards. **Card edits** offers nuisance
declaration changes and `shape?` effects for ROOT inspection. **All effects** also
includes retained declarations. Choose a nuisance to load its templates. The
**Datacard differences** panel below the results lists added and removed channels,
process changes in the selected channels, and the card-wide nuisance shortlist.
Retained shape nuisances can still have different ROOT templates.

The ratio panel labels each channel by the effects used to build its curves,
with separate Before and After labels when the cards are overlaid. A `shape?`
effect that uses its log-normal fallback is labeled log-normal. Sums with both
effect types are labeled mixed. These labels describe the local effects before
plot normalization, and the ratio tooltips show the same information.

### Remote machines

Run the same command on the machine storing your cards, then forward its port:

```bash
ssh -N -L 8501:127.0.0.1:8501 user@remote-host
```

Open http://localhost:8501 locally. The app binds to loopback by default. All plot
assets are supplied locally; normal use needs no internet after dependency installation.
For `view-card`, set `STREAMLIT_SERVER_PORT=8502` to change the port, or use
other standard Streamlit environment variables to override server settings.

## Select shapes

Both modes use the same controls. Choose one or more **Channels** in left-to-right
plot order. Remove and re-add a channel to move it to the end. In Compare, the
channel options are the exact names shared by both cards.

To select channels in bulk, open **Select channels by name**, enter part of a
channel name, and click **Select matching channels**. The button replaces the
selection with all case-insensitive substring matches in datacard order. You can
edit the result in the multiselect.

Choose one or more **Processes** to sum. **Backgrounds**, **Signals**, and
**All** fill the selector, which you can then edit. Backgrounds have
positive process IDs, and signals have zero or negative IDs. In Compare, a
process name qualifies for a preset if either card classifies it that way.
Processes absent from a selected channel or card contribute zero yield there.

Choose a **Nuisance** directly or step through the list with the arrows next to
the picker. **Filter nuisances** narrows the list by datacard group or name.

Under **Plot options**, **Hide unaffected channels** shows only channels where
Up or Down changes a bin yield in the selected process sum before normalization.
Compare keeps a channel when either card shows a change. Normalization, yield
summaries, and scans still use the full channel selection. Plot downloads reflect
the visible channels.

Set **ROOT smoothing passes** under **Plot options** to apply ROOT's
`TH1::SmoothArray` algorithm in either mode. The default of zero disables
smoothing. The viewer smooths **Up / Nominal** and **Down / Nominal** separately
within each channel and card, then multiplies the ratios by the original
nominal to recover the plotted variation yields. When smoothing is enabled, the
ratio panel overlays the unsmoothed ratios. Zero nominal bins keep their
original variation yields and split the ratio into separate segments. Smoothing
requires at least three bins per segment. When normalization removal is enabled,
the smoothed and unsmoothed curves are normalized separately. Statistical bands
use stored bin errors, scaled if plot normalization is enabled. Yield summaries, diagnostics,
unaffected-channel filtering, and scans use the original templates.

## Inspect shapes

Each channel occupies a separate segment in both panels, with its own physical
observable tick labels, channel header, and boundary. Segment widths are proportional
to the observable ranges; bin widths are preserved within each segment. Tooltips
identify the channel and original bin interval. Curves and uncertainty bands stop
at channel boundaries.

The nuisance list includes shape effects declared in any selected channel.
Unaffected channels keep their nominal shapes. A missing or unreadable template
stops the plot and identifies the channel.

Under **Plot options**, **Remove normalization effect** applies one common Up
scale and one common Down scale using the totals across all selected channels. Relative yield
changes between channels remain visible. Integral metrics show raw totals, and the
table shows each channel's raw yields and diagnostics. Diagnostic shape residuals
continue to remove normalization independently within each channel and are distinct
from this global plotting normalization.

CSV exports include channel names, display order, original bin edges, and shifted
display coordinates. HTML exports preserve the segmented figure. **Scan selected
channels** scans all selected channels; whole-card scans are unchanged.

The upper panel overlays nominal, Up, and Down bin yields using physical bin edges.
The lower panel shows variation/nominal. Its y-axis fits the finite Up/nominal and
Down/nominal curves, including the reference at one, with 10% padding. Statistical
bands can extend beyond this range. PNG export is in Plotly's toolbar. The buttons export plotted
bin contents and variances as CSV, or an interactive standalone HTML plot.

Nominal statistical bands use the histogram's stored errors (`Sumw2`), which can
also describe data-driven templates. Up/Down bands are optional. If `Sumw2` is absent,
the reader identifies uproot's Poisson variance fallback. The ratio band is the
nominal fractional error reference, **not** the propagated uncertainty on a ratio.
Normalized bands scale stored errors and do not include normalization covariance.
Process sums add variances in quadrature, assuming independent process templates.

Unaffected processes remain nominal in varied sums. Raw histogram normalization is
preserved: datacard rate mismatches are warned about, and `rate -1` derives its yield
from the histogram. Underflow/overflow are excluded from integrals and plots, with
nonzero contents reported. Negative bins remain visible on a linear y-axis; on a
log y-axis, zero and negative template bins appear as gaps, with their counts
reported above the plot. The table and CSV retain the raw values. Statistical bands
are shown only where the full interval is positive on a log y-axis. Zero
denominators have masked ratios and red markers. Incompatible binning or missing
variations produce an explicit error rather than a partial sum.

Non-unit shape coefficients are displayed, but plotted curves remain stored
endpoints rather than effective ±1σ predictions. `shape?` uses the documented
log-normal fallback only when both templates are absent; one missing template is an
error. Renamed nuisances retain their original template mappings.

## Diagnose nuisances

Start a selected-channel or whole-card scan explicitly. A single background worker
keeps navigation available. Progress, cancellation, and completed pair counts are
shown. Cancelled scans retain their partial results and can be exported.

Each affected channel/process pair reports the integral factors κup/κdown,
maximum fractional residual and nominal-yield-weighted RMS after removing
normalization, and the largest normalized residual relative to the nominal stored
statistical error. The latter is descriptive: nominal/variation event correlations
are unavailable, so it is **not a significance or statistical compatibility test**.

A nuisance is a normalization-only review candidate only if every expected pair
passes the configurable maximum-residual threshold (default 1%) and has valid,
nonnegative templates without newly populated zero-nominal bins. The summary shows
coverage, the worst pair, warnings, and failures. Process-level screening prevents
cancellation in summed shapes from hiding genuine distortions.

CSV exports contain pair-level results and a nuisance summary. For complete candidate
nuisances, unit-coefficient pairs include a suggested asymmetric `lnN` factor in
`down/up` order. These are review suggestions; the app never edits cards. Channel
scan suggestions only cover that channel, not the rest of the card. Input changes
mark scans stale and disable their suggestions; **Open / Reload**, then rescan.

The overview displays rates, nuisance types/groups, file resolution, `autoMCStats`,
and model metadata. This is a prefit **template inspector**. It does not evaluate
rate parameters, external arguments, constraints, analysis-specific physics models,
automatically generated `autoMCStats` nuisance parameters, or fitted workspaces.
RooWorkspace/parametric inputs, exact interpolation, postfit predictions, and
multidimensional histograms are outside the first version.

## Development and validation

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
```

Tests use small weighted ROOT fixtures and, when present, the supplied local analysis
cards. `hist` is a test dependency, not an application dependency. The parser is
vendored from Combine v11.1.0; source and license details are in
`datacard_viewer/vendor/README.md`. Application code is divided into the independent
card/template/scan engine, plotting/export functions, and the Streamlit UI.

Reader caches are bounded to 128 MiB (including array/object overhead), eight open files, and 64 directory
metadata entries per reader. The scan has its own reader, checks cancellation between
pairs, retains scalar results, and closes its files when finished. Cards and files
are tracked by resolved path, size, and nanosecond modification time; Reload also
clears the card cache.
