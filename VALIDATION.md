# Validation — 27 October 2026

Validated on the supplied local examples with the locked Python 3.11 environment.
No input datacards or ROOT files were modified.

- `uv run pytest -q`: 16 passing cases, including weighted ROOT fixtures, both
  supplied cards, scan failure/cancellation/staleness, CSV suggestions, and a
  Streamlit interaction test.
- `uv run ruff check .`: passed. Vendored upstream files are excluded from lint.
- Vendored sources compared with the local Combine v11.1.0 checkout: only the two
  documented parser imports differ; the license is byte-identical.
- Combined-card parse/index: 0.54 seconds.
- Streamlit AppTest, combined card: initial UI execution 1.27 seconds;
  background grouping 0.10 seconds, normalization toggle 0.11 seconds, next
  nuisance 0.10 seconds. These are server-side UI timings, not network latency.
- Browser: both supplied cards rendered, including a background sum with a jet-scale variation and statistical bands. Combined-card plots rendered; nuisance navigation worked during an
  active whole-card scan; cancellation was exercised on an earlier scan.
- Final browser scan completed all 211,772 affected pairs in 350 seconds with
  progress updates and CSV download controls. The worker records failures and
  invalid diagnostics as rows rather than aborting.
- Server `/proc` high-water memory, including full-card export generation:
  1,018,500 KiB (994.6 MiB), below the 1 GiB acceptance target. Subsequent export
  code streams CSV and avoids large cache-data serialization copies to reduce
  temporary allocations further.

The initial reader prototype exceeded the memory target when it counted only
numeric buffer sizes. The shipped cache also accounts for Python/NumPy objects,
provenance strings, and cache overhead. Cached directory metadata prevents repeatedly
parsing ROOT key lists. Re-running timings on other cards/storage is advised.

## Multi-channel display — 28 October 2026

- Full suite: 22 passing cases; lint and whitespace checks pass.
- Added deterministic checks for unequal observable ranges, variable bin widths,
  repeated ticks, channel order, independent band segments, and original-coordinate
  tooltips/exports. Shared normalization preserves channel migrations and scales
  variances by the square of the common factor.
- Streamlit tests cover unaffected channels, common-process selection, empty
  selections, selected-channel scan scope, switching back to single-channel mode,
  and a missing template preventing partial combined plots.
- Fixed and regression-tested the pre-existing same-card Reload path: it must
  recreate a reader after clearing it, even if its identity is unchanged.
- On the combined example, server-side AppTest timings for four backgrounds-sum
  channels were 1.43 seconds for the initial selection, 0.79 seconds for shared
  normalization, and 0.90 seconds for moving to the next nuisance.
- Browser inspection exercised two, three, and four channels, including a jet-scale
  nuisance with summed backgrounds. Channel headers and repeated physical ticks
  are visible in both aligned panels; curve/band segments remain separated.
