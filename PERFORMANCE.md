# Large-card CPU investigation (2026-11-11)

The reported minute-long CPU stall is reproducible without a nuisance scan or
filesystem latency. The main cause is formatting uproot missing-key exceptions.

## Reproduction and evidence

Local input: `HpT_datacards/combined_allfs_HpT_v8.2.txt`, 151,798,625 bytes
(144.8 MiB), 154 channels, 3,102 process columns, 1,392 active supported
nuisance names and 261,311 active declarations. Its optimized directory contains
154 ROOT files. The benchmark uses the same card in both Compare slots, the first
30 channels, all available processes, All effects, and the `shape?` nuisance
`CMS_HIG25019_DY_pTweight_10_2022EE`.
Reader limits are `--max-files 1000 --max-mb 5000`.

Measured with Streamlit AppTest and cProfile around the viewer's actual fragment,
including fragment-only widget reruns (Streamlit 1.66.1, uproot 5.7.7):

| Operation | Before | After both fixes |
| --- | ---: | ---: |
| Select nuisance / first plot | 97.9 s | 4.9 s |
| Toggle log y with warm templates | 99.2 s | 1.8 s |

Before the fix, the toggle called `KeyInFileError.__str__` 208 times, spending
96.7 seconds there. It performed 52,192 Damerau–Levenshtein string comparisons.
Periodic stack samples independently located the active thread in that code.
These are CPU computations, not ROOT decompression or disk reads. Profiling adds
substantial overhead to these Python loops; timings are local comparisons, not
browser latency promises.

A separate unprofiled run of the optional real-card test below, using its default
`CMS_HIG25019_DY_pTweight_10_2022` nuisance, measured 1.94 s opening cards,
1.38 s drawing the first plot, and 0.47/0.49 s toggling log y on/off. CPU times
were approximately the same as wall times. AppTest covers server execution and
plot serialization, but not browser rendering, networking or timer scheduling.

## Causes and fixes

1. `shape?` deliberately probes for Up/Down templates and uses a log-normal
   fallback when both are absent. The comparison code converted each caught
   exception to a string, even though fallback warnings do not need that string.
   Uproot's formatter sorts available keys by edit distance to suggest similar
   names. Many long systematic names make this extremely expensive. Every toggle
   re-entered the same code despite a warm numeric template cache.

   `Reader.read` now converts `KeyInFileError` to a plain `KeyError` at the ROOT
   lookup boundary. The message retains the file, requested full object path and
   missing key, without ranking alternatives. This also covers missing nominal
   histograms, missing directories, and scan failures. Fallback rules and genuine
   missing-endpoint errors are unchanged. Increasing file/template cache limits
   cannot remove this formatting cost.

2. Plotly's incremental `add_annotation` and `add_vline` calls repeatedly validate
   the existing layout collections. That makes construction grow faster than the
   number of channels. Labels and separators are now collected and assigned once
   per figure in both Inspect and Compare, including separate plots for mismatched
   binning. Synthetic 30-channel profiled figure construction fell from 3.21 to
   0.53 s in Inspect and from 1.21 to 0.65 s in Compare. A before/after JSON check
   matched for linear/log axes, statistical bands and smoothing overlays.

The datacard was not reparsed on plot toggles. Scans start only on a scan button
press. File/template limits are per reader; they do not control exception formatting
or plot construction. Low-CPU filesystem waits were not investigated here.

## Regression coverage

`tests/test_performance.py` runs by default with generated fixtures. It covers:

- A ROOT directory with 1,100 similarly named keys and a long nuisance name;
  uproot's suggestion formatter is forbidden, so regressions fail immediately
  instead of burning minutes. Both Inspect and Compare fallbacks are checked,
  along with missing nominal objects and directories.
- An actual 30-channel Compare UI, including log toggles, cache reuse, no card
  reparsing or ROOT file reopening, and correct serialized axis settings.
- 100-channel Inspect/Compare plots, counting validated layout objects to enforce
  linear construction work without a machine-dependent timing threshold.

A real-card check is opt-in because analysis inputs are not distributed:

```bash
DATACARD_VIEWER_LARGE_CARD=/absolute/path/to/large-card.txt \
  uv run pytest tests/test_performance.py::test_local_large_card_compare_toggle -q -s
```

Optional `DATACARD_VIEWER_AFTER_CARD` selects a different comparison card; otherwise
both readers use the input card. `DATACARD_VIEWER_NUISANCE` selects an exact nuisance
name; otherwise the first `shape?` is preferred. The check selects the first 30
common channels and all available processes, prints wall/CPU times, and gives each
warm log toggle a generous 10-second server budget. Opening/first plotting have a
30-second budget. These budgets target interactive stalls on local files; slow
remote filesystems can exceed them independently of CPU performance. Cards needing
path remapping must have locally resolvable shape paths for this helper.
