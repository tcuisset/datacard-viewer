import os
from pathlib import Path

import hist
import numpy as np
import pytest
import uproot

from datacard_viewer.core import (
    Card,
    Comparison,
    Reader,
    Scan,
    Template,
    candidate,
    compare,
    compare_response,
    comparison_shortlist,
    diagnose,
    ratios,
    summarize,
)
from datacard_viewer.plotting import bins_frame, compare_figure, figure


def write_root(path, entries):
    with uproot.recreate(path) as f:
        for name, spec in entries.items():
            values, variances = spec[:2]
            edges = spec[2] if len(spec) > 2 else [0, 1, 3]
            h = hist.Hist(hist.axis.Variable(edges), storage=hist.storage.Weight())
            h.view().value[:] = values
            h.view().variance[:] = variances
            f["c/" + name] = h


@pytest.fixture
def model(tmp_path):
    root = tmp_path / "templates.root"
    write_root(
        root,
        {
            "b": ([10, 20], [4, 9]),
            "s": ([2, 3], [1, 1]),
            "b_normUp": ([12, 24], [5, 10]),
            "b_normDown": ([8, 16], [3, 8]),
            "b_distortUp": ([12, 18], [4, 9]),
            "b_distortDown": ([8, 22], [4, 9]),
            "s_distortUp": ([0, 5], [1, 1]),
            "s_distortDown": ([4, 1], [1, 1]),
        },
    )
    card = tmp_path / "card.txt"
    card.write_text("""imax 1
jmax 1
kmax *
shapes * * templates.root c/$PROCESS c/$PROCESS_$SYSTEMATIC
bin c
observation -1
bin c c
process b s
process 1 0
rate 30 5
norm shape 1 -
distort shape 1 1
fallback shape? 1.1 -
c autoMCStats 0 0 1
all group = norm distort fallback
""")
    return card, root


def test_weighted_sums_and_normalization(model):
    card = Card.load(model[0])
    reader = Reader(card)
    c = compare(reader, "c", ["b", "s"], "norm")
    np.testing.assert_allclose(c.nominal.values, [12, 23])
    np.testing.assert_allclose(c.up.values, [14, 27])
    np.testing.assert_allclose(c.nominal.variances, [5, 10])
    np.testing.assert_allclose(c.down.variances, [4, 9])
    individual = compare(reader, "c", ["b"], "norm")
    assert candidate(diagnose(individual))
    np.testing.assert_allclose(individual.normalized().up.values, [10, 20])
    assert bins_frame(c).low.tolist() == [0, 1]
    assert figure(c).layout.xaxis2.title.text == "Observable"
    reader.close()


def test_cancellation_is_not_a_candidate(model):
    card = Card.load(model[0])
    reader = Reader(card)
    assert candidate(diagnose(compare(reader, "c", ["b", "s"], "distort")))
    rows = []
    for p in ["b", "s"]:
        rows.append(
            dict(
                nuisance="distort",
                channel="c",
                process=p,
                **diagnose(compare(reader, "c", [p], "distort")),
            )
        )
    assert not next(r for r in summarize(card, rows) if r["nuisance"] == "distort")[
        "candidate"
    ]
    assert not next(r for r in summarize(card, rows[:1]) if r["nuisance"] == "distort")[
        "candidate"
    ]
    reader.close()


def test_shape_optional_and_partial_pair(model):
    card = Card.load(model[0])
    reader = Reader(card)
    c = compare(reader, "c", ["b"], "fallback")
    np.testing.assert_allclose(c.up.values, [11, 22])
    np.testing.assert_allclose(c.down.values, np.array([10, 20]) / 1.1)
    fallback_figure = figure(c, remove_normalization=True)
    assert fallback_figure.layout.annotations[0].text == "Log-normal (lnN)"
    assert (
        "fallback" in c.warnings[1]
        if len(c.warnings) > 1
        else "fallback" in c.warnings[0]
    )
    reader.close()
    with uproot.update(model[1]) as f:
        f["c/b_fallbackUp"] = (np.array([11.0, 22.0]), np.array([0.0, 1.0, 3.0]))
    with pytest.raises(ValueError, match="Missing variation"):
        compare(Reader(card), "c", ["b"], "fallback")


def test_before_after_datacard_responses(tmp_path, monkeypatch):
    before_root, after_root = tmp_path / "before.root", tmp_path / "after.root"
    write_root(before_root, {
        "b": ([10, 20], [1, 4]), "gone": ([2, 3], [1, 1]),
        "b_retainedUp": ([12, 19], [1, 4]), "b_retainedDown": ([8, 21], [1, 4]),
        "b_optionalUp": ([11, 22], [1, 4]), "b_optionalDown": ([9, 18], [1, 4]),
        "b_partialUp": ([13, 20], [1, 4]),
    })
    write_root(after_root, {
        "b": ([10, 20], [1, 4], [0, 2, 3]), "new": ([4, 5], [1, 1]),
        "b_retainedUp": ([15, 18], [1, 4], [0, 2, 3]), "b_retainedDown": ([7, 22], [1, 4], [0, 2, 3]),
    })

    def make_card(path, rootname, processes, ids, rates, nuisances):
        bins = " ".join(["c"] * len(processes.split()))
        path.write_text(
            f"imax 1\njmax *\nkmax *\nshapes * * {rootname} c/$PROCESS c/$PROCESS_$SYSTEMATIC\n"
            f"bin c\nobservation -1\nbin {bins}\nprocess {processes}\nprocess {ids}\nrate {rates}\n{nuisances}\n"
        )

    before_path, after_path = tmp_path / "before.txt", tmp_path / "after.txt"
    make_card(before_path, before_root.name, "b gone", "1 2", "30 5", "retained shape 1 -\nsym lnN 1.2 1.4\nasym lnN 0.8/1.25 -\noptional shape? 1.1 -\npartial shape? 1.1 -\ndrop shape - 1")
    make_card(after_path, after_root.name, "new b", "2 1", "9 30", "retained shape - 1\nsym lnN - 1.3\nasym lnN - 0.7/1.4\noptional lnN - 1.15")
    before, after = Card.load(before_path), Card.load(after_path)
    shortlist = comparison_shortlist(before, after)
    assert any(row["nuisance"] == "optional" and row["ROOT check"] for row in shortlist)
    assert any(row["nuisance"] == "drop" for row in shortlist)
    b_reader, a_reader = Reader(before), Reader(after)
    np.testing.assert_allclose(compare_response(b_reader, "c", ["b"], "retained").up.values, [12, 19])
    np.testing.assert_allclose(compare_response(a_reader, "c", ["b"], "retained").up.values, [15, 18])
    _, matching = compare_figure(
        [("c", compare_response(b_reader, "c", ["b"], "retained"))],
        [("c", compare_response(a_reader, "c", ["b"], "retained"))],
    )
    assert not matching
    sym = compare_response(b_reader, "c", ["b"], "sym")
    np.testing.assert_allclose(sym.up.values, [12, 24])
    np.testing.assert_allclose(sym.down.values, np.array([10, 20]) / 1.2)
    grouped = compare_response(b_reader, "c", ["b", "gone"], "sym")
    np.testing.assert_allclose(grouped.up.values, [14.8, 28.2])
    shape = compare_response(b_reader, "c", ["b"], "optional")
    overlaid, matching = compare_figure(
        [("c", shape)], [("c", grouped)], remove_normalization=True
    )
    assert matching
    assert overlaid.data[1].line.dash == "dash"
    assert overlaid.data[6].line.dash == "solid"
    assert overlaid.data[3].line.dash == "dash"
    assert overlaid.data[8].line.dash == "solid"
    assert len(set(overlaid.data[8].y[::2])) > 1
    ratio_label = next(a for a in overlaid.layout.annotations if a.yref == "y2 domain")
    assert ratio_label.text == "Before · <b>Shape</b><br>After · <b>Log-normal (lnN)</b>"
    assert "Log-normal (lnN)" in overlaid.data[8].hovertemplate
    asym = compare_response(a_reader, "c", ["b"], "asym")
    np.testing.assert_allclose(asym.up.values, [14, 28])
    np.testing.assert_allclose(asym.down.values, [7, 14])
    np.testing.assert_allclose(compare_response(b_reader, "c", ["b"], "optional").up.values, [11, 22])
    np.testing.assert_allclose(compare_response(a_reader, "c", ["b"], "optional").up.values, [11.5, 23])
    assert "gone" not in after.channels["c"] and "new" in after.channels["c"]
    # The Up endpoint exists but Down does not, so shape? must report the side and channel.
    with pytest.raises(ValueError, match="Before: c/b/partial: missing variation endpoint"):
        compare_response(Reader(Card.load(before_path)), "c", ["b"], "partial", "Before")
    b_reader.close()
    a_reader.close()
    import sys

    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(sys, "argv", ["app.py", "--card", str(before_path), "--compare-card", str(after_path)])
    at = AppTest.from_file(Path("datacard_viewer/app.py").resolve(), default_timeout=15).run()
    assert not at.exception
    at.radio(key="viewer_mode").set_value("Compare").run()
    assert not at.exception
    assert at.selectbox(key="compare_nuisance").value is None
    assert not at.session_state["reader"].files
    assert not at.session_state["compare_reader"].files
    next(b for b in at.button if b.label == "→").click().run()
    assert at.selectbox(key="compare_nuisance").value == "sym"
    assert at.session_state["reader"].files
    assert at.session_state["compare_reader"].files
    next(b for b in at.button if b.label == "Open / Reload").click().run()
    at.selectbox(key="compare_nuisance").set_value("optional").run()
    assert not at.exception
    assert len(at.session_state["reader"].arrays) == 3
    assert len(at.session_state["compare_reader"].arrays) == 1
    next(b for b in at.button if b.label == "Backgrounds").click().run()
    assert at.multiselect(key="processes").value == ["b", "gone", "new"]
    at.multiselect(key="processes").set_value(["new"]).run()
    assert not at.exception and not at.error
    yields = at.dataframe[0].value
    assert yields["nominal"].tolist() == pytest.approx([0, 9])


def test_variance_cache_reload_and_eviction(model):
    card = Card.load(model[0])
    reader = Reader(card, max_bytes=100, max_files=1)
    first = reader.read("c", "b")
    assert not first.values.flags.writeable
    assert reader.bytes <= 100
    previous = model[1].stat().st_mtime_ns
    write_root(model[1], {"b": ([20, 40], [8, 18])})
    os.utime(model[1], ns=(previous + 1000000, previous + 1000000))
    assert reader.stale()
    np.testing.assert_allclose(reader.read("c", "b").values, [20, 40])
    reader.close()
    assert not reader.files


def test_substitution_mapping_renaming_and_coefficient(model):
    path, root = model
    text = path.read_text().replace(
        "templates.root c/$PROCESS", "output/templates.root c/$PROCESS"
    )
    text += "nuisance edit rename b c norm renamed\n"
    path.write_text(text)
    card = Card.load(path)
    reader = Reader(card, [("output/", "")])
    c = compare(reader, "c", ["b"], "renamed")
    np.testing.assert_allclose(c.up.values, [12, 24])
    assert reader.resolve("c", "b")[0] == root
    reader.close()
    path.write_text(text.replace("norm shape 1 -", "norm shape 0.5 -"))
    c = compare(Reader(Card.load(path), [("output/", "")]), "c", ["b"], "renamed")
    assert any("not effective" in w for w in c.warnings)


def test_zero_negative_and_invalid_normalizations():
    def h(values):
        return Template(
            np.array([0, 1, 3]), np.array(values, dtype=float), np.array([1.0, 4.0])
        )

    c = Comparison(h([0, 2]), h([1, 2]), h([0, 1]))
    assert not diagnose(c)["valid"]
    assert np.isnan(ratios(c.up.values, c.nominal.values)[0])
    assert len(figure(c).data) > 0
    assert not diagnose(Comparison(h([-1, 2]), h([1, 2]), h([1, 2])))["valid"]
    assert not diagnose(Comparison(h([0, 0]), h([1, 2]), h([1, 2])))["valid"]


def test_binning_flow_and_poisson_fallback(model):
    card = Card.load(model[0])
    root = model[1]
    with uproot.update(root) as f:
        h = hist.Hist(hist.axis.Variable([0, 1, 3]), storage=hist.storage.Weight())
        h.view(flow=True).value[:] = [5, 12, 24, 7]
        h.view(flow=True).variance[:] = [5, 4, 9, 7]
        f["c/b_normUp"] = h
        f["c/b_normDown"] = (np.array([8.0, 16.0]), np.array([0.0, 1.0, 4.0]))
    reader = Reader(card)
    assert "under/overflow" in " ".join(reader.read("c", "b", "normUp").warnings)
    assert "Poisson" in " ".join(reader.read("c", "b", "normDown").warnings)
    with pytest.raises(ValueError, match="Incompatible bin edges"):
        compare(reader, "c", ["b"], "norm")
    reader.close()


def test_scan_partial_failures_and_cancellation(model):
    card = Card.load(model[0])
    card.coefficients["missing", "c", "b"] = 1.0
    card.nuisances["missing"] = "shape"
    card.relevant["c"].append("missing")
    job = Scan(card).start()
    job.thread.join(10)
    assert job.done and not job.error
    rows, _ = job.snapshot()
    assert len(rows) == job.total
    assert not next(r for r in rows if r["nuisance"] == "missing")["valid"]
    assert next(r for r in summarize(card, rows) if r["nuisance"] == "norm")[
        "candidate"
    ]
    cancelled = Scan(card)
    cancelled.cancel.set()
    cancelled.start()
    cancelled.thread.join(10)
    assert cancelled.done and cancelled.snapshot()[0] == []
    model[0].write_text(model[0].read_text() + "#changed\n")
    assert job.stale()


@pytest.mark.parametrize(
    "filename,channels,columns,nuisances",
    [
        ("ElMu_2024_highDR_dPhijj_0_pio2_v8.0.txt", 1, 19, 79),
        ("combined_allfs_dPhijj_v8.0.txt", 88, 1696, 992),
    ],
)
def test_supplied_cards(filename, channels, columns, nuisances):
    path = Path("dPhijj_datacards") / filename
    if not path.exists():
        pytest.skip("Local analysis fixtures not installed")
    card = Card.load(path)
    assert len(card.channels) == channels
    assert len(card.parsed.keyline) == columns
    assert len(card.nuisances) == nuisances
    reader = Reader(card, [("output/", "")])
    ch = next(iter(card.channels))
    c = compare(reader, ch, card.processes(ch, "All backgrounds"), card.relevant[ch][0])
    assert np.isfinite(c.nominal.integral)
    reader.close()


def test_override_mass_keywords_and_missing_files(model):
    path, _ = model
    override = path.parent / "override.root"
    write_root(override, {"b_125_2024": ([7, 8], [2, 3])})
    path.write_text(
        path.read_text().replace(
            "bin c\nobservation",
            "shapes b c override.root $CHANNEL/$PROCESS_$MASS_$ERA\nbin c\nobservation",
        )
    )
    card = Card.load(path)
    reader = Reader(card, mass=125, keywords={"ERA": "2024"}, max_files=1)
    np.testing.assert_allclose(reader.read("c", "b").values, [7, 8])
    reader.read("c", "s")
    assert len(reader.files) == 1
    assert all(k[0] != str(override) for k in reader.directories)
    with pytest.raises(ValueError, match="Unresolved substitution"):
        Reader(card).resolve("c", "b")
    reader.close()
    override.unlink()
    with pytest.raises(FileNotFoundError):
        Reader(card, keywords={"ERA": "2024"}).read("c", "b")


@pytest.mark.parametrize(
    "values,variances,error",
    [
        ([float("nan"), 2], [1, 1], "Nonfinite contents"),
        ([1, 2], [float("nan"), 1], "Invalid variances"),
    ],
)
def test_invalid_histograms(model, values, variances, error):
    write_root(model[1], {"b": (values, variances)})
    reader = Reader(Card.load(model[0]))
    with pytest.raises(ValueError, match=error):
        reader.read("c", "b")
    reader.close()


def test_streamlit_workflow(model, monkeypatch):
    import sys

    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(sys, "argv", ["app.py", "--card", str(model[0])])
    at = AppTest.from_file(Path("datacard_viewer/app.py").resolve(), default_timeout=10).run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 1
    next(b for b in at.button if b.label == "→").click().run()
    assert at.selectbox(key="nuisance").value == "distort"
    next(b for b in at.button if b.label == "←").click().run()
    assert at.selectbox(key="nuisance").value == "norm"
    assert not at.exception
    at.checkbox[0].check().run()
    assert not at.exception
    next(b for b in at.button if b.label == "Scan whole card").click().run()
    at.session_state["scan"].thread.join(5)
    at.run()
    assert not at.exception
    assert at.session_state["scan"].done
    assert len(at.session_state["scan"].snapshot()[0]) == 4


def test_scan_detects_changes_even_after_reader_refresh(model, monkeypatch):
    import datacard_viewer.core as core

    original = core.compare
    changed = False

    def compare_and_change(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            changed = True
            stamp = model[1].stat().st_mtime_ns + 1_000_000
            os.utime(model[1], ns=(stamp, stamp))
        return result

    monkeypatch.setattr(core, "compare", compare_and_change)
    job = Scan(Card.load(model[0])).start()
    job.thread.join(5)
    assert job.done and not job.error
    assert job.stale()


def test_csv_suggestions_require_complete_coverage_and_unit_coefficients(model):
    import csv
    import io

    from datacard_viewer.reporting import scan_exports

    card = Card.load(model[0])
    job = Scan(card).start()
    job.thread.join(5)
    rows, _ = job.snapshot()
    pair_csv, _ = scan_exports(card, rows, 0.01)
    pairs = list(csv.DictReader(io.StringIO(pair_csv.decode())))
    norm = next(r for r in pairs if r["nuisance"] == "norm")
    assert norm["suggested_lnN"] == "0.8/1.2"
    assert all(not r["suggested_lnN"] for r in pairs if r["nuisance"] == "distort")
    stale_csv, _ = scan_exports(card, rows, 0.01, stale=True)
    assert all(
        not r["suggested_lnN"] for r in csv.DictReader(io.StringIO(stale_csv.decode()))
    )
