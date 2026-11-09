"""Multi-channel coordinates, shared normalization, and selection workflow."""

import json
import sys
from pathlib import Path

import hist
import numpy as np
import pytest
import uproot
from streamlit.testing.v1 import AppTest

from datacard_viewer.core import Card, Comparison, Reader, Template, compare
from datacard_viewer.plotting import (
    channel_figure,
    channel_segments,
    channels_frame,
    display_channels,
)


def comparison(edges, nominal, up, down):
    def h(values):
        return Template(
            np.array(edges, dtype=float),
            np.array(values, dtype=float),
            np.full(len(values), 4.0),
        )

    return Comparison(h(nominal), h(up), h(down))


def test_segments_ranges_ticks_bands_and_tooltips():
    first = comparison([50, 100, 200], [10, 20], [11, 22], [9, 18])
    second = comparison([0, 10, 200, 400], [2, 3, 5], [2, 4, 4], [1, 4, 5])
    pairs = [("first", first), ("second", second)]
    segments = channel_segments(pairs)
    assert segments[0][2] == 0
    assert segments[1][2] == 240  # first ends at 200; 10% of 400 is the gap.
    frame = channels_frame(pairs)
    assert frame.channel.tolist() == ["first", "first", "second", "second", "second"]
    np.testing.assert_allclose(frame.low, [50, 100, 0, 10, 200])
    np.testing.assert_allclose(frame.display_low, [50, 100, 240, 250, 440])
    fig = channel_figure(pairs, variation_bands=True)
    assert [t.name for t in fig.data if t.showlegend] == ["Nominal", "Up", "Down"]
    assert len(fig.layout.shapes) == 2
    assert [a.text for a in fig.layout.annotations] == ["first", "second"]
    assert fig.layout.xaxis2.ticktext[0] == "50"
    assert fig.layout.xaxis2.ticktext[5] == "0"
    for trace in fig.data:
        # A trace/band belongs to exactly one channel: no connecting across gaps.
        assert max(trace.x) <= 200 or min(trace.x) >= 240
    lines = [t for t in fig.data if t.name == "Up"]
    assert "second" in lines[-1].hovertemplate
    assert list(lines[-1].customdata[0]) == [0, 10]
    assert first.nominal.edges[-1] - first.nominal.edges[0] == 150
    assert second.nominal.edges[-1] - second.nominal.edges[0] == 400


def test_global_normalization_preserves_channel_migration_and_errors():
    pairs = [
        ("one", comparison([0, 1, 2], [10, 10], [20, 20], [5, 5])),
        ("two", comparison([0, 1, 2], [10, 10], [10, 10], [15, 15])),
    ]
    shown = display_channels(pairs, True)
    assert sum(c.up.integral for _, c in shown) == pytest.approx(40)
    assert sum(c.down.integral for _, c in shown) == pytest.approx(40)
    assert shown[0][1].up.integral == pytest.approx(80 / 3)
    assert shown[1][1].up.integral == pytest.approx(40 / 3)
    np.testing.assert_allclose(shown[0][1].up.variances, [16 / 9, 16 / 9])
    frame = channels_frame(pairs, True)
    assert frame.up_over_nominal.iloc[0] == pytest.approx(4 / 3)
    assert frame.up_over_nominal.iloc[2] == pytest.approx(2 / 3)
    assert pairs[0][1].up.integral == 40  # Input templates remain unchanged.
    assert channels_frame(list(reversed(pairs))).channel.iloc[0] == "two"


def test_global_normalization_zero_negative_bins_and_bad_totals():
    pairs = [
        ("one", comparison([0, 1, 2], [0, 10], [0, 12], [0, 8])),
        ("two", comparison([0, 1, 2], [-1, 11], [-1, 11], [-1, 11])),
    ]
    fig = channel_figure(pairs)
    assert any(t.name == "Undefined ratio: nominal zero" for t in fig.data)
    assert channels_frame(pairs).nominal.min() == -1
    assert np.isnan(channels_frame(pairs).up_over_nominal.iloc[0])
    with pytest.raises(ValueError, match="positive totals"):
        display_channels([("zero", comparison([0, 1], [0], [1], [1]))], True)
    with pytest.raises(ValueError, match="at least one"):
        channel_figure([])


@pytest.fixture
def multi_card(tmp_path):
    root = tmp_path / "shapes.root"
    with uproot.recreate(root) as f:
        for ch in ["one", "two", "other"]:
            processes = ["different"] if ch == "other" else ["b", "s"]
            for process in processes:
                h = hist.Hist(
                    hist.axis.Variable([0, 1, 3]), storage=hist.storage.Weight()
                )
                h.view().value[:] = [10, 20]
                h.view().variance[:] = [4, 9]
                f[f"{ch}/{process}"] = h
                if ch == "one" and process == "b":
                    f[f"{ch}/{process}_normUp"] = h * 1.2
                    f[f"{ch}/{process}_normDown"] = h * 0.8
    path = tmp_path / "card.txt"
    path.write_text("""imax 3
jmax 2
kmax *
shapes * * shapes.root $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC
bin one two other
observation -1 -1 -1
bin one one two two other
process b s b s different
process 1 0 1 0 2
rate 30 30 30 30 30
norm shape 1 - - - -
""")
    return path, root


def app(path, monkeypatch, after=None):
    argv = ["app.py", "--card", str(path)]
    if after:
        argv += ["--compare-card", str(after)]
    monkeypatch.setattr(sys, "argv", argv)
    return AppTest.from_file(Path("app.py").resolve(), default_timeout=10).run()


def test_multi_channel_ui_order_unaffected_scans_and_process_presets(
    multi_card, monkeypatch
):
    at = app(multi_card[0], monkeypatch)
    assert not at.exception
    assert at.multiselect(key="channels").value == ["one"]
    at.multiselect(key="channels").set_value(["two", "one"]).run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 1
    next(
        c for c in at.checkbox if c.label == "Remove normalization effect"
    ).check().run()
    assert not at.exception
    # Raw totals: first channel is unaffected; normalization is common.
    assert next(m for m in at.metric if m.label == "Up integral").value == "66"
    assert at.dataframe[0].value.channel.tolist() == ["two", "one"]
    next(b for b in at.button if b.label == "Scan selected channels").click().run()
    job = at.session_state["channel_scan"]
    job.thread.join(5)
    assert job.channels == {"one", "two"}
    at.run()
    assert not at.exception
    at.multiselect(key="channels").set_value(["one", "other"]).run()
    assert not at.exception and not at.error
    assert len(at.get("plotly_chart")) == 1
    assert at.multiselect(key="processes").options == ["b", "s", "different"]
    assert any("zero yield" in w.value for w in at.warning)
    at.multiselect(key="processes").set_value(["different"]).run()
    assert not at.exception and not at.error
    assert len(at.get("plotly_chart")) == 1
    assert next(m for m in at.metric if m.label == "Nominal integral").value == "30"
    assert any("one: No selected processes" in w.value for w in at.warning)
    for label, expected in [
        ("Backgrounds", ["b", "different"]),
        ("Signals", ["s"]),
        ("All", ["b", "s", "different"]),
    ]:
        next(b for b in at.button if b.label == label).click().run()
        assert at.multiselect(key="processes").value == expected
        assert not at.exception
    at.multiselect(key="processes").set_value(["s", "different"]).run()
    assert next(m for m in at.metric if m.label == "Nominal integral").value == "60"
    at.multiselect(key="processes").set_value([]).run()
    assert any("Select at least one process" in info.value for info in at.info)
    at.multiselect(key="channels").set_value([]).run()
    assert any("Select at least one channel" in info.value for info in at.info)
    assert next(b for b in at.button if b.label == "Scan selected channels").disabled


def test_multi_channel_read_failure_never_silently_omits_channel(
    multi_card, monkeypatch
):
    at = app(multi_card[0], monkeypatch)
    at.multiselect(key="channels").set_value(["one", "two"]).run()
    # Remove a nominal, keeping the other channel and variations readable.
    with uproot.update(multi_card[1]) as f:
        del f["two"]["b"]
    at.run()
    assert any("Channel two" in error.value for error in at.error)
    assert not at.get("plotly_chart")


@pytest.mark.parametrize("compare_mode", [False, True], ids=["inspect", "compare"])
def test_hide_unaffected_channels_only_filters_plot(
    multi_card, monkeypatch, compare_mode
):
    path, _ = multi_card
    after = None
    if compare_mode:
        after = path.with_name("after.txt")
        after.write_text(
            path.read_text().replace("norm shape 1 - - - -", "norm lnN - - 1.2 - -")
        )

    at = app(path, monkeypatch, after)
    selected = ["other", "two", "one"]
    at.multiselect(key="channels").set_value(selected).run()
    if compare_mode:
        at.selectbox(key="compare_nuisance").set_value("norm").run()
    at.multiselect(key="processes").set_value(["b"]).run()
    next(
        c for c in at.checkbox if c.label == "Remove normalization effect"
    ).check().run()

    def plotted_names():
        (plot,) = at.get("plotly_chart")
        spec = json.loads(plot.proto.spec)
        return [
            a["text"] for a in spec["layout"]["annotations"] if a["yref"] == "paper"
        ]

    assert plotted_names() == selected
    if not compare_mode:
        full_bins = at.dataframe[1].value
        affected_ratios = full_bins.loc[full_bins.channel == "one", "up_over_nominal"]

    next(c for c in at.checkbox if c.label == "Hide unaffected channels").check().run()
    assert plotted_names() == (["two", "one"] if compare_mode else ["one"])
    assert at.multiselect(key="channels").value == selected
    assert list(dict.fromkeys(at.dataframe[0].value.channel)) == selected
    if not compare_mode:
        np.testing.assert_allclose(
            at.dataframe[1].value.up_over_nominal, affected_ratios
        )

    at.multiselect(key="processes").set_value(["s"]).run()
    assert not at.error and not at.exception
    assert not at.get("plotly_chart")
    assert any("changes no bin yields" in info.value for info in at.info)
    assert at.multiselect(key="channels").value == selected

    next(
        c for c in at.checkbox if c.label == "Hide unaffected channels"
    ).uncheck().run()
    assert plotted_names() == selected


def test_reload_same_card_recreates_reader_without_losing_channels(
    multi_card, monkeypatch
):
    at = app(multi_card[0], monkeypatch)
    at.multiselect(key="channels").set_value(["two", "one"]).run()
    old_reader = at.session_state["reader"]
    next(b for b in at.button if b.label == "Open / Reload").click().run()
    assert not at.error and not at.exception
    assert at.session_state["reader"] is not old_reader
    assert at.multiselect(key="channels").value == ["two", "one"]
    assert len(at.get("plotly_chart")) == 1
    assert not old_reader.files


@pytest.mark.parametrize("variation_bands", [False, True])
def test_ratio_range_ignores_statistical_bands(variation_bands):
    from datacard_viewer.plotting import figure

    c = comparison([0, 1, 2], [10, 10], [11, 10.5], [9, 9.5])
    before = figure(c, variation_bands=variation_bands).layout.yaxis2.range
    c.nominal.variances[:] = 1e8
    c.up.variances[:] = 1e10
    c.down.variances[:] = 1e10
    fig = figure(c, variation_bands=variation_bands)
    assert fig.layout.yaxis2.range == pytest.approx([0.88, 1.12])
    assert fig.layout.yaxis2.range == before
    assert fig.layout.yaxis2.autorange is False
    assert fig.layout.yaxis.range is None  # Absolute panel still uses autoranging.


def test_multi_channel_ratio_range_uses_displayed_ratios_across_channels():
    pairs = [
        ("one", comparison([0, 1], [10], [20], [5])),
        ("two", comparison([0, 1], [10], [10], [15])),
    ]
    for _, c in pairs:
        c.nominal.variances[:] = 1e10
    assert channel_figure(pairs).layout.yaxis2.range == pytest.approx([0.35, 2.15])
    # Shared normalization: Up ratios 4/3 and 2/3; Down ratios 1/2 and 3/2.
    assert channel_figure(pairs, True).layout.yaxis2.range == pytest.approx([0.4, 1.6])


def test_ratio_range_handles_zero_denominators_and_flat_variations():
    from datacard_viewer.plotting import figure

    flat = comparison([0, 1, 2], [0, 10], [100, 10], [100, 10])
    assert figure(flat).layout.yaxis2.range == pytest.approx([0.99, 1.01])
    undefined = comparison([0, 1], [0], [100], [100])
    assert figure(undefined).layout.yaxis2.range == pytest.approx([0.99, 1.01])


@pytest.mark.parametrize("compare_mode", [False, True], ids=["inspect", "compare"])
def test_channel_name_filter_bulk_selection(multi_card, monkeypatch, compare_mode):
    path = multi_card[0]
    after = None
    channels = ["one", "two", "other"]
    if compare_mode:
        after = path.with_name("after.txt")
        after.write_text(path.read_text().replace("other", "after_only"))
        channels = ["one", "two"]
    at = app(path, monkeypatch, after)
    assert at.multiselect(key="channels").options == channels
    at.multiselect(key="channels").set_value(["two", "one"]).run()

    def select_matches():
        return next(b for b in at.button if b.label == "Select matching channels")

    assert select_matches().disabled
    at.text_input(key="channel_filter").set_value("  ONE  ").run()
    assert at.multiselect(key="channels").value == ["two", "one"]
    select_matches().click().run()
    assert at.multiselect(key="channels").value == ["one"]
    assert not at.exception
    at.text_input(key="channel_filter").set_value("o").run()
    select_matches().click().run()
    assert at.multiselect(key="channels").value == channels
    # Manual adjustments remain available, and filtering alone never drops them.
    at.multiselect(key="channels").set_value(["two", "one"]).run()
    at.text_input(key="channel_filter").set_value("missing").run()
    assert select_matches().disabled
    assert at.multiselect(key="channels").value == ["two", "one"]
    assert any("0 matching channel" in c.value for c in at.caption)
    at.text_input(key="channel_filter").set_value("   ").run()
    assert select_matches().disabled
    assert at.multiselect(key="channels").value == ["two", "one"]
    if compare_mode:
        assert not at.session_state["reader"].files
        assert not at.session_state["compare_reader"].files
        at.radio(key="viewer_mode").set_value("Inspect").run()
        assert at.multiselect(key="channels").value == ["two", "one"]
    assert not at.exception


def test_absent_process_has_local_binning_and_zero_errors(multi_card):
    card = Card.load(multi_card[0])
    reader = Reader(card)
    try:
        assert card.processes("other", "One process", "b") == []
        result = compare(reader, "other", [], "norm")
        np.testing.assert_array_equal(result.nominal.edges, [0, 1, 3])
        for template in (result.nominal, result.up, result.down):
            np.testing.assert_array_equal(template.values, [0, 0])
            np.testing.assert_array_equal(template.variances, [0, 0])
    finally:
        reader.close()
