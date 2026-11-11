"""Guard interactive costs with large key lists and many selected channels."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
import uproot
from streamlit.testing.v1 import AppTest

from datacard_viewer.core import (
    Card,
    Comparison,
    Reader,
    Template,
    compare,
    compare_response,
)
from datacard_viewer.plotting import channel_figure, compare_figure


@pytest.fixture
def crowded_card(tmp_path):
    # A shared long prefix prevents uproot's suggestion formatter from cheaply
    # discarding keys. No large binary fixture needs to be checked into git.
    prefix = "CMS_" + "long_systematic_name_" * 4
    nuisance = prefix + "missing"
    with uproot.recreate(tmp_path / "shapes.root") as root:
        root["templates/b"] = (np.array([10., 20.]), np.arange(3.))
        for i in range(1100):
            root[f"templates/b_{prefix}{i}Up"] = "unused key"
    channels = [f"channel_{i}" for i in range(30)]
    path = tmp_path / "card.txt"
    path.write_text(
        "imax 30\njmax *\nkmax *\n"
        "shapes * * shapes.root templates/$PROCESS templates/$PROCESS_$SYSTEMATIC\n"
        + "bin " + " ".join(channels) + "\nobservation " + "-1 " * 30
        + "\nbin " + " ".join(channels)
        + "\nprocess " + "b " * 30 + "\nprocess " + "1 " * 30
        + "\nrate " + "30 " * 30 + f"\n{nuisance} shape? " + "1.1 " * 30 + "\n"
    )
    return path, nuisance


def forbid_suggestions(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Missing ROOT keys must not invoke uproot's fuzzy suggestion formatter")

    monkeypatch.setattr(uproot.exceptions.KeyInFileError, "__str__", fail)


@pytest.mark.parametrize("response", [compare, compare_response])
def test_crowded_directory_fallback_and_missing_objects(crowded_card, monkeypatch, response):
    path, nuisance = crowded_card
    forbid_suggestions(monkeypatch)
    reader = Reader(Card.load(path))
    try:
        result = response(reader, "channel_0", ["b"], nuisance)
        np.testing.assert_allclose(result.up.values, [11, 22])
        assert result.effect_types == frozenset({"lnN"})
        # Also cover direct/nominal failures and missing parent directories:
        # these reach UI and scan error handlers without a fallback catch.
        for mapping in (["shapes.root", "missing"], ["shapes.root", "absent/b"]):
            reader.card.parsed.shapeMap["*"]["*"] = mapping
            with pytest.raises(KeyError) as exc:
                reader.read("channel_0", "b")
            assert str(path.parent / "shapes.root") in str(exc.value)
            assert mapping[1] in str(exc.value)
    finally:
        reader.close()


def test_compare_toggle_with_30_channels_and_crowded_keys(crowded_card, monkeypatch):
    path, nuisance = crowded_card
    forbid_suggestions(monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "app.py", "--card", str(path), "--compare-card", str(path),
        "--max-files", "1000", "--max-mb", "5000",
    ])
    at = AppTest.from_file(Path("datacard_viewer/app.py").resolve(), default_timeout=10).run()
    at.multiselect(key="channels").set_value(at.multiselect(key="channels").options)
    at.selectbox(key="compare_nuisance").set_value(nuisance).run()
    assert not at.exception and not at.error
    reader = at.session_state["reader"]
    after_reader = at.session_state["compare_reader"]
    # Plot-only reruns must retain the parsed cards and warm template caches.
    def unexpected_load(*args, **kwargs):
        pytest.fail("A plot option must not reparse a card or reopen a cached ROOT file")

    monkeypatch.setattr(Card, "load", unexpected_load)
    monkeypatch.setattr(uproot, "open", unexpected_load)
    for log_y in (True, False):
        next(c for c in at.checkbox if c.label == "Logarithmic y-axis").set_value(log_y).run()
        assert not at.exception and not at.error
        assert at.session_state["reader"] is reader
        assert at.session_state["compare_reader"] is after_reader
        (plot,) = at.get("plotly_chart")
        assert json.loads(plot.proto.spec)["layout"]["yaxis"]["type"] == ("log" if log_y else "linear")
    reader.close()
    after_reader.close()


@pytest.mark.parametrize("mode", ["inspect", "compare"])
def test_layout_construction_scales_linearly(monkeypatch, mode):
    # Count validated objects instead of using a hardware-sensitive stopwatch.
    # Incremental Plotly layout appends rebuild the old objects quadratically.
    counts = {"annotations": 0, "shapes": 0}
    for cls, key in ((go.layout.Annotation, "annotations"), (go.layout.Shape, "shapes")):
        original = cls.__init__

        def count(self, *args, _original=original, _key=key, **kwargs):
            counts[_key] += 1
            _original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__init__", count)
    h = Template(np.arange(11.), np.ones(10), np.ones(10))
    comparison = Comparison(h, h, h, effect_types=frozenset({"shape"}))
    channels = [(f"channel_{i}", comparison) for i in range(100)]
    fig = channel_figure(channels) if mode == "inspect" else compare_figure(channels, channels)[0]
    assert len(fig.layout.annotations) == 200
    assert counts["annotations"] <= 4 * len(channels)
    if mode == "inspect":
        assert len(fig.layout.shapes) == 198
        assert counts["shapes"] <= 4 * len(channels)
        assert fig.layout.shapes[0].yref == "y domain"
        assert fig.layout.shapes[1].yref == "y2 domain"


@pytest.mark.skipif(not os.environ.get("DATACARD_VIEWER_LARGE_CARD"), reason="Optional local large-card benchmark")
def test_local_large_card_compare_toggle(monkeypatch):
    """Run the real 100+ MB card without distributing private analysis data."""
    path = os.environ["DATACARD_VIEWER_LARGE_CARD"]
    monkeypatch.setattr(sys, "argv", [
        "app.py", "--card", path, "--compare-card", os.environ.get("DATACARD_VIEWER_AFTER_CARD", path),
        "--max-files", "1000", "--max-mb", "5000",
    ])
    def timed(label, action):
        wall, cpu = time.perf_counter(), time.process_time()
        result = action()
        print(f"{label}: wall={time.perf_counter() - wall:.3f}s CPU={time.process_time() - cpu:.3f}s")
        return result

    at = timed("Open cards", lambda: AppTest.from_file(
        Path("datacard_viewer/app.py").resolve(), default_timeout=30
    ).run())
    assert not at.exception and not at.error
    at.multiselect(key="channels").set_value(at.multiselect(key="channels").options[:30])
    next(r for r in at.radio if r.label == "Nuisance view").set_value("All effects").run()
    at.multiselect(key="processes").set_value(at.multiselect(key="processes").options)
    card = at.session_state["reader"].card
    # Prefer shape? to exercise expected missing endpoints on real cards.
    choices = at.selectbox(key="compare_nuisance").options
    default = next((name for name in choices if card.nuisances.get(name) == "shape?"), choices[0])
    nuisance = os.environ.get("DATACARD_VIEWER_NUISANCE", default)
    print(f"Compare: {len(at.multiselect(key='channels').value)} channels, nuisance={nuisance}")
    timed("First plot", lambda: at.selectbox(key="compare_nuisance").set_value(nuisance).run())
    assert not at.exception and not at.error
    for log_y in (True, False):
        timed(f"Log y={log_y}", lambda: next(
            c for c in at.checkbox if c.label == "Logarithmic y-axis"
        ).set_value(log_y).run(timeout=10))
        assert not at.exception and not at.error
        assert at.get("plotly_chart")
    at.session_state["reader"].close()
    at.session_state["compare_reader"].close()
