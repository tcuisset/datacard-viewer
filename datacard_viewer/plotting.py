"""Plotly figures and data exports, without Streamlit dependencies."""

import html
import textwrap
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .core import ratios


def _base_figure(log_y=False):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.68, 0.32],
    )
    fig.update_yaxes(
        title_text="Events / bin", type="log" if log_y else "linear", row=1, col=1
    )
    fig.update_yaxes(title_text="Variation / nominal", row=2, col=1)
    fig.update_xaxes(title_text="Observable", row=2, col=1)
    fig.update_layout(
        height=610,
        margin=dict(l=55, r=25, t=35, b=45),
        legend=dict(orientation="h", y=1.06),
        template="plotly_white",
    )
    return fig


def _effect_label(comparison):
    types = comparison.effect_types
    if types is None:
        return ""
    if not types:
        return "No local effect"
    if len(types) > 1:
        return "Shape + log-normal"
    return "Shape" if "shape" in types else "Log-normal (lnN)"


def _ratio_label(fig, x, text):
    if text:
        fig.add_annotation(
            x=x, xref="x2", y=1, yref="y2 domain",
            text=text, showarrow=False, yanchor="bottom", yshift=2, align="left",
            font=dict(size=11, color="#364152"),
            bgcolor="rgba(255,255,255,0.9)",
        )


def _draw_comparison(
    fig,
    c,
    log_y=False,
    offset=0.0,
    channel=None,
    showlegend=True,
    variation_bands=False,
    unsmoothed=None,
):
    edges = c.nominal.edges
    # Explicit pairs of edges retain true bin widths and gaps at masked bins.
    x = np.column_stack((edges[:-1], edges[1:])).ravel() + offset
    custom = np.repeat(np.column_stack((edges[:-1], edges[1:])), 2, axis=0)
    effect = _effect_label(c)

    def trace(values, name, color, row, band=None, *, smoothed=False, original=False):
        # Plotly's log axis cannot represent zero or negative values. Leave
        # those bins as gaps rather than disabling log scale or flooring them.
        if log_y and row == 1:
            values = np.where(values > 0, values, np.nan)
        y = np.repeat(values, 2)
        if band is not None:
            lower_values, upper_values = values - band, values + band
            if log_y and row == 1:
                # A band crossing zero cannot be represented faithfully on a
                # log axis, so omit the whole band for that bin.
                valid_band = (lower_values > 0) & (upper_values > 0)
                lower_values = np.where(valid_band, lower_values, np.nan)
                upper_values = np.where(valid_band, upper_values, np.nan)
            lower, upper = np.repeat(lower_values, 2), np.repeat(upper_values, 2)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=lower,
                    mode="lines",
                    line_width=0,
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=name,
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=upper,
                    mode="lines",
                    line_width=0,
                    fill="tonexty",
                    fillcolor=color.replace("rgb", "rgba").replace(")", ",0.14)"),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=name,
                ),
                row=row,
                col=1,
            )
        label = name
        if original:
            label += " (unsmoothed)"
        elif smoothed:
            label += " (smoothed)"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=label,
                line=dict(
                    color=color, width=1.25 if original else 2,
                    dash="dash" if original else "solid",
                ),
                opacity=0.65 if original else 1.0,
                legendgroup=label if original else name,
                showlegend=showlegend and (row == 1 or original),
                connectgaps=False,
                customdata=custom,
                hovertemplate=(html.escape(channel) + "<br>" if channel else "")
                + label
                + (f"<br>{effect}" if row == 2 and name in ("Up", "Down") and effect else "")
                + "<br>bin=[%{customdata[0]:g}, %{customdata[1]:g}]<br>value=%{y:.6g}<extra></extra>",
            ),
            row=row,
            col=1,
        )

    colors = ["rgb(50,65,85)", "rgb(30,125,185)", "rgb(220,100,40)"]
    for h, name, color in zip(
        [c.nominal, c.up, c.down], ["Nominal", "Up", "Down"], colors
    ):
        trace(
            h.values,
            name,
            color,
            1,
            np.sqrt(h.variances) if name == "Nominal" or variation_bands else None,
        )
    n = c.nominal.values
    trace(
        np.where(n != 0, 1.0, np.nan),
        "Nominal reference",
        colors[0],
        2,
        ratios(np.sqrt(c.nominal.variances), np.abs(n)),
    )
    if unsmoothed is not None:
        for h, name, color in zip((unsmoothed.up, unsmoothed.down), ("Up", "Down"), colors[1:]):
            trace(ratios(h.values, n), name, color, 2, original=True)
    for h, name, color in zip((c.up, c.down), ("Up", "Down"), colors[1:]):
        trace(ratios(h.values, n), name, color, 2, smoothed=unsmoothed is not None)
    _ratio_label(fig, (edges[0] + edges[-1]) / 2 + offset, effect)
    if np.any(n == 0):
        centers = (edges[:-1] + edges[1:]) / 2
        fig.add_trace(
            go.Scatter(
                x=centers[n == 0] + offset,
                y=np.ones(np.sum(n == 0)),
                mode="markers",
                marker=dict(symbol="x", color="red", size=10),
                name="Undefined ratio: nominal zero",
                legendgroup="undefined",
                showlegend=not any(
                    t.name == "Undefined ratio: nominal zero" for t in fig.data
                ),
                hovertemplate=(html.escape(channel) + "<br>" if channel else "")
                + "Undefined ratio: nominal zero<extra></extra>",
            ),
            row=2,
            col=1,
        )


def _set_ratio_range(fig, comparisons):
    """Fit displayed Up/Down ratios, never statistical-band envelopes."""
    low = high = 1.0
    for c in comparisons:
        for h in (c.up, c.down):
            values = ratios(h.values, c.nominal.values)
            finite = values[np.isfinite(values)]
            if finite.size:
                low = min(low, float(finite.min()))
                high = max(high, float(finite.max()))
    padding = 0.1 * (high - low) if high > low else 0.01
    fig.update_yaxes(
        range=[low - padding, high + padding], autorange=False, row=2, col=1
    )


def figure(comparison, remove_normalization=False, variation_bands=False, log_y=False):
    c = comparison.normalized() if remove_normalization else comparison
    fig = _base_figure(log_y)
    _draw_comparison(fig, c, log_y, variation_bands=variation_bands)
    _set_ratio_range(fig, [c])
    return fig


def display_channels(comparisons, remove_normalization=False):
    """Return ordered channels, optionally using ONE normalization across them."""
    comparisons = list(comparisons)
    if not comparisons:
        raise ValueError("Select at least one channel")
    if not remove_normalization:
        return comparisons
    nominal = sum(c.nominal.integral for _, c in comparisons)
    up = sum(c.up.integral for _, c in comparisons)
    down = sum(c.down.integral for _, c in comparisons)
    if min(nominal, up, down) <= 0:
        raise ValueError(
            "Normalization removal requires positive totals across selected channels"
        )
    return [
        (
            name,
            replace(
                c,
                up=c.up.scaled(nominal / up),
                down=c.down.scaled(nominal / down),
            ),
        )
        for name, c in comparisons
    ]


def channel_segments(comparisons):
    """Offsets preserve observable units and proportional channel spans."""
    gap = 0.10 * max(c.nominal.edges[-1] - c.nominal.edges[0] for _, c in comparisons)
    cursor = comparisons[0][1].nominal.edges[0]
    result = []
    for name, c in comparisons:
        offset = cursor - c.nominal.edges[0]
        result.append((name, c, offset))
        cursor += c.nominal.edges[-1] - c.nominal.edges[0] + gap
    return result


def channel_figure(
    comparisons, remove_normalization=False, variation_bands=False, log_y=False,
    *,
    unsmoothed=None,
):
    displayed = display_channels(comparisons, remove_normalization)
    originals = dict(display_channels(unsmoothed, remove_normalization)) if unsmoothed else {}
    fig = _base_figure(log_y)
    tickvals, ticktext = [], []
    previous_end = None
    for i, (name, c, offset) in enumerate(channel_segments(displayed)):
        _draw_comparison(
            fig, c, log_y, offset, name, i == 0, variation_bands, originals.get(name)
        )
        edges = c.nominal.edges
        ticks = np.linspace(edges[0], edges[-1], 5)
        tickvals.extend(ticks + offset)
        ticktext.extend(f"{x:g}" for x in ticks)
        label = "<br>".join(
            html.escape(line)
            for line in textwrap.wrap(
                name, width=max(12, min(30, 80 // len(displayed)))
            )
        )
        fig.add_annotation(
            x=(edges[0] + edges[-1]) / 2 + offset,
            y=1.02,
            xref="x",
            yref="paper",
            text=label,
            showarrow=False,
            yanchor="bottom",
            font=dict(size=11),
        )
        if previous_end is not None:
            boundary = (previous_end + edges[0] + offset) / 2
            for row in (1, 2):
                fig.add_vline(
                    x=boundary,
                    line_width=1,
                    line_dash="dot",
                    line_color="#8899aa",
                    row=row,
                    col=1,
                )
        previous_end = edges[-1] + offset
    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=-45 if len(displayed) > 1 else 0,
        tickfont=dict(size=10),
    )
    fig.update_layout(
        margin=dict(l=55, r=25, t=115, b=45),
        height=690,
        legend=dict(orientation="h", y=1.17, yanchor="top"),
    )
    _set_ratio_range(fig, [c for _, c in displayed] + list(originals.values()))
    return fig


def channels_frame(comparisons, remove_normalization=False):
    frames = []
    for i, (name, c, offset) in enumerate(
        channel_segments(display_channels(comparisons, remove_normalization))
    ):
        frame = bins_frame(c).assign(
            channel=name,
            display_order=i + 1,
            display_low=c.nominal.edges[:-1] + offset,
            display_high=c.nominal.edges[1:] + offset,
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def compare_figure(
    before, after, remove_normalization=False, log_y=False,
    *,
    before_unsmoothed=None,
    after_unsmoothed=None,
):
    """Overlay two card predictions. Returns (figure, binning_matches)."""
    before, after = list(before), list(after)
    before_unsmoothed = list(before_unsmoothed or ())
    after_unsmoothed = list(after_unsmoothed or ())
    if [n for n, _ in before] != [n for n, _ in after]:
        raise ValueError("Before and After channel selections must match")
    matches = all(
        np.array_equal(b.nominal.edges, a.nominal.edges)
        for (_, b), (_, a) in zip(before, after)
    )
    if not matches:
        return None, False

    if remove_normalization:
        all_totals = [
            sum(getattr(c, field).integral for _, c in items)
            for items in (before, after, before_unsmoothed, after_unsmoothed)
            if items
            for field in ("nominal", "up", "down")
        ]
        if min(all_totals) <= 0:
            remove_normalization = False
    before = display_channels(before, remove_normalization)
    after = display_channels(after, remove_normalization)
    before_original = (
        dict(display_channels(before_unsmoothed, remove_normalization)) if before_unsmoothed else {}
    )
    after_original = (
        dict(display_channels(after_unsmoothed, remove_normalization)) if after_unsmoothed else {}
    )
    fig = _base_figure(log_y)
    direction = {"Nominal": "rgb(55,65,80)", "Up": "rgb(30,125,185)", "Down": "rgb(220,100,40)"}
    gap = 0.1 * max(c.nominal.edges[-1] - c.nominal.edges[0] for _, c in before)
    cursor = before[0][1].nominal.edges[0]
    for channel_index, ((channel, b), (_, a)) in enumerate(zip(before, after)):
        offset = cursor - b.nominal.edges[0]
        edges = b.nominal.edges
        x = np.column_stack((edges[:-1], edges[1:])).ravel() + offset
        custom = np.repeat(np.column_stack((edges[:-1], edges[1:])), 2, axis=0)
        effect_labels = []
        for side, comparison, original, dash in (
            ("Before", b, before_original.get(channel), "dash"),
            ("After", a, after_original.get(channel), "solid"),
        ):
            effect = _effect_label(comparison)
            if effect:
                effect_labels.append(f"{side} · <b>{effect}</b>")
            effect_hover = f"<br>{effect}" if effect else ""
            for name, template in zip(("Nominal", "Up", "Down"), (comparison.nominal, comparison.up, comparison.down)):
                label = f"{side} {name}"
                fig.add_trace(go.Scatter(x=x, y=np.repeat(template.values, 2), mode="lines", name=label, legendgroup=label, showlegend=channel_index == 0, connectgaps=False, customdata=custom, line=dict(color=direction[name], dash=dash, width=2), hovertemplate=f"{html.escape(channel)}<br>{label}<br>bin=[%{{customdata[0]:g}}, %{{customdata[1]:g}}]<br>value=%{{y:.6g}}<extra></extra>"), row=1, col=1)
            ratio_curves = [(comparison, False)]
            if original is not None:
                ratio_curves.insert(0, (original, True))
            for curve, is_original in ratio_curves:
                suffix = ""
                ratio_dash = dash
                if is_original:
                    suffix = " (unsmoothed)"
                    ratio_dash = "dot" if dash == "dash" else "dashdot"
                elif original is not None:
                    suffix = " (smoothed)"
                for name, template in (("Up", curve.up), ("Down", curve.down)):
                    values = ratios(template.values, curve.nominal.values)
                    label = f"{side} {name} / nominal{suffix}"
                    fig.add_trace(
                        go.Scatter(
                            x=x, y=np.repeat(values, 2), mode="lines", name=label,
                            legendgroup=label if is_original else f"{side} {name}",
                            showlegend=channel_index == 0, connectgaps=False,
                            customdata=custom, opacity=0.65 if is_original else 1.0,
                            line=dict(
                                color=direction[name], dash=ratio_dash,
                                width=1.25 if is_original else 2,
                            ),
                            hovertemplate=f"{html.escape(channel)}<br>{label}{effect_hover}<br>bin=[%{{customdata[0]:g}}, %{{customdata[1]:g}}]<br>ratio=%{{y:.6g}}<extra></extra>",
                        ),
                        row=2, col=1,
                    )
        center = (edges[0] + edges[-1]) / 2 + offset
        fig.add_annotation(x=center, y=1.02, xref="x", yref="paper", text=html.escape(channel), showarrow=False, yanchor="bottom", font=dict(size=11))
        _ratio_label(fig, center, "<br>".join(effect_labels))
        cursor += edges[-1] - edges[0] + gap
    plotted_ratios = [c for _, c in (*before, *after)]
    plotted_ratios.extend(before_original.values())
    plotted_ratios.extend(after_original.values())
    _set_ratio_range(fig, plotted_ratios)
    fig.update_layout(height=690, margin=dict(l=55, r=25, t=115, b=45), legend=dict(orientation="h", y=1.17, yanchor="top"))
    return fig, True


def bins_frame(comparison, remove_normalization=False):
    c = comparison.normalized() if remove_normalization else comparison
    return pd.DataFrame(
        dict(
            low=c.nominal.edges[:-1],
            high=c.nominal.edges[1:],
            nominal=c.nominal.values,
            nominal_variance=c.nominal.variances,
            up=c.up.values,
            up_variance=c.up.variances,
            down=c.down.values,
            down_variance=c.down.variances,
            up_over_nominal=ratios(c.up.values, c.nominal.values),
            down_over_nominal=ratios(c.down.values, c.nominal.values),
        )
    )
