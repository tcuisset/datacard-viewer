"""Plotly figures and data exports, without Streamlit dependencies."""

import html
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .core import Comparison, ratios


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


def _draw_comparison(
    fig,
    c,
    log_y=False,
    offset=0.0,
    channel=None,
    showlegend=True,
    variation_bands=False,
):
    edges = c.nominal.edges
    # Explicit pairs of edges retain true bin widths and gaps at masked bins.
    x = np.column_stack((edges[:-1], edges[1:])).ravel() + offset
    custom = np.repeat(np.column_stack((edges[:-1], edges[1:])), 2, axis=0)

    def trace(values, name, color, row, band=None):
        y = np.repeat(values, 2)
        if band is not None:
            lower, upper = np.repeat(values - band, 2), np.repeat(values + band, 2)
            if log_y and row == 1:
                lower = np.where(lower > 0, lower, np.nan)
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
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
                legendgroup=name,
                showlegend=showlegend and row == 1,
                connectgaps=False,
                customdata=custom,
                hovertemplate=(html.escape(channel) + "<br>" if channel else "")
                + name
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
    for h, name, color in zip([c.up, c.down], ["Up", "Down"], colors[1:]):
        trace(ratios(h.values, n), name, color, 2)
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
            Comparison(
                c.nominal,
                c.up.scaled(nominal / up),
                c.down.scaled(nominal / down),
                c.warnings,
                c.coefficients,
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
    comparisons, remove_normalization=False, variation_bands=False, log_y=False
):
    displayed = display_channels(comparisons, remove_normalization)
    fig = _base_figure(log_y)
    tickvals, ticktext = [], []
    previous_end = None
    for i, (name, c, offset) in enumerate(channel_segments(displayed)):
        _draw_comparison(fig, c, log_y, offset, name, i == 0, variation_bands)
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
    _set_ratio_range(fig, [c for _, c in displayed])
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
