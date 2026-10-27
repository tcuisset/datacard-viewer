"""Plotly figures and data exports, without Streamlit dependencies."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .core import ratios


def figure(comparison, remove_normalization=False, variation_bands=False, log_y=False):
    c = comparison.normalized() if remove_normalization else comparison
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.68, 0.32],
    )
    edges = c.nominal.edges
    # Explicit pairs of edges retain true bin widths and gaps at masked bins.
    x = np.column_stack((edges[:-1], edges[1:])).ravel()

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
                showlegend=row == 1,
                connectgaps=False,
                hovertemplate=name + "<br>x=%{x}<br>value=%{y:.6g}<extra></extra>",
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
                x=centers[n == 0],
                y=np.ones(np.sum(n == 0)),
                mode="markers",
                marker=dict(symbol="x", color="red", size=10),
                name="Undefined ratio: nominal zero",
            ),
            row=2,
            col=1,
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
