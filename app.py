"""Launch with: uv run streamlit run app.py -- --card path/to/card.txt"""

import argparse
import html
import time
from collections import Counter

import pandas as pd
import streamlit as st

from datacard_viewer.core import (
    Card,
    Reader,
    Scan,
    compare,
    compare_response,
    comparison_shortlist,
    diagnose,
    effect_rows,
    fingerprint,
    summarize,
)
from datacard_viewer.plotting import (
    channel_figure,
    channels_frame,
    compare_figure,
    display_channels,
)
from datacard_viewer.reporting import scan_exports

st.set_page_config(page_title="Combine shape viewer", page_icon="📊", layout="wide")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", default="")
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument("--compare-card", default="")
    parser.add_argument("--compare-path-map", action="append", default=[])
    parser.add_argument("--mass", type=float, default=125)
    parser.add_argument("--keyword", action="append", default=[])
    return parser.parse_known_args()[0]


@st.cache_resource(show_spinner="Parsing datacard…", max_entries=2)
def load_card(revision):
    return Card.load(revision[0])


def parse_pairs(text):
    result = []
    for line in text.splitlines():
        if line.strip():
            if "=" not in line:
                raise ValueError(f"Expected prefix=replacement: {line}")
            a, b = line.strip().split("=", 1)
            if not a:
                raise ValueError("Mapping/keyword name cannot be empty")
            result.append((a, b))
    return tuple(result)


def cancel_scans():
    for key in ("scan", "channel_scan"):
        job = st.session_state.get(key)
        if job and not job.done:
            job.cancel.set()
            job.thread.join(timeout=2)
        st.session_state.pop(key, None)


@st.fragment(run_every=1)
def scan_panel(card, mappings, mass, keywords, threshold, channels):
    st.subheader("Nuisance diagnosis")
    st.caption(
        "Candidates are checked separately for every affected channel/process pair. "
        "Residual/statistical-error scores are descriptive, not compatibility tests."
    )
    a, b, c = st.columns(3)
    active = any(
        st.session_state.get(k) and not st.session_state[k].done
        for k in ("scan", "channel_scan")
    )
    if a.button(
        "Scan selected channel" if len(channels) == 1 else "Scan selected channels",
        disabled=active or not channels,
    ):
        st.session_state["channel_scan"] = Scan(
            card, mappings, mass, keywords, channels
        ).start()
    if b.button("Scan whole card", disabled=active):
        st.session_state["scan"] = Scan(card, mappings, mass, keywords).start()
    if c.button("Cancel scan", disabled=not active):
        for key in ("scan", "channel_scan"):
            if st.session_state.get(key):
                st.session_state[key].cancel.set()
    for key, title in [
        ("channel_scan", "Selected-channel results"),
        ("scan", "Whole-card results"),
    ]:
        job = st.session_state.get(key)
        if not job:
            continue
        st.markdown(f"**{title}**")
        if key == "channel_scan":
            st.caption(
                "Scan scope: "
                + " · ".join(ch for ch in card.channels if ch in job.channels)
            )
        rows, _ = job.snapshot()
        elapsed = (job.finished or time.monotonic()) - job.started
        status = (
            "Cancelled"
            if job.cancel.is_set()
            else ("Complete" if job.done else "Scanning")
        )
        st.progress(
            len(rows) / max(job.total, 1),
            text=f"{status}: {len(rows):,}/{job.total:,} pairs · {elapsed:.0f}s",
        )
        if job.error:
            st.error(job.error)
        stale = job.stale()
        if stale:
            st.warning("Results are stale: an input changed. Reload and rescan.")
        if not rows:
            continue
        summary = pd.DataFrame(summarize(card, rows, threshold, job.channels))
        if stale:
            summary["candidate"] = False
        st.dataframe(summary, hide_index=True, width="stretch", key=key + "_summary")
        # Build large CSVs once per completed scan/threshold, not every polling tick.
        if job.done:
            pair_csv, summary_csv = completed_exports(
                job.started, threshold, stale, card, rows, job.channels
            )
            st.download_button(
                "Download all pair results (CSV)",
                pair_csv,
                file_name=key + "_pairs.csv",
                mime="text/csv",
                key=key + "_download",
            )
            st.download_button(
                "Download nuisance summary (CSV)",
                summary_csv,
                file_name=key + "_summary.csv",
                mime="text/csv",
                key=key + "_summary_download",
            )
            with st.expander("Pair results for the selected channels"):
                st.dataframe(
                    pd.DataFrame([r for r in rows if r["channel"] in channels]),
                    hide_index=True,
                    width="stretch",
                )


@st.cache_resource(max_entries=1, show_spinner=False)
def completed_exports(token, threshold, stale, _card, _rows, _channels):
    # Immutable bytes can be shared without cache_data's large pickle copies.
    return scan_exports(_card, _rows, threshold, _channels, stale)


def keep_valid_selection(key, options):
    previous = st.session_state.get(key)
    valid = set(options)
    kept = [value for value in previous or () if value in valid]
    if previous is None or (previous and not kept):
        kept = options[:1]
    if previous != kept:
        st.session_state[key] = kept


def select_channels(options):
    keep_valid_selection("channels", options)
    channels = st.multiselect(
        "Channels",
        options,
        key="channels",
        help="Channels appear in the plot in selection order.",
        placeholder="Choose channels",
    )
    with st.expander("Select channels by name"):
        query = (
            st.text_input("Filter channel names", key="channel_filter")
            .strip()
            .casefold()
        )
        matches = [ch for ch in options if query and query in ch.casefold()]
        if query:
            st.caption(f"{len(matches)} matching channel(s)")
        st.button(
            "Select matching channels",
            disabled=not matches,
            help="Replace the current selection with case-insensitive substring matches in datacard order.",
            on_click=st.session_state.__setitem__,
            args=("channels", matches),
        )
    return channels


def select_processes(cards, channels):
    available = list(
        dict.fromkeys(p for ch in channels for card in cards for p in card.channels[ch])
    )
    keep_valid_selection("processes", available)
    processes = st.multiselect(
        "Processes",
        available,
        key="processes",
        help="Sum the selected processes. A process absent from a channel or card contributes zero yield there.",
        placeholder="Choose processes",
    )
    with st.container(horizontal=True, gap="xsmall"):
        for label, signal, help_text in (
            ("Backgrounds", False, "Select all backgrounds"),
            ("Signals", True, "Select all signals"),
            ("All", None, "Select all processes"),
        ):
            preset = [
                p
                for p in available
                if signal is None
                or any(card.parsed.isSignal.get(p) is signal for card in cards)
            ]
            st.button(
                label,
                help=help_text,
                disabled=not preset,
                on_click=st.session_state.__setitem__,
                args=("processes", preset),
            )
    return processes


def select_nuisance(cards, candidates, allow_empty):
    groups = list(dict.fromkeys(g for card in cards for g in card.parsed.groups))
    with st.expander("Filter nuisances"):
        if st.session_state.get("nuisance_group") not in (None, *groups):
            st.session_state["nuisance_group"] = None
        group = st.selectbox(
            "Nuisance group",
            groups,
            index=None,
            key="nuisance_group",
            placeholder="All groups",
        )
        query = st.text_input("Filter nuisance name").strip().casefold()
    members = {n for card in cards for n in card.parsed.groups.get(group, ())}
    names = [
        n
        for n in candidates
        if query in n.casefold() and (group is None or n in members)
    ]

    key = "compare_nuisance" if allow_empty else "nuisance"
    current = st.session_state.get(key)
    if current not in names:
        current = None if allow_empty or not names else names[0]
        st.session_state[key] = current
    previous = following = None
    if names:
        idx = names.index(current) if current is not None else -1
        previous = names[(idx - 1) % len(names)] if idx >= 0 else names[-1]
        following = names[(idx + 1) % len(names)]
    with st.container(horizontal=True, gap="xsmall", vertical_alignment="bottom"):
        st.button(
            "←",
            help="Previous nuisance",
            disabled=previous == current,
            on_click=st.session_state.__setitem__,
            args=(key, previous),
        )
        nuisance = st.selectbox("Nuisance", names, index=None, key=key)
        st.button(
            "→",
            help="Next nuisance",
            disabled=following == current,
            on_click=st.session_state.__setitem__,
            args=(key, following),
        )
    st.caption(f"{len(names)} matching nuisance(s)")
    return nuisance, len(names)


def processes_by_channel(card, channels, selected):
    return {ch: [p for p in card.channels[ch] if p in selected] for ch in channels}


def visible_channels(comparisons, hide_unaffected, after=()):
    if not hide_unaffected:
        return {ch for ch, _ in comparisons}
    shown = {ch for group in (comparisons, after) for ch, c in group if c.has_effect}
    if not shown:
        st.info(
            "This nuisance changes no bin yields for the selected channels and processes."
        )
    elif len(shown) < len(comparisons):
        st.caption(
            f"Showing {len(shown)} of {len(comparisons)} selected channels in the plot."
        )
    return shown


def smooth_channels(comparisons, passes):
    return [(ch, c.smoothed(passes)) for ch, c in comparisons]


def plotted_channels(comparisons, normalize, shown):
    # Normalize the full selection before hiding channels to preserve plotted values.
    return [
        item for item in display_channels(comparisons, normalize) if item[0] in shown
    ]


def inspect_shapes(
    card,
    reader,
    channels,
    selected,
    nuisance,
    normalize,
    log_y,
    bands,
    hide_unaffected,
    smoothing_passes,
):
    processes = processes_by_channel(card, channels, selected)
    try:
        comparisons = []
        for ch in channels:
            try:
                comparisons.append((ch, compare(reader, ch, processes[ch], nuisance)))
            except Exception as exc:
                raise ValueError(f"Channel {ch}: {exc}") from exc
        warnings = list(
            dict.fromkeys(
                f"{ch}: {warning}"
                for ch, c in comparisons
                for warning in (
                    *c.warnings,
                    *c.nominal.warnings,
                    *c.up.warnings,
                    *c.down.warnings,
                )
            )
        )
        flow_notices = [w for w in warnings if "under/overflow" in w]
        for warning in warnings:
            if (
                warning not in flow_notices
                and "shape? log-normal fallback" not in warning
            ):
                st.warning(warning)
        if flow_notices:
            with st.expander(f"Excluded under/overflow reports ({len(flow_notices)})"):
                for notice in flow_notices:
                    st.write(notice)
        shown = visible_channels(comparisons, hide_unaffected)
        n, u, d = (
            sum(getattr(c, name).integral for _, c in comparisons)
            for name in ("nominal", "up", "down")
        )
        cols = st.columns(3)
        cols[0].metric("Nominal integral", f"{n:,.6g}")
        cols[1].metric(
            "Up integral", f"{u:,.6g}", f"{(u / n - 1) * 100:+.3g}%" if n else None
        )
        cols[2].metric(
            "Down integral",
            f"{d:,.6g}",
            f"{(d / n - 1) * 100:+.3g}%" if n else None,
        )
        if shown:
            plot_inputs = smooth_channels(comparisons, smoothing_passes)
            plotted = plotted_channels(plot_inputs, normalize, shown)
            unsmoothed = (
                plotted_channels(comparisons, normalize, shown) if smoothing_passes else None
            )
            if log_y:
                masked = []
                for ch, comparison in plotted:
                    for label, template in zip(
                        ("Nominal", "Up", "Down"),
                        (comparison.nominal, comparison.up, comparison.down),
                    ):
                        zero = int((template.values == 0).sum())
                        negative = int((template.values < 0).sum())
                        if zero or negative:
                            details = []
                            if zero:
                                details.append(f"{zero} zero")
                            if negative:
                                details.append(f"{negative} negative")
                            masked.append(f"{ch} · {label}: {', '.join(details)}")
                if masked:
                    st.warning(
                        "Log y is active; nonpositive bins are shown as gaps: "
                        + "; ".join(masked)
                        + ". The table and CSV export include those bin values."
                    )
            fig = channel_figure(
                plotted, variation_bands=bands, log_y=log_y, unsmoothed=unsmoothed
            )
            frame = channels_frame(plotted)
            st.plotly_chart(fig, width="stretch", key="shape_plot")
            st.caption(
                "Bands: stored template statistical errors. Ratio band: nominal fractional error reference, "
                "not a propagated ratio uncertainty. Process sums assume independent template errors."
                + (
                    " Normalized bands are scaled stored errors, without normalization covariance."
                    if normalize
                    else ""
                )
            )
        rows = [
            dict(
                channel=ch,
                nominal=c.nominal.integral,
                up=c.up.integral,
                down=c.down.integral,
                **diagnose(c),
            )
            for ch, c in comparisons
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "Diagnostics remove normalization separately within each channel; normalization-only screening uses individual processes."
            + (
                " The plotted normalization instead uses totals across the selected channels."
                if shown and normalize and len(channels) > 1
                else ""
            )
        )
        if shown:
            frame = frame.assign(nuisance=nuisance, normalized=normalize)
            smoothing_label = ""
            if smoothing_passes:
                frame["smoothing_passes"] = smoothing_passes
                smoothing_label = f" · ROOT ratio smoothing passes: {smoothing_passes}"
            frame["processes"] = frame.channel.map(lambda ch: " + ".join(processes[ch]))
            a, b = st.columns(2)
            a.download_button(
                "Download plotted bins (CSV)",
                frame.to_csv(index=False),
                "shape_bins.csv",
                "text/csv",
            )
            plotted_names = " · ".join(ch for ch, _ in plotted)
            b.download_button(
                "Download interactive plot (HTML)",
                fig.to_html(include_plotlyjs=True).replace(
                    "<body>",
                    f"<body><h2>{html.escape(nuisance)}</h2><p>{html.escape(plotted_names)} · {html.escape(' + '.join(selected))} · normalization removed: {normalize}{smoothing_label}</p>",
                ),
                "shape_plot.html",
                "text/html",
            )
        with st.expander("Template provenance and bin values"):
            st.json(
                {
                    ch: dict(
                        coefficients=c.coefficients,
                        nominal=c.nominal.sources,
                        up=c.up.sources,
                        down=c.down.sources,
                    )
                    for ch, c in comparisons
                }
            )
            if shown:
                st.dataframe(frame, hide_index=True)
    except Exception as exc:
        st.error(str(exc))


def compare_shapes(
    before_reader,
    after_reader,
    channels,
    selected,
    nuisance,
    normalize,
    log_y,
    hide_unaffected,
    smoothing_passes,
):
    before, after = before_reader.card, after_reader.card
    b_processes = processes_by_channel(before, channels, selected)
    a_processes = processes_by_channel(after, channels, selected)
    try:
        before_comparisons, after_comparisons = [], []
        for ch in channels:
            before_comparisons.append(
                (
                    ch,
                    compare_response(
                        before_reader, ch, b_processes[ch], nuisance, "Before"
                    ),
                )
            )
            after_comparisons.append(
                (
                    ch,
                    compare_response(
                        after_reader, ch, a_processes[ch], nuisance, "After"
                    ),
                )
            )
        for items in (before_comparisons, after_comparisons):
            for _, comparison in items:
                for warning in comparison.warnings:
                    if "shape? log-normal fallback" not in warning:
                        st.warning(warning)
        shown = visible_channels(before_comparisons, hide_unaffected, after_comparisons)
        if shown:
            before_inputs = smooth_channels(before_comparisons, smoothing_passes)
            after_inputs = smooth_channels(after_comparisons, smoothing_passes)
            normalized_groups = [before_inputs, after_inputs]
            if smoothing_passes:
                normalized_groups.extend((before_comparisons, after_comparisons))
            plot_totals = [
                sum(getattr(c, field).integral for _, c in items)
                for items in normalized_groups
                for field in ("nominal", "up", "down")
            ]
            if normalize and min(plot_totals) <= 0:
                st.warning(
                    "Normalization removal requires positive nominal, Up, and Down totals in both cards. Showing plots without normalization removal."
                )
                normalize = False
            before_plot = plotted_channels(before_inputs, normalize, shown)
            after_plot = plotted_channels(after_inputs, normalize, shown)
            before_original = after_original = None
            if smoothing_passes:
                before_original = plotted_channels(before_comparisons, normalize, shown)
                after_original = plotted_channels(after_comparisons, normalize, shown)
            fig, same_binning = compare_figure(
                before_plot,
                after_plot,
                log_y=log_y,
                before_unsmoothed=before_original,
                after_unsmoothed=after_original,
            )
            if same_binning:
                st.plotly_chart(fig, width="stretch", key="compare_shape_plot")
            else:
                st.warning(
                    "Binning differs between cards; Before and After are shown separately with their own binning."
                )
                a, b = st.columns(2)
                a.markdown("**Before**")
                b.markdown("**After**")
                a.plotly_chart(
                    channel_figure(before_plot, log_y=log_y, unsmoothed=before_original),
                    width="stretch",
                    key="compare_before_plot",
                )
                b.plotly_chart(
                    channel_figure(after_plot, log_y=log_y, unsmoothed=after_original),
                    width="stretch",
                    key="compare_after_plot",
                )
        summaries = []
        for side, items in (
            ("Before", before_comparisons),
            ("After", after_comparisons),
        ):
            summaries.extend(
                {
                    "Card": side,
                    "channel": ch,
                    "nominal": c.nominal.integral,
                    "up": c.up.integral,
                    "down": c.down.integral,
                }
                for ch, c in items
            )
        st.dataframe(pd.DataFrame(summaries), hide_index=True, width="stretch")
    except Exception as exc:
        st.error(str(exc))

    with st.expander("Selected nuisance effects"):
        displayed = {
            ch: list(dict.fromkeys((*b_processes[ch], *a_processes[ch])))
            for ch in channels
        }
        rows = effect_rows(before, after, nuisance, channels, displayed)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "Asymmetric lnN factors are shown in down/up order. A dropped effect "
            "plots at unity when nominal is nonzero."
        )


def comparison_differences(before, after, channels, shortlist):
    with st.expander("Datacard differences"):
        st.markdown("**Channels**")
        added = [ch for ch in after.channels if ch not in before.channels]
        removed = [ch for ch in before.channels if ch not in after.channels]
        st.write("Added · " + (", ".join(added) or "None"))
        st.write("Removed · " + (", ".join(removed) or "None"))
        if channels:
            st.markdown("**Process changes in selected channels**")
            changes = [
                {"channel": ch, "process": p, "change": change}
                for ch in channels
                for source, other, change in (
                    (after, before, "Added"),
                    (before, after, "Removed"),
                )
                for p in source.channels[ch]
                if p not in other.channels[ch]
            ]
            if changes:
                st.dataframe(pd.DataFrame(changes), hide_index=True, width="stretch")
            else:
                st.caption(
                    "The selected channels have the same process names in both cards."
                )
        st.markdown("**Nuisance shortlist across both cards**")
        st.caption(
            "The shortlist uses datacard text and flags shape? effects for a ROOT check. "
            "Retained shape templates may also differ in ROOT."
        )
        if shortlist:
            st.dataframe(pd.DataFrame(shortlist), hide_index=True, width="stretch")
        else:
            st.caption("No nuisance edits or shape? effects found.")


@st.fragment
def viewer(reader, threshold, after_reader=None):
    before = reader.card
    after = after_reader.card if after_reader else None
    cards = (before, after) if after else (before,)
    if any(fingerprint(card.path) != card.revision for card in cards):
        st.rerun()

    options = [ch for ch in before.channels if not after or ch in after.channels]
    shortlist = comparison_shortlist(before, after) if after else []
    if after and not options:
        st.warning("The cards have no exactly matching channels to plot.")
        comparison_differences(before, after, [], shortlist)
        return

    controls, results = st.columns([1, 2], gap="medium")
    with controls:
        channels = select_channels(options)
        selected = select_processes(cards, channels) if channels else []
        if after:
            view = st.radio(
                "Nuisance view",
                ["Card edits", "All effects"],
                horizontal=True,
                help="Card edits shows changed declarations and shape? effects. All effects also includes retained declarations.",
            )
            candidates = (
                [row["nuisance"] for row in shortlist]
                if view == "Card edits"
                else dict.fromkeys((*before.declarations, *after.declarations))
            )
        else:
            relevant = {n for ch in channels for n in before.relevant.get(ch, ())}
            candidates = [n for n in before.nuisances if n in relevant]
        nuisance, count = select_nuisance(
            cards, candidates, allow_empty=after is not None
        )
        with st.expander("Plot options"):
            normalize = st.checkbox("Remove normalization effect")
            log_y = st.checkbox("Logarithmic y-axis")
            smoothing_passes = st.number_input(
                "ROOT smoothing passes",
                min_value=0,
                value=0,
                step=1,
                help="0 disables smoothing. Applies ROOT's TH1::SmoothArray to Up / Nominal "
                "and Down / Nominal separately in each channel. Zero nominal bins split "
                "the ratios into separate segments. Smoothing requires at least three "
                "bins per segment.",
            )
            bands = (
                st.checkbox("Show Up/Down statistical bands") if not after else False
            )
            hide_unaffected = st.checkbox(
                "Hide unaffected channels",
                help="Show channels where Up or Down changes a bin yield for the selected processes. "
                "In Compare, a change in either card keeps the channel visible.",
            )
            if normalize and len(channels) > 1:
                st.caption("Normalization uses totals across the selected channels.")
    with results:
        if not channels:
            st.info("Select at least one channel.")
        elif not selected:
            st.info("Select at least one process.")
        elif not count:
            message = "No matching nuisances. Adjust the nuisance filters."
            if not candidates:
                message = (
                    "No nuisances in this view. Try All effects."
                    if after
                    else "No shape nuisances in the selected channels."
                )
            st.info(message)
        elif nuisance is None:
            st.info("Choose a nuisance to load its templates.")
        else:
            st.subheader(nuisance)
            channel_label = (
                channels[0] if len(channels) == 1 else f"{len(channels)} channels"
            )
            process_label = (
                selected[0] if len(selected) == 1 else f"{len(selected)} processes"
            )
            kind = f" · {before.nuisances[nuisance]}" if not after else ""
            st.caption(f"{channel_label} · {process_label}{kind}")
            if smoothing_passes:
                st.caption(
                    f"Ratio smoothing passes: {smoothing_passes}. The ratio panel overlays "
                    "the unsmoothed ratios. Yield summaries and channel filtering use "
                    "the original templates."
                )
            if after:
                compare_shapes(
                    reader,
                    after_reader,
                    channels,
                    selected,
                    nuisance,
                    normalize,
                    log_y,
                    hide_unaffected,
                    smoothing_passes,
                )
            else:
                inspect_shapes(
                    before,
                    reader,
                    channels,
                    selected,
                    nuisance,
                    normalize,
                    log_y,
                    bands,
                    hide_unaffected,
                    smoothing_passes,
                )
        if after:
            comparison_differences(before, after, channels, shortlist)
    if not after:
        scan_panel(
            before, reader.mappings, reader.mass, reader.keywords, threshold, channels
        )


def main():
    args = arguments()
    st.title("Combine shape viewer")
    st.caption(
        "Inspect nominal and systematic ROOT templates · prefit inputs · no ROOT installation required"
    )
    with st.sidebar:
        with st.form("open_card"):
            path = st.text_input("Datacard path (on this server)", args.card)
            mapping_text = st.text_area(
                "ROOT path mappings (prefix=replacement, one per line)",
                "\n".join(args.path_map),
                help="For the combined example: output/= (empty replacement). Relative targets use the card directory.",
            )
            compare_path = st.text_input("After datacard (optional)", args.compare_card)
            compare_mapping_text = st.text_area(
                "After ROOT path mappings (prefix=replacement, one per line)",
                "\n".join(args.compare_path_map),
            )
            mass = st.number_input("Mass ($MASS)", value=args.mass)
            keyword_text = st.text_area(
                "Additional keywords (NAME=value)", "\n".join(args.keyword)
            )
            submitted = st.form_submit_button("Open / Reload", type="primary")
        threshold = (
            st.number_input(
                "Maximum shape residual for candidates (%)",
                min_value=0.0,
                value=1.0,
                step=0.1,
            )
            / 100
        )
        st.caption(
            "Candidate threshold is a practical review tolerance, not a statistical test."
        )
    if submitted:
        cancel_scans()
        for key in ("reader", "compare_reader"):
            old = st.session_state.pop(key, None)
            if old:
                old.close()
        load_card.clear()
        st.session_state["configuration"] = (
            path,
            mapping_text,
            mass,
            keyword_text,
            compare_path,
            compare_mapping_text,
        )
    elif "configuration" not in st.session_state and args.card:
        st.session_state["configuration"] = (
            args.card,
            "\n".join(args.path_map),
            args.mass,
            "\n".join(args.keyword),
            args.compare_card,
            "\n".join(args.compare_path_map),
        )
    config = st.session_state.get("configuration")
    if not config:
        st.info("Enter a datacard path in the sidebar to begin.")
        return
    try:
        path, mapping_text, mass, keyword_text, compare_path, compare_mapping_text = config
        mappings = parse_pairs(mapping_text)
        keywords = dict(parse_pairs(keyword_text))
        if set(keywords) & {"PROCESS", "CHANNEL", "SYSTEMATIC", "MASS"}:
            raise ValueError("Built-in substitution keywords cannot be overridden")
        card = load_card(fingerprint(path))
        identity = (card.revision, mappings, mass, tuple(keywords.items()))
        if (
            st.session_state.get("reader_identity") != identity
            or "reader" not in st.session_state
        ):
            cancel_scans()
            if st.session_state.get("reader"):
                st.session_state["reader"].close()
            st.session_state["reader"] = Reader(card, mappings, mass, keywords)
            st.session_state["reader_identity"] = identity
        reader = st.session_state["reader"]
        after_card = load_card(fingerprint(compare_path)) if compare_path.strip() else None
        after_mappings = parse_pairs(compare_mapping_text)
        if after_card:
            after_identity = (after_card.revision, after_mappings, mass, tuple(keywords.items()))
            if st.session_state.get("compare_reader_identity") != after_identity or "compare_reader" not in st.session_state:
                old = st.session_state.pop("compare_reader", None)
                if old:
                    old.close()
                st.session_state["compare_reader"] = Reader(after_card, after_mappings, mass, keywords)
                st.session_state["compare_reader_identity"] = after_identity
    except Exception as exc:
        st.error(f"Could not open datacard: {exc}")
        return
    if after_card:
        mode = st.radio("Mode", ["Inspect", "Compare"], index=1, horizontal=True, key="viewer_mode")
        if mode == "Compare":
            viewer(reader, threshold, st.session_state["compare_reader"])
            return
    viewer(reader, threshold)
    with st.expander("Datacard overview and file resolution"):
        a, b, c, d = st.columns(4)
        a.metric("Channels", len(card.channels))
        b.metric("Process columns", len(card.parsed.keyline))
        c.metric("Shape nuisances", len(card.nuisances))
        d.metric("Affected pairs", len(card.coefficients))
        st.write("Nuisance types:", dict(Counter(s[2] for s in card.parsed.systs)))
        channel = st.selectbox("Overview channel", list(card.channels))
        st.dataframe(
            pd.DataFrame(
                [
                    dict(
                        process=p,
                        signal=card.parsed.isSignal[p],
                        rate=card.parsed.exp[channel][p],
                    )
                    for p in card.channels[channel]
                ]
            ),
            hide_index=True,
        )
        st.write(
            "autoMCStats settings (threshold, include-signal, histogram mode):",
            card.parsed.binParFlags,
        )
        st.write(
            "Nuisance groups:",
            {g: len(members) for g, members in card.parsed.groups.items()},
        )
        st.write(
            "Other model metadata (displayed, not evaluated):",
            dict(
                rateParams=card.parsed.rateParams,
                extArgs=card.parsed.extArgs,
                constraints=[
                    s[:4]
                    for s in card.parsed.systs
                    if s[2] in ("constr", "param", "flatParam")
                ],
                frozen=sorted(card.parsed.frozenNuisances),
                edits=card.parsed.nuisanceEditLines,
            ),
        )
        resolutions = []
        for ch, ps in card.channels.items():
            for p in ps:
                try:
                    resolved, _ = reader.resolve(ch, p)
                    resolutions.append(
                        dict(
                            channel=ch,
                            process=p,
                            original=reader.shape_mapping(ch, p)[0],
                            resolved=str(resolved),
                            exists=resolved.exists(),
                        )
                    )
                except Exception as exc:
                    resolutions.append(
                        dict(channel=ch, process=p, resolved=str(exc), exists=False)
                    )
        missing = [r for r in resolutions if not r["exists"]]
        if missing:
            st.warning(
                f"{len(missing)} channel/process mappings have unresolved files."
            )
        st.dataframe(pd.DataFrame(missing or resolutions), hide_index=True, height=200)


main()
