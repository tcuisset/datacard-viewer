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
    bins_frame,
    channel_figure,
    channels_frame,
    compare_figure,
    figure,
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
        "Scan selected channels" if len(channels) > 1 else "Scan selected channel",
        disabled=active,
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


@st.fragment
def inspector(card, reader, threshold, mappings, mass, keywords):
    if fingerprint(card.path) != card.revision:
        st.rerun()
    left, right = st.columns([1, 3])
    with left:
        multiple = st.toggle("Show multiple channels", value=False)
        last = st.session_state.get("last_channel", next(iter(card.channels)))
        if last not in card.channels:
            last = next(iter(card.channels))
        if multiple:
            if "multi_channels" not in st.session_state:
                st.session_state["multi_channels"] = [last]
            st.session_state["multi_channels"] = [
                ch for ch in st.session_state["multi_channels"] if ch in card.channels
            ]
            pattern = st.text_input(
                "Filter channel names",
                key="channel_filter",
                help="Case-insensitive substring matching, like nuisance filtering. "
                "Selecting matches replaces the channel selection in datacard order.",
            ).strip()
            matches = [
                ch
                for ch in card.channels
                if pattern and pattern.casefold() in ch.casefold()
            ]
            if pattern:
                st.caption(f"{len(matches)} matching channel(s)")
            if st.button("Select matching channels", disabled=not matches):
                st.session_state["multi_channels"] = matches
            channels = st.multiselect(
                "Channels (left-to-right order)",
                list(card.channels),
                key="multi_channels",
            )
            if not channels:
                st.info("Select at least one channel.")
                return
        else:
            if st.session_state.get("single_channel") not in card.channels:
                st.session_state["single_channel"] = last
            channel = st.selectbox("Channel", list(card.channels), key="single_channel")
            st.session_state["last_channel"] = channel
            channels = [channel]
        selection = st.radio(
            "Processes", ["One process", "All backgrounds", "All processes"]
        )
        available = list(dict.fromkeys(p for ch in channels for p in card.channels[ch]))
        process = st.selectbox(
            "Process", available, disabled=selection != "One process"
        )
        if multiple:
            st.caption(
                "Processes absent from a channel contribute zero yield in that channel."
            )
        processes = {ch: card.processes(ch, selection, process) for ch in channels}
        group = st.selectbox("Nuisance group", ["All groups", *card.parsed.groups])
        search = st.text_input("Filter nuisance name")
        names = [
            n
            for n in card.nuisances
            if (group == "All groups" or n in card.parsed.groups[group])
            and search.lower() in n.lower()
            and any(
                (n, ch, p) in card.coefficients
                for ch in channels
                for p in (card.channels[ch] if multiple else processes[ch])
            )
        ]
        if not names:
            st.info("No matching shape nuisances for this selection.")
            scan_panel(card, mappings, mass, keywords, threshold, channels)
            return
        if st.session_state.get("nuisance") not in names:
            st.session_state["nuisance"] = names[0]
        prev, nxt = st.columns(2)
        idx = names.index(st.session_state["nuisance"])
        if prev.button("← Previous"):
            st.session_state["nuisance"] = names[(idx - 1) % len(names)]
        if nxt.button("Next →"):
            st.session_state["nuisance"] = names[(idx + 1) % len(names)]
        nuisance = st.selectbox("Shape nuisance", names, key="nuisance")
        normalize = st.checkbox("Remove normalization effect")
        bands = st.checkbox("Show Up/Down statistical bands")
        log_y = st.checkbox("Logarithmic y-axis")
        st.caption(f"{len(channels)} channel(s) · {len(names)} matching nuisances")
        if multiple and normalize:
            st.caption(
                "One common normalization across selected channels; relative channel yield changes remain visible."
            )
    with right:
        st.subheader(nuisance)
        st.caption(f"{' · '.join(channels)} · {selection} · {card.nuisances[nuisance]}")
        try:
            comparisons = []
            for ch in channels:
                try:
                    comparisons.append(
                        (ch, compare(reader, ch, processes[ch], nuisance))
                    )
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
                with st.expander(
                    f"Excluded under/overflow reports ({len(flow_notices)})"
                ):
                    for notice in flow_notices:
                        st.write(notice)
            if log_y and any(
                (h.values <= 0).any()
                for _, c in comparisons
                for h in (c.nominal, c.up, c.down)
            ):
                st.warning(
                    "Log y disabled: selected templates contain nonpositive bins."
                )
                log_y = False
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
            if multiple:
                fig = channel_figure(comparisons, normalize, bands, log_y)
                frame = channels_frame(comparisons, normalize)
            else:
                c = comparisons[0][1]
                fig = figure(c, normalize, bands, log_y)
                frame = bins_frame(c, normalize).assign(channel=channels[0])
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
                    if multiple and normalize
                    else ""
                )
            )
            frame = frame.assign(nuisance=nuisance, normalized=normalize)
            frame["processes"] = frame.channel.map(lambda ch: " + ".join(processes[ch]))
            a, b = st.columns(2)
            a.download_button(
                "Download plotted bins (CSV)",
                frame.to_csv(index=False),
                "shape_bins.csv",
                "text/csv",
            )
            b.download_button(
                "Download interactive plot (HTML)",
                fig.to_html(include_plotlyjs=True).replace(
                    "<body>",
                    f"<body><h2>{html.escape(nuisance)}</h2><p>{html.escape(' · '.join(channels))} · {html.escape(selection)} · normalization removed: {normalize}</p>",
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
                st.dataframe(frame, hide_index=True)
        except Exception as exc:
            st.error(str(exc))
    scan_panel(card, mappings, mass, keywords, threshold, channels)


def comparison_inspector(before, after, before_reader, after_reader):
    st.subheader("Before-and-after comparison")
    st.caption("Shortlist compares datacard text. Retained shape templates may still differ in ROOT.")
    common_channels = [ch for ch in before.channels if ch in after.channels]
    added_channels = [ch for ch in after.channels if ch not in before.channels]
    removed_channels = [ch for ch in before.channels if ch not in after.channels]
    if added_channels or removed_channels:
        st.info(f"Channels added: {', '.join(added_channels) or 'none'} · removed: {', '.join(removed_channels) or 'none'}")
    if not common_channels:
        st.warning("The cards have no exactly matching channels to plot.")
        return
    left, right = st.columns([1, 3])
    with left:
        channel_filter = st.text_input("Filter channel names", key="compare_channel_filter").strip()
        choices = [ch for ch in common_channels if channel_filter.casefold() in ch.casefold()]
        channels = st.multiselect("Channels (left-to-right order)", choices, default=choices[:1], key="compare_channels")
        selection = st.radio("Processes", ["One process", "All backgrounds", "All processes"], key="compare_process_selection")
        all_processes = list(dict.fromkeys(p for ch in channels for card in (before, after) for p in card.channels.get(ch, ())))
        process = st.selectbox("Process", all_processes, key="compare_process", disabled=selection != "One process") if all_processes else None
        group = st.selectbox("Nuisance group", ["All groups", *dict.fromkeys((*before.parsed.groups, *after.parsed.groups))], key="compare_group")
        view = st.radio("Nuisance view", ["Card edits", "All effects"], key="compare_view")
        query = st.text_input("Filter nuisance name", key="compare_nuisance_filter").casefold()
        shortlist = comparison_shortlist(before, after)
        if view == "Card edits":
            candidate_names = [r["nuisance"] for r in shortlist]
        else:
            candidate_names = list(dict.fromkeys((*before.declarations, *after.declarations)))
        group_members = set()
        if group != "All groups":
            group_members = set(before.parsed.groups.get(group, ())) | set(after.parsed.groups.get(group, ()))
        names = [n for n in candidate_names if query in n.casefold() and (group == "All groups" or n in group_members)]
        st.caption(f"{len(names)} matching nuisance(s); shortlist opened without reading ROOT templates.")
        if view == "Card edits":
            shortlist_rows = {row["nuisance"]: row for row in shortlist}
        else:
            shortlist_rows = {
                row["nuisance"]: row
                for row in shortlist
            }
        summaries = []
        for name in names:
            row = shortlist_rows.get(name)
            if row is None:
                b_types = {kind for kind, _ in before.declarations.get(name, {}).values()}
                a_types = {kind for kind, _ in after.declarations.get(name, {}).values()}
                row = {
                    "nuisance": name,
                    "changed pairs": 0,
                    "Before types": ", ".join(sorted(b_types)) or "—",
                    "After types": ", ".join(sorted(a_types)) or "—",
                    "ROOT check": False,
                }
            summaries.append(row)
        st.dataframe(pd.DataFrame(summaries), hide_index=True, width="stretch", height=220)
        nuisance = st.selectbox("Select a nuisance to load its templates", ["Select a nuisance", *names], key="compare_nuisance")
        if channels:
            added = [{"channel": ch, "process": p} for ch in channels for p in after.channels[ch] if p not in before.channels[ch]]
            removed = [{"channel": ch, "process": p} for ch in channels for p in before.channels[ch] if p not in after.channels[ch]]
            with st.expander(f"Process changes · {len(added)} added · {len(removed)} removed"):
                st.markdown("**Added processes**")
                st.dataframe(pd.DataFrame(added, columns=["channel", "process"]), hide_index=True)
                st.markdown("**Removed processes**")
                st.dataframe(pd.DataFrame(removed, columns=["channel", "process"]), hide_index=True)
        normalize = st.checkbox("Remove normalization effect", key="compare_normalize")
        log_y = st.checkbox("Logarithmic y-axis", key="compare_log_y")
    with right:
        if not channels or not names or nuisance == "Select a nuisance":
            st.info("Choose a nuisance above to read only its selected templates.")
            return
        b_processes = {ch: before.processes(ch, selection, process) for ch in channels}
        a_processes = {ch: after.processes(ch, selection, process) for ch in channels}
        displayed = {ch: list(dict.fromkeys((*b_processes[ch], *a_processes[ch]))) for ch in channels}
        rows = effect_rows(before, after, nuisance, channels, displayed)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Effective factors are shown in parser order; asymmetric lnN values are down/up. A dropped effect plots at unity when nominal is nonzero.")
        try:
            before_comparisons, after_comparisons = [], []
            for ch in channels:
                before_comparisons.append((ch, compare_response(before_reader, ch, b_processes[ch], nuisance, "Before")))
                after_comparisons.append((ch, compare_response(after_reader, ch, a_processes[ch], nuisance, "After")))
            for side, items in (("Before", before_comparisons), ("After", after_comparisons)):
                for ch, comparison in items:
                    for warning in comparison.warnings:
                        if "shape? log-normal fallback" not in warning:
                            st.warning(warning)
            plot_totals = [
                sum(getattr(c, field).integral for _, c in items)
                for items in (before_comparisons, after_comparisons)
                for field in ("nominal", "up", "down")
            ]
            if normalize and min(plot_totals) <= 0:
                st.warning("Normalization removal requires positive nominal, Up, and Down totals in both cards. Showing raw plots.")
                normalize = False
            fig, same_binning = compare_figure(before_comparisons, after_comparisons, normalize, log_y)
            if same_binning:
                st.plotly_chart(fig, width="stretch", key="compare_shape_plot")
            else:
                st.warning("Binning differs between cards; Before and After are shown separately with their own binning.")
                a, b = st.columns(2)
                a.plotly_chart(channel_figure(before_comparisons, normalize, False, log_y), width="stretch", key="compare_before_plot")
                b.plotly_chart(channel_figure(after_comparisons, normalize, False, log_y), width="stretch", key="compare_after_plot")
            summaries = []
            for side, items in (("Before", before_comparisons), ("After", after_comparisons)):
                summaries.extend({"Card": side, "channel": ch, "nominal": c.nominal.integral, "up": c.up.integral, "down": c.down.integral} for ch, c in items)
            st.dataframe(pd.DataFrame(summaries), hide_index=True, width="stretch")
        except Exception as exc:
            st.error(str(exc))


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
            comparison_inspector(card, after_card, reader, st.session_state["compare_reader"])
            return
    a, b, c, d = st.columns(4)
    a.metric("Channels", len(card.channels))
    b.metric("Process columns", len(card.parsed.keyline))
    c.metric("Shape nuisances", len(card.nuisances))
    d.metric("Affected pairs", len(card.coefficients))
    with st.expander("Datacard overview and file resolution"):
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
    inspector(card, reader, threshold, mappings, mass, keywords)


main()
