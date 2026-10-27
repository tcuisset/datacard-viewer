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
    diagnose,
    fingerprint,
    summarize,
)
from datacard_viewer.plotting import bins_frame, figure
from datacard_viewer.reporting import scan_exports

st.set_page_config(page_title="Combine shape viewer", page_icon="📊", layout="wide")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", default="")
    parser.add_argument("--path-map", action="append", default=[])
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
def scan_panel(card, mappings, mass, keywords, threshold, channel):
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
    if a.button("Scan selected channel", disabled=active):
        st.session_state["channel_scan"] = Scan(
            card, mappings, mass, keywords, [channel]
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
            with st.expander("Pair results for the current channel"):
                st.dataframe(
                    pd.DataFrame([r for r in rows if r["channel"] == channel]),
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
        st.rerun()  # Refresh the index even when only a fragment was rerun.
    left, right = st.columns([1, 3])
    with left:
        channel = st.selectbox("Channel", list(card.channels))
        selection = st.radio(
            "Processes", ["One process", "All backgrounds", "All processes"]
        )
        process = st.selectbox(
            "Process", card.channels[channel], disabled=selection != "One process"
        )
        processes = card.processes(channel, selection, process)
        group = st.selectbox("Nuisance group", ["All groups", *card.parsed.groups])
        search = st.text_input("Filter nuisance name")
        names = [
            n
            for n in card.relevant.get(channel, [])
            if (group == "All groups" or n in card.parsed.groups[group])
            and search.lower() in n.lower()
            and any((n, channel, p) in card.coefficients for p in processes)
        ]
        if not names:
            st.info("No matching shape nuisances for this selection.")
            scan_panel(card, mappings, mass, keywords, threshold, channel)
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
        st.caption(f"{len(processes)} processes · {len(names)} matching nuisances")
    with right:
        st.subheader(nuisance)
        st.caption(f"{channel} · {selection} · {card.nuisances[nuisance]}")
        try:
            comparison = compare(reader, channel, processes, nuisance)
            warnings = list(
                dict.fromkeys(
                    (
                        *comparison.warnings,
                        *comparison.nominal.warnings,
                        *comparison.up.warnings,
                        *comparison.down.warnings,
                    )
                )
            )
            flow_notices = [w for w in warnings if "under/overflow" in w]
            for warning in warnings:
                if warning not in flow_notices:
                    st.warning(warning)
            if flow_notices:
                with st.expander(
                    f"Excluded under/overflow reports ({len(flow_notices)})"
                ):
                    for notice in flow_notices:
                        st.write(notice)
            if log_y and any(
                (h.values <= 0).any()
                for h in (comparison.nominal, comparison.up, comparison.down)
            ):
                st.warning(
                    "Log y disabled: selected templates contain nonpositive bins."
                )
                log_y = False
            n, u, d = (
                comparison.nominal.integral,
                comparison.up.integral,
                comparison.down.integral,
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
            fig = figure(comparison, normalize, bands, log_y)
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
            diag = diagnose(comparison)
            st.dataframe(pd.DataFrame([diag]), hide_index=True, width="stretch")
            st.caption(
                "Aggregate diagnostics describe this plot; normalization-only screening uses individual processes."
            )
            a, b = st.columns(2)
            a.download_button(
                "Download plotted bins (CSV)",
                bins_frame(comparison, normalize)
                .assign(
                    channel=channel,
                    nuisance=nuisance,
                    processes=" + ".join(processes),
                    normalized=normalize,
                )
                .to_csv(index=False),
                "shape_bins.csv",
                "text/csv",
            )
            b.download_button(
                "Download interactive plot (HTML)",
                fig.to_html(include_plotlyjs=True).replace(
                    "<body>",
                    f"<body><h2>{html.escape(nuisance)}</h2><p>{html.escape(channel)} · {html.escape(selection)} · normalization removed: {normalize}</p>",
                ),
                "shape_plot.html",
                "text/html",
            )
            with st.expander("Template provenance and bin values"):
                st.json(
                    dict(
                        coefficients=comparison.coefficients,
                        nominal=comparison.nominal.sources,
                        up=comparison.up.sources,
                        down=comparison.down.sources,
                    )
                )
                st.dataframe(bins_frame(comparison, normalize), hide_index=True)
        except Exception as exc:
            st.error(str(exc))
    scan_panel(card, mappings, mass, keywords, threshold, channel)


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
        old = st.session_state.pop("reader", None)
        if old:
            old.close()
        load_card.clear()
        st.session_state["configuration"] = (path, mapping_text, mass, keyword_text)
    elif "configuration" not in st.session_state and args.card:
        st.session_state["configuration"] = (
            args.card,
            "\n".join(args.path_map),
            args.mass,
            "\n".join(args.keyword),
        )
    config = st.session_state.get("configuration")
    if not config:
        st.info("Enter a datacard path in the sidebar to begin.")
        return
    try:
        path, mapping_text, mass, keyword_text = config
        mappings = parse_pairs(mapping_text)
        keywords = dict(parse_pairs(keyword_text))
        if set(keywords) & {"PROCESS", "CHANNEL", "SYSTEMATIC", "MASS"}:
            raise ValueError("Built-in substitution keywords cannot be overridden")
        card = load_card(fingerprint(path))
        identity = (card.revision, mappings, mass, tuple(keywords.items()))
        if st.session_state.get("reader_identity") != identity:
            cancel_scans()
            if st.session_state.get("reader"):
                st.session_state["reader"].close()
            st.session_state["reader"] = Reader(card, mappings, mass, keywords)
            st.session_state["reader_identity"] = identity
        reader = st.session_state["reader"]
    except Exception as exc:
        st.error(f"Could not open datacard: {exc}")
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
