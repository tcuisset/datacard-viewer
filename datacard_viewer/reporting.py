"""Streaming CSV reports keep full-card exports from duplicating large tables."""

import csv
import io

from .core import candidate, summarize


def csv_bytes(rows, fields):
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="utf-8", newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    stream.flush()
    result = buffer.getvalue()
    stream.close()
    return result


def scan_exports(card, rows, threshold, channels=None, stale=False):
    summary = summarize(card, rows, threshold, channels)
    if stale:
        for entry in summary:
            entry["candidate"] = False
    approved = {entry["nuisance"] for entry in summary if entry["candidate"]}

    def pair_rows():
        for row in rows:
            entry = dict(row)
            entry["candidate"] = candidate(row, threshold) and not stale
            entry["suggested_lnN"] = (
                f"{row['kappa_down']:.8g}/{row['kappa_up']:.8g}"
                if row["nuisance"] in approved and row["coefficient"] in ("1", "1.0")
                else ""
            )
            yield entry

    pairs = csv_bytes(
        pair_rows(),
        [
            "channel",
            "process",
            "nuisance",
            "coefficient",
            "kappa_up",
            "kappa_down",
            "max_residual",
            "weighted_rms",
            "max_residual_over_stat",
            "valid",
            "warnings",
            "candidate",
            "suggested_lnN",
        ],
    )
    return pairs, csv_bytes(
        summary,
        [
            "nuisance",
            "reviewed",
            "expected",
            "candidate",
            "max_residual",
            "worst_pair",
            "warning_pairs",
            "failures",
        ],
    )
