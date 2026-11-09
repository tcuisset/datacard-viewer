"""ROOT-free card indexing and lazy TH1 template comparisons."""

from __future__ import annotations

import sys
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from optparse import OptionParser
from pathlib import Path

import numpy as np
import uproot

from .vendor.DatacardParser import addDatacardParserOptions, parseCard


def fingerprint(path):
    p = Path(path).expanduser().resolve()
    try:
        s = p.stat()
        return str(p), s.st_size, s.st_mtime_ns
    except OSError:
        return str(p), None, None


@dataclass
class Card:
    """Parsed datacard plus indexes used by the viewer.

    ``path`` and ``revision`` identify the source snapshot; ``parsed`` retains
    the Combine parser result. ``channels`` maps channel names to process names,
    while ``nuisances`` and ``relevant`` index shape nuisances globally and by
    channel. ``coefficients`` stores active shape coefficients by
    (nuisance, channel, process). ``declarations`` stores active supported
    shape/lnN declarations by nuisance and (channel, process), for card diffs.
    """

    path: Path
    revision: tuple
    parsed: object
    channels: dict
    nuisances: dict
    relevant: dict
    coefficients: dict
    declarations: dict

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        parser = OptionParser()
        addDatacardParserOptions(parser)
        options, _ = parser.parse_args([])
        options.allowNoSignal = True
        options.allowNoBackground = True
        revision = fingerprint(path)
        with path.open() as stream:
            dc = parseCard(stream, options)
        if fingerprint(path) != revision:
            raise ValueError("Datacard changed while parsing; reload it")
        channels = defaultdict(list)
        for channel, process, signal in dc.keyline:
            channels[channel].append(process)
        nuisances = {}
        coefficients = {}
        declarations = defaultdict(dict)
        relevant = defaultdict(list)
        for name, _, kind, _, effects in dc.systs:
            # Keep only active shape/lnN pair declarations. Missing pairs then
            # naturally mean a dropped effect, and the sparse index avoids
            # copying every nuisance's mostly empty channel/process matrix.
            if kind in ("lnN", "shape", "shapeN", "shape?") and isinstance(effects, dict):
                for channel, processes in channels.items():
                    for process in processes:
                        value = effects.get(channel, {}).get(process, 0)
                        if value != 0 and not (kind == "lnN" and value == 1):
                            declarations[name][channel, process] = (kind, value)
            if not kind.startswith("shape"):
                continue
            if kind not in ("shape", "shapeN", "shape?"):
                raise ValueError(f"Unsupported shape algorithm: {kind}")
            nuisances[name] = kind
            for channel, processes in channels.items():
                for process in processes:
                    value = effects[channel][process]
                    if value != 0:
                        coefficients[name, channel, process] = value
                        if name not in relevant[channel]:
                            relevant[channel].append(name)
        return cls(
            path,
            revision,
            dc,
            dict(channels),
            nuisances,
            dict(relevant),
            coefficients,
            dict(declarations),
        )

    def processes(self, channel, selection, process=None):
        ps = self.channels[channel]
        if selection == "One process":
            return [process or ps[0]] if (process or ps[0]) in ps else []
        if selection == "All backgrounds":
            return [p for p in ps if not self.parsed.isSignal[p]]
        return list(ps)


@dataclass(slots=True)
class Template:
    edges: np.ndarray
    values: np.ndarray
    variances: np.ndarray
    sources: tuple = ()
    warnings: tuple = ()

    @property
    def integral(self):
        return float(self.values.sum())

    @property
    def nbytes(self):
        return self.edges.nbytes + self.values.nbytes + self.variances.nbytes

    @property
    def cache_size(self):
        # Include Python/NumPy object and cache-key overhead, not just numeric
        # buffers: small histograms otherwise consume far beyond the byte cap.
        return (
            self.nbytes
            + sum(sys.getsizeof(a) for a in (self.edges, self.values, self.variances))
            + sys.getsizeof(self)
            + sys.getsizeof(self.sources)
            + sys.getsizeof(self.warnings)
            + sum(sys.getsizeof(s) for s in (*self.sources, *self.warnings))
            + 768
        )

    def scaled(self, factor):
        return Template(
            self.edges,
            self.values * factor,
            self.variances * factor**2,
            self.sources,
            self.warnings,
        )


class Reader:
    def __init__(
        self,
        card,
        mappings=(),
        mass=125,
        keywords=None,
        max_bytes=128 * 1024**2,
        max_files=8,
    ):
        self.card = card
        self.mappings = tuple(mappings)
        self.mass = f"{float(mass):g}"
        self.keywords = keywords or {}
        self.max_bytes = max_bytes
        self.max_files = max_files
        self.files = OrderedDict()
        self.directories = OrderedDict()
        self.arrays = OrderedDict()
        self.bytes = 0
        self.revisions = {}

    def shape_mapping(self, channel, process):
        maps = self.card.parsed.shapeMap
        entry = maps.get(channel, maps.get("*", {}))
        names = entry.get(process, entry.get("*"))
        if names is None:
            names = maps.get("*", {}).get(process, maps.get("*", {}).get("*"))
        if not names or names == ["FAKE"]:
            raise ValueError(f"No TH1 mapping for {channel}/{process}")
        return names

    def resolve(self, channel, process, systematic=""):
        names = self.shape_mapping(channel, process)
        if len(names) < 2:
            raise ValueError(
                f"Malformed histogram mapping for {channel}/{process}: {names}"
            )
        if systematic and len(names) < 3:
            raise KeyError(f"No systematic mapping for {channel}/{process}")
        pattern = names[2] if systematic else names[1]
        replacements = dict(
            self.keywords,
            PROCESS=process,
            CHANNEL=channel,
            SYSTEMATIC=systematic,
            MASS=self.mass,
        )

        def expand(s):
            # Longest keys first prevents a short keyword consuming a prefix.
            for key in sorted(replacements, key=len, reverse=True):
                s = s.replace("$" + key, str(replacements[key]))
            if "$" in s:
                raise ValueError(f"Unresolved substitution: {s}")
            return s

        filename, key = expand(names[0]), expand(pattern)
        for source, target in self.mappings:
            if filename.startswith(source):
                filename = target + filename[len(source) :]
                break
        path = Path(filename).expanduser()
        if not path.is_absolute():
            path = self.card.path.parent / path
        if ":" in key:
            raise ValueError(f"Workspace/parametric input is unsupported: {key}")
        return path.resolve(), key

    def _file(self, path):
        revision = fingerprint(path)
        self.revisions[str(path)] = revision
        if revision[1] is None:
            raise FileNotFoundError(str(path))
        if path in self.files:
            old_revision, handle = self.files.pop(path)
            if old_revision == revision:
                self.files[path] = old_revision, handle
                return handle, revision
            handle.close()
            self._drop_directories(path)
            for cache_key in list(self.arrays):
                if cache_key[0] == str(path):
                    self.bytes -= self.arrays.pop(cache_key).cache_size
        handle = uproot.open(path, object_cache=None, array_cache=None)
        self.files[path] = revision, handle
        while len(self.files) > self.max_files:
            old_path, (_, old) = self.files.popitem(last=False)
            old.close()
            self._drop_directories(old_path)
        return handle, revision

    def _drop_directories(self, path):
        for key in list(self.directories):
            if key[0] == str(path):
                del self.directories[key]

    def read(self, channel, process, systematic=""):
        path, key = self.resolve(channel, process, systematic)
        handle, revision = self._file(path)
        cache_key = (str(path), revision, key)
        if cache_key in self.arrays:
            result = self.arrays.pop(cache_key)
            self.arrays[cache_key] = result
            return result
        directory, _, leaf = key.rpartition("/")
        if directory:
            directory_key = (str(path), revision, directory)
            if directory_key in self.directories:
                parent = self.directories.pop(directory_key)
            else:
                parent = handle[directory]
            self.directories[directory_key] = parent
            while len(self.directories) > 64:
                self.directories.popitem(last=False)
            h = parent[leaf]
        else:
            h = handle[key]
        if not h.classname.startswith("TH1") or len(h.axes) != 1:
            raise ValueError(
                f"Expected one-dimensional TH1: {path}:{key} ({h.classname})"
            )
        # uproot's variances() sanitizes some invalid Sumw2 entries to zero;
        # inspect the stored buffer first so invalid input is not hidden.
        stored_variances = np.asarray(h.member("fSumw2"), dtype=float)
        if stored_variances.size and (
            not np.all(np.isfinite(stored_variances)) or np.any(stored_variances < 0)
        ):
            raise ValueError(f"Invalid variances: {path}:{key}")
        values = np.asarray(h.values(), dtype=float)
        variances = np.asarray(h.variances(), dtype=float)
        edges = np.asarray(h.axis().edges(), dtype=float)
        warnings = []
        if not h.weighted:
            warnings.append("No Sumw2: Poisson variance fallback")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite contents: {path}:{key}")
        if not np.all(np.isfinite(variances)) or np.any(variances < 0):
            raise ValueError(f"Invalid variances: {path}:{key}")
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError(f"Invalid bin edges: {path}:{key}")
        if np.any(values < 0):
            warnings.append("Negative bin contents")
        flows = h.values(flow=True)[[0, -1]]
        if np.any(flows != 0):
            warnings.append(f"Excluded under/overflow contents: {flows.tolist()}")
        if not systematic:
            rate = self.card.parsed.exp[channel][process]
            if rate != -1 and abs(values.sum() - rate) > 0.01 * max(1, abs(rate)):
                warnings.append(
                    f"Rate mismatch: card={rate:g}, histogram={values.sum():g}"
                )
        for a in (edges, values, variances):
            a.flags.writeable = False
        result = Template(edges, values, variances, (f"{path}:{key}",), tuple(warnings))
        if result.cache_size <= self.max_bytes:
            self.arrays[cache_key] = result
            self.bytes += result.cache_size
            while self.bytes > self.max_bytes:
                _, old = self.arrays.popitem(last=False)
                self.bytes -= old.cache_size
        return result

    def close(self):
        for _, handle in self.files.values():
            handle.close()
        self.files.clear()
        self.directories.clear()
        self.arrays.clear()
        self.bytes = 0

    def stale(self):
        return fingerprint(self.card.path) != self.card.revision or any(
            fingerprint(p) != rev for p, rev in self.revisions.items()
        )


def add(templates):
    templates = list(templates)
    if not templates:
        raise ValueError("No processes selected")
    first = templates[0]
    if any(not np.array_equal(first.edges, h.edges) for h in templates[1:]):
        raise ValueError("Incompatible bin edges between selected templates")
    return Template(
        first.edges,
        np.sum([h.values for h in templates], axis=0),
        np.sum([h.variances for h in templates], axis=0),
        tuple(s for h in templates for s in h.sources),
        tuple(dict.fromkeys(w for h in templates for w in h.warnings)),
    )


@dataclass
class Comparison:
    nominal: Template
    up: Template
    down: Template
    warnings: tuple = ()
    coefficients: dict = field(default_factory=dict)

    def normalized(self):
        if min(self.nominal.integral, self.up.integral, self.down.integral) <= 0:
            raise ValueError("Normalization removal requires positive integrals")
        return Comparison(
            self.nominal,
            self.up.scaled(self.nominal.integral / self.up.integral),
            self.down.scaled(self.nominal.integral / self.down.integral),
            self.warnings,
            self.coefficients,
        )


def compare(reader, channel, processes, nuisance):
    if not processes:
        # Use local physical binning for an absent process or an empty group.
        reference = reader.read(channel, reader.card.channels[channel][0])
        zero = Template(
            reference.edges.copy(),
            np.zeros_like(reference.values),
            np.zeros_like(reference.variances),
        )
        return Comparison(
            zero,
            zero,
            zero,
            ("No selected processes in this channel: zero yield.",),
            {},
        )
    nominal, up, down, warnings, coefficients = [], [], [], [], {}
    for process in processes:
        n = reader.read(channel, process)
        nominal.append(n)
        coefficient = reader.card.coefficients.get((nuisance, channel, process), 0)
        if not coefficient:
            up.append(n)
            down.append(n)
            continue
        coefficients[process] = coefficient
        mapped = reader.card.parsed.systematicsShapeMap.get(
            (nuisance, channel, process), nuisance
        )
        variations = []
        missing = []
        for suffix in ("Up", "Down"):
            try:
                variations.append(reader.read(channel, process, mapped + suffix))
            except KeyError as exc:
                variations.append(None)
                missing.append(str(exc))
        if len(missing) == 2 and reader.card.nuisances[nuisance] == "shape?":
            if isinstance(coefficient, list):
                low, high = coefficient
            else:
                high, low = coefficient, 1 / coefficient
            if min(low, high) <= 0:
                raise ValueError("Invalid shape? log-normal factor")
            variations = [n.scaled(high), n.scaled(low)]
            warnings.append(f"{process}: shape? log-normal fallback")
        elif missing:
            raise ValueError(
                f"Missing variation for {channel}/{process}/{nuisance}: "
                + "; ".join(missing)
            )
        elif coefficient != 1:
            warnings.append(
                f"{process}: coefficient {coefficient}; stored endpoints are not effective ±1σ predictions"
            )
        # Validate against the nominal even when comparing just one process.
        add([n, *variations])
        up.append(variations[0])
        down.append(variations[1])
    return Comparison(
        add(nominal),
        add(up),
        add(down),
        tuple(warnings),
        coefficients,
    )


def _effective_declaration(card, nuisance, channel, process):
    """Return the effective datacard declaration for one nuisance pair."""
    declaration = card.declarations.get(nuisance, {}).get((channel, process))
    if declaration is None:
        return "dropped", 0
    kind, value = declaration
    if value == 0 or (kind == "lnN" and value == 1):
        return "dropped", 0
    return kind, value


def comparison_shortlist(before, after):
    """Summarize text-level nuisance changes without opening ROOT files."""
    rows = []
    for name in dict.fromkeys((*before.declarations, *after.declarations)):
        b_entries = before.declarations.get(name, {})
        a_entries = after.declarations.get(name, {})
        pairs = set(b_entries) | set(a_entries)
        changed = [
            pair
            for pair in pairs
            if _effective_declaration(before, name, *pair)
            != _effective_declaration(after, name, *pair)
        ]
        b_kinds = {kind for kind, _ in b_entries.values()}
        a_kinds = {kind for kind, _ in a_entries.values()}
        root_check = "shape?" in b_kinds | a_kinds
        if changed or root_check:
            rows.append(
                {
                    "nuisance": name,
                    "changed pairs": len(changed),
                    "Before types": ", ".join(sorted(b_kinds)) or "—",
                    "After types": ", ".join(sorted(a_kinds)) or "—",
                    "ROOT check": root_check,
                }
            )
    return rows


def compare_response(reader, channel, processes, nuisance, side="card"):
    """Build effective nominal/Up/Down predictions for shape and lnN effects."""
    if not processes:
        reference = reader.read(channel, reader.card.channels[channel][0])
        zero = Template(reference.edges.copy(), np.zeros_like(reference.values), np.zeros_like(reference.variances))
        return Comparison(zero, zero, zero, ("No selected processes in this channel: zero yield.",), {})
    nominal, up, down, warnings, coefficients = [], [], [], [], {}
    for process in processes:
        n = reader.read(channel, process)
        kind, coefficient = _effective_declaration(reader.card, nuisance, channel, process)
        nominal.append(n)
        if kind == "dropped":
            up.append(n)
            down.append(n)
            continue
        coefficients[process] = coefficient
        if kind == "lnN":
            if isinstance(coefficient, list):
                low, high = coefficient  # Combine stores asymmetric lnN as down/up.
            else:
                high, low = coefficient, 1 / coefficient
            up.append(n.scaled(high))
            down.append(n.scaled(low))
            continue
        if not kind.startswith("shape"):
            up.append(n)
            down.append(n)
            continue
        mapped = reader.card.parsed.systematicsShapeMap.get((nuisance, channel, process), nuisance)
        variations, missing = [], []
        for suffix in ("Up", "Down"):
            try:
                variations.append(reader.read(channel, process, mapped + suffix))
            except KeyError as exc:
                variations.append(None)
                missing.append(str(exc))
        if len(missing) == 2 and kind == "shape?":
            if isinstance(coefficient, list):
                low, high = coefficient
            else:
                high, low = coefficient, 1 / coefficient
            variations = [n.scaled(high), n.scaled(low)]
            warnings.append(f"{side}: {channel}/{process}: shape? log-normal fallback")
        elif missing:
            raise ValueError(
                f"{side}: {channel}/{process}/{nuisance}: missing variation endpoint: "
                + "; ".join(missing)
            )
        elif coefficient != 1:
            warnings.append(f"{side}: {channel}/{process}: shape coefficient {coefficient}; stored endpoints are shown")
        add([n, *variations])
        up.append(variations[0])
        down.append(variations[1])
    return Comparison(
        add(nominal),
        add(up),
        add(down),
        tuple(warnings),
        coefficients,
    )


def effect_rows(before, after, nuisance, channels, processes_by_channel):
    rows = []
    for channel in channels:
        for process in dict.fromkeys((*before.channels.get(channel, ()), *after.channels.get(channel, ()))):
            if process not in processes_by_channel[channel]:
                continue
            b = _effective_declaration(before, nuisance, channel, process)
            a = _effective_declaration(after, nuisance, channel, process)
            b_exists, a_exists = process in before.channels.get(channel, ()), process in after.channels.get(channel, ())
            if not b_exists or not a_exists:
                status = "Removed process" if b_exists else "Added process"
            elif b[0] != "dropped" and a[0] == "dropped":
                status = "Effect dropped"
            elif b[0] == "dropped" and a[0] != "dropped":
                status = "Effect added"
            else:
                status = "Changed" if b != a else "Retained"
            rows.append({"channel": channel, "process": process, "Before type": b[0] if b_exists else "—", "Before factor/coefficient": str(b[1]) if b_exists else "—", "After type": a[0] if a_exists else "—", "After factor/coefficient": str(a[1]) if a_exists else "—", "status": status})
    return rows


def ratios(values, nominal):
    return np.divide(
        values,
        nominal,
        out=np.full_like(values, np.nan, dtype=float),
        where=nominal != 0,
    )


def diagnose(comparison):
    n, u, d = comparison.nominal, comparison.up, comparison.down
    warnings = list(
        dict.fromkeys((*n.warnings, *u.warnings, *d.warnings, *comparison.warnings))
    )
    result = dict(
        kappa_up=np.nan,
        kappa_down=np.nan,
        max_residual=np.nan,
        weighted_rms=np.nan,
        max_residual_over_stat=np.nan,
        valid=False,
        warnings="; ".join(warnings),
    )
    if min(n.integral, u.integral, d.integral) <= 0:
        result["warnings"] += "; Nonpositive integral"
        return result
    ku, kd = u.integral / n.integral, d.integral / n.integral
    result.update(kappa_up=ku, kappa_down=kd)
    if any(np.any(h.values < 0) for h in (n, u, d)):
        result["warnings"] += "; Negative bins prevent candidate screening"
        return result
    if np.any((n.values == 0) & ((u.values != 0) | (d.values != 0))):
        result["warnings"] += "; Newly populated zero-nominal bins"
        return result
    mask = n.values > 0
    residuals = np.array([u.values / ku - n.values, d.values / kd - n.values])[:, mask]
    fractional = residuals / n.values[mask]
    stat = np.sqrt(n.variances[mask])
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.divide(
            np.abs(residuals), stat, out=np.full_like(residuals, np.inf), where=stat > 0
        )
        scores[(residuals == 0) & (stat == 0)] = 0
    result.update(
        max_residual=float(np.max(np.abs(fractional))),
        weighted_rms=float(
            np.max(np.sqrt(np.sum(fractional**2 * n.values[mask], axis=1) / n.integral))
        ),
        max_residual_over_stat=float(np.max(scores)),
        valid=True,
    )
    return result


def candidate(row, threshold=0.01):
    return bool(row.get("valid", False) and row["max_residual"] <= threshold)


class Scan:
    """One cancellable worker; scalar snapshots are protected by a lock."""

    def __init__(self, card, mappings=(), mass=125, keywords=None, channels=None):
        self.card = card
        self.reader_args = (card, mappings, mass, keywords)
        self.channels = set(channels or card.channels)
        self.total = sum(c in self.channels for _, c, _ in card.coefficients)
        self.rows = []
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.done = False
        self.error = None
        self.revisions = {}
        self.started = time.monotonic()
        self.finished = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        reader = Reader(*self.reader_args)
        try:
            # Snapshot nominal files before scanning so a change during a scan
            # cannot be hidden by later cache invalidation/re-reads.
            for channel, processes in self.card.channels.items():
                if channel not in self.channels:
                    continue
                for process in processes:
                    try:
                        path, _ = reader.resolve(channel, process)
                        self.revisions.setdefault(str(path), fingerprint(path))
                    except (ValueError, KeyError):
                        pass  # Report invalid mappings in their pair result.
            # Channel order minimizes file churn; the worker never shares readers with UI.
            for channel in self.card.channels:
                if channel not in self.channels:
                    continue
                for nuisance in self.card.relevant.get(channel, []):
                    for process in self.card.channels[channel]:
                        coefficient = self.card.coefficients.get(
                            (nuisance, channel, process), 0
                        )
                        if not coefficient:
                            continue
                        if self.cancel.is_set():
                            return
                        row = dict(
                            channel=channel,
                            process=process,
                            nuisance=nuisance,
                            coefficient=str(coefficient),
                        )
                        try:
                            row.update(
                                diagnose(compare(reader, channel, [process], nuisance))
                            )
                        except Exception as exc:
                            row.update(
                                valid=False, warnings=str(exc), max_residual=np.nan
                            )
                        with self.lock:
                            self.rows.append(row)
                            for path, revision in reader.revisions.items():
                                self.revisions.setdefault(path, revision)
        except Exception as exc:
            self.error = str(exc)
        finally:
            reader.close()
            self.finished = time.monotonic()
            self.done = True

    def snapshot(self):
        with self.lock:
            return list(self.rows), dict(self.revisions)

    def stale(self):
        _, revisions = self.snapshot()
        return fingerprint(self.card.path) != self.card.revision or any(
            fingerprint(p) != rev for p, rev in revisions.items()
        )


def summarize(card, rows, threshold=0.01, channels=None):
    allowed = set(channels or card.channels)
    expected = defaultdict(int)
    for name, channel, _ in card.coefficients:
        if channel in allowed:
            expected[name] += 1
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["nuisance"]].append(row)
    result = []
    for name, total in expected.items():
        subset = grouped[name]
        valid = [r for r in subset if r.get("valid")]
        worst = max(valid, key=lambda r: r["max_residual"], default=None)
        result.append(
            dict(
                nuisance=name,
                reviewed=len(subset),
                expected=total,
                candidate=len(subset) == total
                and all(candidate(r, threshold) for r in subset),
                max_residual=worst["max_residual"] if worst else np.nan,
                worst_pair=f"{worst['channel']}/{worst['process']}" if worst else "",
                warning_pairs=sum(bool(r.get("warnings")) for r in subset),
                failures=len(subset) - len(valid),
            )
        )
    return result
