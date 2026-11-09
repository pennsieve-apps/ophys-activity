"""
Finding the inputs: traces NWB files, stimulus tables (NWB or CSV) and the sessions.csv sheet.

Files are found by name and content, recursively, following symlinks (INPUT_DIR entries are
symlinks), never by fixed paths, so the app works with linked ports (`traces/`, `session/`,
`metadata/` subfolders) and with everything merged at the root of INPUT_DIR.
"""
from __future__ import annotations

import csv
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("ophys-activity")

UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
BIDS_RE = re.compile(r"sub-([A-Za-z0-9-]+)_ses-([A-Za-z0-9-]+)")
NWB_SUFFIXES = (".nwb",)
STIM_CSV_COLUMNS = {"start_time"}


def sanitise(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", name).strip("-") or "session"


def bids_session(text: str) -> str | None:
    """`sub-<a>_ses-<b>` in a name, with <b> cut to its first 8 characters when it is a UUID."""
    m = BIDS_RE.search(text)
    if not m:
        return None
    sub, ses = m.group(1), m.group(2)
    if UUID_RE.match(ses):
        ses = ses[:8]
    return f"sub-{sub}_ses-{ses}"


def walk_files(root: Path):
    """Every regular file under root, following symlinks (each real directory visited once)."""
    seen: set = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        try:
            real = os.path.realpath(dirpath)
        except OSError:
            continue
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            p = Path(dirpath) / fn
            if p.is_file():
                yield p


def read_sheet(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(fh)]
    return [r for r in rows if r.get("session_id") and r.get("path")]


def match_sheet(path: Path, root: Path, rows: list[dict]) -> dict | None:
    """The sheet row whose `path` matches this file: the relative path's tail, else its name."""
    rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
    best = None
    for r in rows:
        p = r["path"].strip("/").replace("\\", "/")
        if rel == p or rel.endswith("/" + p):
            return r
        if Path(p).name == path.name or Path(p).stem == path.stem:
            best = best or r
    return best


def session_id_for(path: Path, root: Path, rows: list[dict]) -> tuple[str, dict, str]:
    """(session_id, sheet row or {}, how): sheet, else sub-X_ses-Y in the name or a parent, else the stem."""
    row = match_sheet(path, root, rows) if rows else None
    if row:
        return sanitise(row["session_id"]), row, "sessions.csv"
    sid = bids_session(path.name)
    if sid:
        return sid, {}, "file name"
    for parent in path.parents:
        if parent == root or not parent.is_relative_to(root):
            break
        sid = bids_session(parent.name)
        if sid:
            return sid, {}, "folder name"
    stem = path.name
    for suf in (".nwb", ".csv", ".tsv"):
        if stem.lower().endswith(suf):
            stem = stem[: -len(suf)]
    return sanitise(stem), {}, "file stem"


@dataclass
class Session:
    session_id: str
    traces: Path
    sheet: dict = field(default_factory=dict)
    id_source: str = ""
    stim_nwb: list[Path] = field(default_factory=list)
    stim_csv: list[Path] = field(default_factory=list)


@dataclass
class Found:
    traces: list[Path] = field(default_factory=list)       # NWB files with processing/*/RoiResponseSeries
    stim_nwb: list[Path] = field(default_factory=list)     # NWB files without ophys (session port)
    stim_csv: list[Path] = field(default_factory=list)     # CSV files with start_time (stop_time) columns
    sheets: list[Path] = field(default_factory=list)       # sessions.csv
    unreadable: list[str] = field(default_factory=list)


def _csv_header(path: Path) -> list[str]:
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            first = fh.readline()
    except (OSError, UnicodeDecodeError):
        return []
    delim = "\t" if first.count("\t") > first.count(",") else ","
    return [c.strip().strip('"').lower() for c in first.strip().split(delim)]


def discover(root: Path) -> Found:
    from .nwbread import probe_nwb

    found = Found()
    for p in walk_files(root):
        name = p.name.lower()
        if name.endswith(NWB_SUFFIXES):
            kind = probe_nwb(p)
            # A file linked into the `session` port is a stimulus source even when it also has ROI
            # traces (the Allen companion NWB next to ophys-traces' output of the same session).
            if kind == "traces" and p.relative_to(root).parts[0] == "session":
                kind = "session"
            if kind == "traces":
                found.traces.append(p)
            elif kind == "session":
                found.stim_nwb.append(p)
            elif kind == "error":
                found.unreadable.append(str(p))
        elif name.endswith((".csv", ".tsv")):
            header = _csv_header(p)
            if name == "sessions.csv" or ("session_id" in header and "path" in header):
                found.sheets.append(p)
            elif STIM_CSV_COLUMNS <= set(header):
                found.stim_csv.append(p)
    return found


def build_sessions(found: Found, root: Path) -> tuple[list[Session], list[str]]:
    """One session per traces NWB; stimulus files matched to it by session_id."""
    warnings: list[str] = []
    rows: list[dict] = []
    for s in found.sheets:
        try:
            rows += read_sheet(s)
        except (OSError, csv.Error) as exc:
            warnings.append(f"could not read {s.name}: {exc}")
    sessions: dict[str, Session] = {}
    base_ids: dict[str, list[str]] = {}
    for p in found.traces:
        sid, row, how = session_id_for(p, root, rows)
        base_ids.setdefault(sid, [])
        if sid in sessions:
            # Two traces files with one id (e.g. two planes): keep both, distinguished by stem.
            alt = sanitise(f"{sid}_{p.stem}")
            warnings.append(f"{p.name}: session_id {sid} already taken by {sessions[sid].traces.name}; using {alt}")
            sid = alt
        sessions[sid] = Session(sid, p, row, how)
        base_ids[session_id_for(p, root, rows)[0]].append(sid)
    # Traces files of one session (e.g. ours and the source's, merged side by side): each one's
    # stimulus tables are candidates for the others.
    for sids in base_ids.values():
        for a in sids:
            sessions[a].stim_nwb += [sessions[b].traces for b in sids if b != a]

    def owner(path: Path) -> str | None:
        sid, _, _ = session_id_for(path, root, rows)
        if sid in sessions:
            return sid
        for k in sessions:          # a name that contains the session id (`<session_id>_stimulus.csv`)
            if k in path.name or k in path.as_posix():
                return k
        return None

    for p in found.stim_nwb:
        sid = owner(p)
        if sid is None and len(sessions) == 1:
            sid = next(iter(sessions))
        if sid is None:
            warnings.append(f"{p.name}: session NWB matches no traces file (session ids: {', '.join(sessions)}); ignored")
            continue
        sessions[sid].stim_nwb.append(p)
    for p in found.stim_csv:
        sid = owner(p)
        if sid is None and len(sessions) == 1:
            sid = next(iter(sessions))
        if sid is None:
            # A shared stimulus.csv with a session_id column applies to every session it names.
            if "session_id" in _csv_header(p):
                for s in sessions.values():
                    s.stim_csv.append(p)
                continue
            warnings.append(f"{p.name}: stimulus CSV matches no session and has no session_id column; ignored")
            continue
        sessions[sid].stim_csv.append(p)
    return list(sessions.values()), warnings
