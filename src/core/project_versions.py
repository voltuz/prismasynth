"""Project version snapshots — durable cross-session rollback.

Stores timestamped copies of a `.psynth` file in a sibling
`<project>.psynth.versions/` folder, alongside an `index.json` manifest
that lets the UI render the version list without parsing every snapshot.

Triggers:
    - "autosave"            — every 60s autosave when project is dirty
    - "manual"              — user-clicked "Snapshot Now" with optional label
    - "pre_detect_cuts"     — before SceneDetector replaces clips
    - "pre_multi_delete"    — before W/D delete on >5 selected clips
    - "pre_group_delete"    — before People-group deletion
    - "pre_source_removal"  — before bulk source remove
    - "pre_restore"         — before restoring an older version

Retention: keep newest 50 always; older versions thinned to one per hour
for the last day, one per day for the last week, one per week beyond.

All filesystem ops are best-effort: failures are logged, never raised, so
the app never crashes because the versions directory is read-only or
permissioned away. The user's primary `.psynth` save is unaffected.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Newest N versions are always kept regardless of bucket.
_KEEP_NEWEST = 50

# Tier boundaries (relative to "now") for retention bucketing.
_HOURLY_WINDOW = timedelta(days=1)
_DAILY_WINDOW = timedelta(days=7)


_TS_FMT = "%Y-%m-%d_%H-%M-%S"
_FNAME_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(?:_([a-z0-9_-]+))?\.psynth$",
    re.IGNORECASE,
)

VALID_TRIGGERS = {
    "autosave", "manual",
    "pre_detect_cuts", "pre_multi_delete",
    "pre_group_delete", "pre_source_removal",
    "pre_restore",
}


@dataclass
class VersionEntry:
    filename: str
    timestamp: str    # ISO-8601 ("2026-05-10T14:23:45")
    trigger: str
    label: Optional[str]
    clip_count: int
    source_count: int
    size_bytes: int

    def datetime(self) -> datetime:
        try:
            return datetime.fromisoformat(self.timestamp)
        except ValueError:
            return datetime.fromtimestamp(0)


class ProjectVersionStore:
    """Manages versions for one project file."""

    def __init__(self, project_path: str):
        self._project_path = project_path
        self._dir = Path(project_path + ".versions")
        self._index = self._dir / "index.json"

    # ------------------------------------------------------------------
    # Public API

    @property
    def versions_dir(self) -> Path:
        return self._dir

    @property
    def project_path(self) -> str:
        return self._project_path

    def list_versions(self) -> List[VersionEntry]:
        """Return all versions, newest first.

        Self-healing: reconciles `index.json` against the directory contents
        on every call so manual deletions / external copies don't desync.
        """
        if not self._dir.exists():
            return []
        manifest = self._read_manifest()
        on_disk = {p.name for p in self._dir.glob("*.psynth")}

        # Drop manifest entries for missing files
        kept = [e for e in manifest if e.filename in on_disk]

        # Surface orphan files (e.g. user copied a version manually)
        known = {e.filename for e in kept}
        orphans = on_disk - known
        for fname in orphans:
            entry = self._derive_entry(fname)
            if entry is not None:
                kept.append(entry)

        if len(kept) != len(manifest):
            self._write_manifest(kept)

        kept.sort(key=lambda e: e.datetime(), reverse=True)
        return kept

    def create(
        self,
        trigger: str,
        label: Optional[str] = None,
    ) -> Optional[VersionEntry]:
        """Copy the current `.psynth` into the versions dir, update the
        manifest, then prune. Returns the new entry, or None on failure.
        """
        if trigger not in VALID_TRIGGERS:
            logger.warning("project_versions.create: unknown trigger %r", trigger)
            return None
        if not os.path.isfile(self._project_path):
            logger.debug("project_versions.create: project not on disk yet")
            return None

        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("project_versions: cannot create %s: %s", self._dir, e)
            return None

        now = datetime.now()
        ts = now.strftime(_TS_FMT)
        label_slug = _slugify(label) if label else None
        # Trigger gets baked into the filename (sans the autosave default for
        # readability), separately from any user label, so the dir is browsable
        # without consulting the manifest.
        suffix_parts: List[str] = []
        if trigger != "autosave":
            suffix_parts.append(trigger.replace("_", "-"))
        if label_slug:
            suffix_parts.append(label_slug)
        suffix = ("_" + "-".join(suffix_parts)) if suffix_parts else ""
        fname = f"{ts}{suffix}.psynth"
        # Collision-safe (same-second snapshots): append _2, _3, ...
        target = self._dir / fname
        i = 2
        while target.exists():
            fname = f"{ts}{suffix}_{i}.psynth"
            target = self._dir / fname
            i += 1

        try:
            shutil.copy2(self._project_path, target)
        except (OSError, shutil.SameFileError) as e:
            logger.warning("project_versions: copy failed %s -> %s: %s",
                           self._project_path, target, e)
            return None

        clip_count, source_count = _peek_counts(target)
        size_bytes = _safe_size(target)

        entry = VersionEntry(
            filename=fname,
            timestamp=now.replace(microsecond=0).isoformat(),
            trigger=trigger,
            label=label or None,
            clip_count=clip_count,
            source_count=source_count,
            size_bytes=size_bytes,
        )

        # Read-only fetch of the current manifest. Avoid list_versions() here:
        # it self-heals orphan .psynth files into the manifest, and the file
        # we just copied is on disk but not yet in the manifest — that path
        # would re-derive it as an orphan, then insert() below would dupe it.
        manifest = self._read_manifest()
        manifest.insert(0, entry)
        manifest.sort(key=lambda e: e.datetime(), reverse=True)
        self._write_manifest(manifest)

        try:
            self.prune()
        except Exception:
            logger.exception("project_versions: prune failed (non-fatal)")

        return entry

    def restore(self, filename: str) -> Optional[Path]:
        """Return the absolute path to the version `.psynth` for loading,
        or None if it's missing. Caller is responsible for taking a
        pre_restore snapshot BEFORE calling this.
        """
        target = self._dir / filename
        if not target.is_file():
            logger.warning("project_versions.restore: missing %s", target)
            return None
        return target

    def delete(self, filename: str) -> bool:
        target = self._dir / filename
        try:
            target.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("project_versions.delete: %s: %s", target, e)
            return False
        manifest = [e for e in self._read_manifest() if e.filename != filename]
        self._write_manifest(manifest)
        return True

    def prune(self) -> int:
        """Apply retention. Returns number of versions removed."""
        entries = self.list_versions()  # newest first
        if len(entries) <= _KEEP_NEWEST:
            return 0

        keep: List[VersionEntry] = list(entries[:_KEEP_NEWEST])
        # For older entries, bucket and keep the newest in each bucket.
        seen_buckets: set = set()
        now = datetime.now()
        for e in entries[_KEEP_NEWEST:]:
            ts = e.datetime()
            age = now - ts
            if age <= _HOURLY_WINDOW:
                bucket = ("h", ts.strftime("%Y-%m-%d_%H"))
            elif age <= _DAILY_WINDOW:
                bucket = ("d", ts.strftime("%Y-%m-%d"))
            else:
                bucket = ("w", ts.strftime("%G-%V"))  # ISO year + ISO week
            if bucket in seen_buckets:
                continue
            seen_buckets.add(bucket)
            keep.append(e)

        keep_filenames = {e.filename for e in keep}
        removed = 0
        for e in entries:
            if e.filename in keep_filenames:
                continue
            try:
                (self._dir / e.filename).unlink(missing_ok=True)
                removed += 1
            except OSError as ex:
                logger.warning("prune: cannot remove %s: %s", e.filename, ex)

        if removed:
            keep.sort(key=lambda e: e.datetime(), reverse=True)
            self._write_manifest(keep)
        return removed

    # ------------------------------------------------------------------
    # Internal

    def _read_manifest(self) -> List[VersionEntry]:
        if not self._index.exists():
            return []
        try:
            with self._index.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("project_versions: index read failed: %s", e)
            return []
        out: List[VersionEntry] = []
        for d in data.get("versions", []):
            try:
                out.append(VersionEntry(
                    filename=d["filename"],
                    timestamp=d["timestamp"],
                    trigger=d.get("trigger", "manual"),
                    label=d.get("label"),
                    clip_count=int(d.get("clip_count", 0)),
                    source_count=int(d.get("source_count", 0)),
                    size_bytes=int(d.get("size_bytes", 0)),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def _write_manifest(self, entries: List[VersionEntry]) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = self._index.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(
                    {"versions": [asdict(e) for e in entries]},
                    f, indent=2,
                )
            os.replace(tmp, self._index)
        except OSError as e:
            logger.warning("project_versions: index write failed: %s", e)

    def _derive_entry(self, fname: str) -> Optional[VersionEntry]:
        """Build a VersionEntry from a file we found on disk that's not
        in the manifest (e.g. user copied a snapshot in)."""
        m = _FNAME_RE.match(fname)
        if not m:
            return None
        try:
            ts = datetime.strptime(m.group(1), _TS_FMT)
        except ValueError:
            return None
        trigger_slug = m.group(2) or "manual"
        # Filename slug is hyphenated; convert back to underscored trigger.
        trigger_guess = trigger_slug.replace("-", "_")
        if trigger_guess not in VALID_TRIGGERS:
            trigger_guess = "manual"
        target = self._dir / fname
        clip_count, source_count = _peek_counts(target)
        return VersionEntry(
            filename=fname,
            timestamp=ts.replace(microsecond=0).isoformat(),
            trigger=trigger_guess,
            label=None,
            clip_count=clip_count,
            source_count=source_count,
            size_bytes=_safe_size(target),
        )


# ----------------------------------------------------------------------
# Helpers

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    s = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return s[:40]


def _peek_counts(path: Path) -> tuple:
    """Read clip_count and source_count from a .psynth without fully
    deserialising. Returns (0, 0) on any failure."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return (len(data.get("clips", [])), len(data.get("sources", [])))
    except (OSError, json.JSONDecodeError):
        return (0, 0)


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
