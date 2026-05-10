"""Warning dialog for sources whose container time_base can't exactly
represent their declared fps. Replaces the previous QMessageBox with a
custom dialog that lets the user trigger an in-app ffmpeg remux to fix
each file (sidesteps the user having to copy a CLI command into a shell).

Per-source we precompute three things:

  1. **Input** for the remux. If a sibling ``<basename>.mkv`` exists next
     to the listed file, prefer it — that's almost always the original
     Bluray rip the .mov was made from, so remuxing from it is one step
     cleaner than re-remuxing the broken-timebase intermediary.
  2. **Output** = ``<basename>_fixed.mov`` next to the original. We pin
     ``.mov`` because Resolve imports MOV cleanly and that's why the
     user produced a .mov in the first place; even when the input is a
     sibling .mkv, the output stays .mov so PrismaSynth ends up with the
     same container the user already had working in their pipeline.
  3. **Target timescale** = the denominator returned by
     ``core.video_source._frame_duration_for_fps`` for the source's fps.
     That value is exactly the FCPXML denominator the exporter writes,
     so every PTS in the resulting container is an integer number of
     ticks and Resolve's time-seek lands on the right source frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFrame, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from core.video_source import VideoSource, _frame_duration_for_fps


# Plan tuple: (source_id, input_path, output_path, target_timescale)
RemuxPlan = Tuple[str, str, str, int]


@dataclass
class _RowPlan:
    source: VideoSource
    input_path: str
    output_path: str
    target_timescale: int
    used_mkv_sibling: bool


def _pick_remux_input(source_path: str) -> Tuple[str, bool]:
    """Pick the file to feed into ``ffmpeg -i``.

    If a sibling ``<basename>.mkv`` exists and the listed file isn't
    already a .mkv, prefer the sibling — it's the original rip.
    """
    base, ext = os.path.splitext(source_path)
    sibling_mkv = base + ".mkv"
    if ext.lower() != ".mkv" and os.path.exists(sibling_mkv):
        return sibling_mkv, True
    return source_path, False


def _fixed_output_path(source_path: str) -> str:
    """``<basename>_fixed.mov`` in the original's folder. MOV regardless of
    input container — see module docstring."""
    base, _ = os.path.splitext(source_path)
    return base + "_fixed.mov"


def build_plan(sources: List[VideoSource]) -> List[_RowPlan]:
    rows: List[_RowPlan] = []
    for s in sources:
        in_path, used_mkv = _pick_remux_input(s.file_path)
        out_path = _fixed_output_path(s.file_path)
        _, target_ts = _frame_duration_for_fps(s.fps)
        rows.append(_RowPlan(
            source=s, input_path=in_path, output_path=out_path,
            target_timescale=target_ts, used_mkv_sibling=used_mkv,
        ))
    return rows


def _truncate_middle(text: str, max_len: int = 56) -> str:
    """Long Bluray filenames are unreadable when wrapped — keep the head
    + tail so the user can still recognise it."""
    if len(text) <= max_len:
        return text
    keep = (max_len - 1) // 2
    return text[:keep] + "…" + text[-keep:]


class TimebaseWarningDialog(QDialog):
    """Lists unsafe sources, shows the remux plan per row, and offers
    Auto-fix / Dismiss. Auto-fix emits ``auto_fix_requested`` with a list
    of ``RemuxPlan`` tuples so the caller can run the worker."""

    auto_fix_requested = Signal(list)  # List[RemuxPlan]

    def __init__(self, unsafe_sources: List[VideoSource], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Source timebase warning")
        self.setModal(True)
        self.setMinimumWidth(620)

        self._plans = build_plan(unsafe_sources)

        layout = QVBoxLayout(self)

        header = QLabel(
            "These source files have a container time_base that can't "
            "exactly represent their declared fps. FCPXML / OTIO export "
            "still works, but NLE imports (Resolve, Premiere) may drift "
            "clip start/end frames by ±1 because the NLE seeks the source "
            "by time and rounds onto the broken tick grid. Video / audio "
            "/ image-sequence exports are unaffected.\n\n"
            "Auto-fix runs ffmpeg with -c copy (no re-encode, no quality "
            "loss) and writes a *_fixed.mov next to each original. "
            "PrismaSynth then relinks the project to the fixed file — "
            "your timeline edits are preserved."
        )
        header.setWordWrap(True)
        header.setStyleSheet("color: #ccc;")
        layout.addWidget(header)

        # Per-source rows in a scroll area so a project with many sources
        # doesn't push the buttons off-screen.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        host = QWidget()
        rows_layout = QVBoxLayout(host)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setSpacing(8)

        for plan in self._plans:
            rows_layout.addWidget(self._build_row(plan))
        rows_layout.addStretch(1)

        scroll.setWidget(host)
        layout.addWidget(scroll, 1)

        btns = QDialogButtonBox()
        self._fix_btn = QPushButton("Auto-fix")
        self._fix_btn.setDefault(True)
        self._fix_btn.clicked.connect(self._on_auto_fix)
        btns.addButton(self._fix_btn, QDialogButtonBox.ButtonRole.AcceptRole)

        dismiss_btn = QPushButton("Dismiss")
        dismiss_btn.clicked.connect(self.reject)
        btns.addButton(dismiss_btn, QDialogButtonBox.ButtonRole.RejectRole)
        layout.addWidget(btns)

    # --- internals ----------------------------------------------------

    def _build_row(self, plan: _RowPlan) -> QWidget:
        row = QFrame()
        row.setFrameShape(QFrame.Shape.StyledPanel)
        row.setStyleSheet(
            "QFrame { background-color: #2a2a2a; border-radius: 4px; }"
            "QLabel { background: transparent; }"
        )
        v = QVBoxLayout(row)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(2)

        s = plan.source
        name = QLabel(_truncate_middle(os.path.basename(s.file_path)))
        name.setStyleSheet("color: #ddd; font-weight: bold;")
        name.setToolTip(s.file_path)
        v.addWidget(name)

        meta = QLabel(
            f"timebase {s.time_base_str}, {s.fps:.3f} fps  →  "
            f"target -video_track_timescale {plan.target_timescale}")
        meta.setStyleSheet("color: #aaa; font-size: 11px;")
        v.addWidget(meta)

        in_text = f"Input:  {_truncate_middle(plan.input_path, 70)}"
        if plan.used_mkv_sibling:
            in_text += "  (sibling .mkv detected)"
        in_lbl = QLabel(in_text)
        in_lbl.setToolTip(plan.input_path)
        in_lbl.setStyleSheet(
            "color: #8fbf6f; font-size: 11px;" if plan.used_mkv_sibling
            else "color: #bbb; font-size: 11px;"
        )
        v.addWidget(in_lbl)

        out_lbl = QLabel(f"Output: {_truncate_middle(plan.output_path, 70)}")
        out_lbl.setToolTip(plan.output_path)
        out_lbl.setStyleSheet("color: #bbb; font-size: 11px;")
        v.addWidget(out_lbl)

        return row

    def _on_auto_fix(self):
        plans: List[RemuxPlan] = [
            (p.source.id, p.input_path, p.output_path, p.target_timescale)
            for p in self._plans
        ]
        self.auto_fix_requested.emit(plans)
        self.accept()
