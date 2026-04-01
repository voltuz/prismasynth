from typing import List, Optional, Tuple
from PySide6.QtCore import QObject, Signal
from core.clip import Clip


class TimelineModel(QObject):
    clips_changed = Signal()
    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._clips: List[Clip] = []
        self._selected_ids: set = set()
        self._color_counter: int = 0

    # --- Item access ---

    @property
    def clips(self) -> List[Clip]:
        """All items (clips + gaps) in timeline order."""
        return list(self._clips)

    @property
    def clip_count(self) -> int:
        """Total items including gaps."""
        return len(self._clips)

    @property
    def real_clip_count(self) -> int:
        """Count of real clips (not gaps)."""
        return sum(1 for c in self._clips if not c.is_gap)

    def get_clip_by_id(self, clip_id: str) -> Optional[Clip]:
        for c in self._clips:
            if c.id == clip_id:
                return c
        return None

    def get_clip_index(self, clip_id: str) -> int:
        for i, c in enumerate(self._clips):
            if c.id == clip_id:
                return i
        return -1

    # --- Timeline position mapping ---

    def get_total_duration_frames(self) -> int:
        return sum(c.duration_frames for c in self._clips)

    def get_clip_timeline_start(self, clip_id: str) -> int:
        pos = 0
        for c in self._clips:
            if c.id == clip_id:
                return pos
            pos += c.duration_frames
        return -1

    def get_clip_at_position(self, timeline_frame: int) -> Optional[Tuple[Clip, int]]:
        """Return (clip_or_gap, offset_within_item) for a given timeline frame."""
        pos = 0
        for c in self._clips:
            if pos <= timeline_frame < pos + c.duration_frames:
                return (c, timeline_frame - pos)
            pos += c.duration_frames
        return None

    def timeline_frame_to_source_frame(self, timeline_frame: int) -> Optional[Tuple[Clip, int]]:
        """Convert timeline frame to (clip, source_frame_number).
        Returns None if position is on a gap or beyond the timeline."""
        result = self.get_clip_at_position(timeline_frame)
        if result is None:
            return None
        clip, offset = result
        if clip.is_gap:
            return None
        return (clip, clip.source_in + offset)

    # --- Mutations ---

    def add_clips(self, clips: List[Clip], assign_colors: bool = True):
        if assign_colors:
            for c in clips:
                if not c.is_gap:
                    c.color_index = self._color_counter % 8
                    self._color_counter += 1
        self._clips.extend(clips)
        self.clips_changed.emit()

    def add_clip(self, clip: Clip, index: Optional[int] = None):
        if not clip.is_gap:
            clip.color_index = self._color_counter % 8
            self._color_counter += 1
        if index is None:
            self._clips.append(clip)
        else:
            self._clips.insert(index, clip)
        self.clips_changed.emit()

    def remove_clips(self, clip_ids: set):
        """Replace real clips with gaps, remove gaps outright. Merge adjacent gaps."""
        new_list = []
        for c in self._clips:
            if c.id in clip_ids:
                if c.is_gap:
                    # Deleting a gap: collapse it (don't add anything)
                    pass
                else:
                    # Deleting a real clip: replace with a gap of same duration
                    new_list.append(Clip.make_gap(c.duration_frames))
            else:
                new_list.append(c)

        self._clips = new_list
        self._merge_adjacent_gaps()
        self._selected_ids -= clip_ids
        self.clips_changed.emit()
        self.selection_changed.emit()

    def _merge_adjacent_gaps(self):
        """Merge consecutive gap items into a single gap."""
        if not self._clips:
            return
        merged = [self._clips[0]]
        for c in self._clips[1:]:
            if c.is_gap and merged[-1].is_gap:
                # Merge: extend the previous gap
                prev = merged[-1]
                combined_duration = prev.duration_frames + c.duration_frames
                merged[-1] = Clip.make_gap(combined_duration)
                merged[-1].id = prev.id  # keep the first gap's ID
            else:
                merged.append(c)
        self._clips = merged

    def split_clip_at(self, clip_id: str, timeline_frame: int) -> bool:
        """Split a clip at the given timeline frame position.
        Only works on real clips, not gaps. Returns True if split was performed."""
        result = self.get_clip_at_position(timeline_frame)
        if result is None:
            return False
        clip, offset = result
        if clip.id != clip_id or clip.is_gap:
            return False
        if offset <= 0 or offset >= clip.duration_frames - 1:
            return False

        idx = self.get_clip_index(clip_id)
        if idx < 0:
            return False

        clip_a = Clip(
            source_id=clip.source_id,
            source_in=clip.source_in,
            source_out=clip.source_in + offset - 1,
            label=clip.label,
            color_index=clip.color_index,
        )
        clip_b = Clip(
            source_id=clip.source_id,
            source_in=clip.source_in + offset,
            source_out=clip.source_out,
            label=clip.label,
            color_index=clip.color_index,
        )

        self._clips[idx:idx + 1] = [clip_a, clip_b]
        if clip_id in self._selected_ids:
            self._selected_ids.discard(clip_id)
            self._selected_ids.add(clip_a.id)
            self._selected_ids.add(clip_b.id)

        self.clips_changed.emit()
        self.selection_changed.emit()
        return True

    def clear(self):
        self._clips.clear()
        self._selected_ids.clear()
        self._color_counter = 0
        self.clips_changed.emit()
        self.selection_changed.emit()

    # --- Selection ---

    @property
    def selected_ids(self) -> set:
        return set(self._selected_ids)

    def select_clip(self, clip_id: str, exclusive: bool = True):
        if exclusive:
            self._selected_ids = {clip_id}
        else:
            self._selected_ids.add(clip_id)
        self.selection_changed.emit()

    def deselect_clip(self, clip_id: str):
        self._selected_ids.discard(clip_id)
        self.selection_changed.emit()

    def toggle_select(self, clip_id: str):
        if clip_id in self._selected_ids:
            self._selected_ids.discard(clip_id)
        else:
            self._selected_ids.add(clip_id)
        self.selection_changed.emit()

    def select_range(self, from_id: str, to_id: str):
        """Select all items between from_id and to_id (inclusive)."""
        idx_a = self.get_clip_index(from_id)
        idx_b = self.get_clip_index(to_id)
        if idx_a < 0 or idx_b < 0:
            return
        lo, hi = min(idx_a, idx_b), max(idx_a, idx_b)
        self._selected_ids = {self._clips[i].id for i in range(lo, hi + 1)}
        self.selection_changed.emit()

    def clear_selection(self):
        self._selected_ids.clear()
        self.selection_changed.emit()

    def select_to_gap_left(self, timeline_frame: int):
        """Select all real clips between the playhead and the first gap to the left."""
        result = self.get_clip_at_position(timeline_frame)
        if result is None:
            return
        clip, _ = result
        idx = self.get_clip_index(clip.id)
        if idx < 0:
            return

        # Walk left from idx, collecting real clips until we hit a gap
        selected = set()
        for i in range(idx, -1, -1):
            item = self._clips[i]
            if item.is_gap:
                break
            selected.add(item.id)
        self._selected_ids = selected
        self.selection_changed.emit()

    def select_all(self):
        self._selected_ids = {c.id for c in self._clips}
        self.selection_changed.emit()

    def get_selected_clips(self) -> List[Clip]:
        return [c for c in self._clips if c.id in self._selected_ids]

    def delete_selected(self):
        if self._selected_ids:
            self.remove_clips(set(self._selected_ids))
