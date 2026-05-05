import copy
from typing import Dict, Iterable, List, Optional, Tuple
from PySide6.QtCore import QObject, Signal
from core.clip import Clip
from core.group import Group, GROUP_COLOR_PALETTE

MAX_UNDO = 50


class TimelineModel(QObject):
    clips_changed = Signal()
    selection_changed = Signal()
    in_out_changed = Signal()
    groups_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._clips: List[Clip] = []
        self._selected_ids: set = set()
        self._color_counter: int = 0
        self._undo_stack: list = []
        self._redo_stack: list = []
        self._in_point: Optional[int] = None
        self._out_point: Optional[int] = None
        self._groups: Dict[str, Group] = {}

    # --- Undo / Redo ---

    def _snapshot(self) -> tuple:
        # deepcopy of clips so mutable Clip.group_ids lists don't leak
        # between the live model and the snapshot.
        return (
            [copy.deepcopy(c) for c in self._clips],
            set(self._selected_ids),
            self._color_counter,
            copy.deepcopy(self._groups),
        )

    def _restore(self, snapshot: tuple):
        # Tolerate legacy 3-tuple snapshots in case any in-memory ones
        # predate the groups field (defensive).
        if len(snapshot) == 4:
            clips, selected, color_counter, groups = snapshot
        else:
            clips, selected, color_counter = snapshot
            groups = {}
        self._clips = clips
        self._selected_ids = selected
        self._color_counter = color_counter
        self._groups = groups
        self.clips_changed.emit()
        self.selection_changed.emit()
        self.groups_changed.emit()

    def _push_undo(self):
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(self._snapshot())
        self._restore(self._undo_stack.pop())

    def redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())

    def clear_undo(self):
        self._undo_stack.clear()
        self._redo_stack.clear()

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
        if not clips:
            return
        self._push_undo()
        if assign_colors:
            for c in clips:
                if not c.is_gap:
                    c.color_index = self._color_counter % 8
                    self._color_counter += 1
        self._clips.extend(clips)
        self.clips_changed.emit()

    def add_clip(self, clip: Clip, index: Optional[int] = None):
        self._push_undo()
        if not clip.is_gap:
            clip.color_index = self._color_counter % 8
            self._color_counter += 1
        if index is None:
            self._clips.append(clip)
        else:
            self._clips.insert(index, clip)
        self.clips_changed.emit()

    def insert_clips_at_frame(self, clips: List[Clip], timeline_frame: int,
                              assign_colors: bool = True):
        """Insert clips at the given timeline frame, splitting an existing
        clip if the frame falls inside one. Used by the Media Panel's
        drag-to-timeline drop. Earlier clips stay put; later clips ripple
        back by the inserted duration."""
        if not clips:
            return
        self._push_undo()
        if assign_colors:
            for c in clips:
                if not c.is_gap:
                    c.color_index = self._color_counter % 8
                    self._color_counter += 1

        total = self.get_total_duration_frames()
        if timeline_frame >= total or not self._clips:
            self._clips.extend(clips)
            self.clips_changed.emit()
            return
        if timeline_frame <= 0:
            self._clips = list(clips) + self._clips
            self.clips_changed.emit()
            return

        # Find the clip containing the target frame and the offset within it.
        pos = 0
        insert_idx = len(self._clips)
        for i, c in enumerate(self._clips):
            if pos <= timeline_frame < pos + c.duration_frames:
                offset = timeline_frame - pos
                if offset == 0:
                    insert_idx = i
                else:
                    # Split this clip in two so the new clips slot between halves.
                    if c.is_gap:
                        left = Clip.make_gap(offset)
                        right = Clip.make_gap(c.duration_frames - offset)
                        # Preserve the original gap's id on the left half so undo
                        # snapshots referencing it stay valid.
                        left.id = c.id
                    else:
                        import uuid as _uuid
                        from copy import copy
                        left = copy(c)
                        right = copy(c)
                        right.id = _uuid.uuid4().hex[:12]  # fresh id for the right half
                        left.source_out = c.source_in + offset - 1
                        right.source_in = c.source_in + offset
                    self._clips[i:i + 1] = [left, right]
                    insert_idx = i + 1
                break
            pos += c.duration_frames

        self._clips[insert_idx:insert_idx] = clips
        self.clips_changed.emit()

    def remove_source_clips(self, source_id: str) -> int:
        """Replace every clip referencing the given source with a gap.
        Returns the number of clips removed. Used by the Media Panel's
        'Remove source AND its clips' action — we keep gaps so the timeline
        layout stays stable; the user can then ripple-delete those gaps if
        they want compaction."""
        affected = [c for c in self._clips if c.source_id == source_id]
        if not affected:
            return 0
        self.remove_clips({c.id for c in affected})
        return len(affected)

    def count_clips_for_source(self, source_id: str) -> int:
        """How many real clips reference this source. Used by the remove dialog."""
        return sum(1 for c in self._clips if c.source_id == source_id)

    def remove_clips(self, clip_ids: set):
        """Replace real clips with gaps, remove gaps outright. Merge adjacent gaps."""
        self._push_undo()
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

    def split_clip_at(self, clip_id: str, timeline_frame: int,
                      select_left_only: bool = False) -> bool:
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

        self._push_undo()
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
            if not select_left_only:
                self._selected_ids.add(clip_b.id)

        self.clips_changed.emit()
        self.selection_changed.emit()
        return True

    def clear(self):
        self._clips.clear()
        self._selected_ids.clear()
        self._color_counter = 0
        self._in_point = None
        self._out_point = None
        self._groups.clear()
        self.clear_undo()
        self.clips_changed.emit()
        self.selection_changed.emit()
        self.in_out_changed.emit()
        self.groups_changed.emit()

    # --- People / Groups ---

    @property
    def groups(self) -> Dict[str, Group]:
        """Live dict — callers must not mutate it directly."""
        return self._groups

    def get_group_by_digit(self, digit: Optional[int]) -> Optional[Group]:
        if digit is None:
            return None
        for g in self._groups.values():
            if g.digit == digit:
                return g
        return None

    def _next_group_color(self) -> str:
        used = {g.color for g in self._groups.values()}
        for c in GROUP_COLOR_PALETTE:
            if c not in used:
                return c
        # Palette exhausted — recycle by index (count modulo palette size).
        return GROUP_COLOR_PALETTE[len(self._groups) % len(GROUP_COLOR_PALETTE)]

    def add_group(self, name: str, color: Optional[str] = None,
                  digit: Optional[int] = None) -> Group:
        """Create and register a new group. ``color`` defaults to the next
        free palette entry. ``digit`` must be either None or a free digit
        (0-9); raises ValueError if that digit is already in use."""
        if digit is not None and self.get_group_by_digit(digit) is not None:
            raise ValueError(f"digit {digit} already in use")
        if color is None:
            color = self._next_group_color()
        self._push_undo()
        g = Group(name=name, color=color, digit=digit)
        self._groups[g.id] = g
        self.groups_changed.emit()
        return g

    def remove_group(self, group_id: str):
        """Delete a group and strip its id from every clip's group_ids."""
        if group_id not in self._groups:
            return
        self._push_undo()
        del self._groups[group_id]
        for c in self._clips:
            if group_id in c.group_ids:
                c.group_ids = [gid for gid in c.group_ids if gid != group_id]
        self.groups_changed.emit()
        self.clips_changed.emit()

    def update_group(self, group_id: str,
                     name: Optional[str] = None,
                     color: Optional[str] = None,
                     digit=Ellipsis) -> Optional[str]:
        """Rename / recolour / re-bind digit. Pass digit=Ellipsis (default)
        to leave it unchanged; pass None to clear; pass 0-9 to assign.

        Returns:
          - ``None`` on success.
          - The conflicting group's display name when the requested digit
            is already held by a different group.
        """
        g = self._groups.get(group_id)
        if g is None:
            return None
        if digit is not Ellipsis and digit is not None:
            owner = self.get_group_by_digit(digit)
            if owner is not None and owner.id != group_id:
                return owner.name
        self._push_undo()
        if name is not None:
            g.name = name
        if color is not None:
            g.color = color
        if digit is not Ellipsis:
            g.digit = digit
        self.groups_changed.emit()
        return None

    def toggle_clip_group(self, clip_ids, group_id: str):
        """Toggle group membership for one or many clip ids. Behaviour
        on multi-select: if every targeted clip already has the group,
        the group is removed from all of them; otherwise it's added to
        every clip that doesn't yet have it (so all become members)."""
        if group_id not in self._groups:
            return
        if isinstance(clip_ids, str):
            clip_ids = [clip_ids]
        ids_set = set(clip_ids)
        targets = [c for c in self._clips
                   if c.id in ids_set and not c.is_gap]
        if not targets:
            return
        all_have = all(group_id in c.group_ids for c in targets)
        self._push_undo()
        for c in targets:
            if all_have:
                if group_id in c.group_ids:
                    c.group_ids = [gid for gid in c.group_ids
                                   if gid != group_id]
            else:
                if group_id not in c.group_ids:
                    c.group_ids = list(c.group_ids) + [group_id]
        self.clips_changed.emit()

    def set_groups_bulk(self, groups_iter: Iterable[Group]):
        """Replace the entire groups dict from an iterable of Group objects.
        Used by project load. Does NOT push undo (load is a fresh state)."""
        self._groups = {g.id: g for g in groups_iter}
        self.groups_changed.emit()

    # --- In/Out Points ---

    @property
    def in_point(self) -> Optional[int]:
        return self._in_point

    @property
    def out_point(self) -> Optional[int]:
        return self._out_point

    def set_in_point(self, frame: Optional[int]):
        # Reject values that would invalidate the pair instead of silently
        # wiping the opposite marker (in/out is not undoable, so a wipe
        # destroys data the user cannot recover).
        if frame is not None and self._out_point is not None:
            # The visible Out marker is rendered at out_pt+1's pixel (so the
            # dim overlay covers everything past the Out frame), so a click
            # on the marker line lands the playhead at out_pt+1. Accept both
            # out_pt and out_pt+1 as "at Out" for the collision-nudge.
            if frame == self._out_point or frame == self._out_point + 1:
                last_frame = self.get_total_duration_frames() - 1
                if frame >= last_frame:
                    return  # No room to push Out past the timeline tail
                self._out_point = frame + 1
            elif frame > self._out_point:
                return
        if self._in_point == frame:
            return
        self._in_point = frame
        self.in_out_changed.emit()

    def set_out_point(self, frame: Optional[int]):
        if frame is not None and self._in_point is not None:
            if frame == self._in_point:
                # Collision-nudge: pressing R at the In frame shoves In
                # back by one so the gesture sets Out here without losing In.
                if self._in_point <= 0:
                    return
                self._in_point = self._in_point - 1
            elif frame < self._in_point:
                return
        if self._out_point == frame:
            return
        self._out_point = frame
        self.in_out_changed.emit()

    def clear_in_out(self):
        self._in_point = None
        self._out_point = None
        self.in_out_changed.emit()

    def get_render_range(self) -> Tuple[int, int]:
        total = self.get_total_duration_frames()
        if total == 0:
            return (0, 0)
        start = self._in_point if self._in_point is not None else 0
        end = self._out_point if self._out_point is not None else total - 1
        start = min(start, total - 1)
        end = min(end, total - 1)
        return (start, end)

    def get_used_source_ids(self, use_render_range: bool) -> list:
        """Return source IDs referenced by non-gap clips that would be exported,
        in first-encountered order. Mirrors the iteration in xml_exporter /
        otio_exporter so the export dialog's audio summary matches what the
        actual exporter would emit."""
        if use_render_range:
            r_start, r_end = self.get_render_range()
        else:
            total = self.get_total_duration_frames()
            r_start, r_end = 0, max(total - 1, 0)

        pos = 0
        used = []
        seen = set()
        for c in self._clips:
            clip_start = pos
            clip_end = pos + c.duration_frames - 1
            pos += c.duration_frames
            if clip_end < r_start or clip_start > r_end:
                continue
            if c.is_gap or c.source_id is None:
                continue
            if c.source_id in seen:
                continue
            seen.add(c.source_id)
            used.append(c.source_id)
        return used

    def get_export_audio_summary(self, sources: dict,
                                 use_render_range: bool) -> str:
        """One-line audio summary for the sources that would be exported.

        Returns:
          - 'none' if no source has audio (or no sources used)
          - the single source's format_audio() string when all referenced
            sources share the same audio config
          - 'mixed (N ch and silent)' when some have audio and others don't
          - 'mixed' when multiple distinct audio configs are present
        """
        used = self.get_used_source_ids(use_render_range)
        descs = []
        for sid in used:
            src = sources.get(sid)
            if src is None:
                continue
            descs.append(src.format_audio())
        if not descs:
            return "none"
        unique = list(dict.fromkeys(descs))
        if len(unique) == 1:
            return unique[0]
        # Distinguish "some have audio, some don't" from "all have audio but configs differ"
        if "none" in unique and len(unique) == 2:
            other = next(d for d in unique if d != "none")
            return f"mixed ({other} and silent)"
        return "mixed"

    def compute_export_extent(self, include_gaps: bool,
                              use_render_range: bool) -> Tuple[int, int]:
        """Counts (real_clips, frames) that would be exported under the given flags.
        Mirrors what xml_exporter / otio_exporter actually emit so the export
        dialog's info text matches the produced file.
        """
        if use_render_range:
            r_start, r_end = self.get_render_range()
        else:
            total = self.get_total_duration_frames()
            r_start, r_end = 0, max(total - 1, 0)

        pos = 0
        clips = 0
        frames = 0
        for c in self._clips:
            clip_start = pos
            clip_end = pos + c.duration_frames - 1
            pos += c.duration_frames
            if clip_end < r_start or clip_start > r_end:
                continue
            in_range_dur = min(clip_end, r_end) - max(clip_start, r_start) + 1
            if c.is_gap:
                if include_gaps:
                    frames += in_range_dur
            else:
                clips += 1
                frames += in_range_dur
        return clips, frames

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
        """Select all real clips between the playhead and the first gap to the left.
        If the playhead is on a gap, starts from the clip just before it."""
        result = self.get_clip_at_position(timeline_frame)
        if result is None:
            return
        clip, _ = result
        idx = self.get_clip_index(clip.id)
        if idx < 0:
            return

        # If on a gap, skip to the clip before it
        if clip.is_gap:
            idx -= 1
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

    def set_selection(self, ids: set):
        """Replace selection with the given set of clip IDs."""
        self._selected_ids = set(ids)
        self.selection_changed.emit()

    def select_all(self):
        self._selected_ids = {c.id for c in self._clips}
        self.selection_changed.emit()

    def get_selected_clips(self) -> List[Clip]:
        return [c for c in self._clips if c.id in self._selected_ids]

    def delete_selected(self):
        """Delete selected clips, replacing them with gaps (non-ripple)."""
        if self._selected_ids:
            self.remove_clips(set(self._selected_ids))

    def ripple_delete_selected(self):
        """Ripple delete selected clips — removes them and collapses the space."""
        if not self._selected_ids:
            return
        self._push_undo()
        self._clips = [c for c in self._clips if c.id not in self._selected_ids]
        self._merge_adjacent_gaps()
        self._selected_ids.clear()
        self.clips_changed.emit()
        self.selection_changed.emit()

    def ripple_delete_by_source(self, source_id: str):
        """Remove all clips belonging to a source. Gaps are preserved. Used when re-detecting cuts."""
        self._push_undo()
        self._clips = [c for c in self._clips if c.source_id != source_id]
        self._merge_adjacent_gaps()
        self._selected_ids.clear()
        self.clips_changed.emit()
        self.selection_changed.emit()

    def replace_detected(self, replacements: dict, assign_colors: bool = True):
        """Replace clips by ID with lists of detected sub-clips.
        replacements: {clip_id: [Clip, ...]}. Gaps and non-matched clips are preserved."""
        # Drop entries with empty sub-clips — would silently delete the original
        # without leaving a gap, violating the "non-matched clips preserved" rule.
        replacements = {k: v for k, v in replacements.items() if v}
        if not replacements:
            return
        self._push_undo()
        new_list = []
        for c in self._clips:
            if c.id in replacements:
                sub_clips = replacements[c.id]
                if assign_colors:
                    for sc in sub_clips:
                        if not sc.is_gap:
                            sc.color_index = self._color_counter % 8
                            self._color_counter += 1
                new_list.extend(sub_clips)
            else:
                new_list.append(c)
        self._clips = new_list
        self._selected_ids.clear()
        self.clips_changed.emit()
        self.selection_changed.emit()
