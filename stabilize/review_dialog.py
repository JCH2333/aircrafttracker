"""CustomTkinter dialog for correcting low-confidence tracking segments."""

from __future__ import annotations

import tkinter as tk

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from stabilize.review import ReviewRequest
from stabilize.tracking.models import TrackingState
from stabilize.tracking.track_file import ManualAnchor


class TrackReviewDialog(ctk.CTkToplevel):
    def __init__(self, parent, request: ReviewRequest):
        super().__init__(parent)
        self.request = request
        self.result: list[ManualAnchor] = []
        self._new_anchors: dict[int, ManualAnchor] = {}
        self._undo_stack: list[tuple[int, ManualAnchor | None]] = []
        self._segment_index = 0
        self._frame_idx = self._segment_midpoint(0)
        self._capture = cv2.VideoCapture(str(request.video_path))
        self._photo = None
        self._display_scale = 1.0
        self._image_origin = (0, 0)
        self._image_size = (1, 1)
        self._drag_start = None
        self._selection_id = None

        self.title("Tracking review")
        self.geometry("1120x780")
        self.minsize(920, 680)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._finish)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._status = ctk.CTkLabel(
            self,
            text="",
            anchor="w",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self._status.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 8))

        self._canvas = tk.Canvas(
            self,
            bg="#111113",
            highlightthickness=1,
            highlightbackground="#34343a",
            cursor="crosshair",
        )
        self._canvas.grid(row=1, column=0, sticky="nsew", padx=16)
        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Configure>", lambda _event: self._load_frame())

        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.grid(row=2, column=0, sticky="ew", padx=16, pady=14)

        ctk.CTkButton(
            toolbar,
            text="Previous issue",
            width=110,
            command=lambda: self._move_segment(-1),
        ).pack(side="left", padx=(0, 6))
        ctk.CTkButton(
            toolbar,
            text="Frame -1",
            width=82,
            command=lambda: self._move_frame(-1),
        ).pack(side="left", padx=3)
        ctk.CTkButton(
            toolbar,
            text="Frame +1",
            width=82,
            command=lambda: self._move_frame(1),
        ).pack(side="left", padx=3)
        ctk.CTkButton(
            toolbar,
            text="Next issue",
            width=100,
            command=lambda: self._move_segment(1),
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            toolbar,
            text="Prev reliable",
            width=105,
            fg_color="#34343a",
            hover_color="#46464d",
            command=lambda: self._move_reliable(-1),
        ).pack(side="left", padx=3)
        ctk.CTkButton(
            toolbar,
            text="Next reliable",
            width=105,
            fg_color="#34343a",
            hover_color="#46464d",
            command=lambda: self._move_reliable(1),
        ).pack(side="left", padx=3)
        ctk.CTkButton(
            toolbar,
            text="Undo",
            width=72,
            fg_color="#34343a",
            hover_color="#46464d",
            command=self._undo,
        ).pack(side="right", padx=(6, 0))
        ctk.CTkButton(
            toolbar,
            text="Continue rendering",
            width=145,
            command=self._finish,
        ).pack(side="right")

        self.after(50, self._load_frame)

    def _segment_midpoint(self, index: int) -> int:
        segment = self.request.segments[index]
        return (segment.start_frame + segment.end_frame) // 2

    def _move_segment(self, delta: int) -> None:
        self._segment_index = max(
            0,
            min(
                len(self.request.segments) - 1,
                self._segment_index + delta,
            ),
        )
        self._frame_idx = self._segment_midpoint(self._segment_index)
        self._load_frame()

    def _move_frame(self, delta: int) -> None:
        self._frame_idx = max(
            0,
            min(len(self.request.predicted_centers) - 1, self._frame_idx + delta),
        )
        self._load_frame()

    def _move_reliable(self, direction: int) -> None:
        idx = self._frame_idx + direction
        while 0 <= idx < len(self.request.observations):
            observation = self.request.observations[idx]
            if (
                observation.center is not None
                and observation.bbox is not None
                and observation.confidence >= 0.65
                and observation.state
                in (TrackingState.TRACKING, TrackingState.MANUAL_ANCHOR)
            ):
                self._frame_idx = idx
                self._load_frame()
                return
            idx += direction

    def _load_frame(self) -> None:
        canvas_w = max(self._canvas.winfo_width(), 100)
        canvas_h = max(self._canvas.winfo_height(), 100)
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, self._frame_idx)
        ok, frame = self._capture.read()
        if not ok:
            return

        mask = self.request.masks.get(self._frame_idx)
        if mask is not None:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            overlay = frame.copy()
            overlay[mask] = (40, 200, 80)
            frame = cv2.addWeighted(frame, 0.72, overlay, 0.28, 0)

        path_start = max(0, self._frame_idx - 15)
        path_end = min(
            len(self.request.predicted_centers),
            self._frame_idx + 16,
        )
        path_points = np.asarray(
            [
                self.request.predicted_centers[idx]
                for idx in range(path_start, path_end)
            ],
            dtype=np.int32,
        )
        if len(path_points) >= 2:
            cv2.polylines(
                frame,
                [path_points],
                False,
                (0, 220, 255),
                3,
            )

        observation = (
            self.request.observations[self._frame_idx]
            if self._frame_idx < len(self.request.observations)
            else None
        )
        if observation is not None and observation.bbox is not None:
            x, y, w, h = observation.bbox
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (80, 255, 80),
                3,
            )

        center = self.request.predicted_centers[self._frame_idx]
        cv2.drawMarker(
            frame,
            (int(center[0]), int(center[1])),
            (0, 220, 255),
            cv2.MARKER_TILTED_CROSS,
            28,
            3,
        )
        frame_h, frame_w = frame.shape[:2]
        scale = min(canvas_w / frame_w, canvas_h / frame_h, 1.0)
        display_w = max(1, int(round(frame_w * scale)))
        display_h = max(1, int(round(frame_h * scale)))
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (display_w, display_h),
                interpolation=cv2.INTER_AREA,
            )
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(frame))
        origin_x = (canvas_w - display_w) // 2
        origin_y = (canvas_h - display_h) // 2
        self._display_scale = scale
        self._image_origin = (origin_x, origin_y)
        self._image_size = (display_w, display_h)

        self._canvas.delete("all")
        self._canvas.create_image(
            origin_x,
            origin_y,
            image=self._photo,
            anchor="nw",
        )
        anchor = self._new_anchors.get(self._frame_idx)
        if anchor is not None:
            self._draw_anchor(anchor)

        segment = self.request.segments[self._segment_index]
        before, after = self._reliable_neighbors(self._frame_idx)
        state_text = observation.state.value if observation else "unknown"
        confidence = observation.confidence if observation else 0.0
        self._status.configure(
            text=(
                f"Issue {self._segment_index + 1}/{len(self.request.segments)}  "
                f"Frames {segment.start_frame}-{segment.end_frame}  "
                f"Current frame {self._frame_idx}  "
                f"State {state_text}  Confidence {confidence:.2f}  "
                f"Reliable {before if before is not None else '-'} / "
                f"{after if after is not None else '-'}  "
                "Drag a box around the aircraft."
            )
        )

    def _reliable_neighbors(
        self,
        frame_idx: int,
    ) -> tuple[int | None, int | None]:
        before = None
        after = None
        for idx in range(frame_idx - 1, -1, -1):
            if self._is_reliable(idx):
                before = idx
                break
        for idx in range(frame_idx + 1, len(self.request.observations)):
            if self._is_reliable(idx):
                after = idx
                break
        return before, after

    def _is_reliable(self, frame_idx: int) -> bool:
        observation = self.request.observations[frame_idx]
        return (
            observation.center is not None
            and observation.bbox is not None
            and observation.confidence >= 0.65
            and observation.state
            in (TrackingState.TRACKING, TrackingState.MANUAL_ANCHOR)
        )

    def _on_press(self, event) -> None:
        point = self._clamp_canvas_point(event.x, event.y)
        if point is None:
            return
        self._drag_start = point
        if self._selection_id is not None:
            self._canvas.delete(self._selection_id)
        self._selection_id = self._canvas.create_rectangle(
            point[0],
            point[1],
            point[0],
            point[1],
            outline="#ffb224",
            width=2,
        )

    def _on_drag(self, event) -> None:
        if self._drag_start is None or self._selection_id is None:
            return
        point = self._clamp_canvas_point(event.x, event.y)
        if point is None:
            return
        self._canvas.coords(
            self._selection_id,
            self._drag_start[0],
            self._drag_start[1],
            point[0],
            point[1],
        )

    def _on_release(self, event) -> None:
        if self._drag_start is None:
            return
        point = self._clamp_canvas_point(event.x, event.y)
        if point is None:
            self._drag_start = None
            return
        x1, x2 = sorted((self._drag_start[0], point[0]))
        y1, y2 = sorted((self._drag_start[1], point[1]))
        self._drag_start = None
        if x2 - x1 < 8 or y2 - y1 < 8:
            return

        ox, oy = self._image_origin
        scale = self._display_scale
        bbox = (
            int(round((x1 - ox) / scale)),
            int(round((y1 - oy) / scale)),
            int(round((x2 - x1) / scale)),
            int(round((y2 - y1) / scale)),
        )
        previous = self._new_anchors.get(self._frame_idx)
        self._undo_stack.append((self._frame_idx, previous))
        anchor = ManualAnchor(frame_idx=self._frame_idx, bbox=bbox)
        self._new_anchors[self._frame_idx] = anchor
        self._load_frame()

    def _clamp_canvas_point(self, x: int, y: int):
        ox, oy = self._image_origin
        width, height = self._image_size
        if x < ox or y < oy or x > ox + width or y > oy + height:
            return None
        return (
            max(ox, min(x, ox + width)),
            max(oy, min(y, oy + height)),
        )

    def _draw_anchor(self, anchor: ManualAnchor) -> None:
        x, y, w, h = anchor.bbox
        ox, oy = self._image_origin
        scale = self._display_scale
        self._canvas.create_rectangle(
            ox + x * scale,
            oy + y * scale,
            ox + (x + w) * scale,
            oy + (y + h) * scale,
            outline="#30d980",
            width=3,
        )

    def _undo(self) -> None:
        if not self._undo_stack:
            return
        frame_idx, previous = self._undo_stack.pop()
        if previous is None:
            self._new_anchors.pop(frame_idx, None)
        else:
            self._new_anchors[frame_idx] = previous
        self._frame_idx = frame_idx
        self._load_frame()

    def _finish(self) -> None:
        self.result = sorted(self._new_anchors.values())
        self._capture.release()
        self.grab_release()
        self.destroy()


def show_review_dialog(parent, request: ReviewRequest) -> list[ManualAnchor]:
    dialog = TrackReviewDialog(parent, request)
    parent.wait_window(dialog)
    return dialog.result
