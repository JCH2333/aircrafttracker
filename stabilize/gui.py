"""Aircraft Tracker GUI — dark industrial minimalist design.

Built with CustomTkinter for modern widgets.
Supports: multi-file queue, add/remove/reorder during processing,
CPU/GPU toggle, real-time dual progress bars.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import customtkinter as ctk
from tkinter import filedialog, messagebox

from stabilize.config import StabilizerConfig
from stabilize.pipeline import StabilizationPipeline

logger = logging.getLogger(__name__)

# ── Theme ────────────────────────────────────────────────────
BG_DARK = "#141416"
BG_PANEL = "#1c1c1f"
BG_INPUT = "#26262b"
FG_PRIMARY = "#e4e4e7"
FG_SECONDARY = "#8b8b90"
ACCENT = "#5b8def"
ACCENT_HOVER = "#7aa3f4"
DANGER = "#e5484d"
SUCCESS = "#30a46c"
WARN = "#e5a343"
BORDER = "#2e2e33"


class ProcessingQueue:
    """Thread-safe queue of video files to process."""

    def __init__(self):
        self._lock = threading.Lock()
        self._items: list[dict] = []  # {path, name, frames, size_mb, status: pending|running|done|error}

    def add(self, paths: list[Path]) -> int:
        """Add files to queue. Returns number added."""
        with self._lock:
            existing = {str(it["path"]) for it in self._items}
            added = 0
            for p in paths:
                if str(p) not in existing:
                    self._items.append({
                        "path": p,
                        "name": p.name,
                        "frames": "?",
                        "size_mb": f"{p.stat().st_size / 1e6:.0f}",
                        "status": "pending",
                    })
                    added += 1
            return added

    def remove(self, index: int) -> bool:
        """Remove item at index. Returns True if removed."""
        with self._lock:
            if 0 <= index < len(self._items):
                if self._items[index]["status"] != "running":
                    self._items.pop(index)
                    return True
            return False

    def move_up(self, index: int) -> bool:
        with self._lock:
            if index > 0 and self._items[index]["status"] == "pending":
                self._items[index], self._items[index - 1] = self._items[index - 1], self._items[index]
                return True
            return False

    def move_down(self, index: int) -> bool:
        with self._lock:
            if index < len(self._items) - 1 and self._items[index]["status"] == "pending":
                self._items[index], self._items[index + 1] = self._items[index + 1], self._items[index]
                return True
            return False

    def get_all(self) -> list[dict]:
        with self._lock:
            return [dict(it) for it in self._items]

    def set_status(self, index: int, status: str):
        with self._lock:
            if 0 <= index < len(self._items):
                self._items[index]["status"] = status

    def get_next_pending(self) -> tuple[int, dict] | None:
        with self._lock:
            for i, item in enumerate(self._items):
                if item["status"] == "pending":
                    return i, item
            return None

    def __len__(self):
        with self._lock:
            return len(self._items)


class AircraftTrackerApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        # Theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("Aircraft Tracker")
        self.geometry("820x600")
        self.minsize(700, 450)
        self.configure(fg_color=BG_DARK)

        # State
        self.queue = ProcessingQueue()
        self._processing = False
        self._cancel = False
        self._progress_queue = queue.Queue()
        self._output_dir = Path("处理结果")
        self._selected_index: int | None = None

        # Build UI
        self._build_ui()

        # Poll progress
        self._poll_progress()

    # ── Build ─────────────────────────────────────────────────

    def _build_ui(self):
        """Construct the two-panel layout."""
        self.grid_columnconfigure(0, weight=0)  # left panel
        self.grid_columnconfigure(1, weight=1)  # right panel
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)  # status bar

        # ── Left panel: Queue ──
        left = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=10, border_width=1, border_color=BORDER)
        left.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=0)
        left.grid_rowconfigure(1, weight=1)
        left.grid_rowconfigure(2, weight=0)

        ctk.CTkLabel(left, text="QUEUE", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=FG_SECONDARY).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 6))

        # Scrollable queue list
        list_frame = ctk.CTkScrollableFrame(left, fg_color="transparent", scrollbar_button_color=BORDER,
                                            scrollbar_button_hover_color=ACCENT)
        list_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        list_frame.grid_columnconfigure(0, weight=1)
        self._queue_cells: list[dict] = []  # {frame, status_dot, name_label, info_label}
        self._list_frame = list_frame

        # Buttons row
        btn_row = ctk.CTkFrame(left, fg_color="transparent")
        btn_row.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        btn_row.grid_columnconfigure((0, 1, 2, 3), weight=0)

        ctk.CTkButton(btn_row, text="＋ Add Files", width=90, height=30,
                       fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#fff",
                       font=ctk.CTkFont(size=12), corner_radius=6,
                       command=self._add_files).grid(row=0, column=0, padx=(0, 4))
        ctk.CTkButton(btn_row, text="↑", width=32, height=30,
                       fg_color=BG_INPUT, hover_color=BORDER, text_color=FG_PRIMARY,
                       font=ctk.CTkFont(size=14), corner_radius=6,
                       command=self._move_up).grid(row=0, column=1, padx=2)
        ctk.CTkButton(btn_row, text="↓", width=32, height=30,
                       fg_color=BG_INPUT, hover_color=BORDER, text_color=FG_PRIMARY,
                       font=ctk.CTkFont(size=14), corner_radius=6,
                       command=self._move_down).grid(row=0, column=2, padx=2)
        ctk.CTkButton(btn_row, text="✕", width=32, height=30,
                       fg_color=BG_INPUT, hover_color=DANGER, text_color=FG_SECONDARY,
                       font=ctk.CTkFont(size=14), corner_radius=6,
                       command=self._remove_selected).grid(row=0, column=3, padx=(4, 0))

        # ── Right panel: Controls + Progress ──
        right = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=10, border_width=1, border_color=BORDER)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=0)
        right.grid_rowconfigure(1, weight=0)
        right.grid_rowconfigure(2, weight=1)
        right.grid_rowconfigure(3, weight=0)

        # ── Settings section ──
        settings = ctk.CTkFrame(right, fg_color="transparent")
        settings.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 4))
        settings.grid_columnconfigure(0, weight=0)
        settings.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(settings, text="MODE", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=FG_SECONDARY).grid(row=0, column=0, sticky="w", padx=(0, 12))

        self._mode_var = ctk.StringVar(value="GPU")
        mode_frame = ctk.CTkFrame(settings, fg_color="transparent")
        mode_frame.grid(row=0, column=1, sticky="w")
        self._nvenc_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(mode_frame, text="NVENC Encode", variable=self._nvenc_var,
                        fg_color=ACCENT, hover_color=ACCENT_HOVER,
                        font=ctk.CTkFont(size=12), text_color=FG_PRIMARY).grid(
                            row=1, column=0, sticky="w", pady=(6, 0))

        for i, (label, val) in enumerate([("GPU", "GPU"), ("CPU", "CPU")]):
            ctk.CTkRadioButton(mode_frame, text=label, variable=self._mode_var, value=val,
                               fg_color=ACCENT, hover_color=ACCENT_HOVER,
                               font=ctk.CTkFont(size=12), text_color=FG_PRIMARY,
                               command=self._on_mode_change).grid(row=0, column=i, padx=(0, 16))

        # Output dir
        out_frame = ctk.CTkFrame(right, fg_color="transparent")
        out_frame.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        out_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(out_frame, text="OUTPUT", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=FG_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        self._out_var = ctk.StringVar(value="处理结果")
        ctk.CTkEntry(out_frame, textvariable=self._out_var, height=28,
                     fg_color=BG_INPUT, border_color=BORDER, text_color=FG_PRIMARY,
                     corner_radius=6).grid(row=0, column=1, sticky="ew", padx=(0, 6))
        ctk.CTkButton(out_frame, text="Browse", width=60, height=28,
                       fg_color=BG_INPUT, hover_color=BORDER, text_color=FG_PRIMARY,
                       font=ctk.CTkFont(size=11), corner_radius=6,
                       command=self._browse_output).grid(row=0, column=2)

        # ── Progress section ──
        prog_frame = ctk.CTkFrame(right, fg_color="transparent")
        prog_frame.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 4))
        prog_frame.grid_columnconfigure(0, weight=1)
        prog_frame.grid_rowconfigure((0, 1, 2, 3), weight=0)

        # Current file
        self._current_label = ctk.CTkLabel(prog_frame, text="Ready", font=ctk.CTkFont(size=13),
                                           text_color=FG_SECONDARY, anchor="w")
        self._current_label.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        # Pass 1
        ctk.CTkLabel(prog_frame, text="Pass 1 · Tracking", font=ctk.CTkFont(size=11),
                     text_color=FG_SECONDARY, anchor="w").grid(row=1, column=0, sticky="ew")
        self._p1_bar = ctk.CTkProgressBar(prog_frame, height=6, fg_color=BG_INPUT,
                                          progress_color=ACCENT, corner_radius=3)
        self._p1_bar.grid(row=2, column=0, sticky="ew", pady=(2, 8))
        self._p1_bar.set(0)
        self._p1_label = ctk.CTkLabel(prog_frame, text="—", font=ctk.CTkFont(size=10),
                                      text_color=FG_SECONDARY, anchor="e")
        self._p1_label.grid(row=2, column=0, sticky="e", padx=(0, 4))

        # Pass 2
        ctk.CTkLabel(prog_frame, text="Pass 2 · Rendering", font=ctk.CTkFont(size=11),
                     text_color=FG_SECONDARY, anchor="w").grid(row=3, column=0, sticky="ew", pady=(0, 2))
        self._p2_bar = ctk.CTkProgressBar(prog_frame, height=6, fg_color=BG_INPUT,
                                          progress_color=SUCCESS, corner_radius=3)
        self._p2_bar.grid(row=4, column=0, sticky="ew", pady=(2, 8))
        self._p2_bar.set(0)
        self._p2_label = ctk.CTkLabel(prog_frame, text="—", font=ctk.CTkFont(size=10),
                                      text_color=FG_SECONDARY, anchor="e")
        self._p2_label.grid(row=4, column=0, sticky="e", padx=(0, 4))

        # Overall
        self._overall_label = ctk.CTkLabel(prog_frame, text="", font=ctk.CTkFont(size=11),
                                           text_color=FG_SECONDARY, anchor="e")
        self._overall_label.grid(row=5, column=0, sticky="e")

        # ── Action buttons ──
        btn_frame = ctk.CTkFrame(right, fg_color="transparent")
        btn_frame.grid(row=3, column=0, sticky="ew", padx=14, pady=(4, 14))

        self._start_btn = ctk.CTkButton(btn_frame, text="▶ Start", height=34, width=110,
                                         fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#fff",
                                         font=ctk.CTkFont(size=13, weight="bold"), corner_radius=8,
                                         command=self._start)
        self._start_btn.pack(side="left", padx=(0, 8))

        self._stop_btn = ctk.CTkButton(btn_frame, text="■ Stop", height=34, width=80,
                                        fg_color=BG_INPUT, hover_color=DANGER, text_color=FG_PRIMARY,
                                        font=ctk.CTkFont(size=13), corner_radius=8,
                                        command=self._stop, state="disabled")
        self._stop_btn.pack(side="left")

        # ── Status bar ──
        status = ctk.CTkFrame(self, fg_color=BG_PANEL, height=28, corner_radius=0)
        status.grid(row=1, column=0, columnspan=2, sticky="ew")
        self._status_label = ctk.CTkLabel(status, text="● Ready", font=ctk.CTkFont(size=10),
                                          text_color=SUCCESS, anchor="w")
        self._status_label.pack(side="left", padx=(14, 0))

        self._status_count = ctk.CTkLabel(status, text="", font=ctk.CTkFont(size=10),
                                          text_color=FG_SECONDARY, anchor="e")
        self._status_count.pack(side="right", padx=(0, 14))

        # Initial refresh
        self._refresh_queue_display()

    # ── Queue display ──────────────────────────────────────────

    def _refresh_queue_display(self):
        """Rebuild the queue list from current queue state."""
        for cell in self._queue_cells:
            for w in cell.values():
                w.destroy()
        self._queue_cells.clear()

        items = self.queue.get_all()
        for i, item in enumerate(items):
            self._add_queue_row(i, item)
        self._update_status_count()

    def _add_queue_row(self, index: int, item: dict):
        """Add a single row to the queue display."""
        status_color = {"pending": FG_SECONDARY, "running": ACCENT,
                        "done": SUCCESS, "error": DANGER}.get(item["status"], FG_SECONDARY)

        row = ctk.CTkFrame(self._list_frame, fg_color=BG_INPUT if item["status"] == "running" else "transparent",
                           corner_radius=6, height=36)
        row.grid(row=index, column=0, sticky="ew", pady=2, padx=2)
        row.grid_columnconfigure(1, weight=1)

        # Status dot
        dot = ctk.CTkLabel(row, text="●", font=ctk.CTkFont(size=10), text_color=status_color, width=20)
        dot.grid(row=0, column=0, padx=(8, 0))

        # Name
        name = ctk.CTkLabel(row, text=item["name"], font=ctk.CTkFont(size=12),
                            text_color=FG_PRIMARY if item["status"] != "done" else FG_SECONDARY,
                            anchor="w")
        name.grid(row=0, column=1, sticky="w", padx=(4, 8))

        # Info (frames + size)
        info = ctk.CTkLabel(row, text=f"{item['frames']}f · {item['size_mb']}MB",
                            font=ctk.CTkFont(size=10), text_color=FG_SECONDARY, anchor="e")
        info.grid(row=0, column=2, sticky="e", padx=(4, 8))

        # Make row clickable
        idx = index
        for w in (row, dot, name, info):
            w.bind("<Button-1>", lambda e, i=idx: self._select_row(i))

        self._queue_cells.append({"frame": row, "dot": dot, "name": name, "info": info})

    def _update_status_count(self):
        items = self.queue.get_all()
        done = sum(1 for it in items if it["status"] == "done")
        total = len(items)
        if total > 0:
            self._status_count.configure(text=f"{done}/{total} completed")
        else:
            self._status_count.configure(text="")

    # ── Actions ────────────────────────────────────────────────

    def _add_files(self):
        paths_str = filedialog.askopenfilenames(
            title="Select video files",
            filetypes=[("Video", "*.MOV *.MP4 *.mov *.mp4"), ("All", "*.*")],
        )
        if paths_str:
            paths = [Path(p) for p in paths_str]
            added = self.queue.add(paths)
            if added > 0:
                self._refresh_queue_display()

    def _select_row(self, index: int):
        """Select a queue row by index."""
        self._selected_index = index
        for i, cell in enumerate(self._queue_cells):
            if i == index:
                cell["frame"].configure(fg_color=BG_INPUT)
            else:
                cell["frame"].configure(fg_color="transparent")

    def _move_up(self):
        if self._selected_index is not None:
            if self.queue.move_up(self._selected_index):
                self._selected_index -= 1
            self._refresh_queue_display()

    def _move_down(self):
        if self._selected_index is not None:
            if self.queue.move_down(self._selected_index):
                self._selected_index += 1
            self._refresh_queue_display()

    def _remove_selected(self):
        if self._selected_index is not None:
            items = self.queue.get_all()
            if 0 <= self._selected_index < len(items):
                if items[self._selected_index]["status"] == "pending":
                    self.queue.remove(self._selected_index)
                    self._selected_index = None
            self._refresh_queue_display()

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self._out_var.set(path)

    def _on_mode_change(self):
        mode = self._mode_var.get()
        self._status_label.configure(text=f"● {mode} mode selected")

    # ── Processing ─────────────────────────────────────────────

    def _start(self):
        """Start processing the queue."""
        if len(self.queue) == 0:
            messagebox.showinfo("Info", "Add files to the queue first.")
            return

        self._output_dir = Path(self._out_var.get())
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Update pending to total count
        items = self.queue.get_all()
        pending = sum(1 for it in items if it["status"] == "pending")
        if pending == 0:
            messagebox.showinfo("Info", "All files already processed.")
            return

        self._processing = True
        self._cancel = False
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._mode_var.set(self._mode_var.get())  # lock mode

        thread = threading.Thread(target=self._process_loop, daemon=True)
        thread.start()

    def _stop(self):
        self._cancel = True
        self._status_label.configure(text="● Stopping...", text_color=WARN)

    def _process_loop(self):
        """Background thread: process queue items sequentially."""
        while not self._cancel:
            next_item = self.queue.get_next_pending()
            if next_item is None:
                break

            index, item = next_item
            self.queue.set_status(index, "running")
            self.after(0, self._refresh_queue_display)
            self.after(0, lambda: self._status_label.configure(
                text=f"● Processing: {item['name']}", text_color=ACCENT))

            path = item["path"]
            output_path = self._output_dir / f"{path.stem}_stabilized.MOV"

            try:
                device = "cuda" if self._mode_var.get() == "GPU" else "cpu"
                codec = "h264_nvenc" if self._nvenc_var.get() else "libx264"
                config = StabilizerConfig(
                    input_path=path,
                    output_path=output_path,
                    output_dir=self._output_dir,
                    device=device,
                    video_codec=codec,
                )
                pipeline = StabilizationPipeline(config)
                pipeline.set_progress_callback(self._on_pipeline_progress)
                pipeline.run()
                self.queue.set_status(index, "done")
            except Exception as e:
                logger.error("Failed: %s — %s", path.name, e)
                self.queue.set_status(index, "error")

            self.after(0, self._refresh_queue_display)

        self._processing = False
        self.after(0, lambda: self._start_btn.configure(state="normal"))
        self.after(0, lambda: self._stop_btn.configure(state="disabled"))
        self.after(0, lambda: self._status_label.configure(
            text="● Complete" if not self._cancel else "● Stopped",
            text_color=SUCCESS if not self._cancel else WARN))
        self.after(0, self._update_status_count)

    def _on_pipeline_progress(self, phase: int, current: int, total: int):
        """Callback from pipeline — push to UI queue."""
        self._progress_queue.put({"phase": phase, "current": current, "total": total})

    # ── Progress polling ───────────────────────────────────────

    def _poll_progress(self):
        try:
            while True:
                msg = self._progress_queue.get_nowait()
                phase = msg["phase"]
                current = msg["current"]
                total = msg["total"]
                frac = current / total if total > 0 else 0

                if phase == 1:
                    self._p1_bar.set(frac)
                    self._p1_label.configure(text=f"{current}/{total}" if total else "—")
                elif phase == 2:
                    self._p2_bar.set(frac)
                    self._p2_label.configure(text=f"{current}/{total}" if total else "—")
        except queue.Empty:
            pass
        self.after(200, self._poll_progress)


def launch_gui():
    app = AircraftTrackerApp()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
