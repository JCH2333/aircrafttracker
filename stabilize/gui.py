"""Tkinter GUI for Aircraft Tracker.

Provides single-file and batch processing modes with real-time
progress bars. Uses the same StabilizationPipeline as the CLI.
"""

import logging
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from stabilize.config import StabilizerConfig
from stabilize.pipeline import StabilizationPipeline

logger = logging.getLogger(__name__)


class AircraftTrackerGUI:
    """Main GUI window for the Aircraft Tracker application."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Aircraft Tracker")
        self.root.geometry("720x560")
        self.root.minsize(600, 400)
        self.root.resizable(True, True)

        # State
        self._input_paths: list[Path] = []
        self._output_dir = Path("处理结果")
        self._batch_mode = tk.BooleanVar(value=False)
        self._processing = False
        self._cancel = False
        self._progress_queue = queue.Queue()

        # Build UI
        self._build_ui()

        # Start progress poller
        self._poll_progress()

    # ── UI construction ─────────────────────────────────────────

    def _build_ui(self):
        """Build the complete UI layout."""
        # Main frame with padding
        main = ttk.Frame(self.root, padding="10")
        main.pack(fill=tk.BOTH, expand=True)

        # Row 0: Input selection
        ttk.Label(main, text="输入文件:").grid(
            row=0, column=0, sticky=tk.W, pady=(0, 2)
        )
        self._input_var = tk.StringVar()
        ttk.Entry(main, textvariable=self._input_var, width=60).grid(
            row=0, column=1, sticky=tk.EW, padx=(5, 2), pady=(0, 2)
        )
        ttk.Button(main, text="浏览...", command=self._browse_input).grid(
            row=0, column=2, pady=(0, 2)
        )

        # Row 1: Output directory
        ttk.Label(main, text="输出目录:").grid(
            row=1, column=0, sticky=tk.W, pady=(0, 2)
        )
        self._output_var = tk.StringVar(value=str(self._output_dir))
        ttk.Entry(main, textvariable=self._output_var, width=60).grid(
            row=1, column=1, sticky=tk.EW, padx=(5, 2), pady=(0, 2)
        )
        ttk.Button(main, text="浏览...", command=self._browse_output).grid(
            row=1, column=2, pady=(0, 2)
        )

        # Row 2: Batch mode toggle
        ttk.Checkbutton(
            main, text="批量模式（处理文件夹内所有视频）",
            variable=self._batch_mode,
            command=self._toggle_batch,
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(5, 2))

        # Row 3: File list (batch mode)
        list_frame = ttk.LabelFrame(main, text="文件列表", padding="5")
        list_frame.grid(row=3, column=0, columnspan=3, sticky=tk.NSEW, pady=(5, 5))

        columns = ("include", "filename", "frames", "size")
        self._file_tree = ttk.Treeview(
            list_frame, columns=columns, show="headings", height=8,
            selectmode=tk.NONE,
        )
        self._file_tree.heading("include", text="")
        self._file_tree.heading("filename", text="文件名")
        self._file_tree.heading("frames", text="帧数")
        self._file_tree.heading("size", text="大小")
        self._file_tree.column("include", width=30, anchor=tk.CENTER)
        self._file_tree.column("filename", width=300)
        self._file_tree.column("frames", width=80, anchor=tk.CENTER)
        self._file_tree.column("size", width=100, anchor=tk.E)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._file_tree.yview)
        self._file_tree.configure(yscrollcommand=scrollbar.set)
        self._file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._file_tree.bind("<ButtonRelease-1>", self._on_tree_click)

        # Row 4-5: Progress bars
        progress_frame = ttk.LabelFrame(main, text="进度", padding="5")
        progress_frame.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=(0, 5))

        # Pass 1 bar
        ttk.Label(progress_frame, text="Pass 1 (追踪):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self._pass1_bar = ttk.Progressbar(progress_frame, length=400, mode="determinate")
        self._pass1_bar.grid(row=0, column=1, sticky=tk.EW, padx=(0, 5))
        self._pass1_label = ttk.Label(progress_frame, text="--", width=10, anchor=tk.E)
        self._pass1_label.grid(row=0, column=2)

        # Pass 2 bar
        ttk.Label(progress_frame, text="Pass 2 (渲染):").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self._pass2_bar = ttk.Progressbar(progress_frame, length=400, mode="determinate")
        self._pass2_bar.grid(row=1, column=1, sticky=tk.EW, padx=(0, 5))
        self._pass2_label = ttk.Label(progress_frame, text="--", width=10, anchor=tk.E)
        self._pass2_label.grid(row=1, column=2)

        # Overall bar
        ttk.Label(progress_frame, text="总体:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5))
        self._overall_bar = ttk.Progressbar(progress_frame, length=400, mode="determinate")
        self._overall_bar.grid(row=2, column=1, sticky=tk.EW, padx=(0, 5))
        self._overall_label = ttk.Label(progress_frame, text="--", width=10, anchor=tk.E)
        self._overall_label.grid(row=2, column=2)

        # Current file
        self._current_label = ttk.Label(progress_frame, text="就绪", foreground="gray")
        self._current_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))

        progress_frame.columnconfigure(1, weight=1)

        # Row 6: Action buttons
        button_frame = ttk.Frame(main)
        button_frame.grid(row=5, column=0, columnspan=3, pady=(5, 0))

        self._start_btn = ttk.Button(
            button_frame, text="开始处理", command=self._start_processing
        )
        self._start_btn.pack(side=tk.LEFT, padx=(0, 10))

        self._stop_btn = ttk.Button(
            button_frame, text="停止", command=self._stop_processing, state=tk.DISABLED
        )
        self._stop_btn.pack(side=tk.LEFT)

        # Configure grid weights
        main.columnconfigure(1, weight=1)
        main.rowconfigure(3, weight=1)

    # ── Event handlers ──────────────────────────────────────────

    def _browse_input(self):
        if self._batch_mode.get():
            path = filedialog.askdirectory(title="选择输入文件夹")
            if path:
                self._input_var.set(path)
                self._scan_folder(path)
        else:
            path = filedialog.askopenfilename(
                title="选择视频文件",
                filetypes=[("视频文件", "*.MOV *.MP4 *.mov *.mp4"), ("所有文件", "*.*")],
            )
            if path:
                self._input_var.set(path)
                self._file_tree.delete(*self._file_tree.get_children())

    def _browse_output(self):
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            self._output_var.set(path)

    def _toggle_batch(self):
        self._input_var.set("")
        self._file_tree.delete(*self._file_tree.get_children())

    def _scan_folder(self, folder):
        """Scan folder for video files and populate the tree."""
        self._file_tree.delete(*self._file_tree.get_children())
        folder = Path(folder)
        if not folder.is_dir():
            return

        video_exts = {".mov", ".mp4", ".MOV", ".MP4"}
        files = sorted(
            [f for f in folder.iterdir() if f.suffix in video_exts and f.is_file()]
        )
        self._input_paths = files

        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            self._file_tree.insert(
                "",
                tk.END,
                values=("✓", f.name, "?", f"{size_mb:.0f} MB"),
                tags=("checked",),
            )

    def _on_tree_click(self, event):
        """Toggle checkmark on tree item click."""
        item = self._file_tree.identify_row(event.y)
        if not item:
            return
        col = self._file_tree.identify_column(event.x)
        if col != "#1":
            return  # only toggle on the checkmark column

        values = self._file_tree.item(item, "values")
        if values[0] == "✓":
            self._file_tree.item(item, values=("✗", *values[1:]), tags=("unchecked",))
        else:
            self._file_tree.item(item, values=("✓", *values[1:]), tags=("checked",))

        self._file_tree.tag_configure("checked", foreground="black")
        self._file_tree.tag_configure("unchecked", foreground="gray")

    # ── Processing ──────────────────────────────────────────────

    def _start_processing(self):
        """Start batch or single-file processing in a background thread."""
        if self._batch_mode.get():
            items = self._file_tree.get_children()
            paths = []
            for item in items:
                values = self._file_tree.item(item, "values")
                if values[0] == "✓":
                    idx = self._file_tree.index(item)
                    paths.append(self._input_paths[idx])
            if not paths:
                messagebox.showwarning("提示", "没有选中任何文件")
                return
        else:
            path = Path(self._input_var.get())
            if not path.exists():
                messagebox.showerror("错误", f"文件不存在: {path}")
                return
            paths = [path]

        self._output_dir = Path(self._output_var.get())
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._processing = True
        self._cancel = False
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)

        thread = threading.Thread(
            target=self._process_files, args=(paths,), daemon=True
        )
        thread.start()

    def _stop_processing(self):
        self._cancel = True
        self._current_label.config(text="正在停止...", foreground="orange")

    def _process_files(self, paths: list[Path]):
        """Background thread: process all files sequentially."""
        total = len(paths)
        for i, path in enumerate(paths):
            if self._cancel:
                break

            self._progress_queue.put({
                "current": f"处理中: {path.name}",
                "overall_val": i,
                "overall_max": total,
                "overall_text": f"{i}/{total}",
                "pass1_val": 0,
                "pass1_text": "等待中",
                "pass2_val": 0,
                "pass2_text": "等待中",
            })

            output_path = self._output_dir / f"{path.stem}_stabilized.MOV"

            try:
                config = StabilizerConfig(
                    input_path=path,
                    output_path=output_path,
                    output_dir=self._output_dir,
                )
                pipeline = StabilizationPipeline(config)

                # Run with progress callbacks
                pipeline.set_progress_callback(
                    lambda phase, val, total_f: self._progress_queue.put({
                        "current": f"处理中: {path.name}",
                        "overall_val": i + (val / total_f if total_f else 0),
                        "overall_max": total,
                        "overall_text": f"{i}/{total}",
                        "pass1_val" if phase == 1 else "pass2_val": val,
                        "pass1_max" if phase == 1 else "pass2_max": total_f,
                        "pass1_text" if phase == 1 else "pass2_text":
                            f"{val}/{total_f}",
                    })
                )
                pipeline.run()

            except Exception as e:
                logger.error("Failed: %s — %s", path.name, e)
                self._progress_queue.put({
                    "current": f"失败: {path.name} — {e}",
                    "overall_text": f"{i+1}/{total} (错误)",
                })

        # Done
        self._progress_queue.put({
            "current": "完成!" if not self._cancel else "已停止",
            "done": True,
            "overall_val": total,
            "overall_max": total,
            "overall_text": f"{total}/{total}",
        })

    def _poll_progress(self):
        """Poll the progress queue and update UI (runs on main thread)."""
        try:
            while True:
                msg = self._progress_queue.get_nowait()
                if "current" in msg:
                    self._current_label.config(text=msg["current"], foreground="blue")
                if "overall_val" in msg:
                    self._overall_bar["maximum"] = msg.get("overall_max", 100)
                    self._overall_bar["value"] = msg["overall_val"]
                    self._overall_label.config(text=msg.get("overall_text", "--"))
                if "pass1_val" in msg:
                    self._pass1_bar["maximum"] = msg.get("pass1_max", 100)
                    self._pass1_bar["value"] = msg["pass1_val"]
                    self._pass1_label.config(text=msg.get("pass1_text", "--"))
                if "pass2_val" in msg:
                    self._pass2_bar["maximum"] = msg.get("pass2_max", 100)
                    self._pass2_bar["value"] = msg["pass2_val"]
                    self._pass2_label.config(text=msg.get("pass2_text", "--"))
                if msg.get("done"):
                    self._done()
        except queue.Empty:
            pass
        finally:
            self.root.after(200, self._poll_progress)

    def _done(self):
        self._processing = False
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._current_label.config(text="就绪", foreground="gray")


def launch_gui():
    """Entry point for GUI mode."""
    root = tk.Tk()
    app = AircraftTrackerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
