#!/usr/bin/env python3
"""Control a Raspberry Pi HQ Camera for astronomical imaging and recording.

The application provides live preview, still-image sequences, and planetary
recording in SER or AVI format. Preview stretching and crosshairs affect only
the on-screen view; saved image data are written from the captured camera
frames.

AI-assistance disclosure: The main scientific functions, supporting utility
functions, and debugging code were developed with AI assistance. The operator
is responsible for validating camera settings, metadata, and scientific output
for the intended observing workflow.
"""

import os
import sys
import time
import threading
import queue
import struct
from dataclasses import dataclass
from datetime import datetime, timezone

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk
from PIL.PngImagePlugin import PngInfo

from picamera2 import Picamera2

try:
    from astropy.io import fits
    HAVE_ASTROPY = True
except Exception:
    HAVE_ASTROPY = False

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False


# =========================
# SER writer for RGB8 frames with optional timestamp trailer
# =========================
SER_FILE_ID = b"LUCAM-RECORDER"  # 14 bytes


def _safe_ascii_bytes(s: str, n: int) -> bytes:
    s = (s or "").strip()
    b = s.encode("ascii", errors="replace")[:n]
    return b + b"\x00" * (n - len(b))


def _ser_ticks_from_naive(dt_naive: datetime) -> int:
    # Convert to 100-nanosecond ticks measured from 0001-01-01 00:00:00.
    epoch = datetime(1, 1, 1)
    delta = dt_naive - epoch
    return int(delta.total_seconds() * 10_000_000)


@dataclass
class SERMeta:
    observer: str = ""
    instrument: str = "Picamera2"
    telescope: str = ""


class SERWriter:
    """Write packed 8-bit RGB frames in the SER format.

    The fixed header is 178 bytes:
      FileID(14),
      LuID(4), ColorID(4), LittleEndian(4), Width(4), Height(4),
      PixelDepthPerPlane(4), FrameCount(4),
      Observer(40), Instrument(40), Telescope(40),
      DateTimeLocalTicks(8), DateTimeUTCTicks(8)

    Each frame contains packed RGB bytes. When enabled, the trailer stores one
    eight-byte UTC timestamp for every frame.

    The SER header has no OBJECT field, so recordings also receive a .hdr.txt
    sidecar containing the target name and other astronomy metadata.
    """

    COLOR_ID_RGB = 100

    def __init__(self, path: str, width: int, height: int, meta: SERMeta, write_trailer: bool = True):
        self.path = path
        self.w = int(width)
        self.h = int(height)
        self.meta = meta
        self.write_trailer = bool(write_trailer)

        self._f = None
        self._frame_count = 0
        self._utc_ticks = []
        self._t0_local_naive = datetime.now().replace(tzinfo=None)
        self._t0_utc_naive = datetime.utcnow().replace(tzinfo=None)

    def open(self):
        self._f = open(self.path, "xb")
        self._write_header(frame_count=0)
        return self

    def _write_header(self, frame_count: int):
        lu_id = 0
        little_endian = 1
        pix_depth = 8

        observer_b = _safe_ascii_bytes(self.meta.observer, 40)
        instrument_b = _safe_ascii_bytes(self.meta.instrument, 40)
        telescope_b = _safe_ascii_bytes(self.meta.telescope, 40)

        dt_local_ticks = _ser_ticks_from_naive(self._t0_local_naive)
        dt_utc_ticks = _ser_ticks_from_naive(self._t0_utc_naive)

        # Field sizes: 14 + (7 * 4) + (3 * 40) + (2 * 8) = 178 bytes.
        header = struct.pack(
            "<14s"      # FileID
            "I"         # LuID
            "I"         # ColorID
            "I"         # LittleEndian
            "I"         # Width
            "I"         # Height
            "I"         # PixelDepthPerPlane
            "I"         # FrameCount
            "40s"       # Observer
            "40s"       # Instrument
            "40s"       # Telescope
            "q"         # DateTimeLocal
            "q",        # DateTimeUTC
            SER_FILE_ID,
            lu_id,
            self.COLOR_ID_RGB,
            little_endian,
            self.w,
            self.h,
            pix_depth,
            int(frame_count),
            observer_b,
            instrument_b,
            telescope_b,
            int(dt_local_ticks),
            int(dt_utc_ticks),
        )

        if len(header) != 178:
            raise RuntimeError(f"SER header size mismatch: {len(header)} != 178")

        self._f.seek(0)
        self._f.write(header)

    def add_frame_rgb8(self, rgb8: np.ndarray):
        if self._f is None:
            raise RuntimeError("SERWriter not opened.")
        if rgb8.dtype != np.uint8 or rgb8.ndim != 3 or rgb8.shape[2] != 3:
            raise ValueError("SERWriter expects RGB uint8 (H,W,3).")
        if rgb8.shape[0] != self.h or rgb8.shape[1] != self.w:
            raise ValueError(f"Frame size mismatch: got {rgb8.shape[1]}x{rgb8.shape[0]}, expected {self.w}x{self.h}")

        self._f.write(rgb8.tobytes(order="C"))
        self._frame_count += 1

        if self.write_trailer:
            utc_naive = datetime.utcnow().replace(tzinfo=None)
            self._utc_ticks.append(_ser_ticks_from_naive(utc_naive))

    def close(self):
        if self._f is None:
            return

        # Append the optional per-frame UTC timestamp trailer.
        if self.write_trailer and self._utc_ticks:
            for t in self._utc_ticks:
                self._f.write(struct.pack("<q", int(t)))

        # Update the frame-count field after all frames have been written.
        # Its byte offset is 14 + (6 * 4) = 38.
        self._f.flush()
        self._f.seek(38)
        self._f.write(struct.pack("<I", int(self._frame_count)))

        self._f.flush()
        self._f.close()
        self._f = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        except Exception:
            pass
        return False


# =========================
# User-interface helper for a vertically scrollable frame
# =========================
class ScrollableFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.inner = ttk.Frame(self.canvas, padding=6)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.bind("<Enter>", lambda _e: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.canvas.bind("<Leave>", lambda _e: self.canvas.unbind_all("<MouseWheel>"))

    def _on_inner_configure(self, _e):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, e):
        self.canvas.itemconfigure(self._win, width=e.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


# =========================
# Main camera-control application
# =========================
class PiAstroCamApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Pi AstroCam (Picamera2) — Imaging + Planetary Recording (SER/AVI)")
        self.root.geometry("1280x800")

        # Open the first camera exposed by Picamera2.
        try:
            self.picam2 = Picamera2(0)
        except Exception as e:
            messagebox.showerror("Camera error", f"Could not open camera 0:\n{e}")
            self.root.destroy()
            return

        self.cam_lock = threading.Lock()
        self.started = False

        # Worker state and thread-safe queues.
        self.preview_running = False
        self.recording_running = False
        self.closing = False

        self.stop_preview_evt = threading.Event()
        self.stop_capture_evt = threading.Event()
        self.stop_record_evt = threading.Event()

        self.preview_thread = None
        self.capture_thread = None
        self.record_thread = None

        self.ui_q = queue.Queue()
        self.frame_q = queue.Queue(maxsize=2)

        # Preview display state.
        self.zoom_factor = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 6.0
        self.base_scale = None
        self.last_canvas_size = (None, None)
        self.last_scale_mode = None
        self.tk_image = None
        self.last_frame_rgb = None

        # Preview frame-rate tracking; this does not measure recording cadence.
        self.last_frame_time = None
        self.fps_smooth = 0.0

        # Astronomy-preview toolbar controls.
        self.stretch_var = tk.StringVar(value="Dim")  # Dimmest, Dim, Bright, or Brightest
        self.crosshair_var = tk.BooleanVar(value=True)
        self.scale_mode_var = tk.StringVar(value="fit")  # fit | 1:1
        self.zoom_label_var = tk.StringVar(value="Zoom: 1.00×")

        self._stretch_last_calc = 0.0
        self._stretch_mode_cache = None
        self._stretch_lut3_cache = None  # Three 256-entry channels for PIL.Image.point().
        self._preview_start_pending = False
        self.allow_preview_restart = True

        # General output settings.
        self.output_dir = tk.StringVar(value=os.path.expanduser("~/Pictures"))
        self.base_name = tk.StringVar(value="moz")

        # Astronomy session metadata, displayed with Object before Observer.
        self.object_var = tk.StringVar(value="")
        self.observer_var = tk.StringVar(value="")
        self.instrument_var = tk.StringVar(value=self._default_instrument_name())
        self.telescope_var = tk.StringVar(value="")

        # Exposure-unit multipliers and permitted user-interface ranges.
        self.exposure_modes = {"µs": 1, "ms": 1000, "s": 1_000_000}
        self.exposure_ranges = {"µs": (50, 1_000_000), "ms": (1, 5000), "s": (1, 200)}

        # Still-imaging settings.
        self.format_var = tk.StringVar(value="fits" if HAVE_ASTROPY else "png")
        self.frame_type_var = tk.StringVar(value="Light")
        self.bin_var = tk.IntVar(value=1)

        self.img_ae_var = tk.BooleanVar(value=False)
        self.img_exposure_mode_var = tk.StringVar(value="ms")
        self.img_exposure_val_var = tk.DoubleVar(value=10.0)
        self.img_gain_var = tk.DoubleVar(value=1.0)

        self.count_var = tk.IntVar(value=3)
        self.interval_var = tk.DoubleVar(value=1.0)
        self.countdown_var = tk.IntVar(value=3)

        # High-frame-rate planetary-recording settings.
        self.rec_format_var = tk.StringVar(value="ser")
        self.rec_resolution_var = tk.StringVar(value="")
        self.rec_write_trailer_var = tk.BooleanVar(value=True)

        self.rec_ae_var = tk.BooleanVar(value=False)
        self.rec_exposure_mode_var = tk.StringVar(value="ms")
        self.rec_exposure_val_var = tk.DoubleVar(value=5.0)
        self.rec_gain_var = tk.DoubleVar(value=1.0)

        self.rec_stop_mode_var = tk.StringVar(value="manual")  # manual, frames, or seconds
        self.rec_frames_var = tk.IntVar(value=2000)
        self.rec_seconds_var = tk.DoubleVar(value=30.0)

        # AVI playback rate controls file timing, not the camera capture cadence.
        self.avi_playback_fps_var = tk.DoubleVar(value=30.0)

        # Live recording status shown in the user interface.
        self.rec_progress_value = tk.DoubleVar(value=0.0)
        self.rec_progress_max = 1.0
        self.rec_status_var = tk.StringVar(value="Idle.")

        # General application status.
        self.status_var = tk.StringVar(value="Ready.")
        self.fps_var = tk.StringVar(value="FPS: 0.0")

        self._build_ui()

        try:
            self._configure_camera(profile="preview")
            with self.cam_lock:
                self._apply_controls_locked("img")
        except Exception as e:
            messagebox.showerror("Camera error", f"Failed to configure camera:\n{e}")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll_queues()

    # ---------- defaults ----------
    def _default_instrument_name(self) -> str:
        try:
            props = getattr(self.picam2, "camera_properties", {}) or {}
            model = props.get("Model") or props.get("SensorModel") or ""
            if model:
                return str(model)
        except Exception:
            pass
        return "Picamera2"

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)

        self.preview_btn = ttk.Button(top, text="Start Live", command=self.toggle_preview)
        self.preview_btn.pack(side=tk.LEFT)

        ttk.Button(top, text="Save current frame", command=self.save_single_frame).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Start sequence", command=self.start_sequence).pack(side=tk.LEFT, padx=6)

        self.rec_top_btn = ttk.Button(top, text="Start recording", command=self.start_recording)
        self.rec_top_btn.pack(side=tk.LEFT, padx=6)

        ttk.Button(top, text="Stop", command=self.stop_all).pack(side=tk.LEFT)

        ttk.Label(top, textvariable=self.fps_var).pack(side=tk.LEFT, padx=14)
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT)

        # Resizable control and preview panes.
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        self.controls_panel = ScrollableFrame(main)
        main.add(self.controls_panel, weight=0)

        self.preview_container = ttk.Frame(main, padding=6)
        main.add(self.preview_container, weight=1)

        c = self.controls_panel.inner

        # Output folder and base filename.
        ttk.Label(c, text="Save folder").pack(anchor="w")
        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 8))
        ttk.Entry(row, textvariable=self.output_dir).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse…", command=self.choose_folder).pack(side=tk.LEFT, padx=5)

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row, text="Base name:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.base_name, width=16).pack(side=tk.LEFT, padx=6)

        ttk.Separator(c).pack(fill=tk.X, pady=8)

        # Astronomy session metadata.
        ttk.Label(c, text="Session metadata (written to headers)").pack(anchor="w")

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(row, text="Object:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.object_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(row, text="Observer:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.observer_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(row, text="Instrument:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.instrument_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 8))
        ttk.Label(row, text="Telescope:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.telescope_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        # Imaging and recording tabs.
        self.nb = ttk.Notebook(c)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.tab_imaging = ttk.Frame(self.nb, padding=6)
        self.tab_recording = ttk.Frame(self.nb, padding=6)
        self.nb.add(self.tab_imaging, text="Imaging")
        self.nb.add(self.tab_recording, text="Recording")

        self._build_imaging_tab(self.tab_imaging)
        self._build_recording_tab(self.tab_recording)

        # Preview toolbar and image canvas.
        self.preview_container.rowconfigure(1, weight=1)
        self.preview_container.columnconfigure(0, weight=1)

        # Preview controls.
        self.preview_toolbar = ttk.Frame(self.preview_container, padding=(2, 2))
        self.preview_toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")

        ttk.Label(self.preview_toolbar, text="Stretch:").grid(row=0, column=0, padx=(2, 4))
        self.stretch_combo = ttk.Combobox(
            self.preview_toolbar,
            textvariable=self.stretch_var,
            values=["Dimmest", "Dim", "Bright", "Brightest"],
            state="readonly",
            width=10
        )
        self.stretch_combo.grid(row=0, column=1, padx=(0, 10))
        self.stretch_combo.bind("<<ComboboxSelected>>", lambda _e: self._render_frame())

        ttk.Checkbutton(
            self.preview_toolbar,
            text="Crosshair",
            variable=self.crosshair_var,
            command=self._render_frame
        ).grid(row=0, column=2, padx=(0, 12))

        ttk.Label(self.preview_toolbar, text="Scale:").grid(row=0, column=3, padx=(0, 4))
        ttk.Radiobutton(
            self.preview_toolbar, text="Fit", value="fit", variable=self.scale_mode_var,
            command=self._on_scale_mode_changed
        ).grid(row=0, column=4)
        ttk.Radiobutton(
            self.preview_toolbar, text="1:1", value="1:1", variable=self.scale_mode_var,
            command=self._on_scale_mode_changed
        ).grid(row=0, column=5, padx=(0, 12))

        ttk.Button(self.preview_toolbar, text="−", width=3, command=lambda: self._zoom(-1)).grid(row=0, column=6)
        ttk.Button(self.preview_toolbar, text="+", width=3, command=lambda: self._zoom(+1)).grid(row=0, column=7, padx=(4, 8))
        ttk.Button(self.preview_toolbar, text="Reset", command=self._reset_zoom).grid(row=0, column=8)

        ttk.Label(self.preview_toolbar, textvariable=self.zoom_label_var, foreground="gray").grid(
            row=0, column=9, padx=(10, 2), sticky="e"
        )
        self.preview_toolbar.grid_columnconfigure(9, weight=1)

        # Image canvas and scrollbars.
        self.canvas = tk.Canvas(self.preview_container, background="#404040")
        self.canvas.grid(row=1, column=0, sticky="nsew")

        vbar = ttk.Scrollbar(self.preview_container, orient=tk.VERTICAL, command=self.canvas.yview)
        vbar.grid(row=1, column=1, sticky="ns")
        hbar = ttk.Scrollbar(self.preview_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        hbar.grid(row=2, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        # Mouse-wheel zoom and click-drag panning.
        self.canvas.bind("<MouseWheel>", self._on_mousewheel_zoom)
        self.canvas.bind("<Button-4>", lambda _e: self._zoom(+1))  # Linux scroll-wheel up.
        self.canvas.bind("<Button-5>", lambda _e: self._zoom(-1))
        self.canvas.bind("<ButtonPress-1>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B1-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))
        try:
            self.canvas.configure(cursor="fleur")
        except Exception:
            pass

    def _build_imaging_tab(self, parent: ttk.Frame):
        c = parent

        # Format
        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row, text="Format:").pack(side=tk.LEFT)
        fmt_values = ["jpg", "png", "fits"] if HAVE_ASTROPY else ["jpg", "png"]
        fmt = ttk.Combobox(row, textvariable=self.format_var, values=fmt_values, state="readonly", width=6)
        fmt.pack(side=tk.LEFT, padx=6)
        fmt.bind("<<ComboboxSelected>>", lambda _e: self._on_format_changed())
        if not HAVE_ASTROPY:
            ttk.Label(row, text="(Install python3-astropy for FITS)", foreground="gray").pack(side=tk.LEFT, padx=10)

        ttk.Label(c, text="Frame type (IMAGETYP)").pack(anchor="w")
        ttk.Combobox(
            c, textvariable=self.frame_type_var,
            values=["Bias", "Dark Flat", "Dark", "Flat", "Light"],
            state="readonly", width=12
        ).pack(anchor="w", pady=(2, 10))

        ttk.Separator(c).pack(fill=tk.X, pady=8)

        # Still-imaging exposure and gain.
        ttk.Label(c, text="Imaging exposure / gain").pack(anchor="w")

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 6))
        ttk.Label(row, text="Mode:").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.img_exposure_mode_var, values=list(self.exposure_modes.keys()),
                     state="readonly", width=5).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Apply", command=self.apply_settings).pack(side=tk.LEFT, padx=8)

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(row, text="Exposure:").pack(side=tk.LEFT)
        self.img_exp_value_label = ttk.Label(row, text="")
        self.img_exp_value_label.pack(side=tk.LEFT, padx=6)

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(row, text="Set:").pack(side=tk.LEFT)
        self.img_exp_entry = ttk.Entry(row, width=10)
        self.img_exp_entry.pack(side=tk.LEFT, padx=6)
        self.img_exp_entry.insert(0, "10")
        self.img_exp_entry.bind("<Return>", lambda _e: self._on_exposure_entry(kind="img"))
        self.img_exp_entry.bind("<FocusOut>", lambda _e: self._on_exposure_entry(kind="img"))

        self.img_exp_scale = ttk.Scale(c, from_=1, to=5000, orient=tk.HORIZONTAL,
                                       command=lambda v: self._on_exposure_slider(v, kind="img"))
        self.img_exp_scale.set(10.0)
        self.img_exp_scale.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(c, text="Analogue gain").pack(anchor="w")
        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 2))
        self.img_gain_value_label = ttk.Label(row, text="1.00")
        self.img_gain_value_label.pack(side=tk.LEFT)

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(row, text="Set:").pack(side=tk.LEFT)
        self.img_gain_entry = ttk.Entry(row, width=10)
        self.img_gain_entry.pack(side=tk.LEFT, padx=6)
        self.img_gain_entry.insert(0, "1.0")
        self.img_gain_entry.bind("<Return>", lambda _e: self._on_gain_entry(kind="img"))
        self.img_gain_entry.bind("<FocusOut>", lambda _e: self._on_gain_entry(kind="img"))

        self.img_gain_scale = ttk.Scale(c, from_=1.0, to=32.0, orient=tk.HORIZONTAL,
                                        command=lambda v: self._on_gain_slider(v, kind="img"))
        self.img_gain_scale.set(1.0)
        self.img_gain_scale.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(c, text="Auto Exposure (AE)", variable=self.img_ae_var,
                        command=self.apply_settings).pack(anchor="w", pady=(2, 6))

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 10))
        ttk.Label(row, text="Bin:").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.bin_var, values=[1, 2, 4], state="readonly", width=5).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Reconfigure", command=self.reconfigure_preview).pack(side=tk.LEFT, padx=8)

        ttk.Separator(c).pack(fill=tk.X, pady=8)

        # Sequence
        ttk.Label(c, text="Capture sequence").pack(anchor="w")
        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(row, text="# images:").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=1, to=9999, textvariable=self.count_var, width=8).pack(side=tk.LEFT, padx=6)
        ttk.Label(row, text="Interval (s):").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=0.0, to=3600.0, increment=0.1, textvariable=self.interval_var, width=8).pack(side=tk.LEFT, padx=6)

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 10))
        ttk.Label(row, text="Countdown (s):").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=0, to=3600, textvariable=self.countdown_var, width=8).pack(side=tk.LEFT, padx=6)

        ttk.Separator(c).pack(fill=tk.X, pady=8)

        # Log
        ttk.Label(c, text="Capture log").pack(anchor="w")
        log_frame = ttk.Frame(c)
        log_frame.pack(fill=tk.BOTH, expand=False, pady=(2, 6))

        self.log = ttk.Treeview(log_frame, columns=("idx", "type", "fmt", "status", "path"),
                                show="headings", height=10)
        for col, w in [("idx", 50), ("type", 90), ("fmt", 55), ("status", 90), ("path", 650)]:
            self.log.heading(col, text=col.upper())
            self.log.column(col, width=w, anchor="w")
        self.log.column("idx", anchor="center")

        log_v = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=log_v.set)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_v.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(c, text="Clear log", command=self._clear_log).pack(anchor="w", pady=(0, 8))

        # Initialize the still-imaging exposure controls.
        self.img_exposure_mode_var.trace_add("write", lambda *_: self._update_exposure_slider_range(kind="img"))
        self._update_exposure_slider_range(kind="img")
        self._update_exposure_label(kind="img")

    def _build_recording_tab(self, parent: ttk.Frame):
        c = parent

        ttk.Label(c, text="Recording format").pack(anchor="w")
        fmts = ["ser"] + (["avi"] if HAVE_CV2 else [])
        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 8))
        ttk.Combobox(row, textvariable=self.rec_format_var, values=fmts, state="readonly", width=6).pack(side=tk.LEFT)
        if not HAVE_CV2:
            ttk.Label(row, text="(Install OpenCV to enable AVI)", foreground="gray").pack(side=tk.LEFT, padx=10)

        self.rec_format_var.trace_add("write", lambda *_: self._update_avi_fps_enabled())

        ttk.Label(c, text="Resolution").pack(anchor="w")
        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 8))
        sizes = self._get_recording_sizes()
        if sizes and not self.rec_resolution_var.get():
            self.rec_resolution_var.set(sizes[0])

        self.rec_size_combo = ttk.Combobox(row, textvariable=self.rec_resolution_var, values=sizes,
                                           state="readonly", width=16)
        self.rec_size_combo.pack(side=tk.LEFT)
        ttk.Button(row, text="Refresh", command=self._refresh_record_sizes).pack(side=tk.LEFT, padx=8)

        # AVI playback rate; it does not throttle camera acquisition.
        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 8))
        ttk.Label(row, text="AVI playback FPS (metadata only):").pack(side=tk.LEFT)
        self.avi_fps_spin = ttk.Spinbox(row, from_=1.0, to=240.0, increment=1.0,
                                        textvariable=self.avi_playback_fps_var, width=8)
        self.avi_fps_spin.pack(side=tk.LEFT, padx=6)
        ttk.Label(row, text="(does not limit capture)").pack(side=tk.LEFT, padx=6)
        self._update_avi_fps_enabled()

        ttk.Separator(c).pack(fill=tk.X, pady=8)

        ttk.Label(c, text="Recording exposure / gain (max throughput)").pack(anchor="w")

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 6))
        ttk.Label(row, text="Mode:").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.rec_exposure_mode_var, values=list(self.exposure_modes.keys()),
                     state="readonly", width=5).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Apply", command=self.apply_settings).pack(side=tk.LEFT, padx=8)

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(row, text="Exposure:").pack(side=tk.LEFT)
        self.rec_exp_value_label = ttk.Label(row, text="")
        self.rec_exp_value_label.pack(side=tk.LEFT, padx=6)

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(row, text="Set:").pack(side=tk.LEFT)
        self.rec_exp_entry = ttk.Entry(row, width=10)
        self.rec_exp_entry.pack(side=tk.LEFT, padx=6)
        self.rec_exp_entry.insert(0, "5")
        self.rec_exp_entry.bind("<Return>", lambda _e: self._on_exposure_entry(kind="rec"))
        self.rec_exp_entry.bind("<FocusOut>", lambda _e: self._on_exposure_entry(kind="rec"))

        self.rec_exp_scale = ttk.Scale(c, from_=1, to=5000, orient=tk.HORIZONTAL,
                                       command=lambda v: self._on_exposure_slider(v, kind="rec"))
        self.rec_exp_scale.set(5.0)
        self.rec_exp_scale.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(c, text="Analogue gain").pack(anchor="w")
        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 2))
        self.rec_gain_value_label = ttk.Label(row, text="1.00")
        self.rec_gain_value_label.pack(side=tk.LEFT)

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(row, text="Set:").pack(side=tk.LEFT)
        self.rec_gain_entry = ttk.Entry(row, width=10)
        self.rec_gain_entry.pack(side=tk.LEFT, padx=6)
        self.rec_gain_entry.insert(0, "1.0")
        self.rec_gain_entry.bind("<Return>", lambda _e: self._on_gain_entry(kind="rec"))
        self.rec_gain_entry.bind("<FocusOut>", lambda _e: self._on_gain_entry(kind="rec"))

        self.rec_gain_scale = ttk.Scale(c, from_=1.0, to=32.0, orient=tk.HORIZONTAL,
                                        command=lambda v: self._on_gain_slider(v, kind="rec"))
        self.rec_gain_scale.set(1.0)
        self.rec_gain_scale.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(c, text="Auto Exposure (AE)", variable=self.rec_ae_var,
                        command=self.apply_settings).pack(anchor="w", pady=(2, 6))

        ttk.Checkbutton(c, text="SER: write per-frame timestamp trailer",
                        variable=self.rec_write_trailer_var).pack(anchor="w", pady=(2, 10))

        ttk.Separator(c).pack(fill=tk.X, pady=8)

        ttk.Label(c, text="Stop condition").pack(anchor="w")

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 4))
        ttk.Radiobutton(row, text="Manual (stop button)", value="manual", variable=self.rec_stop_mode_var).pack(side=tk.LEFT)

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 4))
        ttk.Radiobutton(row, text="After frames:", value="frames", variable=self.rec_stop_mode_var).pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=1, to=50_000_000, increment=1, textvariable=self.rec_frames_var, width=12).pack(side=tk.LEFT, padx=6)

        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 8))
        ttk.Radiobutton(row, text="After seconds:", value="seconds", variable=self.rec_stop_mode_var).pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=0.1, to=36000.0, increment=0.1, textvariable=self.rec_seconds_var, width=12).pack(side=tk.LEFT, padx=6)

        # Show this progress row only for a frame-count recording target.
        self.rec_prog_row = ttk.Frame(c)
        self.rec_prog = ttk.Progressbar(self.rec_prog_row, variable=self.rec_progress_value, maximum=100.0)
        self.rec_prog.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.rec_prog_pct = ttk.Label(self.rec_prog_row, text="0%")
        self.rec_prog_pct.pack(side=tk.LEFT, padx=8)

        # Buttons
        row = ttk.Frame(c)
        row.pack(fill=tk.X, pady=(2, 8))
        self.rec_btn = ttk.Button(row, text="Start recording", command=self.start_recording)
        self.rec_btn.pack(side=tk.LEFT)
        ttk.Button(row, text="Stop recording", command=self.stop_recording).pack(side=tk.LEFT, padx=8)

        ttk.Label(c, textvariable=self.rec_status_var, foreground="gray").pack(anchor="w")

        # Update progress-bar visibility when the stop mode changes.
        self.rec_stop_mode_var.trace_add("write", lambda *_: self._update_rec_progressbar_visibility())
        self.rec_frames_var.trace_add("write", lambda *_: self._update_rec_progressbar_visibility())
        self._update_rec_progressbar_visibility()

        # Initialize the recording exposure controls.
        self.rec_exposure_mode_var.trace_add("write", lambda *_: self._update_exposure_slider_range(kind="rec"))
        self._update_exposure_slider_range(kind="rec")
        self._update_exposure_label(kind="rec")

    def _update_avi_fps_enabled(self):
        is_avi = self.rec_format_var.get().strip().lower() == "avi"
        state = "normal" if is_avi else "disabled"
        try:
            self.avi_fps_spin.configure(state=state)
        except Exception:
            pass

    def _update_rec_progressbar_visibility(self):
        mode = self.rec_stop_mode_var.get().strip().lower()
        if mode == "frames":
            target = max(1, int(self.rec_frames_var.get() or 1))
            self.rec_progress_max = float(target)
            self.rec_prog.configure(maximum=self.rec_progress_max)
            if not self.rec_prog_row.winfo_ismapped():
                self.rec_prog_row.pack(fill=tk.X, pady=(2, 6))
        else:
            if self.rec_prog_row.winfo_ismapped():
                self.rec_prog_row.pack_forget()

    # ---------- common UI helpers ----------
    def choose_folder(self):
        d = filedialog.askdirectory(initialdir=self.output_dir.get())
        if d:
            self.output_dir.set(d)

    def _clear_log(self):
        for item in self.log.get_children():
            self.log.delete(item)

    def _log_add(self, idx, frame_type, fmt, status, path):
        self.log.insert("", "end", values=(idx, frame_type, fmt, status, path))

    def _log_update(self, idx, status, path):
        for item in reversed(self.log.get_children()):
            vals = self.log.item(item, "values")
            if vals and int(vals[0]) == int(idx):
                self.log.item(item, values=(vals[0], vals[1], vals[2], status, path))
                return

    def _validate_common(self, fmt: str):
        outdir = self.output_dir.get().strip()
        base = self.base_name.get().strip()
        fmt = (fmt or "").strip().lower()

        if not outdir or not os.path.isdir(outdir):
            messagebox.showerror("Folder", "Choose a valid save folder.")
            return None
        if not base:
            messagebox.showerror("Name", "Enter a base name (e.g., moz).")
            return None
        if fmt == "fits" and not HAVE_ASTROPY:
            messagebox.showerror("FITS", "FITS requires astropy (python3-astropy).")
            return None
        if fmt == "avi" and not HAVE_CV2:
            messagebox.showerror("AVI", "AVI requires OpenCV (python3-opencv / opencv-python).")
            return None

        return outdir, base, fmt

    def _next_available_indices(self, base: str, ext: str, n: int, outdir=None):
        if outdir is None:
            outdir = self.output_dir.get().strip()
        indices = []
        i = 1
        while len(indices) < n:
            path = os.path.join(outdir, f"{base}_{i}.{ext}")
            if not os.path.exists(path):
                indices.append(i)
            i += 1
        return indices

    # ---------- camera config ----------
    def _choose_mode_for_bin(self, bin_factor: int):
        modes = getattr(self.picam2, "sensor_modes", None)
        if not modes:
            return None
        props = getattr(self.picam2, "camera_properties", {}) or {}
        full = props.get("PixelArraySize")
        if full and isinstance(full, (tuple, list)) and len(full) == 2:
            target = (max(1, int(full[0] / bin_factor)), max(1, int(full[1] / bin_factor)))
        else:
            target = (4056 // bin_factor, 3040 // bin_factor)

        def score(mode):
            w, h = mode["size"]
            return abs(w - target[0]) + abs(h - target[1])

        return min(modes, key=score)

    def _choose_mode_for_size(self, target_w: int, target_h: int):
        modes = getattr(self.picam2, "sensor_modes", None)
        if not modes:
            return None

        def score(mode):
            w, h = mode["size"]
            return abs(w - target_w) + abs(h - target_h)

        return min(modes, key=score)

    def _configure_camera(self, profile: str, rec_size=None, bin_factor=None):
        profile = profile.strip().lower()
        with self.cam_lock:
            if self.started:
                self.picam2.stop()
                self.started = False

            if profile in ("preview", "still"):
                if bin_factor is None:
                    bin_factor = int(self.bin_var.get())
                bin_factor = max(1, int(bin_factor))
                mode = self._choose_mode_for_bin(bin_factor)
                if mode:
                    size = tuple(mode["size"])
                else:
                    size = (4056, 3040) if bin_factor == 1 else (2028, 1520) if bin_factor == 2 else (1014, 760)

                if profile == "preview":
                    cfg = self.picam2.create_preview_configuration(main={"size": size, "format": "RGB888"})
                else:
                    cfg = self.picam2.create_still_configuration(main={"size": size, "format": "RGB888"})

            elif profile == "video":
                if rec_size is None:
                    raise ValueError("rec_size required for video profile.")
                tw, th = rec_size
                mode = self._choose_mode_for_size(tw, th)
                size = tuple(mode["size"]) if mode else (tw, th)
                cfg = self.picam2.create_video_configuration(main={"size": size, "format": "RGB888"})
            else:
                raise ValueError(f"Unknown profile: {profile}")

            self.picam2.configure(cfg)
            self.picam2.start()
            self.started = True

    # ---------- controls ----------
    def _current_exposure_us(self, kind: str) -> int:
        if kind == "img":
            mode = self.img_exposure_mode_var.get()
            factor = self.exposure_modes.get(mode, 1000)
            return int(float(self.img_exposure_val_var.get()) * factor)
        else:
            mode = self.rec_exposure_mode_var.get()
            factor = self.exposure_modes.get(mode, 1000)
            return int(float(self.rec_exposure_val_var.get()) * factor)

    def _set_controls_values(self, ae_enabled: bool, exposure_us: int, gain: float):
        """Apply camera controls from plain values without reading Tk in a worker."""
        if ae_enabled:
            self.picam2.set_controls({"AeEnable": True})
            return

        exposure_us = max(1, int(exposure_us))
        self.picam2.set_controls({
            "AeEnable": False,
            "ExposureTime": exposure_us,
            "AnalogueGain": float(gain),
            "FrameDurationLimits": (exposure_us, exposure_us),
        })

    def _apply_controls_locked(self, kind: str):
        # Select the still-imaging ("img") or recording ("rec") controls.
        if kind == "img":
            ae = bool(self.img_ae_var.get())
            exp_us = self._current_exposure_us("img")
            gain = float(self.img_gain_var.get())
        else:
            ae = bool(self.rec_ae_var.get())
            exp_us = self._current_exposure_us("rec")
            gain = float(self.rec_gain_var.get())

        self._set_controls_values(ae, exp_us, gain)

    def _snapshot_metadata(self):
        """Copy Tk-backed astronomy metadata into thread-safe strings."""
        return {
            "object": self.object_var.get().strip(),
            "observer": self.observer_var.get().strip(),
            "instrument": self.instrument_var.get().strip() or "Picamera2",
            "telescope": self.telescope_var.get().strip(),
        }

    def _snapshot_imaging_settings(self):
        """Snapshot all still-imaging settings on Tk's main thread."""
        return {
            "binning": int(self.bin_var.get()),
            "ae_enabled": bool(self.img_ae_var.get()),
            "exposure_us": self._current_exposure_us("img"),
            "gain": float(self.img_gain_var.get()),
            "frame_type": self._imagetyp_value(),
            "metadata": self._snapshot_metadata(),
        }

    def _snapshot_recording_settings(self):
        """Snapshot all recording settings on Tk's main thread."""
        return {
            "ae_enabled": bool(self.rec_ae_var.get()),
            "exposure_us": self._current_exposure_us("rec"),
            "gain": float(self.rec_gain_var.get()),
            "stop_mode": self.rec_stop_mode_var.get().strip().lower(),
            "max_frames": max(1, int(self.rec_frames_var.get())),
            "max_seconds": max(0.0, float(self.rec_seconds_var.get())),
            "avi_playback_fps": float(self.avi_playback_fps_var.get()),
            "write_trailer": bool(self.rec_write_trailer_var.get()),
            "metadata": self._snapshot_metadata(),
            "preview_binning": int(self.bin_var.get()),
            "preview_ae_enabled": bool(self.img_ae_var.get()),
            "preview_exposure_us": self._current_exposure_us("img"),
            "preview_gain": float(self.img_gain_var.get()),
        }

    def apply_settings(self):
        try:
            with self.cam_lock:
                if not self.started:
                    return
                if self.recording_running:
                    self._apply_controls_locked("rec")
                else:
                    self._apply_controls_locked("img")
            self.status_var.set("Settings applied.")
        except Exception as e:
            self.status_var.set("Error applying settings.")
            messagebox.showerror("Apply error", str(e))

    # ---------- exposure/gain UI (shared) ----------
    def _update_exposure_slider_range(self, kind: str):
        mode = self.img_exposure_mode_var.get() if kind == "img" else self.rec_exposure_mode_var.get()
        mn, mx = self.exposure_ranges.get(mode, (1, 5000))

        if kind == "img":
            self.img_exp_scale.configure(from_=mn, to=mx)
            v = float(self.img_exposure_val_var.get())
            v = max(mn, min(mx, v))
            self.img_exposure_val_var.set(v)
            self.img_exp_scale.set(v)
            self.img_exp_entry.delete(0, tk.END)
            self.img_exp_entry.insert(0, f"{v:g}")
        else:
            self.rec_exp_scale.configure(from_=mn, to=mx)
            v = float(self.rec_exposure_val_var.get())
            v = max(mn, min(mx, v))
            self.rec_exposure_val_var.set(v)
            self.rec_exp_scale.set(v)
            self.rec_exp_entry.delete(0, tk.END)
            self.rec_exp_entry.insert(0, f"{v:g}")

        self._update_exposure_label(kind)

    def _update_exposure_label(self, kind: str):
        if kind == "img":
            mode = self.img_exposure_mode_var.get()
            v = float(self.img_exposure_val_var.get())
            lbl = self.img_exp_value_label
        else:
            mode = self.rec_exposure_mode_var.get()
            v = float(self.rec_exposure_val_var.get())
            lbl = self.rec_exp_value_label

        if mode == "µs":
            lbl.config(text=f"{v:.0f} µs")
        elif mode == "ms":
            lbl.config(text=f"{v:.0f} ms")
        else:
            lbl.config(text=f"{v:.2f} s")

    def _on_exposure_slider(self, value, kind: str):
        try:
            v = float(value)
        except ValueError:
            v = float(self.img_exp_scale.get() if kind == "img" else self.rec_exp_scale.get())

        if kind == "img":
            self.img_exposure_val_var.set(v)
            self.img_exp_entry.delete(0, tk.END)
            self.img_exp_entry.insert(0, f"{v:g}")
        else:
            self.rec_exposure_val_var.set(v)
            self.rec_exp_entry.delete(0, tk.END)
            self.rec_exp_entry.insert(0, f"{v:g}")

        self._update_exposure_label(kind)
        self.apply_settings()

    def _on_exposure_entry(self, kind: str):
        entry = self.img_exp_entry if kind == "img" else self.rec_exp_entry
        txt = entry.get().strip()
        if not txt:
            return
        try:
            v = float(txt)
        except ValueError:
            return

        mode = self.img_exposure_mode_var.get() if kind == "img" else self.rec_exposure_mode_var.get()
        mn, mx = self.exposure_ranges.get(mode, (1, 5000))
        v = max(mn, min(mx, v))

        if kind == "img":
            self.img_exposure_val_var.set(v)
            self.img_exp_scale.set(v)
        else:
            self.rec_exposure_val_var.set(v)
            self.rec_exp_scale.set(v)

        self._update_exposure_label(kind)
        self.apply_settings()

    def _on_gain_slider(self, value, kind: str):
        try:
            v = float(value)
        except ValueError:
            v = float(self.img_gain_scale.get() if kind == "img" else self.rec_gain_scale.get())

        v = max(1.0, min(32.0, v))

        if kind == "img":
            self.img_gain_var.set(v)
            self.img_gain_entry.delete(0, tk.END)
            self.img_gain_entry.insert(0, f"{v:.2f}")
            self.img_gain_value_label.config(text=f"{v:.2f}")
        else:
            self.rec_gain_var.set(v)
            self.rec_gain_entry.delete(0, tk.END)
            self.rec_gain_entry.insert(0, f"{v:.2f}")
            self.rec_gain_value_label.config(text=f"{v:.2f}")

        self.apply_settings()

    def _on_gain_entry(self, kind: str):
        entry = self.img_gain_entry if kind == "img" else self.rec_gain_entry
        txt = entry.get().strip()
        if not txt:
            return
        try:
            v = float(txt)
        except ValueError:
            return

        v = max(1.0, min(32.0, v))

        if kind == "img":
            self.img_gain_var.set(v)
            self.img_gain_scale.set(v)
            self.img_gain_value_label.config(text=f"{v:.2f}")
        else:
            self.rec_gain_var.set(v)
            self.rec_gain_scale.set(v)
            self.rec_gain_value_label.config(text=f"{v:.2f}")

        self.apply_settings()

    def reconfigure_preview(self):
        try:
            was_running = self.preview_running
            if was_running:
                self.stop_preview()
            self._configure_camera(profile="preview")
            with self.cam_lock:
                self._apply_controls_locked("img")
            self.status_var.set("Reconfigured (bin/mode).")
            if was_running:
                self.start_preview()
        except Exception as e:
            messagebox.showerror("Reconfigure error", str(e))

    def _on_format_changed(self):
        if self.format_var.get().lower() == "fits" and not HAVE_ASTROPY:
            messagebox.showerror("FITS unavailable", "Install python3-astropy to enable FITS.")
            self.format_var.set("png")
        self._render_frame()

    # ---------- preview stretch helpers ----------
    def _get_stretch_profile(self):
        # Each profile is (low percentile, high percentile, gamma).
        mode = (self.stretch_var.get() or "").strip()
        profiles = {
            "Dimmest":   (0.5, 99.5, 1.35),
            "Dim":       (1.0, 99.0, 1.15),
            "Bright":    (2.0, 98.0, 0.95),
            "Brightest": (5.0, 95.0, 0.75),
        }
        return profiles.get(mode, profiles["Dim"])

    def _ensure_stretch_lut_cached(self, rgb: np.ndarray):
        # Recompute at most twice per second, or immediately after a mode change.
        mode = (self.stretch_var.get() or "").strip()
        now = time.time()

        if (
            self._stretch_lut3_cache is not None
            and self._stretch_mode_cache == mode
            and (now - self._stretch_last_calc) < 0.5
        ):
            return

        p_lo, p_hi, gamma = self._get_stretch_profile()

        samp = rgb[::8, ::8, :]
        r = samp[..., 0].astype(np.float32)
        g = samp[..., 1].astype(np.float32)
        b = samp[..., 2].astype(np.float32)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

        lo = float(np.percentile(lum, p_lo))
        hi = float(np.percentile(lum, p_hi))
        if hi <= lo + 1e-6:
            lo, hi = 0.0, 255.0

        x = np.arange(256, dtype=np.float32)
        y = (x - lo) / (hi - lo)
        y = np.clip(y, 0.0, 1.0)
        y = np.power(y, gamma)
        lut = (y * 255.0 + 0.5).astype(np.uint8).tolist()

        self._stretch_lut3_cache = lut * 3
        self._stretch_mode_cache = mode
        self._stretch_last_calc = now

    # ---------- preview ----------
    def toggle_preview(self):
        if self.preview_running:
            self.stop_preview()
        else:
            self.start_preview()

    def start_preview(self):
        if self.closing:
            return
        self.allow_preview_restart = True
        if self.recording_running:
            messagebox.showinfo("Preview", "Stop recording before starting live preview.")
            return

        if self.capture_thread and self.capture_thread.is_alive():
            messagebox.showinfo("Preview", "Wait for the current capture operation to finish.")
            return

        if self.preview_thread and self.preview_thread.is_alive():
            self._preview_start_pending = True
            self.status_var.set("Waiting for the previous preview thread to stop…")
            self.root.after(100, self._start_preview_when_ready)
            return

        try:
            self._configure_camera(profile="preview")
            with self.cam_lock:
                self._apply_controls_locked("img")
        except Exception as e:
            messagebox.showerror("Preview error", f"Could not start preview:\n{e}")
            return

        self.stop_preview_evt.clear()
        self.preview_running = True
        self.preview_btn.config(text="Stop Live")
        self.status_var.set("Live preview running…")
        self.preview_thread = threading.Thread(target=self._preview_worker, daemon=True)
        self.preview_thread.start()

    def _start_preview_when_ready(self):
        if self.closing or not self._preview_start_pending:
            self._preview_start_pending = False
            return
        if self.recording_running:
            self._preview_start_pending = False
            return
        if self.capture_thread and self.capture_thread.is_alive():
            self.root.after(100, self._start_preview_when_ready)
            return
        if self.preview_thread and self.preview_thread.is_alive():
            self.root.after(100, self._start_preview_when_ready)
            return
        self._preview_start_pending = False
        self.start_preview()

    def _halt_preview_worker(self, wait=False):
        """Stop the preview worker, optionally waiting outside Tk's main thread."""
        self.stop_preview_evt.set()
        self.preview_running = False
        thread = self.preview_thread
        if wait and thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join()
        if thread and not thread.is_alive():
            self.preview_thread = None

    def stop_preview(self):
        self._preview_start_pending = False
        self._halt_preview_worker(wait=False)
        self.preview_btn.config(text="Start Live")
        self.fps_var.set("FPS: 0.0")
        self.last_frame_time = None
        self.fps_smooth = 0.0
        self.status_var.set("Live preview stopped.")

    def _preview_worker(self):
        try:
            while not self.stop_preview_evt.is_set():
                with self.cam_lock:
                    frame = self.picam2.capture_array()

                now = time.time()
                if self.last_frame_time is not None:
                    dt = now - self.last_frame_time
                    if dt > 0:
                        inst = 1.0 / dt
                        self.fps_smooth = inst if self.fps_smooth == 0 else (0.85 * self.fps_smooth + 0.15 * inst)
                self.last_frame_time = now

                if self.frame_q.full():
                    try:
                        _ = self.frame_q.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_q.put(frame)

                time.sleep(0.03)
        except Exception as e:
            self.ui_q.put(("error", f"Preview error: {e}"))
        finally:
            self.preview_running = False
            self.ui_q.put(("fps", 0.0))

    # ---------- headers ----------
    def _imagetyp_value(self) -> str:
        t = self.frame_type_var.get().strip().lower()
        return {
            "light": "LIGHT",
            "dark": "DARK",
            "bias": "BIAS",
            "flat": "FLAT",
            "dark flat": "DARKFLAT",
        }.get(t, "LIGHT")

    def _build_header_pairs(self, width: int, height: int, bitdepth: int, is_fits_rgb_cube: bool, extra: dict | None = None, metadata: dict | None = None):
        now = datetime.now(timezone.utc)
        date_only = now.strftime("%Y-%m-%d")
        time_only = now.strftime("%H:%M:%S.%f")[:-3]
        date_obs_full = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

        if metadata is None:
            metadata = self._snapshot_metadata()

        hdr = {
            "DATE-OBS": (date_only, "UTC date at start"),
            "TIME-OBS": (time_only, "UTC time at start"),
            "DATEOBS": (date_obs_full, "UTC date-time at start (ISO8601)"),
            "TIMESYS": ("UTC", "Time system"),
            "BITDEPTH": (int(bitdepth), "Bits per channel stored"),

            # Preserve the display order used by the session-metadata controls.
            "OBJECT": (metadata.get("object", ""), "Target object"),
            "OBSERVER": (metadata.get("observer", ""), "Observer"),
            "INSTRUME": (metadata.get("instrument", "Picamera2"), "Instrument / camera"),
            "TELESCOP": (metadata.get("telescope", ""), "Telescope"),
        }

        if is_fits_rgb_cube:
            hdr["NAXIS"] = (3, "Number of axes (RGB cube)")
            hdr["NAXIS1"] = (int(width), "Axis 1 length (X)")
            hdr["NAXIS2"] = (int(height), "Axis 2 length (Y)")
            hdr["NAXIS3"] = (3, "Axis 3 length (RGB planes)")
        else:
            hdr["NAXIS"] = (2, "Number of axes")
            hdr["NAXIS1"] = (int(width), "Axis 1 length (X)")
            hdr["NAXIS2"] = (int(height), "Axis 2 length (Y)")

        if extra:
            for k, (v, c) in extra.items():
                hdr[k] = (v, c)

        return hdr

    def _header_as_text(self, hdr_pairs: dict) -> str:
        lines = []
        for k, (v, c) in hdr_pairs.items():
            v_str = f"'{v}'" if isinstance(v, str) else str(v)
            lines.append(f"{k}={v_str} / {c}")
        return "\n".join(lines)

    def _write_sidecar_header(self, media_path: str, width: int, height: int, bitdepth: int, extra: dict | None = None, metadata: dict | None = None):
        hdr = self._build_header_pairs(width, height, bitdepth=bitdepth, is_fits_rgb_cube=False, extra=extra, metadata=metadata)
        txt = self._header_as_text(hdr)
        sidecar = media_path + ".hdr.txt"
        with open(sidecar, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        return sidecar

    # ---------- imaging save ----------
    def _save_fits_rgb_with_header(self, rgb8: np.ndarray, out_path: str, settings: dict):
        if not HAVE_ASTROPY:
            raise RuntimeError("FITS selected but astropy is not installed.")

        h, w = rgb8.shape[:2]
        rgb16 = (rgb8.astype(np.uint16) * 257)
        cube = np.transpose(rgb16, (2, 0, 1))

        exp_s = settings["exposure_us"] / 1_000_000.0
        gain = settings["gain"]
        binning = settings["binning"]

        extra = {
            "EXPTIME": (float(exp_s), "Exposure time [s]"),
            "GAIN": (float(gain), "Analogue gain setting"),
            "XBINNING": (int(binning), "Binning factor in X"),
            "YBINNING": (int(binning), "Binning factor in Y"),
            "IMAGETYP": (settings["frame_type"], "Frame type"),
        }
        hdr_pairs = self._build_header_pairs(w, h, bitdepth=16, is_fits_rgb_cube=True, extra=extra, metadata=settings["metadata"])

        hdu = fits.PrimaryHDU(data=cube)
        hdr = hdu.header
        for k, (v, c) in hdr_pairs.items():
            if k in ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND"):
                continue
            hdr[k] = (v, c)

        hdr["CTYPE3"] = ("COLOR", "Axis 3 meaning")
        hdr["CNAME3"] = ("RGB", "RGB planes order")

        hdu.writeto(out_path, overwrite=False)

    def _save_png_with_metadata(self, rgb8: np.ndarray, out_path: str, settings: dict):
        h, w = rgb8.shape[:2]
        exp_s = settings["exposure_us"] / 1_000_000.0
        gain = settings["gain"]
        binning = settings["binning"]

        extra = {
            "EXPTIME": (float(exp_s), "Exposure time [s]"),
            "GAIN": (float(gain), "Analogue gain setting"),
            "XBINNING": (int(binning), "Binning factor in X"),
            "YBINNING": (int(binning), "Binning factor in Y"),
            "IMAGETYP": (settings["frame_type"], "Frame type"),
        }
        hdr_pairs = self._build_header_pairs(w, h, bitdepth=8, is_fits_rgb_cube=False, extra=extra, metadata=settings["metadata"])
        header_text = self._header_as_text(hdr_pairs)

        img = Image.fromarray(rgb8)
        meta = PngInfo()
        for k, (v, _c) in hdr_pairs.items():
            meta.add_text(k, str(v))
        meta.add_text("ASTROHDR", header_text)
        img.save(out_path, pnginfo=meta)

    def _save_jpeg_with_exif(self, rgb8: np.ndarray, out_path: str, settings: dict):
        h, w = rgb8.shape[:2]
        exp_s = settings["exposure_us"] / 1_000_000.0
        gain = settings["gain"]
        binning = settings["binning"]

        extra = {
            "EXPTIME": (float(exp_s), "Exposure time [s]"),
            "GAIN": (float(gain), "Analogue gain setting"),
            "XBINNING": (int(binning), "Binning factor in X"),
            "YBINNING": (int(binning), "Binning factor in Y"),
            "IMAGETYP": (settings["frame_type"], "Frame type"),
        }
        hdr_pairs = self._build_header_pairs(w, h, bitdepth=8, is_fits_rgb_cube=False, extra=extra, metadata=settings["metadata"])
        header_text = self._header_as_text(hdr_pairs)

        img = Image.fromarray(rgb8)
        desc = (
            f"OBJECT={settings['metadata'].get('object', '')} "
            f"IMAGETYP={extra['IMAGETYP'][0]} EXPTIME={extra['EXPTIME'][0]}s "
            f"GAIN={extra['GAIN'][0]} BIN={binning}"
        )

        try:
            exif = img.getexif()
        except Exception:
            exif = None

        if exif is not None:
            exif[270] = desc  # EXIF ImageDescription.
            exif[37510] = b"ASCII\0\0\0" + header_text.encode("ascii", errors="replace")
            exif[306] = datetime.now(timezone.utc).strftime("%Y:%m:%d %H:%M:%S")
            img.save(out_path, quality=95, exif=exif.tobytes())
        else:
            img.save(out_path, quality=95)

    def _save_frame_embedded(self, rgb: np.ndarray, out_path: str, fmt: str, settings: dict):
        fmt = fmt.lower()
        if fmt == "fits":
            self._save_fits_rgb_with_header(rgb, out_path, settings)
        elif fmt == "png":
            self._save_png_with_metadata(rgb, out_path, settings)
        elif fmt in ("jpg", "jpeg"):
            self._save_jpeg_with_exif(rgb, out_path, settings)
        else:
            raise RuntimeError(f"Unknown format: {fmt}")

    # ---------- actions ----------
    def stop_all(self):
        self.allow_preview_restart = False
        self.stop_capture_evt.set()
        self.stop_preview_evt.set()
        self.stop_record_evt.set()
        self._preview_start_pending = False

        self.preview_running = False

        self.preview_btn.config(text="Start Live")
        self.status_var.set("Stopping…")
        self.rec_status_var.set("Stopping…")

    def save_single_frame(self):
        if self.recording_running:
            messagebox.showinfo("Imaging", "Stop recording before saving a still frame.")
            return
        if self.capture_thread and self.capture_thread.is_alive():
            messagebox.showinfo("Imaging", "Another capture operation is already running.")
            return

        v = self._validate_common(self.format_var.get())
        if not v:
            return
        outdir, base, fmt = v

        idx = self._next_available_indices(base, fmt, 1)[0]
        out_path = os.path.join(outdir, f"{base}_{idx}.{fmt}")
        settings = self._snapshot_imaging_settings()
        self._log_add(idx, settings["frame_type"], fmt, "Saving…", out_path)

        def worker():
            was_preview = self.preview_running
            try:
                if was_preview:
                    self._halt_preview_worker(wait=True)
                    self.ui_q.put(("preview_stopped_ui", None))

                self._configure_camera(profile="still", bin_factor=settings["binning"])
                with self.cam_lock:
                    self._set_controls_values(settings["ae_enabled"], settings["exposure_us"], settings["gain"])
                    rgb = self.picam2.capture_array()

                self._save_frame_embedded(rgb, out_path, fmt, settings)
                self.ui_q.put(("saved", idx, out_path, rgb))
            except Exception as e:
                self.ui_q.put(("error", str(e)))
            finally:
                if not self.closing:
                    try:
                        self._configure_camera(profile="preview", bin_factor=settings["binning"])
                        with self.cam_lock:
                            self._set_controls_values(settings["ae_enabled"], settings["exposure_us"], settings["gain"])
                    except Exception:
                        pass
                if was_preview and self.allow_preview_restart and not self.closing:
                    self.ui_q.put(("start_preview", None))

        self.capture_thread = threading.Thread(target=worker, daemon=True)
        self.capture_thread.start()

    def start_sequence(self):
        if self.recording_running:
            messagebox.showinfo("Sequence", "Stop recording before starting a sequence.")
            return
        if self.capture_thread and self.capture_thread.is_alive():
            messagebox.showinfo("Sequence", "A sequence is already running.")
            return

        v = self._validate_common(self.format_var.get())
        if not v:
            return

        outdir, base, fmt = v
        settings = self._snapshot_imaging_settings()
        settings.update({
            "outdir": outdir,
            "base": base,
            "fmt": fmt,
            "count": max(1, int(self.count_var.get())),
            "interval": max(0.0, float(self.interval_var.get())),
            "countdown": max(0, int(self.countdown_var.get())),
        })

        self.stop_capture_evt.clear()
        self.capture_thread = threading.Thread(target=self._sequence_worker, args=(settings,), daemon=True)
        self.capture_thread.start()

    def _sequence_worker(self, settings):
        was_preview = self.preview_running
        try:
            if was_preview:
                self._halt_preview_worker(wait=True)
                self.ui_q.put(("preview_stopped_ui", None))

            self.ui_q.put(("status", "Configuring still capture…"))
            self._configure_camera(profile="still", bin_factor=settings["binning"])

            countdown = settings["countdown"]
            for t in range(countdown, 0, -1):
                if self.stop_capture_evt.is_set():
                    self.ui_q.put(("status", "Sequence cancelled."))
                    return
                self.ui_q.put(("status", f"Countdown: {t}s"))
                time.sleep(1)

            outdir, base, fmt = settings["outdir"], settings["base"], settings["fmt"]
            n = settings["count"]
            interval = settings["interval"]
            frame_type = settings["frame_type"]
            indices = self._next_available_indices(base, fmt, n, outdir=outdir)

            for k, idx in enumerate(indices, start=1):
                if self.stop_capture_evt.is_set():
                    self.ui_q.put(("status", "Sequence stopped."))
                    break

                out_path = os.path.join(outdir, f"{base}_{idx}.{fmt}")
                self.ui_q.put(("log_add", idx, frame_type, fmt, "Saving…", out_path))
                self.ui_q.put(("status", f"Capturing {k}/{n} → {os.path.basename(out_path)}"))

                with self.cam_lock:
                    self._set_controls_values(settings["ae_enabled"], settings["exposure_us"], settings["gain"])
                    rgb = self.picam2.capture_array()

                self._save_frame_embedded(rgb, out_path, fmt, settings)
                self.ui_q.put(("saved", idx, out_path, rgb))

                if k < n and interval > 0:
                    remain = interval
                    while remain > 0 and not self.stop_capture_evt.is_set():
                        step = min(0.1, remain)
                        time.sleep(step)
                        remain -= step

            self.ui_q.put(("status", "Sequence finished."))
        except Exception as e:
            self.ui_q.put(("error", str(e)))
        finally:
            if not self.closing:
                try:
                    self._configure_camera(profile="preview", bin_factor=settings["binning"])
                    with self.cam_lock:
                        self._set_controls_values(settings["ae_enabled"], settings["exposure_us"], settings["gain"])
                except Exception:
                    pass
            if was_preview and self.allow_preview_restart and not self.closing:
                self.ui_q.put(("start_preview", None))

    # ---------- recording ----------
    def _get_recording_sizes(self):
        sizes = []
        try:
            modes = getattr(self.picam2, "sensor_modes", None) or []
            unique = set()
            for m in modes:
                s = m.get("size")
                if s and isinstance(s, (tuple, list)) and len(s) == 2:
                    unique.add((int(s[0]), int(s[1])))

            props = getattr(self.picam2, "camera_properties", {}) or {}
            full = props.get("PixelArraySize")
            if full and isinstance(full, (tuple, list)) and len(full) == 2:
                unique.add((int(full[0]), int(full[1])))

            for w, h in sorted(unique, key=lambda x: x[0] * x[1], reverse=True):
                sizes.append(f"{w}x{h}")
        except Exception:
            pass
        return sizes

    def _refresh_record_sizes(self):
        sizes = self._get_recording_sizes()
        self.rec_size_combo.configure(values=sizes)
        if sizes and self.rec_resolution_var.get() not in sizes:
            self.rec_resolution_var.set(sizes[0])

    def _parse_size(self, s: str):
        s = s.lower().replace(" ", "")
        w, h = s.split("x", 1)
        return int(w), int(h)

    def start_recording(self):
        if self.recording_running:
            self.stop_recording()
            return
        if self.capture_thread and self.capture_thread.is_alive():
            messagebox.showinfo("Recording", "A capture sequence is running. Stop it first.")
            return

        fmt = self.rec_format_var.get().strip().lower()
        v = self._validate_common(fmt)
        if not v:
            return
        outdir, base, fmt = v

        if not self.rec_resolution_var.get():
            messagebox.showerror("Recording", "Pick a resolution.")
            return
        try:
            tw, th = self._parse_size(self.rec_resolution_var.get())
        except Exception:
            messagebox.showerror("Recording", "Invalid resolution.")
            return

        idx = self._next_available_indices(base, fmt, 1)[0]
        out_path = os.path.join(outdir, f"{base}_{idx}.{fmt}")
        settings = self._snapshot_recording_settings()

        # Reset the recording progress display.
        self.rec_progress_value.set(0.0)
        try:
            self.rec_prog_pct.config(text="0%")
        except Exception:
            pass
        self._update_rec_progressbar_visibility()

        self._log_add(idx, "VIDEO", fmt.upper(), "Recording…", out_path)
        self.stop_record_evt.clear()
        self.recording_running = True
        self.rec_top_btn.config(text="Stop recording")
        self.rec_btn.config(text="Stop recording")
        self.rec_status_var.set(f"Recording → {os.path.basename(out_path)}")

        def worker():
            was_preview = self.preview_running
            t0 = time.time()
            frames_written = 0
            dropped_est = 0

            fps_ema = 0.0
            dt_ema = 0.0
            last_frame_t = None
            last_ui_t = time.time()

            sidecar = None

            try:
                if was_preview:
                    self._halt_preview_worker(wait=True)
                    self.ui_q.put(("preview_stopped_ui", None))

                self.ui_q.put(("status", "Configuring video…"))
                self._configure_camera(profile="video", rec_size=(tw, th))

                with self.cam_lock:
                    self._set_controls_values(settings["ae_enabled"], settings["exposure_us"], settings["gain"])

                with self.cam_lock:
                    first = self.picam2.capture_array()
                h, w = first.shape[:2]

                rec_exp_s = settings["exposure_us"] / 1_000_000.0
                rec_gain = settings["gain"]

                stop_mode = settings["stop_mode"]
                max_frames = settings["max_frames"]
                max_seconds = settings["max_seconds"]

                extra = {
                    "RECFMT": (fmt.upper(), "Recording container"),
                    "REC_SIZE": (f"{w}x{h}", "Actual capture size"),
                    "EXPTIME": (float(rec_exp_s), "Recording exposure time [s]"),
                    "GAIN": (float(rec_gain), "Recording analogue gain"),
                    "IMAGETYP": ("VIDEO", "Frame type"),
                }
                if fmt == "avi":
                    extra["AVI_FPS"] = (settings["avi_playback_fps"], "AVI playback fps (metadata)")
                sidecar = self._write_sidecar_header(
                    out_path,
                    width=w,
                    height=h,
                    bitdepth=8,
                    extra=extra,
                    metadata=settings["metadata"],
                )

                if stop_mode == "frames":
                    self.ui_q.put(("rec_set_target", max_frames))

                if fmt == "ser":
                    meta = SERMeta(
                        observer=settings["metadata"]["observer"],
                        instrument=settings["metadata"]["instrument"],
                        telescope=settings["metadata"]["telescope"],
                    )
                    write_trailer = settings["write_trailer"]

                    with SERWriter(out_path, width=w, height=h, meta=meta, write_trailer=write_trailer) as ser:
                        ser.add_frame_rgb8(first)
                        frames_written = 1
                        now = time.time()
                        last_frame_t = now

                        while not self.stop_record_evt.is_set():
                            if stop_mode == "frames" and frames_written >= max_frames:
                                break
                            if stop_mode == "seconds" and (time.time() - t0) >= max_seconds:
                                break

                            with self.cam_lock:
                                frame = self.picam2.capture_array()

                            if frame.shape[0] != h or frame.shape[1] != w:
                                raise RuntimeError(
                                    f"Frame size changed during recording: got {frame.shape[1]}x{frame.shape[0]}, expected {w}x{h}"
                                )

                            ser.add_frame_rgb8(frame)
                            frames_written += 1

                            now = time.time()
                            dt = now - last_frame_t if last_frame_t else None
                            last_frame_t = now
                            if dt and dt > 0:
                                inst_fps = 1.0 / dt
                                fps_ema = inst_fps if fps_ema == 0 else (0.85 * fps_ema + 0.15 * inst_fps)
                                dt_ema = dt if dt_ema == 0 else (0.85 * dt_ema + 0.15 * dt)

                                if dt_ema > 0 and dt > 1.8 * dt_ema:
                                    missed = int(dt / dt_ema) - 1
                                    if missed > 0:
                                        dropped_est += missed

                            if now - last_ui_t >= 0.2:
                                elapsed = now - t0
                                fps_avg = frames_written / max(1e-9, elapsed)
                                self.ui_q.put(("rec_progress", frames_written, elapsed, fps_ema, fps_avg, dropped_est))
                                last_ui_t = now

                else:
                    play_fps = settings["avi_playback_fps"]
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    vw = cv2.VideoWriter(out_path, fourcc, play_fps, (w, h))
                    if not vw.isOpened():
                        raise RuntimeError("OpenCV VideoWriter could not open AVI output (MJPG).")

                    try:
                        vw.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
                        frames_written = 1
                        now = time.time()
                        last_frame_t = now

                        while not self.stop_record_evt.is_set():
                            if stop_mode == "frames" and frames_written >= max_frames:
                                break
                            if stop_mode == "seconds" and (time.time() - t0) >= max_seconds:
                                break

                            with self.cam_lock:
                                frame = self.picam2.capture_array()

                            if frame.shape[0] != h or frame.shape[1] != w:
                                raise RuntimeError(
                                    f"Frame size changed during recording: got {frame.shape[1]}x{frame.shape[0]}, expected {w}x{h}"
                                )

                            vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                            frames_written += 1

                            now = time.time()
                            dt = now - last_frame_t if last_frame_t else None
                            last_frame_t = now
                            if dt and dt > 0:
                                inst_fps = 1.0 / dt
                                fps_ema = inst_fps if fps_ema == 0 else (0.85 * fps_ema + 0.15 * inst_fps)
                                dt_ema = dt if dt_ema == 0 else (0.85 * dt_ema + 0.15 * dt)

                                if dt_ema > 0 and dt > 1.8 * dt_ema:
                                    missed = int(dt / dt_ema) - 1
                                    if missed > 0:
                                        dropped_est += missed

                            if now - last_ui_t >= 0.2:
                                elapsed = now - t0
                                fps_avg = frames_written / max(1e-9, elapsed)
                                self.ui_q.put(("rec_progress", frames_written, elapsed, fps_ema, fps_avg, dropped_est))
                                last_ui_t = now
                    finally:
                        vw.release()

                dur = max(1e-9, time.time() - t0)
                avg_fps = frames_written / dur
                with open(sidecar, "a", encoding="utf-8") as f:
                    f.write(f"REC_FRAMES={frames_written} / Total frames written\n")
                    f.write(f"REC_DUR={dur:.3f} / Duration [s]\n")
                    f.write(f"REC_FPSAVG={avg_fps:.3f} / Average capture FPS\n")
                    f.write(f"REC_DROPEST={dropped_est} / Dropped-frames estimate (cadence gaps)\n")

                self.ui_q.put(("record_done", idx, out_path, sidecar))

            except Exception as e:
                self.ui_q.put(("record_fail", idx, out_path, str(e)))
            finally:
                self.recording_running = False
                if not self.closing:
                    try:
                        self._configure_camera(profile="preview", bin_factor=settings["preview_binning"])
                        with self.cam_lock:
                            self._set_controls_values(
                                settings["preview_ae_enabled"],
                                settings["preview_exposure_us"],
                                settings["preview_gain"],
                            )
                    except Exception:
                        pass
                if was_preview and self.allow_preview_restart and not self.closing:
                    self.ui_q.put(("start_preview", None))
                self.ui_q.put(("record_btn_reset", None))

        self.record_thread = threading.Thread(target=worker, daemon=True)
        self.record_thread.start()

    def stop_recording(self):
        self.stop_record_evt.set()
        self.rec_status_var.set("Stopping…")
        self.status_var.set("Stopping recording…")

    # ---------- queues + preview ----------
    def _poll_queues(self):
        # Display the newest preview frame.
        try:
            while True:
                rgb = self.frame_q.get_nowait()
                self.last_frame_rgb = rgb
                self._render_frame()
                self.fps_var.set(f"FPS: {self.fps_smooth:.1f}")
        except queue.Empty:
            pass

        # Apply worker messages on Tk's main thread.
        try:
            while True:
                msg = self.ui_q.get_nowait()
                t = msg[0]

                if t == "status":
                    self.status_var.set(msg[1])

                elif t == "error":
                    self.status_var.set("Error.")
                    messagebox.showerror("Error", msg[1])

                elif t == "fps":
                    self.fps_var.set(f"FPS: {float(msg[1]):.1f}")

                elif t == "stop_preview":
                    if self.preview_running:
                        self.stop_preview()

                elif t == "preview_stopped_ui":
                    self.preview_btn.config(text="Start Live")
                    self.fps_var.set("FPS: 0.0")
                    self.status_var.set("Preview paused for capture.")

                elif t == "start_preview":
                    if not self.preview_running and not self.recording_running:
                        self._preview_start_pending = True
                        self.root.after(50, self._start_preview_when_ready)

                elif t == "log_add":
                    _, idx, ft, fmt, st, path = msg
                    self._log_add(idx, ft, fmt, st, path)

                elif t == "saved":
                    _, idx, out_path, rgb = msg
                    self._log_update(idx, "Saved", out_path)
                    self.last_frame_rgb = rgb
                    self._render_frame()
                    self.status_var.set(f"Saved: {os.path.basename(out_path)}")

                elif t == "rec_set_target":
                    target = max(1, int(msg[1]))
                    self.rec_progress_max = float(target)
                    self.rec_prog.configure(maximum=self.rec_progress_max)
                    self.rec_progress_value.set(0.0)
                    self.rec_prog_pct.config(text="0%")

                elif t == "rec_progress":
                    _, frames, elapsed, fps_inst_smooth, fps_avg, dropped_est = msg
                    self.rec_status_var.set(
                        f"Recording… frames={frames}  time={elapsed:.1f}s  instFPS={fps_inst_smooth:.1f}  avgFPS={fps_avg:.1f}  drop≈{dropped_est}"
                    )

                    if self.rec_stop_mode_var.get().strip().lower() == "frames":
                        self.rec_progress_value.set(float(frames))
                        pct = int(min(100, max(0, (float(frames) / max(1.0, self.rec_progress_max)) * 100)))
                        self.rec_prog_pct.config(text=f"{pct}%")

                elif t == "record_done":
                    _, idx, out_path, sidecar = msg
                    self._log_update(idx, "Saved", out_path)
                    self.status_var.set(f"Recording saved: {os.path.basename(out_path)}")
                    self.rec_status_var.set(f"Saved. Header: {os.path.basename(sidecar)}")

                elif t == "record_fail":
                    _, idx, out_path, err = msg
                    self._log_update(idx, "Error", out_path)
                    self.status_var.set("Recording error.")
                    self.rec_status_var.set("Error.")
                    messagebox.showerror("Recording error", err)

                elif t == "record_btn_reset":
                    self.rec_top_btn.config(text="Start recording")
                    self.rec_btn.config(text="Start recording")

        except queue.Empty:
            pass

        if not self.closing:
            self.root.after(40, self._poll_queues)

    # ---------- zoom / render ----------
    def _on_mousewheel_zoom(self, event):
        self._zoom(+1 if event.delta > 0 else -1)

    def _zoom(self, direction: int):
        if self.last_frame_rgb is None:
            return
        step = 1.03
        new_zoom = self.zoom_factor * step if direction > 0 else self.zoom_factor / step
        self.zoom_factor = max(self.min_zoom, min(self.max_zoom, new_zoom))
        self._render_frame()

    def _on_scale_mode_changed(self):
        self.zoom_factor = 1.0
        self.base_scale = None
        self.last_canvas_size = (None, None)
        self.last_scale_mode = None
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self._render_frame()

    def _reset_zoom(self):
        self.zoom_factor = 1.0
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self._render_frame()

    def _render_frame(self):
        if self.last_frame_rgb is None:
            return

        rgb = self.last_frame_rgb
        h, w = rgb.shape[:2]

        canvas_w = self.canvas.winfo_width() or w
        canvas_h = self.canvas.winfo_height() or h
        canvas_size = (canvas_w, canvas_h)

        # Recalculate the base scale after a mode or canvas-size change.
        scale_mode = self.scale_mode_var.get()
        if scale_mode != self.last_scale_mode:
            self.base_scale = None
            self.last_canvas_size = (None, None)
            self.last_scale_mode = scale_mode

        if scale_mode == "1:1":
            self.base_scale = 1.0
            self.last_canvas_size = canvas_size
        else:
            if self.base_scale is None or canvas_size != self.last_canvas_size:
                self.base_scale = min(canvas_w / w, canvas_h / h)
                self.last_canvas_size = canvas_size

        display_scale = float(self.base_scale) * float(self.zoom_factor)
        new_w = max(1, int(w * display_scale))
        new_h = max(1, int(h * display_scale))

        img = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)

        # Apply a fast lookup-table stretch to the preview only.
        try:
            self._ensure_stretch_lut_cached(rgb)
            if self._stretch_lut3_cache:
                img = img.point(self._stretch_lut3_cache)
        except Exception as e:
            self._stretch_lut3_cache = None
            print(f"Preview stretch error: {e}", file=sys.stderr)

        self.tk_image = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        offset_x = max(0, (canvas_w - new_w) // 2)
        offset_y = max(0, (canvas_h - new_h) // 2)
        self.canvas.create_image(offset_x, offset_y, image=self.tk_image, anchor="nw")

        # Draw a centered crosshair over the preview.
        if bool(self.crosshair_var.get()):
            cx = offset_x + (new_w // 2)
            cy = offset_y + (new_h // 2)
            x0, x1 = offset_x, offset_x + new_w
            y0, y1 = offset_y, offset_y + new_h
            self.canvas.create_line(x0, cy, x1, cy, width=1, fill="#00ff00")
            self.canvas.create_line(cx, y0, cx, y1, width=1, fill="#00ff00")

        max_w = max(canvas_w, offset_x + new_w)
        max_h = max(canvas_h, offset_y + new_h)
        self.canvas.configure(scrollregion=(0, 0, max_w, max_h))

        self.zoom_label_var.set(f"Zoom: {self.zoom_factor:.2f}×")

    # ---------- shutdown ----------
    def on_close(self):
        self.closing = True
        self.stop_all()

        deadline = time.time() + 2.0
        threads = (self.preview_thread, self.capture_thread, self.record_thread)
        for thread in threads:
            if thread and thread.is_alive() and thread is not threading.current_thread():
                remaining = deadline - time.time()
                if remaining > 0:
                    thread.join(timeout=remaining)

        workers_alive = any(thread and thread.is_alive() for thread in threads)
        if not workers_alive:
            try:
                with self.cam_lock:
                    if self.started:
                        self.picam2.stop()
                        self.started = False
            except Exception as e:
                print(f"Camera shutdown error: {e}", file=sys.stderr)
        else:
            print("Camera worker still active during shutdown; allowing process exit to release it.", file=sys.stderr)

        self.root.destroy()


def main():
    root = tk.Tk()
    PiAstroCamApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
