"""Live Transcriber - real-time local STT of microphone + system audio (Windows).

Captures the microphone (input) and whatever you hear (output loopback), transcribes
both locally (NVIDIA Parakeet or Whisper) and shows them live. Transcripts are saved to disk.
"""
import os, sys, queue, threading, time, tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from datetime import datetime
from fractions import Fraction

# windowed exe has no stdout/stderr; some libs write to them
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import models
models.add_dll_dirs()   # must run before onnxruntime / ctranslate2 are imported
APP_DIR, MODELS_DIR = models.APP_DIR, models.MODELS_DIR

import traceback
def _crash_log(exc_type, exc, tb):
    try:
        with open(os.path.join(models.DATA_DIR, "app.log"), "a", encoding="utf-8") as f:
            f.write(datetime.now().isoformat() + "\n" + "".join(traceback.format_exception(exc_type, exc, tb)) + "\n")
    except Exception:
        pass
sys.excepthook = _crash_log

import numpy as np
import pyaudiowpatch as pyaudio
from scipy.signal import resample_poly

SR = 16000
SILENCE_END, MAX_SEG, MIN_SEG, PREROLL = 0.8, 25.0, 0.4, 0.3
PARTIAL_EVERY = 0.7   # seconds between live updates while someone is still talking
DEFAULT_SAVE_DIR = os.path.join(os.path.expanduser("~"), "Documents", "LiveTranscriber")

# ---------------------------------------------------------------- engine ----

def to16k(raw, rate, ch):
    x = np.frombuffer(raw, dtype=np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != SR:
        fr = Fraction(SR, int(rate)).limit_denominator(1000)
        x = resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)
    return x


def list_devices():
    pa = pyaudio.PyAudio()
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        mics, loops = [], []
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if d["hostApi"] != wasapi["index"] or d["maxInputChannels"] < 1:
                continue
            (loops if d.get("isLoopbackDevice") else mics).append(d)
        dm = pa.get_device_info_by_index(wasapi["defaultInputDevice"])
        dl = pa.get_default_wasapi_loopback()
        return mics, loops, dm["index"], dl["index"]
    finally:
        pa.terminate()


class Engine:
    def __init__(self, ui_q, model_name, lang, mic_idx, loop_idx, log_path, use_gpu,
                 diarize=False, names=None, thresh=0.62):
        self.ui_q, self.model_name, self.lang = ui_q, model_name, lang
        self.mic_idx, self.loop_idx, self.log_path, self.use_gpu = mic_idx, loop_idx, log_path, use_gpu
        self.want_diar, self.names, self.thresh, self.diar = diarize, names if names is not None else {}, thresh, None
        self.stop_ev = threading.Event()
        self.jobs = queue.Queue()
        self.partial = {}                      # label -> latest in-progress audio (only newest kept)
        self.seg = {"MIC": 0, "SYS": 0}        # utterance ids
        self.pa = None
        self.streams = []

    def post(self, kind, *a):
        self.ui_q.put((kind, *a))

    def start(self):
        threading.Thread(target=self._boot, daemon=True).start()

    def _progress(self, what):
        def cb(done, total):
            pct = f" {100 * done // total}%" if total else ""
            self.post("status", f"Downloading {what} (first run only){pct}  {done / 1e6:.0f} MB")
        return cb

    def _boot(self):
        try:
            dev = "cpu"
            if self.use_gpu:
                ok, why = models.cuda_status()
                if ok:
                    dev = "cuda"
                else:
                    self.post("info", f"GPU not used: {why}. Running on CPU.")
            self.parakeet = self.model_name.startswith("parakeet")
            if self.parakeet:
                import onnx_asr
                self.post("status", f"Preparing {self.model_name}...")
                mdir, quant = models.ensure_parakeet(self.model_name, dev == "cuda", self._progress(self.model_name))
                self.post("status", f"Loading {self.model_name}...")
                prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if dev == "cuda" else ["CPUExecutionProvider"]
                self.model = onnx_asr.load_model(models.PARAKEET[self.model_name][0], mdir,
                                                 quantization=quant, providers=prov)
                self.model.recognize(np.zeros(SR, dtype=np.float32))  # warmup
                self._start_audio(dev)
                return
            from faster_whisper import WhisperModel
            self.post("status", f"Loading Whisper '{self.model_name}' (downloads on first run)...")
            kw = dict(download_root=MODELS_DIR)
            try:
                self.model = WhisperModel(self.model_name, device=dev,
                                          compute_type="float16" if dev == "cuda" else "int8", **kw)
                if dev == "cuda":  # force a CUDA lib load now so failure falls back cleanly
                    list(self.model.transcribe(np.zeros(SR, dtype=np.float32), beam_size=1)[0])
            except Exception as e:
                if dev == "cpu":
                    raise
                self.post("info", f"GPU failed ({str(e)[:80]}); using CPU.")
                dev = "cpu"
                self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8", **kw)
            self._start_audio(dev)
        except Exception as e:
            self.post("error", f"{type(e).__name__}: {e}")

    def _start_audio(self, dev):
        self.dev = dev
        if self.want_diar:
            try:
                from diarizer import Diarizer
                self.post("status", "Preparing speaker model...")
                mp = models.ensure_speaker_model(self._progress("speaker model"))
                self.diar = Diarizer(mp, threshold=self.thresh)
            except Exception as e:
                self.post("info", f"Speaker detection unavailable: {e}")
        self.pa = pyaudio.PyAudio()
        self.streams = [self._source("MIC", self.pa.get_device_info_by_index(self.mic_idx)),
                        self._source("SYS", self.pa.get_device_info_by_index(self.loop_idx))]
        threading.Thread(target=self._transcriber, daemon=True).start()
        self.post("running", dev)

    def stop(self):
        self.stop_ev.set()
        for s in self.streams:
            try:
                s.stop_stream(); s.close()
            except Exception:
                pass
        try:
            self.pa and self.pa.terminate()
        except Exception:
            pass

    def _transcriber(self):
        while not self.stop_ev.is_set():
            try:  # finished utterances always go first
                label, audio, t0, sid = self.jobs.get_nowait()
                final = True
            except queue.Empty:
                final = False
                p = None
                for lb in ("MIC", "SYS"):  # then the freshest in-progress audio (stale ones are dropped)
                    p = self.partial.pop(lb, None)
                    if p:
                        label = lb
                        break
                if not p:
                    time.sleep(0.04)
                    continue
                sid, audio, t0 = p
            try:
                if self.parakeet:
                    text = str(self.model.recognize(audio)).strip()
                else:
                    segs, _ = self.model.transcribe(audio, language=self.lang, vad_filter=True,
                                                    beam_size=1, condition_on_previous_text=False)
                    text = " ".join(s.text.strip() for s in segs).strip()
            except Exception as e:
                text = ""
                self.post("info", f"transcribe error: {e}")
            if not final:
                if text:
                    self.post("partial", label, text, sid)
                continue
            ts = datetime.fromtimestamp(t0).strftime("%H:%M:%S")
            spk = None
            if text and self.diar and label == "SYS":   # the mic is you; only the meeting audio needs splitting
                try:
                    spk = self.diar.identify(audio)
                except Exception as e:
                    self.post("info", f"speaker id failed: {e}")
            if text:
                who = f" {self.names.get(spk, f'Speaker {spk + 1}')}:" if spk is not None else ""
                try:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(f"{ts} [{label}]{who} {text}\n")
                except Exception as e:
                    self.post("info", f"log write failed: {e}")
            self.post("line", ts, label, text, sid, spk)  # empty text just clears the live line

    def _source(self, label, dev):
        rate, ch = int(dev["defaultSampleRate"]), int(dev["maxInputChannels"])
        q = queue.Queue()

        def cb(data, frames, ti, status):
            q.put(data)
            return (None, pyaudio.paContinue)

        stream = self.pa.open(format=pyaudio.paFloat32, channels=ch, rate=rate, input=True,
                              input_device_index=dev["index"], frames_per_buffer=int(rate * 0.03),
                              stream_callback=cb)

        def emit(buf, t0):
            a = np.concatenate(buf)
            if len(a) / SR >= MIN_SEG:
                self.jobs.put((label, a, t0, self.seg[label]))
            else:
                self.post("line", "", label, "", self.seg[label], None)

        def loop():
            buf, pre = [], []
            in_speech, last_voice, seg_start, floor, last_partial = False, 0.0, 0.0, 0.003, 0.0
            while not self.stop_ev.is_set():
                try:
                    raw = q.get(timeout=0.2)
                except queue.Empty:  # loopback sends nothing during silence
                    if in_speech and time.time() - last_voice > SILENCE_END:
                        emit(buf, seg_start); buf, in_speech = [], False
                    continue
                x = to16k(raw, rate, ch)
                rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
                self.post("level", label, rms)
                now = time.time()
                voiced = rms > max(0.006, floor * 3)
                if not voiced:
                    floor = 0.98 * floor + 0.02 * rms
                else:
                    if not in_speech:
                        in_speech, seg_start, buf = True, now, list(pre)
                        self.seg[label] += 1
                        last_partial = now
                    last_voice = now
                if in_speech:
                    buf.append(x)
                    dur = sum(map(len, buf)) / SR
                    if now - last_voice > SILENCE_END or dur > MAX_SEG:
                        emit(buf, seg_start); buf, in_speech = [], False
                    elif now - last_partial >= PARTIAL_EVERY and dur >= 0.6:
                        # live: hand the latest audio of this utterance to the decoder
                        self.partial[label] = (self.seg[label], np.concatenate(buf), seg_start)
                        last_partial = now
                else:
                    pre.append(x); pre[:] = pre[-int(PREROLL / 0.03):]

        threading.Thread(target=loop, daemon=True).start()
        return stream

# ------------------------------------------------------------------- GUI ----

BG, PANEL, FG, MUTED = "#14161a", "#1d2026", "#e6e8eb", "#8b919a"
C_MIC, C_SYS, ACCENT = "#5eead4", "#fbbf24", "#3b82f6"
SPK_COLORS = ["#60a5fa", "#f472b6", "#a3e635", "#fb923c", "#c084fc", "#2dd4bf",
              "#facc15", "#f87171", "#38bdf8", "#4ade80", "#e879f9", "#fda4af"]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Live Transcriber")
        self.geometry("980x560+60+10"); self.minsize(720, 440)
        self.configure(bg=BG)
        self.engine = None
        self.names, self.spk_tags = {}, set()
        self.ui_q = queue.Queue()
        self.levels = {"MIC": 0.0, "SYS": 0.0}
        self._style()
        self._build()
        try:
            self.mics, self.loops, dm, dl = list_devices()
            self.mic_cb["values"] = [f"{d['index']}: {d['name']}" for d in self.mics]
            self.loop_cb["values"] = [f"{d['index']}: {d['name']}" for d in self.loops]
            self.mic_cb.set(next(v for v in self.mic_cb["values"] if v.startswith(f"{dm}:")))
            self.loop_cb.set(next(v for v in self.loop_cb["values"] if v.startswith(f"{dl}:")))
        except Exception as e:
            messagebox.showerror("Audio", f"Could not list audio devices:\n{e}")
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(50, self._pump)
        if "--autostart" in sys.argv:
            self.after(600, self.toggle)

    def _style(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, fieldbackground=PANEL, font=("Segoe UI", 10))
        s.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=FG, arrowcolor=FG)
        s.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", FG)])
        s.configure("TLabel", background=BG, foreground=MUTED)
        s.configure("TCheckbutton", background=BG, foreground=FG)
        s.configure("Go.TButton", background=ACCENT, foreground="white", padding=(18, 7), font=("Segoe UI", 10, "bold"), borderwidth=0)
        s.map("Go.TButton", background=[("active", "#2563eb")])
        s.configure("Stop.TButton", background="#dc2626", foreground="white", padding=(18, 7), font=("Segoe UI", 10, "bold"), borderwidth=0)
        s.map("Stop.TButton", background=[("active", "#b91c1c")])
        s.configure("TButton", background=PANEL, foreground=FG, padding=(10, 5), borderwidth=0)
        s.map("TButton", background=[("active", "#2a2f38")])
        for n, c in (("Mic", C_MIC), ("Sys", C_SYS)):
            s.configure(f"{n}.Horizontal.TProgressbar", troughcolor=PANEL, background=c, borderwidth=0, thickness=6)

    def _build(self):
        top = tk.Frame(self, bg=BG); top.pack(fill="x", padx=16, pady=(14, 6))
        top.columnconfigure(1, weight=1); top.columnconfigure(3, weight=1)
        ttk.Label(top, text="Microphone").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.mic_cb = ttk.Combobox(top, state="readonly"); self.mic_cb.grid(row=0, column=1, sticky="ew", padx=(0, 16))
        ttk.Label(top, text="System audio").grid(row=0, column=2, sticky="w", padx=(0, 8))
        self.loop_cb = ttk.Combobox(top, state="readonly"); self.loop_cb.grid(row=0, column=3, sticky="ew")
        self.mic_bar = ttk.Progressbar(top, style="Mic.Horizontal.TProgressbar", maximum=0.2)
        self.mic_bar.grid(row=1, column=1, sticky="ew", pady=(4, 0), padx=(0, 16))
        self.sys_bar = ttk.Progressbar(top, style="Sys.Horizontal.TProgressbar", maximum=0.2)
        self.sys_bar.grid(row=1, column=3, sticky="ew", pady=(4, 0))

        opt = tk.Frame(self, bg=BG); opt.pack(fill="x", padx=16, pady=6)
        ttk.Label(opt, text="Model").pack(side="left")
        self.model_cb = ttk.Combobox(opt, state="readonly", width=14, values=["parakeet-v2", "parakeet-v3", "small", "base", "medium", "large-v3-turbo"])
        self.model_cb.set("parakeet-v2"); self.model_cb.pack(side="left", padx=(6, 14))
        ttk.Label(opt, text="Language").pack(side="left")
        self.lang_cb = ttk.Combobox(opt, state="readonly", width=8, values=["en", "auto", "es", "de", "fr", "ro"])
        self.lang_cb.set("en"); self.lang_cb.pack(side="left", padx=(6, 14))
        self.gpu_var = tk.BooleanVar(value=models.cuda_status()[0])  # on by default only if a usable NVIDIA GPU is found
        ttk.Checkbutton(opt, text="Use GPU", variable=self.gpu_var).pack(side="left", padx=(0, 14))
        self.scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Auto-scroll", variable=self.scroll_var).pack(side="left", padx=(0, 14))
        self.diar_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Identify speakers", variable=self.diar_var).pack(side="left", padx=(0, 6))
        self.thr_var = tk.DoubleVar(value=0.62)
        ttk.Scale(opt, from_=0.40, to=0.85, variable=self.thr_var, length=90, command=self._thr_changed).pack(side="left")
        self.btn = ttk.Button(opt, text="Start", style="Go.TButton", command=self.toggle)
        self.btn.pack(side="right")

        sv = tk.Frame(self, bg=BG); sv.pack(fill="x", padx=16, pady=(0, 6))
        ttk.Label(sv, text="Save to").pack(side="left")
        self.dir_var = tk.StringVar(value=DEFAULT_SAVE_DIR)
        tk.Entry(sv, textvariable=self.dir_var, bg=PANEL, fg=FG, insertbackground=FG, relief="flat",
                 highlightthickness=0).pack(side="left", fill="x", expand=True, padx=8, ipady=4)
        ttk.Button(sv, text="Browse", command=self._browse).pack(side="left", padx=(0, 6))
        ttk.Button(sv, text="Open folder", command=lambda: os.startfile(self.dir_var.get()) if os.path.isdir(self.dir_var.get()) else None).pack(side="left")

        body = tk.Frame(self, bg=PANEL); body.pack(fill="both", expand=True, padx=16, pady=(0, 6))
        self.txt = tk.Text(body, bg=PANEL, fg=FG, wrap="word", relief="flat", padx=14, pady=10,
                           font=("Segoe UI", 11), spacing3=6, state="disabled", highlightthickness=0, height=6)
        sb = ttk.Scrollbar(body, command=self.txt.yview); self.txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); self.txt.pack(fill="both", expand=True)
        self.txt.tag_configure("ts", foreground=MUTED, font=("Consolas", 9))
        self.txt.tag_configure("MIC", foreground=C_MIC, font=("Segoe UI", 9, "bold"))
        self.txt.tag_configure("SYS", foreground=C_SYS, font=("Segoe UI", 9, "bold"))
        self.txt.tag_configure("info", foreground=MUTED, font=("Segoe UI", 9, "italic"))

        live = tk.Frame(self, bg=PANEL); live.pack(fill="x", padx=16, pady=(0, 6))
        self.live_var, self.live_lbl, self.done = {}, {}, {"MIC": 0, "SYS": 0}
        for name, color in (("MIC", C_MIC), ("SYS", C_SYS)):
            row = tk.Frame(live, bg=PANEL); row.pack(fill="x", padx=14, pady=(6, 0))
            tk.Label(row, text=name, bg=PANEL, fg=color, font=("Segoe UI", 9, "bold"), width=4, anchor="w").pack(side="left")
            self.live_var[name] = tk.StringVar(value="")
            lb = tk.Label(row, textvariable=self.live_var[name], bg=PANEL, fg=MUTED, font=("Segoe UI", 11, "italic"),
                          anchor="w", justify="left", wraplength=820)
            lb.pack(side="left", fill="x", expand=True); self.live_lbl[name] = lb
        tk.Frame(live, bg=PANEL, height=6).pack()
        self.bind("<Configure>", lambda e: [l.configure(wraplength=max(200, self.winfo_width() - 130)) for l in self.live_lbl.values()])

        bot = tk.Frame(self, bg=BG); bot.pack(fill="x", padx=16, pady=(0, 12))
        self.status = tk.StringVar(value="Idle")
        ttk.Label(bot, textvariable=self.status).pack(side="left")
        ttk.Button(bot, text="Clear", command=self._clear).pack(side="right")
        ttk.Button(bot, text="Copy all", command=self._copy).pack(side="right", padx=6)

    def _browse(self):
        d = filedialog.askdirectory(initialdir=self.dir_var.get())
        if d: self.dir_var.set(d)

    def _clear(self):
        self.txt.configure(state="normal"); self.txt.delete("1.0", "end"); self.txt.configure(state="disabled")

    def _copy(self):
        self.clipboard_clear(); self.clipboard_append(self.txt.get("1.0", "end").strip())

    def _spk_name(self, spk):
        return self.names.get(spk, f"Speaker {spk + 1}")

    def _spk_tag(self, spk):
        tag = f"spk{spk}"
        if tag not in self.spk_tags:
            self.spk_tags.add(tag)
            self.txt.tag_configure(tag, foreground=SPK_COLORS[spk % len(SPK_COLORS)], font=("Segoe UI", 10, "bold"))
            self.txt.tag_bind(tag, "<Double-Button-1>", lambda e, s=spk: self._rename(s))
            self.txt.tag_bind(tag, "<Enter>", lambda e: self.txt.configure(cursor="hand2"))
            self.txt.tag_bind(tag, "<Leave>", lambda e: self.txt.configure(cursor=""))
        return tag

    def _rename(self, spk):
        new = simpledialog.askstring("Rename speaker", f"Name for {self._spk_name(spk)}:",
                                     initialvalue=self._spk_name(spk), parent=self)
        if not new or not new.strip():
            return
        self.names[spk] = new.strip()
        tag = self._spk_tag(spk)
        self.txt.configure(state="normal")
        r = self.txt.tag_ranges(tag)
        for i in range(len(r) - 2, -2, -2):  # back to front so earlier ranges stay valid
            self.txt.delete(r[i], r[i + 1]); self.txt.insert(r[i], new.strip(), (tag,))
        self.txt.configure(state="disabled")

    def _append(self, ts, label, text, spk=None):
        self.txt.configure(state="normal")
        self.txt.insert("end", ts + "  ", "ts")
        self.txt.insert("end", label + "  ", label)
        if spk is not None:
            self.txt.insert("end", self._spk_name(spk), (self._spk_tag(spk),))
            self.txt.insert("end", "  ")
        self.txt.insert("end", text + "\n")
        self.txt.configure(state="disabled")
        if self.scroll_var.get(): self.txt.see("end")

    def _info(self, msg):
        self.txt.configure(state="normal"); self.txt.insert("end", msg + "\n", "info")
        self.txt.configure(state="disabled")
        if self.scroll_var.get(): self.txt.see("end")

    def toggle(self):
        if self.engine:
            self._stop(); return
        try:
            mic = int(self.mic_cb.get().split(":")[0]); lp = int(self.loop_cb.get().split(":")[0])
        except Exception:
            messagebox.showerror("Audio", "Pick a microphone and system-audio device."); return
        d = self.dir_var.get()
        try:
            os.makedirs(d, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Save folder", str(e)); return
        path = os.path.join(d, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log")
        lang = self.lang_cb.get(); lang = None if lang == "auto" else lang
        self.names = {}
        self.engine = Engine(self.ui_q, self.model_cb.get(), lang, mic, lp, path, self.gpu_var.get(),
                             diarize=self.diar_var.get(), names=self.names, thresh=self.thr_var.get())
        self.btn.configure(text="Starting...", state="disabled")
        self._info(f"Saving to {path}")
        self.engine.start()

    def _thr_changed(self, _=None):
        d = getattr(self.engine, "diar", None) if self.engine else None
        if d: d.threshold = float(self.thr_var.get())  # live: takes effect on the next utterance

    def _stop(self):
        if self.engine:
            self.engine.stop(); self.engine = None
        self.done = {"MIC": 0, "SYS": 0}
        for v in self.live_var.values(): v.set("")
        self.btn.configure(text="Start", style="Go.TButton", state="normal")
        self.status.set("Stopped"); self.mic_bar["value"] = self.sys_bar["value"] = 0

    def _pump(self):
        try:
            while True:
                kind, *a = self.ui_q.get_nowait()
                if kind == "line":
                    ts, label, text, sid, spk = a
                    self.done[label] = max(self.done[label], sid)
                    self.live_var[label].set("")
                    if text: self._append(ts, label, text, spk)
                elif kind == "partial":
                    label, text, sid = a
                    if sid > self.done[label]:  # ignore late partials of an already-finished utterance
                        self.live_var[label].set(text[-400:])
                elif kind == "info": self._info(a[0])
                elif kind == "status": self.status.set(a[0])
                elif kind == "level": self.levels[a[0]] = max(a[1], self.levels[a[0]] * 0.8)
                elif kind == "running":
                    self.btn.configure(text="Stop", style="Stop.TButton", state="normal")
                    self.status.set(f"Listening  -  {'GPU' if a[0] == 'cuda' else 'CPU'}")
                elif kind == "error":
                    self._info("ERROR: " + a[0]); self._stop(); messagebox.showerror("Error", a[0])
        except queue.Empty:
            pass
        if self.engine:
            self.mic_bar["value"] = self.levels["MIC"]; self.sys_bar["value"] = self.levels["SYS"]
            for k in self.levels: self.levels[k] *= 0.85
        self.after(50, self._pump)

    def _close(self):
        if self.engine: self.engine.stop()
        self.destroy()
        os._exit(0)


if __name__ == "__main__":
    App().mainloop()
