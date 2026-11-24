"""Audio capture for Windows (WASAPI incl. loopback) and Linux (PulseAudio / PipeWire).

Public API (identical on both platforms):
    list_devices()  -> (mics, loops, default_mic_id, default_loop_id)
                       each device is {"id", "name"}; "loops" capture what you hear (output monitor)
    open_input(dev_id, callback) -> handle with .rate, .ch, .close()
                       callback(raw: bytes) gets interleaved float32 PCM, ~30 ms at a time
    shutdown()
"""
import subprocess, sys, threading

WIN = sys.platform == "win32"

# ------------------------------------------------------------------ Windows --
if WIN:
    import pyaudiowpatch as pyaudio

    _pa = None

    def _get_pa():
        global _pa
        if _pa is None:
            _pa = pyaudio.PyAudio()
        return _pa

    def list_devices():
        pa = pyaudio.PyAudio()
        try:
            wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            mics, loops = [], []
            for i in range(pa.get_device_count()):
                d = pa.get_device_info_by_index(i)
                if d["hostApi"] != wasapi["index"] or d["maxInputChannels"] < 1:
                    continue
                (loops if d.get("isLoopbackDevice") else mics).append({"id": d["index"], "name": d["name"]})
            dm = pa.get_device_info_by_index(wasapi["defaultInputDevice"])["index"]
            dl = pa.get_default_wasapi_loopback()["index"]
            return mics, loops, dm, dl
        finally:
            pa.terminate()

    class _Handle:
        def __init__(self, stream, rate, ch):
            self.stream, self.rate, self.ch = stream, rate, ch

        def close(self):
            try:
                self.stream.stop_stream(); self.stream.close()
            except Exception:
                pass

    def open_input(dev_id, callback):
        pa = _get_pa()
        dev = pa.get_device_info_by_index(int(dev_id))
        rate, ch = int(dev["defaultSampleRate"]), int(dev["maxInputChannels"])

        def cb(data, frames, ti, status):
            callback(data)
            return (None, pyaudio.paContinue)

        s = pa.open(format=pyaudio.paFloat32, channels=ch, rate=rate, input=True,
                    input_device_index=dev["index"], frames_per_buffer=int(rate * 0.03), stream_callback=cb)
        return _Handle(s, rate, ch)

    def shutdown():
        global _pa
        try:
            _pa and _pa.terminate()
        except Exception:
            pass
        _pa = None

# -------------------------------------------------------------------- Linux --
else:
    def _pactl(*args):
        return subprocess.run(["pactl", *args], capture_output=True, text=True, timeout=10).stdout

    def list_devices():
        """PulseAudio/PipeWire 'sources': real inputs are microphones, '*.monitor' sources are loopbacks."""
        desc, names = {}, []
        cur = None
        for line in _pactl("list", "sources").splitlines():
            s = line.strip()
            if s.startswith("Name:"):
                cur = s.split(":", 1)[1].strip(); names.append(cur)
            elif s.startswith("Description:") and cur:
                desc[cur] = s.split(":", 1)[1].strip()
        mics = [{"id": n, "name": desc.get(n, n)} for n in names if not n.endswith(".monitor")]
        loops = [{"id": n, "name": desc.get(n, n)} for n in names if n.endswith(".monitor")]
        dm = _pactl("get-default-source").strip()
        dl = _pactl("get-default-sink").strip() + ".monitor"
        ids = {d["id"] for d in mics}
        if dm not in ids and mics:
            dm = mics[0]["id"]
        if dl not in {d["id"] for d in loops} and loops:
            dl = loops[0]["id"]
        if not mics or not loops:
            raise RuntimeError("No PulseAudio/PipeWire sources found. Is `pactl` installed "
                               "(package pulseaudio-utils) and a sound server running?")
        return mics, loops, dm, dl

    class _Handle:
        rate, ch = 16000, 1   # parec resamples/downmixes for us

        def __init__(self, proc):
            self.proc = proc

        def close(self):
            try:
                self.proc.terminate(); self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    def open_input(dev_id, callback):
        proc = subprocess.Popen(
            ["parec", "-d", str(dev_id), "--raw", "--format=float32le", "--rate=16000",
             "--channels=1", "--latency-msec=30"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        chunk = int(16000 * 0.03) * 4     # 30 ms of float32 mono

        def pump():
            while True:
                data = proc.stdout.read(chunk)   # blocks until a full chunk (or EOF)
                if not data or len(data) < chunk:
                    return
                callback(data)

        threading.Thread(target=pump, daemon=True).start()
        return _Handle(proc)

    def shutdown():
        pass
