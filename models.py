"""Model locations, first-run downloads and GPU detection (no machine-specific paths)."""
import ctypes, os, sys, threading, time, urllib.request

WIN = sys.platform == "win32"
try:  # use the bundled CA list so downloads work even on systems without a certificate store
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass
APP_DIR = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
BASE_DIR = getattr(sys, "_MEIPASS", APP_DIR)
if WIN:
    _data_root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
else:
    _data_root = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
DATA_DIR = os.path.join(_data_root, "LiveTranscriber")
MODELS_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

PARAKEET = {  # menu name -> (onnx-asr model id, Hugging Face repo)
    "parakeet-v2": ("nemo-parakeet-tdt-0.6b-v2", "istupakov/parakeet-tdt-0.6b-v2-onnx"),   # English
    "parakeet-v3": ("nemo-parakeet-tdt-0.6b-v3", "istupakov/parakeet-tdt-0.6b-v3-onnx"),   # 25 EU languages
}
SPEAKER_MODEL = "wespeaker_en_voxceleb_resnet34.onnx"
SPEAKER_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
               "speaker-recongition-models/" + SPEAKER_MODEL)


def add_dll_dirs():
    """Make optional GPU libraries visible: <app>/cuda (the GPU pack) or anything already installed."""
    if WIN:
        for d in (APP_DIR, BASE_DIR, os.path.join(APP_DIR, "cuda"), os.path.join(BASE_DIR, "cuda")):
            if os.path.isdir(d):
                try:
                    os.add_dll_directory(d)
                except Exception:
                    pass
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        return
    # Linux: the loader reads LD_LIBRARY_PATH only at process start, so restart once with it set.
    pack = next((d for d in (os.path.join(APP_DIR, "cuda"), os.path.join(BASE_DIR, "cuda")) if os.path.isdir(d)), None)
    if pack and pack not in os.environ.get("LD_LIBRARY_PATH", "").split(":") and not os.environ.get("LT_REEXEC"):
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = pack + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        env["LT_REEXEC"] = "1"
        args = [sys.executable] + (sys.argv[1:] if getattr(sys, "frozen", False) else sys.argv)
        try:
            os.execve(sys.executable, args, env)
        except OSError:
            pass


def cuda_status():
    """(usable, reason). Needs an NVIDIA driver plus CUDA 12 cuBLAS and cuDNN 9 libraries."""
    if WIN:
        load, driver = ctypes.WinDLL, "nvcuda.dll"
        libs = ("cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll", "cufft64_11.dll", "cudnn64_9.dll")
    else:
        load, driver = ctypes.CDLL, "libcuda.so.1"
        libs = ("libcublas.so.12", "libcublasLt.so.12", "libcudart.so.12", "libcufft.so.11", "libcudnn.so.9")
    try:
        load(driver)
    except OSError:
        return False, "no NVIDIA driver found"
    for lib in libs:
        try:
            load(lib)
        except OSError:
            return False, f"{lib} not found (install the optional GPU pack, see README)"
    return True, ""


def find_local(*names):
    """First existing path among the data dir and the folder next to the exe."""
    for root in (MODELS_DIR, os.path.join(APP_DIR, "models")):
        for n in names:
            p = os.path.join(root, n)
            if os.path.exists(p):
                return p
    return None


def _watch(paths_fn, total, progress, stop):
    while not stop.is_set():
        done = sum(os.path.getsize(p) for p in paths_fn() if os.path.exists(p))
        progress(done, total)
        time.sleep(0.7)


def ensure_parakeet(name, use_gpu_weights, progress):
    """Download (first run only) and return (local_dir, quantization).

    GPU -> fp32 weights (~2.4 GB, fastest on CUDA); CPU -> int8 weights (~650 MB).
    Files are written as real files (no symlinks) because onnxruntime refuses symlinked weights.
    """
    from huggingface_hub import HfApi, hf_hub_download
    _, repo = PARAKEET[name]
    quant = None if use_gpu_weights else "int8"
    sfx = "" if quant is None else ".int8"
    files = [f"encoder-model{sfx}.onnx", f"decoder_joint-model{sfx}.onnx", "nemo128.onnx", "vocab.txt", "config.json"]
    if quant is None:
        files.insert(1, "encoder-model.onnx.data")
    d = os.path.join(MODELS_DIR, f"{name}-{'int8' if quant else 'fp32'}")
    marker = os.path.join(d, ".complete")
    if os.path.exists(marker):
        return d, quant
    os.makedirs(d, exist_ok=True)
    try:
        sizes = {s.rfilename: s.size for s in HfApi().model_info(repo, files_metadata=True).siblings}
        total = sum(sizes.get(f, 0) for f in files)
    except Exception:
        total = 0
    stop = threading.Event()
    threading.Thread(target=_watch, daemon=True, args=(
        lambda: [os.path.join(dp, f) for dp, _, fs in os.walk(d) for f in fs], total, progress, stop)).start()
    try:
        for f in files:
            hf_hub_download(repo, f, local_dir=d)
    finally:
        stop.set()
    open(marker, "w").close()
    return d, quant


def ensure_speaker_model(progress):
    p = find_local(SPEAKER_MODEL)
    if p:
        return p
    dst = os.path.join(MODELS_DIR, SPEAKER_MODEL)
    tmp = dst + ".part"
    with urllib.request.urlopen(SPEAKER_URL, timeout=60) as r, open(tmp, "wb") as out:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 16):
            out.write(chunk); done += len(chunk); progress(done, total)
    os.replace(tmp, dst)
    return dst
