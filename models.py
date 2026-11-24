"""Model locations, first-run downloads and GPU detection (no machine-specific paths)."""
import ctypes, os, sys, threading, time, urllib.request

APP_DIR = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
BASE_DIR = getattr(sys, "_MEIPASS", APP_DIR)
DATA_DIR = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "LiveTranscriber")
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
    """Make optional GPU libraries visible: <app>\\cuda (the GPU pack) or anything already installed."""
    for d in (APP_DIR, BASE_DIR, os.path.join(APP_DIR, "cuda"), os.path.join(BASE_DIR, "cuda")):
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def cuda_status():
    """(usable, reason). Needs an NVIDIA driver plus CUDA 12 cuBLAS and cuDNN 9 DLLs."""
    try:
        ctypes.WinDLL("nvcuda.dll")
    except OSError:
        return False, "no NVIDIA driver found"
    for dll in ("cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll", "cufft64_11.dll", "cudnn64_9.dll"):
        try:
            ctypes.WinDLL(dll)
        except OSError:
            return False, f"{dll} not found (install the optional GPU pack, see README)"
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
