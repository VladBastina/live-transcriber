# Live Transcriber

Real-time, **fully local** speech-to-text for Windows. It listens to your **microphone** *and* to **whatever you hear** (meeting audio, video, calls), transcribes both live, tells the people in a meeting apart, and saves everything to a text file. Nothing is sent to any cloud service.

- **Two sources at once:** `MIC` (you) and `SYS` (the other side of the call / any audio playing on your PC)
- **Live streaming text:** words appear while someone is still speaking, then the line is committed
- **Speaker identification:** meeting audio is split into *Speaker 1, Speaker 2, …*; double-click a name to rename it (e.g. "Nadine")
- **Fast models:** NVIDIA **Parakeet TDT 0.6B** (default, very fast, punctuation included) or **Whisper** (small / base / medium / large-v3-turbo)
- **Runs on any Windows PC.** CPU works out of the box; an NVIDIA GPU makes it faster (optional)

## Download & run

1. Go to the **[Releases](../../releases/latest)** page and download **`LiveTranscriber-win64.zip`**.
2. Unzip it anywhere (not inside a protected folder like `Program Files`) and run **`LiveTranscriber.exe`**.
3. Press **Start**. On the **first run only** it downloads the speech model (Parakeet CPU model ≈ 650 MB, speaker model ≈ 25 MB) into `%LOCALAPPDATA%\LiveTranscriber\models`. A progress percentage is shown in the status bar. After that it works offline.

Windows SmartScreen may warn about an unsigned app the first time: *More info → Run anyway*.

**Requirements:** Windows 10/11, 64-bit. About 3 GB free RAM. Roughly 1 GB disk for the CPU model (≈ 3 GB if you use the GPU model).

### Optional: NVIDIA GPU acceleration
If you have an NVIDIA GPU with a recent driver:

1. Download **`LiveTranscriber-gpu-pack.zip`** from the same release.
2. Extract it **into the app folder**, so you get `LiveTranscriber\cuda\*.dll` next to `LiveTranscriber.exe`.
3. Restart the app and keep **Use GPU** ticked. The status bar shows `Listening - GPU`.

If the GPU can't be used for any reason, the app tells you why in the transcript pane and **falls back to CPU automatically**. Use GPU is only pre-ticked when a usable GPU is detected. (The GPU pack contains NVIDIA cuBLAS and cuDNN redistributable libraries, which are subject to NVIDIA's license terms.)

## Using it

| Control | What it does |
|---|---|
| **Microphone / System audio** | Pick the input and the output device to capture. System audio is a *loopback* of the speakers/headphones you choose, so choose the one you're actually listening on. Switching output devices needs Stop → Start. |
| **Model** | `parakeet-v2` (English), `parakeet-v3` (25 European languages, auto-detect), or a Whisper model. Bigger Whisper models are more accurate but slower. |
| **Language** | Used by Whisper models. `auto` detects it. Parakeet-v2 is English only. |
| **Use GPU** | NVIDIA only, see above. |
| **Identify speakers** + slider | Splits the *System audio* stream by voice. Slider right = more willing to create new speakers; left = more willing to merge. Takes effect on the next sentence. |
| **Save to** | Transcripts are saved as `YYYY-MM-DD_HH-MM-SS.log`, default `Documents\LiveTranscriber`. |

The grey strip under the transcript is the **live** text of whoever is currently talking; when they pause it is committed above and written to the log, e.g.

```
17:51:01 [SYS] Speaker 3: That is useful to know, can you share what is blocking the team right now?
17:51:07 [MIC] Understood, I will raise it with the client today.
```

Tip: use a **headset**. With speakers, your microphone also hears the meeting, so the same line shows up under both `MIC` and `SYS`.

### Limitations
- Windows only (uses WASAPI loopback).
- Speaker identification is a best effort, based on voiceprints of each sentence. Very short utterances ("yeah", "okay") are attributed to the previous speaker, people interrupting each other without a pause end up in one line, and similar voices can be merged. Adjust the slider if needed.
- The log keeps the names that were current when each line was written; renaming a speaker updates the on-screen text and later log lines.
- Only English for `parakeet-v2`; for other languages use `parakeet-v3` or Whisper.
- Please make sure you have the right to transcribe the conversations you capture. Many places require telling participants you are recording or transcribing.

## Privacy
Audio is processed entirely on your computer. The app connects to the internet only to download models on first use (Hugging Face for Parakeet/Whisper, GitHub for the speaker model).

## Troubleshooting
- **Nothing appears for `SYS`:** make sure *System audio* is the loopback of the device you are actually listening on, and that something is playing.
- **Crashes / errors:** details are written to `%LOCALAPPDATA%\LiveTranscriber\app.log`.
- **Re-download a model:** delete its folder in `%LOCALAPPDATA%\LiveTranscriber\models`.
- **Slow on CPU:** try the `parakeet-v2` model (fastest on CPU) or a smaller Whisper model, or add the GPU pack.

## Build from source
```powershell
git clone <this repo>
cd live-transcriber
.\build.ps1                       # creates .venv, installs deps, builds dist\LiveTranscriber-win64.zip
python app.py                     # or just run from source after pip install -r requirements.txt
```
Optional GPU pack (needs CUDA 12 cuBLAS and cuDNN 9 DLLs on your machine):
`.\build.ps1 -CudaBin "C:\...\CUDA\v12.x\bin" -CudnnDir "C:\...\cudnn\bin"` produces `dist\LiveTranscriber-gpu-pack.zip`.

## Credits & licenses
App code: MIT (see `LICENSE`). It builds on:
[NVIDIA Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) (CC-BY-4.0) via [onnx-asr](https://github.com/istupakov/onnx-asr) ·
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) / OpenAI Whisper (MIT) ·
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) with a [WeSpeaker](https://github.com/wenet-e2e/wespeaker) speaker model ·
[PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch) · ONNX Runtime · PyInstaller.
