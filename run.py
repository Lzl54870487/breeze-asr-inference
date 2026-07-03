import argparse
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

CT2_MODEL_ID = "SoybeanMilk/faster-whisper-Breeze-ASR-25"


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"  [{self.name}] {elapsed:.2f}s")


def list_devices():
    devices = sd.query_devices()
    print("=== 可用的音訊輸入裝置 ===")
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"  [{i}] {dev['name']} (輸入頻道: {dev['max_input_channels']})")
    print()


def select_device():
    list_devices()
    idx = int(input("請選擇麥克風裝置編號: "))
    return idx


def record_audio(duration, samplerate=16000, device=None):
    print(f"\n開始錄音 {duration} 秒... (請說話)")
    frames = int(duration * samplerate)
    recording = sd.rec(frames, samplerate=samplerate, channels=1, dtype=np.float32, device=device)
    sd.wait()
    print("錄音完成\n")
    return recording.flatten()


def resample(audio, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_len = int(duration * target_sr)
    return np.interp(
        np.linspace(0, len(audio) - 1, target_len),
        np.arange(len(audio)),
        audio,
    )


def load_audio(file_path):
    audio, sr = sf.read(file_path, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample(audio, sr)


def load_model():
    device = "cuda"
    compute_type = "float16"
    model = WhisperModel(CT2_MODEL_ID, device=device, compute_type=compute_type)
    return model


def transcribe(model, audio):
    segments, info = model.transcribe(audio, beam_size=5, language="zh")
    text = "".join(seg.text for seg in segments)
    return text


def main():
    parser = argparse.ArgumentParser(description="Breeze ASR 25 - USB 麥克風語音辨識")
    parser.add_argument("--duration", type=int, default=10, help="錄音秒數 (預設: 10)")
    parser.add_argument("--device", type=int, default=None, help="音訊裝置編號 (預設: 列出裝置供選擇)")
    parser.add_argument("--list-devices", action="store_true", help="列出可用音訊裝置後結束")
    parser.add_argument("--file", type=str, default=None, help="直接辨識音檔 (跳過錄音)")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    print("===== Breeze ASR 25 語音辨識 (faster-whisper) =====")
    total_start = time.perf_counter()

    if args.file:
        with Timer("載入音檔"):
            audio = load_audio(args.file)
    else:
        device = args.device if args.device is not None else select_device()
        with Timer("錄音"):
            audio = record_audio(args.duration, device=device)

    with Timer("載入模型"):
        model = load_model()

    with Timer("模型推論"):
        text = transcribe(model, audio)

    total_elapsed = time.perf_counter() - total_start
    print(f"  ───────────────────────────")
    print(f"  總計: {total_elapsed:.2f}s\n")

    print("=== 辨識結果 ===")
    print(text)
    print("================")


if __name__ == "__main__":
    main()
