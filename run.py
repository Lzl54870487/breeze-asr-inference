import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import sounddevice as sd
import torch
import torchaudio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutomaticSpeechRecognitionPipeline,
)

MODEL_ID = "MediaTek-Research/Breeze-ASR-25"


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


def load_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy()


def load_model():
    print("正在下載/載入 Breeze ASR 25 模型... (第一次執行需下載約 3GB)")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("使用 GPU 加速")
    else:
        print("使用 CPU 推論 (速度較慢)")
    model.eval()
    return processor, model


def transcribe(processor, model, audio):
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=0,
    )
    result = pipe(audio, return_timestamps=True)
    return result["text"]


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

    if args.file:
        print(f"載入音檔: {args.file}")
        audio = load_audio(args.file)
    else:
        device = args.device if args.device is not None else select_device()
        audio = record_audio(args.duration, device=device)

    processor, model = load_model()
    text = transcribe(processor, model, audio)

    print("=== 辨識結果 ===")
    print(text)
    print("================")


if __name__ == "__main__":
    main()
