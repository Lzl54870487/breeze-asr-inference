# breeze-asr-inference — Breeze ASR 25 語音辨識

基於聯發科 **Breeze ASR 25** 模型的離線語音辨識工具，支援 USB 麥克風即時收音與音檔辨識，使用 CTranslate2 推論引擎達到 GPU 加速。

---

## 環境需求

- **OS**: Windows 10 / 11
- **GPU**: NVIDIA GPU (CUDA 12+)，建議 VRAM 4GB 以上
- **套件管理**: [pixi](https://pixi.sh)
- **Git**: 2.x

---

## 快速開始

```bash
# 安裝依賴
pixi install

# 執行語音辨識（互動模式：選麥克風 → 錄音 10 秒 → 辨識）
pixi run python run.py
```

首次執行會自動從 HuggingFace 下載 Breeze ASR 25 的 CTranslate2 模型（約 3GB）。

---

## 使用方式

| 指令 | 說明 |
|---|---|
| `pixi run python run.py` | 互動模式：列出裝置 → 選擇麥克風 → 錄音 10 秒 → 辨識 |
| `pixi run python run.py --duration 5` | 自訂錄音秒數 |
| `pixi run python run.py --device 4` | 指定音訊裝置編號，跳過選擇步驟 |
| `pixi run python run.py --file audio.wav` | 直接辨識音檔（跳過錄音） |
| `pixi run list-devices` | 列出所有可用的音訊輸入裝置 |

### 參數說明

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--duration` | 10 | 錄音秒數 |
| `--device` | 無 | 音訊裝置編號（預設為互動選擇） |
| `--list-devices` | — | 列出裝置後結束 |
| `--file` | 無 | 指定音檔路徑（跳過錄音） |

---

## 依賴說明

### conda 依賴

| 套件 | 用途 |
|---|---|
| `python >=3.11,<3.13` | 執行環境 |
| `pip` | Python 套件安裝器 |
| `python-sounddevice >=0.5.1` | USB 麥克風音訊擷取（底層 portaudio） |
| `numpy >=1.26.0` | 音訊資料陣列處理與重採樣 |

### pip 依賴

| 套件 | 用途 |
|---|---|
| `faster-whisper >=1.1.0` | CTranslate2 推論引擎（含模型載入、特徵提取、解碼） |
| `soundfile >=0.12.0` | 音檔讀取（支援 wav, flac, ogg 等格式） |

---

## 模型資訊

| 項目 | 內容 |
|---|---|
| **原始模型** | [MediaTek Research Breeze ASR 25](https://huggingface.co/MediaTek-Research/Breeze-ASR-25) |
| **基礎架構** | OpenAI Whisper-large-v2（1.5B 參數） |
| **推論格式** | CTranslate2（[SoybeanMilk/faster-whisper-Breeze-ASR-25](https://huggingface.co/SoybeanMilk/faster-whisper-Breeze-ASR-25)） |
| **量化精度** | FP16（可選 INT8） |
| **推論設備** | NVIDIA GPU（CUDA） |
| **授權** | Apache 2.0 |

### 模型特色

- 針對**台灣國語口音**進行微調
- 強化**中英夾雜（Code-switching）**辨識，準確度提升 56%
- 強化時間戳記對齊，適合字幕生成
- 中文訓練資料完全使用合成語音（BreezyVoice TTS）

### 論文

> **A Self-Refining Framework for Enhancing ASR Using TTS-Synthesized Data**
>
> Cheng-Kang Chou\*, Chan-Jan Hsu\*, Ho-Lam Chung, Liang-Hsuan Tseng, Hsi-Chun Cheng, Yu-Kuan Fu, Kuan-Po Huang, Hung-yi Lee
>
> arXiv: 2506.11130（2025）
>
> https://arxiv.org/abs/2506.11130

---

## 效能

### 實測數據（RTX 3070 Ti, test.wav, 約 10 秒音檔）

| 環節 | 舊版（HF Transformers） | 新版（faster-whisper） | 加速倍數 |
|---|---|---|---|
| 載入音檔 | 0.02s | 0.01s | — |
| 載入模型 | 6.94s | 4.03s | 1.7x |
| **模型推論** | **16.33s** | **0.99s** | **16.5x** |
| 總計 | 23.31s | **5.03s** | **4.6x** |

### 加速原理說明

加速來自三個層面：

1. **CTranslate2 推論引擎** — 底層使用 C++ 實作，避免 Python 逐層執行的 overhead，並針對 Transformer 結構進行 kernel 融合與記憶體優化。

2. **FP16 半精度運算** — 模型權重儲存為 FP16（16-bit 浮點數），相較 FP32 減少 50% 記憶體頻寬需求，Volta 以上 GPU 有專用 Tensor Core 加速 FP16 運算。

3. **INT8 量化支援** — 若改用 `compute_type="int8_float16"` 可再提升約 30% 速度（本專案預設 FP16 以維持最佳辨識品質）。

---

## 實作細節

### Windows CP950 編碼處理

Windows 終端機預設使用 cp950 編碼，部分音訊裝置名稱含特殊字元（如 `®`）會導致 `UnicodeEncodeError`。`run.py` 啟動時將 stdout 重新設定為 UTF-8 編碼來解決此問題：

```python
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
```

### OpenMP 衝突處理

`faster-whisper` 使用 LLVM OpenMP（`libomp.dll`），但 conda 環境中的 Intel MKL 使用 `libiomp5md.dll`，載入時會發生衝突。設定環境變數跳過此檢查：

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

此變數僅影響 OpenMP 初始化檢查，不影響效能與正確性。

---

## 專案架構

```
breeze-asr-inference/
├── .git/                  # Git 版本控制
├── .gitignore             # 排除 .pixi/、__pycache__、*.wav 等
├── .pixi/                 # pixi 虛擬環境（自動產生，不納入版本控制）
├── LICENSE                # Apache 2.0 授權
├── pixi.toml              # 專案環境設定檔
├── pixi.lock              # 依賴鎖定檔（pixi 自動管理）
├── README.md              # 本文件
├── run.py (126 lines)     # 主程式
└── test.wav               # 測試音檔（供 --file 模式使用）
```

### `run.py` 架構

```
main()
├── 解析命令列參數
├── 載入音檔（soundfile）或麥克風錄音（sounddevice）
├── 載入模型（WhisperModel, faster-whisper）
├── 模型推論（transcribe, beam_size=5, language=zh）
└── 輸出結果 + 各環節計時
```

---

## Git 提交歷史

```
154d8ce docs: add Apache 2.0 license
51b001a docs: add README with project overview, usage guide, and performance benchmarks
959f43f feat: switch to faster-whisper (CTranslate2) inference engine
e4752ac feat: add Timer class to profile each stage (audio load, model load, inference)
f8dab5b fix: resolve torchaudio incompatibility and cp950 encoding on Windows
740f582 chore: initial commit with pixi env and USB microphone STT
```

---

## 授權

- **模型權重**：[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)（MediaTek Research Breeze ASR 25）
- **專案程式碼**：[Apache 2.0](LICENSE)
