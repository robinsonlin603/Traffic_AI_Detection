# 機車行車記錄器 AI 分析

這是一個以離線處理為優先、朝正式應用環境設計的機車行車記錄器影片分析專案。目前階段提供車輛偵測、具持續 ID 的 BoT-SORT 物件追蹤、軌跡資料、結構化分析成果及標註影片。車道變換、切入事件與方向燈判定將於後續里程碑實作。

## 系統需求

- Python 3.12 以上
- 建議安裝 FFmpeg，用於檢查及驗證影音檔案
- Apple Silicon Mac 可使用 MPS
- NVIDIA RTX 電腦可使用 CUDA
- CPU 是所有平台共用的備援模式

## 安裝

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cv,dev]'
```

第一次使用模型名稱執行 Ultralytics 時，系統可能會自動下載模型權重。如需完全離線執行，可以透過 `--model` 指定本機模型權重路徑。

## 推論裝置

未指定裝置時，`auto` 會依照 CUDA、MPS、CPU 的順序自動選擇。可先檢查目前電腦支援的裝置：

```bash
dashcam-ai devices
```

Apple Silicon Mac：

```bash
dashcam-ai analyze \
  --input ./samples/ride.mp4 \
  --output ./output/ride-mac \
  --config ./configs/mac.yaml
```

NVIDIA RTX 4070 SUPER：

```bash
dashcam-ai analyze \
  --input ./samples/ride.mp4 \
  --output ./output/ride-nvidia \
  --config ./configs/nvidia.yaml
```

CPU 備援模式：

```bash
dashcam-ai analyze \
  --input ./samples/ride.mp4 \
  --output ./output/ride-cpu \
  --device cpu
```

NVIDIA 電腦應先依照其驅動與 CUDA 環境安裝相容的 PyTorch，再安裝本專案。兩台電腦應使用相同的 Python 主版本、設定檔及模型權重。

## 使用方式

```bash
dashcam-ai analyze \
  --input ./samples/ride.mp4 \
  --output ./output/ride \
  --model yolo26m.pt \
  --imgsz 1280
```

預設設定位於 `configs/default.yaml`。分析完成後，輸出目錄包含：

```text
metadata.json
frames.jsonl
tracks.json
events.json
annotated.mp4
```

目前階段尚未實作事件偵測，因此 `events.json` 會是一個空陣列。

## 架構

特定函式庫產生的結果會在轉接層邊界被轉換成標準化領域模型。應用層與儲存層只使用領域模型，讓未來可以替換不同的偵測器與追蹤器。整個處理流程均以原始影像座標作為標準座標系。

`metadata.json` 會記錄實際使用的裝置、Python／PyTorch／Ultralytics／OpenCV 版本、模型名稱與本機模型檔案的 SHA-256，以利比較 Mac 與 NVIDIA 電腦的分析結果。

Milestone 1 已完成功能與驗證狀態請參閱 [`docs/MILESTONE_1.md`](docs/MILESTONE_1.md)，已核准的里程碑計畫請參閱 [`docs/EXEC_PLAN.md`](docs/EXEC_PLAN.md)。

## 測試

```bash
pytest
ruff check .
mypy src
```

核心測試不需要 CUDA、模型權重或網路連線。
