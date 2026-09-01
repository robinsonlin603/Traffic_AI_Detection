# 機車行車記錄器 AI 分析

這是一個以離線處理為優先、朝正式應用環境設計的機車行車記錄器影片分析專案。目前提供車輛偵測、具持續 ID 的 BoT-SORT 物件追蹤、configured multi-lane geometry、ego-motion、一般換道時間線、結構化事件與標註影片。

```text
MP4 -> YOLO + BoT-SORT -> lane geometry -> ego-motion
    -> temporal lane membership -> general lane-change events
    -> JSON / JSONL + annotated MP4
```

Milestone 2 不使用 LLM／VLM，也不判斷 cut-in、方向燈、真實距離、精準 TTC、碰撞風險、法律責任或執法結論。

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

只指定 `samples` 內的影片時，系統會使用 `configs/default.yaml` 自動選擇 CUDA、MPS 或 CPU，並將結果輸出到 `output/<影片檔名>/`：

```bash
dashcam-ai analyze --input ./samples/test1.mp4
```

上述範例的輸出目錄是 `./output/test1/`。如需自訂輸出目錄、模型或推論尺寸，可以使用完整參數：

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

啟用 `lane_geometry` 時，`frames.jsonl` 會包含逐幀 scene analysis，`events.json` 只保存去重後的一般 lane-change event。停用時仍可執行 Milestone 1 perception-only pipeline，此時逐幀 `analysis` 為 `null`，`events.json` 為空陣列。

## Milestone 2 事件狀態

- `candidate`：目標由穩定來源車道接近共享邊界，仍待跨線與目的車道停留證據。
- `confirmed`：目標跨越共享邊界、穩定停留於相鄰目的車道，且 ego-motion／相對運動證據有效。
- `rejected`：目標返回來源車道、候選逾時，或影片結束前未完成換道。
- `unknown`：車道幾何、追蹤觀察或 ego-motion 證據不足，不能安全確認。

事件不是由單一影格決定。Ego-motion 無效時不會產生 confirmed event。影片結束時仍未完成的 candidate 會以明確原因 finalize 為 rejected。

Confirmed event 的主要欄位如下；candidate、rejected 或 unknown 可缺少後段時間：

```json
{
  "event_type": "lane_change",
  "status": "confirmed",
  "track_id": 17,
  "source_lane": "lane_right",
  "target_lane": "lane_center",
  "direction": "left",
  "started_at": 42.7,
  "lane_crossed_at": 43.2,
  "completed_at": 43.8,
  "confidence": 0.86,
  "confidence_breakdown": {
    "timeline": 1.0,
    "membership_stability": 0.8,
    "ego_motion": 0.9,
    "relative_motion": 0.8,
    "overall": 0.86
  },
  "evidence": {
    "boundary_id": "boundary_right",
    "frame_ids": [1281, 1295, 1314]
  }
}
```

## 車道幾何校正

第一版使用 normalized configured lane polygons 與共享 boundaries。數值 `x`、`y` 的範圍為 `0.0` 到 `1.0`，分析時會映射到來源影片的原始解析度。`lateral_order` 必須依畫面由左至右遞增；換道方向由來源與目的 order 決定，不由 bbox 位移直接決定。

三車道範例：

```yaml
lane_geometry:
  enabled: true
  lanes:
    - id: lane_left
      lateral_order: 0
      polygon: [{x: 0.32, y: 0.45}, {x: 0.44, y: 0.45}, {x: 0.10, y: 1.00}, {x: 0.00, y: 1.00}]
    - id: lane_center
      lateral_order: 1
      polygon: [{x: 0.44, y: 0.45}, {x: 0.56, y: 0.45}, {x: 0.90, y: 1.00}, {x: 0.10, y: 1.00}]
  boundaries:
    - id: boundary_left
      left_lane_id: lane_left
      right_lane_id: lane_center
      points: [{x: 0.44, y: 0.45}, {x: 0.10, y: 1.00}]
```

校正時應讓每個 polygon 與 boundary 沿著實際道路車道，並抽查影片開頭、中段、彎道與結尾。configured geometry 不會動態跟隨彎道或鏡頭姿態；其他攝影機位置、安裝角度或道路環境必須重新校正，不能直接沿用既有座標。

## 標註影片圖例

- 黃青色 polygon 與文字：configured lane 與 Lane ID。
- 綠色線：相鄰車道的共享 boundary。
- 綠／黃／紅／灰色矩形：idle／candidate／confirmed／rejected 或 unknown。
- 橘色線：Track bottom-center 歷史軌跡。
- bottom-center 圓點：目前 membership；靠近 boundary 時為黃色，unknown 時為灰色。
- `LANE CHANGE LEFT/RIGHT`：一般換道方向與事件狀態。
- `#ID code`：單行精簡標籤；`C/T/B/M/P/BC` 分別代表 car、truck、bus、motorcycle、person、bicycle。

## 架構

特定函式庫產生的結果會在轉接層邊界被轉換成標準化領域模型。應用層與儲存層只使用領域模型，讓未來可以替換不同的偵測器與追蹤器。整個處理流程均以原始影像座標作為標準座標系。

`metadata.json` 會記錄實際使用的裝置、Python／PyTorch／Ultralytics／OpenCV 版本、模型名稱與本機模型檔案的 SHA-256，以利比較 Mac 與 NVIDIA 電腦的分析結果。

Milestone 1 已完成功能與驗證狀態請參閱 [`docs/MILESTONE_1.md`](docs/MILESTONE_1.md)，目前 Milestone 2 計畫請參閱 [`docs/EXEC_PLAN_MILESTONE2_GENERAL_LANE_CHANGE.md`](docs/EXEC_PLAN_MILESTONE2_GENERAL_LANE_CHANGE.md)，人工驗收格式請參閱 [`docs/MILESTONE2_ACCEPTANCE.md`](docs/MILESTONE2_ACCEPTANCE.md)。

## 測試

```bash
pytest
ruff check .
mypy src
```

核心測試不需要 CUDA、模型權重或網路連線。

## macOS／Linux 跨平台驗證紀錄

專案根目錄的 `AGENTS.md` 定義所有 agent 共用的驗證規則，`validation/` 則保存可透過 Git 在不同電腦間交換的測試證據。平台結果互相獨立：macOS MPS 不證明 Linux CUDA，Linux CUDA 也不證明 macOS MPS；報告只適用於其中記錄的精確 source commit。

在目前平台執行共通 gates 並產生 JSON 與 Markdown 報告：

```bash
dashcam-ai validate --milestone 2 --platform macos-mps
dashcam-ai validate --milestone 2 --platform linux-cuda
```

檢查單一報告是否仍適用於目前 checkout，或彙整 Milestone 2 所需平台：

```bash
dashcam-ai validation-status validation/milestone-2/linux-cuda.json
dashcam-ai milestone-status --milestone 2
```

若報告的 source commit 與目前 commit 不同，狀態會是 `stale`；缺少報告、dirty worktree、缺少工具或 requested accelerator 不可用時會是 `blocked`。工具只產生報告，不會自動 commit 或 push。Linux 與 macOS 的完整 Git handoff 與資料安全規則請參閱 [`validation/README.md`](validation/README.md)。

## 限制與驗收狀態

- Configured polygon 不會隨彎道、坡度、鏡頭姿態或道路幾何自動改變。
- Homography 可能在低紋理、夜間、雨天或大量動態物體時失敗；品質不足會輸出 unknown，而非 confirmed event。
- Track ID switch 可能切斷 temporal evidence；目前不包含跨 ID re-identification。
- 本版本只判斷 configured lanes 之間的一般換道，不判斷是否進入機車車道或 cut-in。
- Synthetic integration test 已覆蓋一般 confirmed lane-change 與事件 artifact；真實影片仍需依 `docs/MILESTONE2_ACCEPTANCE.md` 人工抽查。
- 既有 macOS MPS 與 Linux CUDA 報告若 source commit 不同即為 stale，必須在最終乾淨 commit 分別重跑。
