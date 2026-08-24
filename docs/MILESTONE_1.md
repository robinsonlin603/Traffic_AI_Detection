# Milestone 1：車輛偵測與追蹤基礎流程

Milestone 1 建立了 Motorcycle Dashcam AI 的第一條可用分析流程。使用者可以輸入 MP4 行車影片，透過 YOLO 偵測車輛、使用 BoT-SORT 維持物件 ID，並取得軌跡資料、結構化分析結果與標註影片。

```text
MP4 影片
  -> YOLO 車輛偵測
  -> BoT-SORT 物件追蹤
  -> 軌跡歷史
  -> JSON／JSONL 分析結果
  -> 標註 MP4 影片
```

## 已完成功能

### 專案與領域基礎

- 建立 Python 3.12 專案、設定模型與結構化日誌。
- 定義與特定電腦視覺函式庫解耦的領域模型及 Protocol 介面。
- 將原始影像座標設為標準座標系，並提供可逆的推論影像座標轉換。
- 透過轉接層將 Ultralytics 結果轉換成專案內部的標準資料模型。

### 車輛偵測與物件追蹤

- 整合 Ultralytics YOLO 車輛偵測。
- 使用 BoT-SORT 為跨影格物件維持持續 ID。
- 累積每個追蹤物件的位置、信心分數及軌跡歷史。
- 提供不依賴模型權重的假資料轉接器，讓核心流程可以穩定測試。

### 影片與分析產物

- 使用 OpenCV 讀取 MP4 影片並保留原始解析度資訊。
- 產生包含邊界框、類別、追蹤 ID 與信心分數的標註影片。
- 將逐影格資料串流寫入 JSONL，避免長影片必須將所有影格保留在記憶體中。
- 產生追蹤摘要、事件占位資料及執行環境 metadata。

每次分析的輸出目錄包含：

```text
metadata.json
frames.jsonl
tracks.json
events.json
annotated.mp4
```

目前尚未實作事件判定，因此 `events.json` 會是空陣列。

## CLI 操作

分析影片：

```bash
dashcam-ai analyze \
  --input ./samples/ride.mp4 \
  --output ./output/ride \
  --model yolo26m.pt \
  --imgsz 1280
```

檢查目前電腦可用的推論裝置：

```bash
dashcam-ai devices
```

## CPU、MPS 與 CUDA 支援

預設裝置為 `auto`，系統會依序選擇：

1. NVIDIA CUDA
2. Apple Metal Performance Shaders（MPS）
3. CPU

裝置設定會驗證格式、索引與實際可用性，並在指定裝置不可用時提供明確錯誤。專案另外提供：

- `configs/default.yaml`：自動選擇裝置。
- `configs/mac.yaml`：Apple Silicon MPS 設定。
- `configs/nvidia.yaml`：NVIDIA CUDA 設定。

## 執行紀錄與可重現性

`metadata.json` 會記錄下列資訊，方便比較不同電腦或不同執行環境的結果：

- 使用者指定與實際解析後的裝置。
- CPU、MPS 或 CUDA 裝置名稱。
- Python、PyTorch、CUDA、Ultralytics 與 OpenCV 版本。
- 模型名稱及本機模型檔案的 SHA-256。
- 推論影像尺寸與信心分數門檻。

比較 Mac 與 NVIDIA 電腦的分析結果時，應使用相同影片、Python 主版本、設定檔及模型權重。

## 驗證狀態

Milestone 1 已涵蓋下列自動化驗證：

- 領域模型、幾何與座標轉換。
- 裝置選擇、格式驗證及可用性錯誤。
- 軌跡歷史與序列化。
- 假資料分析流程整合測試。
- OpenCV 暫存 MP4 讀寫與標註影片輸出。
- CLI 與 runtime metadata。

目前 Apple Silicon Mac 已確認 CPU 與 MPS 可用，CUDA 不可用符合預期。下列實機工作仍待完成：

- 使用真實行車影片驗證 MPS 上的 YOLO／BoT-SORT。
- 在 RTX 4070 SUPER 驗證 CUDA 推論。
- 使用同一影片比較 MPS 與 CUDA 的輸出結果。
- 比較長影片的處理速度、資源使用及穩定性。

## 不在 Milestone 1 範圍內

Milestone 1 專注於穩定的偵測、追蹤與資料輸出基礎。以下功能將在後續階段處理：

- Phase 2：車道幾何、自車運動、lane change 與 cut-in。
- Phase 3：方向燈辨識與事件融合。
- Phase 4：FastAPI、GPS 整合與事件影片片段。
- Phase 5：VLM／LLM 分析與進階語意功能。

完整的執行決策與技術範圍請參閱 [Execution Plan](./EXEC_PLAN.md)。
