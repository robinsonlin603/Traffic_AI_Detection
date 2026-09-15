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
- 產生包含邊界框、類別、追蹤 ID 與精簡類別標籤的標註影片。
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

目前 Apple Silicon Mac 已確認 CPU 與 MPS 可用，CUDA 不可用符合預期。真實行車影片已在 MPS 完成 YOLO／BoT-SORT 分析，並確認逐幀資料、軌跡摘要與標註影片均完整產生。

標註影片亦已完成人工視覺抽查，確認下列項目沒有明顯問題：

- 物件邊界框位置合理。
- Track ID 與物件移動情況相符。
- 移動軌跡線能跟隨對應物件。
- 標註影片可完整播放，開頭、中段與結尾畫面正常。

下列實機工作仍待完成：

- 在 RTX 4070 SUPER 驗證 CUDA 推論。
- 使用同一影片比較 MPS 與 CUDA 的輸出結果。
- 比較長影片的處理速度、資源使用及穩定性。

## 2026-09-03 恢復 Milestone 1

舊 Milestone 2 實作已移除。現在只畫綠色物件框、單行 #ID 類別縮寫、深色標籤底板、底部軌跡點及橘色軌跡；保留標籤避讓，不顯示信心數字、車道或事件橫幅。偵測信心仍保存於資料中。

frames.jsonl 每筆只有 frame_id、timestamp、objects；events.json 固定為 []。保留最短追蹤長度修正、進度顯示與自動輸出路徑。先前章節的 MPS 實片結果是歷史紀錄，不代表此次修改已獲雙平台驗收。

新階段見 [Roadmap](ROADMAP.md)。舊紀錄見 [歷史封存](history/legacy-milestone-2/README.md)。

## 本次回退驗證結果（2026-09-03）

55 項 pytest、Ruff、strict Mypy（36 個來源檔案）通過，git diff --check 通過。
以既有 yolo26m.pt、imgsz 1280 在 CPU 完整分析 test1.mp4，產生並成功逐幀解碼 625 幀 1920×1080 影片；88 筆符合最短長度的追蹤摘要，0 個事件。全部逐幀資料只含 frame_id、timestamp、objects。

人工畫面抽查影格 0、312、624，確認保留綠色框、#ID 類別縮寫、深色底板、標籤避讓與橘色軌跡，沒有車道／事件疊圖。此抽查驗證回退後輸出樣式，不代表偵測準確率驗收；模型仍會把畫面左下自車儀表部位誤認為 car，偵測模型本次未改動。

輸入 SHA-256：4b4d7a363070066011b80e3b777ff51a0bd16bf0bc5e81986f9413a05785a964。模型 SHA-256：401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7。

macOS MPS：blocked，本機執行環境無法使用 MPS，且工作目錄包含本次未提交修改。Linux CUDA：missing，未在該平台重跑。CPU 實片成功僅證明本次 dirty worktree 的 CPU 行為，不能完成 clean-commit 雙平台驗收。機器產生報告見 ../validation/milestone-1/macos-mps.json。


## 2026-09-03 小型車輛追蹤前篩選

使用者核准以框占畫面比例排除過小車輛，保留主要車輛。此為局部變更，依已核准的三步計畫完成尺寸量測、實作、短片與自動測試驗證。

新增 detection.minimum_vehicle_area_ratio，預設 0.001（0.1%），0 停用。依原始影像面積篩選 car/motorcycle/bus/truck，在 Ultralytics 追蹤 callback 執行前移除過小偵測；不改 person/bicycle，不合併類別或重複框。runtime metadata 保存門檻。

基準資料 frame 116：原 #40 約 23.2×29.9 px、面積占比 0.0334%；原 #3 約 0.7712%、原 #5 約 2.8669%。使用相同既有影片及模型（SHA-256 見上節），在 CPU 對 frame 110–125 共 16 幀分別以門檻 0 與 0.001 分析。檢查追蹤前 callback，確認框尚無 ID 且所有車輛均符合面積門檻。frame 116 偵測由 8 筆降為 6 筆，原 #40 對應目標及另一過小車框被排除，原 #3/#5 對應車輛保留。已目視檢查該幀；短片從中途開始，ID 不與全片編號對齊。

61 項 pytest、Ruff、strict Mypy（36 source files）通過。macOS MPS 報告重新產生，source_commit 為 5fdf4e6def033d5021e94197f80331adc30749b2，但 dirty worktree 且 MPS unavailable，故 blocked；Linux CUDA missing，兩 GPU 平台均待乾淨來源版本實機重跑。未重跑完整影片，未 commit/push。

限制：此為大小篩選，並非距離估計；小型機車或遮擋車輛可能被排除。門檻附近的物件可能斷續提供給追蹤器；追蹤器仍保留其既有 lost-track 狀態，重現不保證 ID 延續。不同鏡頭或影片仍需調整門檻。


## 2026-09-03 預設本機模型解析

使用者核准簡化 analyze，只需 --input、--output（輸出仍可省略）。此局部改動不需 ExecPlan。預設模型檔名先解析目前目錄，其次透過 Git common-dir 找主要工作目錄的既有權重；明確路徑必須存在，不靜默替換，不自動下載。保留 --model、--config 與其他覆寫選項。

64 項 pytest、Ruff、strict Mypy（36 來源檔）通過；涵蓋目前目錄優先、worktree 共用權重、明確路徑及缺失錯誤。實際從現有影片取一幀，以僅 --input 和 --output 的 CLI 完成 CPU 分析，metadata 確認 auto 選 CPU、既有模型雜湊與 0.001 尺寸門檻。此 smoke test 僅驗證簡化入口，不代替完整影片或 GPU 驗收。

本機報告 source_commit 仍為 5fdf4e6def033d5021e94197f80331adc30749b2，工作樹未提交且 MPS unavailable，macOS MPS blocked；Linux CUDA missing，兩平台待乾淨版本重跑。未 commit/push。

## 2026-09-15 移除人物追蹤

預設偵測類別移除 person，人物不再送入 BoT-SORT，也不會產生人物 ID、軌跡或逐幀物件資料。保留 car、motorcycle、bus、truck、bicycle。此變更可減少騎士與機車同時標註，但模型若將人物誤判成 motorcycle，仍可能形成重複 M 框；重複框處理不在本次範圍。

設定測試與完整 64 項 pytest、Ruff、strict Mypy（36 來源檔）通過，git diff --check 通過。本次依使用者縮減後的核准範圍未執行實拍影片分析，macOS MPS 與 Linux CUDA 報告均未更新；未 commit/push。
