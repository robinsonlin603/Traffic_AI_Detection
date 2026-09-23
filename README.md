# 機車行車記錄器 AI 分析

目前提供 Milestone 1：離線 YOLO 偵測、BoT-SORT 持續追蹤、結構化資料及簡單標註影片。舊 Milestone 2 已移除；動態白線與一般換道分為新的 Milestone 2、3，尚未實作。

## 安裝與分析

需要 Python 3.12 以上。FFmpeg 可用於額外影音檢查。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cv,dev]'
dashcam-ai devices
dashcam-ai analyze --input ./samples/test1.mp4
```

自動輸出到 output/test1/。可指定 --output、--config、--model、--imgsz、--confidence、--device，以及 --no-save-video 或 --no-save-frames。

預設設定 configs/default.yaml 自動依 CUDA、MPS、CPU 順序選擇裝置；configs/mac.yaml 使用 MPS，configs/nvidia.yaml 使用 cuda:0。NVIDIA 環境需安裝相容的 PyTorch。分析僅使用既有本機權重，不自動下載。預設 yolo26m.pt 先從目前目錄尋找；若只提供檔名且目前目錄沒有，則從主要 Git 工作目錄尋找，讓獨立 worktree 共用權重。明確指定的路徑不存在時會報錯，不改用其他模型。

## 輸出與標註

標註僅顯示綠色物件框、單行 #ID 類別縮寫、深色文字底板與橘色軌跡。預設只偵測 car、truck、bus、motorcycle，追蹤與輸出時統一為 vehicle；不追蹤 person 或 bicycle。畫面標籤只顯示 #ID，保留標籤避讓；不顯示類別、車道、事件或信心數字。

| 檔案 | 內容 |
|---|---|
| metadata.json | 影片、裝置、套件版本及模型雜湊 |
| frames.jsonl | frame_id、timestamp、objects；包含偵測信心，沒有場景分析欄位 |
| tracks.json | 達到 minimum_track_length 的追蹤摘要 |
| events.json | 空陣列 [] |
| annotated.mp4 | 簡單標註影片 |

--no-save-frames 與 --no-save-video 分別停用逐幀資料與影片輸出。

## 驗證

```bash
pytest
ruff check .
mypy src
dashcam-ai validate --milestone 1 --platform macos-mps
dashcam-ai milestone-status --milestone 1
```

Linux 電腦使用 --platform linux-cuda。各平台必須在同一乾淨來源版本分別驗證，dirty 或 stale 報告不能完成驗收。驗證工具不會 commit 或 push。

## 階段與限制

目前只偵測與追蹤物件，不判斷白線、車道、換道、cut-in、方向燈、距離或責任。短暫遮擋仍可能造成追蹤 ID 改變。

[Milestone 1](docs/MILESTONE_1.md) · [新 Roadmap](docs/ROADMAP.md) · [回退計畫](docs/EXEC_PLAN_RESTORE_MILESTONE1.md) · [平台驗證](validation/README.md) · [舊 Milestone 2 歷史](docs/history/legacy-milestone-2/README.md)

## 小型車輛篩選

detection.minimum_vehicle_area_ratio 預設為 0，不依尺寸排除有動力車輛。detection.duplicate_vehicle_iou_threshold 預設 0.85；同一幀高度重疊的 car、motorcycle、bus、truck 偵測只保留信心較高者。巢狀框另以 0.9 的較小框包含比例與 0.22 的中心距離比例判斷，再統一為 vehicle 送入追蹤。設定均記錄於 metadata.json。追蹤輸出會再次排除巢狀框；只有中間漏掉一至兩幀且端點框高度重疊的新 ID，才會接回原 ID。

1920×1080 影像的門檻為 2073.6 平方像素，按原始解析度比例計算。遠處機車或被遮擋的車輛也可能被排除；車輛達到門檻後才提供給追蹤器。既有追蹤若暫時低於門檻，仍依 BoT-SORT 的遺失追蹤規則保留內部狀態，重新出現時不保證沿用 ID。


日常分析只需輸入與輸出（在專案根目錄、已啟用此專案 Python 環境時）：

```bash
dashcam-ai analyze --input samples/test4.mp4 --output output/test4-size-filter
```

--output 也可省略，自動使用 output/test4。預設使用 yolo26m.pt、imgsz 1280、裝置 auto、停用尺寸篩選及 0.85 重複框 IoU 門檻；--model 與 --config 等覆寫選項仍保留。輸入影片路徑仍須指向實際存在的檔案。
