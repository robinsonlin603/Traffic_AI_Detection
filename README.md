# 機車行車記錄器 AI 分析

目前提供 Milestone 1 的離線 YOLO 偵測與 BoT-SORT 追蹤，以及 Milestone 2 的 YOLOP 動態道路白線疊圖。一般換道屬於 Milestone 3，尚未實作。

## 安裝與分析

需要 Python 3.12 以上。FFmpeg 可用於額外影音檢查。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cv,dev]'
mkdir -p models
curl -L --fail --output models/yolop-640-640.onnx \
  https://github.com/hustvl/YOLOP/raw/main/weights/yolop-640-640.onnx
dashcam-ai devices
dashcam-ai analyze --input ./samples/test1.mp4
```

自動輸出到 output/test1/。可指定 --output、--config、--model、--imgsz、--confidence、--device，以及 --no-save-video 或 --no-save-frames。

預設設定 configs/default.yaml 自動依 CUDA、MPS、CPU 順序選擇裝置；configs/mac.yaml 使用 MPS，configs/nvidia.yaml 使用 cuda:0。NVIDIA 環境需安裝相容的 PyTorch。分析僅使用既有本機權重，不自動下載。車輛模型預設為 yolo26m.pt；白線模型預設為 models/yolop-640-640.onnx。明確指定的路徑不存在時會報錯，不改用其他模型。ONNX 權重已由 `.gitignore` 排除，不應提交到 Git。

## 輸出與標註

標註顯示動態白色道路曲線、綠色物件框、單行 #ID、深色文字底板與橘色軌跡。預設只偵測 car、truck、bus、motorcycle，追蹤與輸出時統一為 vehicle；不追蹤 person 或 bicycle。畫面標籤只顯示 #ID，保留標籤避讓；不顯示類別、車道歸屬、事件或信心數字。

白線偵測由 YOLOP 的車道線與可行駛區域分割提出候選。此 ONNX 輸出已經過 Sigmoid，兩類分數先除以總和，0.5 表示前景分數高於背景；相對分數不等同經校準的正確率。minimum_drivable_probability 預設為 0.5，候選中心支持不足時需由兩側道路證據補足。原圖漆面的主方向、寬度膨脹與非細長形狀協助排除箭頭；道路／非道路的兩側差異協助拒絕路緣。漆面形狀在車框裁切前辨識，超過半數被遮擋的元件不作箭頭形狀證據；內部寬度突變補足粗箭身的判斷。淺斜曲線需能延伸回畫面中的道路遠端，以排除橫跨行車方向的停止線，近垂直真線仍保留。紅／黃色路緣須同時有一側道路支持不足才拒絕。候選先通過道路與漆面檢查，再以雙向垂直連接誤差判斷能否合併；低於原本跨度門檻的片段需有端點鄰近遮擋、可見長度及上下文支持，避免放行孤立小圖案。

大型貼底角落車框以高分數道路區域細化，其餘車框仍完整遮擋；偵測與繪圖共用同一像素遮罩，避免自車誤框清空可見道路。白線遮罩本身不刪除車框或 ID；自車 ID 由下述追蹤前排除處理。候選需確認兩幀，跨幀以共同高度內的位置與方向配對，只在共同觀測高度內平滑，位移不超過當前候選半寬，新出現的端點保留當幀位置。低道路支持的候選若只有一端缺乏白漆，會嘗試裁去該端再驗證。短暫缺失最多沿用一幀，並以灰色顯示；未確認、衰減後低信心或與新白線重疊的曲線不會沿用。模型是在 BDD100K 道路資料上訓練，台灣道路、機車鏡頭位置及極端逆光仍可能產生領域差異。

| 檔案 | 內容 |
|---|---|
| metadata.json | 影片、裝置、套件版本及模型雜湊 |
| frames.jsonl | frame_id、timestamp、objects；包含偵測信心，沒有場景分析欄位 |
| lane-lines.jsonl | 每幀白線狀態、曲線與確認／沿用幀數，以及模型、形狀、像素數、候選數與淘汰原因診斷 |
| tracks.json | 達到 minimum_track_length 的追蹤摘要 |
| events.json | 空陣列 [] |
| annotated.mp4 | 車輛追蹤與動態白線標註影片 |

--no-save-frames 與 --no-save-video 分別停用逐幀資料與影片輸出。

### 固定攝影機的自車排除

目前三份設定檔的 `detection.ego_vehicle_polygon` 已標出本專案影片左下角自車外殼；頂點是 0..1 的影像比例。偵測用影像副本在此區填入灰色，原始輸出畫面與白線輸入仍保留原圖。進入 BoT-SORT 前，遮罩覆蓋偵測框至少 `ego_vehicle_overlap_threshold`（預設 0.8）才排除殘留誤框；局部重疊的旁車仍保留。為避免遮罩改變整張影像的偵測信心，另用同一權重在原圖做一次純偵測，補回完全不與自車輪廓相交的候選，再一起去重。原圖偵測器不分配 ID。這會增加一次推論；設定、門檻與補回狀態會寫入 runtime metadata。

換用不同鏡頭位置時必須重新標定輪廓；設為 `ego_vehicle_polygon: []` 可停用。這個輪廓適用於目前固定鏡頭，不是所有行車紀錄器的共用遮罩。

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

目前偵測與追蹤物件並畫出道路白線，但不建立車道歸屬，也不判斷換道、cut-in、方向燈、距離或責任。YOLOP 的訓練場景與本地機車行車記錄器仍有差異；Milestone 2 已完成兩支各 60 秒實拍的成對 CPU 白線重播，但完整品質與跨平台驗收尚未通過。短暫遮擋仍可能造成追蹤 ID 改變。

[Milestone 1](docs/MILESTONE_1.md) · [Milestone 2](docs/EXEC_PLAN_MILESTONE2_DYNAMIC_WHITE_LINES.md) · [白線聚焦驗證](docs/MILESTONE2_FOCUSED_REVIEW.md) · [v5 修正重播](docs/MILESTONE2_V5_REVIEW.md) · [v7 穩定性驗證](docs/MILESTONE2_STABILITY_REVIEW.md) · [v8 人工回報驗證](docs/MILESTONE2_HUMAN_REVIEW.md) · [左白漆恢復驗證](docs/MILESTONE2_LEFT_PAINT_REVIEW.md) · [公車格排除驗證](docs/MILESTONE2_BUS_BAY_REVIEW.md) · [新 Roadmap](docs/ROADMAP.md) · [平台驗證](validation/README.md) · [舊 Milestone 2 歷史](docs/history/legacy-milestone-2/README.md)

## 小型車輛篩選

detection.minimum_vehicle_area_ratio 預設為 0，不依尺寸排除有動力車輛。detection.duplicate_vehicle_iou_threshold 預設 0.85；同一幀高度重疊的 car、motorcycle、bus、truck 偵測只保留信心較高者。巢狀框另以 0.9 的較小框包含比例與 0.22 的中心距離比例判斷，再統一為 vehicle 送入追蹤。設定均記錄於 metadata.json。追蹤輸出會再次排除巢狀框；只有中間漏掉一至兩幀且端點框高度重疊的新 ID，才會接回原 ID。

1920×1080 影像的門檻為 2073.6 平方像素，按原始解析度比例計算。遠處機車或被遮擋的車輛也可能被排除；車輛達到門檻後才提供給追蹤器。既有追蹤若暫時低於門檻，仍依 BoT-SORT 的遺失追蹤規則保留內部狀態，重新出現時不保證沿用 ID。


日常分析只需輸入與輸出（在專案根目錄、已啟用此專案 Python 環境時）：

```bash
dashcam-ai analyze --input samples/test4.mp4 --output output/test4-size-filter
```

--output 也可省略，自動使用 output/test4。預設使用 yolo26m.pt、imgsz 1280、裝置 auto、停用尺寸篩選及 0.85 重複框 IoU 門檻；--model 與 --config 等覆寫選項仍保留。輸入影片路徑仍須指向實際存在的檔案。
