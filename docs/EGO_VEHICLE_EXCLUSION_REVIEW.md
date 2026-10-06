# 固定攝影機的自車 ID 排除

2026-10-04 使用者確認攝影機固定並批准：標定左下角自車外殼、只遮車輛偵測副本、追蹤前過濾殘留自車框，重播兩支原片並檢查旁車。本次是局部偵測輸入／過濾修正，採已批准的分段計畫，未另建 ExecPlan。沒有 commit/push、模型替換或標線原型整合。

## 進度與結果

- [x] 檢查 HEAD、dirty 工作樹、現有 M2 計畫與平台紀錄，保存修改前來源。
- [x] 人工檢視兩片第 0、600、1230、1799 幀，固定正規化車體多邊形及 11 個旁車控制。
- [x] 接入偵測副本遮罩、追蹤前過濾、設定與 runtime 記錄。
- [x] 182 個 pytest、Ruff、Mypy（42 個來源檔）、diff 檢查通過。
- [x] 兩支各 1800 幀完整 MPS 重播、逐幀記錄與完整解碼通過；11/11 旁車控制通過、自車形狀框 0，來源／輸出雜湊一致；抽查疊圖通過。

本次批准的自車排除修正已完成，專項檢查 **passed**；M2 整體驗收仍未完成。

| 影片 | 原片／解碼幀數 | 自車形狀框 | 旁車控制 | 控制最低 IoU | 分析秒數 | 實測 FPS |
|---|---:|---:|---:|---:|---:|---:|
| sample_1 | 1800 / 1800 | 0 | 5/5 | 0.9756 | 950.749 | 1.893 |
| sample_2 | 1800 / 1800 | 0 | 6/6 | 0.7500 | 1143.677 | 1.574 |

兩個模型實際 device 均為 `mps`，不只記錄 requested/resolved 值。elapsed 包含讀片、雙推論、追蹤與編碼；第二片期間另有短片整合檢查，所以這不是隔離的硬體效能 benchmark。輸出仍是原片 30fps、60 秒，分析耗時不是播放長度。

預設 CLI 另以 sample_1 第 0..19 幀重新編碼片段完成整合：20 幀／20 白線記錄、影片完整解碼、自車形狀框 0，runtime 顯示 MPS 與 context recovery。只證明正常入口可用，不證明全片白線品質。

完整記錄掃描配合每 60 幀快照及控制影格目視檢查，未見自車追蹤框；兩片 contact sheet 與首幀歷史／新版本比較均保存在 output。停等時部分近距離機車在原偵測流程就有漏檢，本次未調整門檻／追蹤參數，不宣稱全片逐車召回完整。

初步 CPU 同影格、同模型比較：兩片第 0 幀原始影像各有一個左下角自車誤框，信心分別約 0.6375、0.6897；遮罩副本皆沒有該框。這只證明這兩個開發影格，不是整片結論。

## 實作與設定

`src/dashcam_ai/detection/ego_mask.py` 依原圖尺寸將 0..1 多邊形縮放，快取遮罩及積分影像。偵測副本中的車體填入灰色 114；原始輸入不被修改，Analyzer 的影片繪製及白線辨識仍用原圖。

`UltralyticsDetectorTracker._prepare_vehicle_detections` 原本已在 BoT-SORT 分配 ID 前處理候選，新增殘留框過濾。僅當遮罩覆蓋偵測框面積至少 80% 才排除；少量重疊不刪除旁車。既有自車誤框包含大量道路，未必達 80%，因此不能只用框重疊過濾代替偵測前遮罩。

首輪灰色遮罩使 sample_1 第 600 幀右側機車信心由約 0.40 降至 0.31，跌破 0.35 門檻；改填色不能解決。最終另用同權重、同設定在原圖做純偵測，只補回完全不與自車輪廓相交的候選，與遮罩結果共同去重。原圖模型不註冊 tracker，所以不會先消耗 ID。遮罩外候選保留原圖信心；與車體重疊的原圖候選不補回，避免重新帶入自車。代價是每幀增加一次推論及一份模型記憶體。該機車已加入開發回歸控制，沒有稱為未見資料。

三份設定檔已標定目前固定鏡頭，實際像素輪廓近似 `(0,386),(67,402),(172,462),(238,515),(352,578),(453,633),(627,719),(0,719)`，原片為 1280×720。頂點比例與 `ego_vehicle_overlap_threshold` 一併寫入 runtime metadata。未提供多邊形時預設停用，明確設 `ego_vehicle_polygon: []` 也可停用。

此輪廓沿車體外緣保留小幅邊界餘量，人工檢查避開鄰近可見車輛。固定鏡頭仍可能有少量震動；新影片若裁切、鏡頭位置或可見外殼改變，必須重新標定。本次不承諾跨攝影機通用。

## 驗證範圍與證據

HEAD `b0cf8be22d735aeed9d7ce9e7d49cbe8ea441996`、分支 `codex/restore-milestone1`，來源 dirty；既有未提交工作全部保留。實測來源由 run manifest 的 SHA256 識別，HEAD 本身不足以代表此次修正。

最終重播使用正式 Analyzer／YOLO26m／BoT-SORT、imgsz 1280、confidence 0.35，停用白線以單獨驗證車輛 ID；未更動白線辨識算法。兩片各 1800 幀、60 秒。輸出影片仍顯示自車原始外觀，而非灰色遮罩。

`tests/fixtures/ego_vehicle_review.json` 固定 11 個從原片目視確認的旁車控制，以 IoU ≥0.5 檢查有對應追蹤框。它們覆蓋可見鄰車，不是整片人工逐車召回真值，也不要求修正前後數字 ID 相同。自車排除本來就可能改變後續 ID 編號。

`scripts/check_ego_vehicle_review.py` 驗證 manifest complete、來源／輸入／輸出雜湊、完整影格序列與 runtime 輪廓，並列出控制結果。另用固定車體高度、左／下邊緣及 ≥45% 重疊診斷大型自車形狀框；這是 review 輔助，沒有替代正式 80% 過濾條件，也不能只依該數字宣稱全片無所有誤判。需配合原片／疊圖目視。

完整重播產物在 ignored `output/ego-exclusion-mps-v3`。首輪灰色重播在 `output/ego-exclusion-mps-v1` 因旁車退步拒絕並中斷，`v2` 為型別修正前中斷的準備重播，皆不列作最後驗收。先前 CPU 片段在 `output/ego-exclusion-v1`，已中斷且 `complete=false`，不能作完整影片證據。

## 重現

兩片重播命令使用現有 `.venv` 及本機權重，不下載／安裝：

```bash
.venv/bin/dashcam-ai analyze --input samples/sample_1.mp4 \
  --output output/ego-reproduce-sample1 --config output/ego-exclusion-mps-v3/tracking.yaml
.venv/bin/dashcam-ai analyze --input samples/sample_2.mp4 \
  --output output/ego-reproduce-sample2 --config output/ego-exclusion-mps-v3/tracking.yaml

PYTHONPATH=src .venv/bin/python scripts/check_ego_vehicle_review.py \
  --run-dir output/ego-exclusion-mps-v3 --output output/ego-exclusion-mps-v3/review.json

.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src
git diff --check
```

正式 Analyzer 的一般 CLI 命令不會自動產生這個專項 manifest；完整雜湊／解碼守衛重現用本輪 ignored `run.py`（須選新目錄，避免覆蓋）。既有影片、模型、原始 log 與大輸出均 ignored。

## 平台狀態

沙箱內 devices：CPU available、MPS/CUDA unavailable；沙箱外本機 devices：CPU/MPS available、CUDA unavailable。已確認原先「MPS unavailable」是執行環境差異，不能當成本機硬體不可用。本輪完整重播要求實際 resolved_device=mps，但 dirty 專項重播仍不是正式 milestone acceptance。

M2 macOS MPS 正式驗收仍 blocked（驗證入口只支持 M1、來源 dirty；正式 JSON missing），Linux CUDA missing。M1 macOS 16bedd0 原 verdict blocked，對目前 HEAD stale；M1 CUDA missing。legacy M2 macOS cef1205、CUDA c7d77e3 雖原 verdict passed，對新來源全 stale。不能將任何一個舊／其他平台結果套用本輪。先前 M2 CPU／原型報告保留其凍結來源效力，不能代表這次新增自車遮罩設定與重新追蹤的輸出。

## 最終來源與產物雜湊

| 內容 | SHA256 |
|---|---|
| `src/dashcam_ai/detection/ego_mask.py` | `66c624cde214c707d45c5db0c850ca4f3ba84179be2f88563a70930e768e55aa` |
| `src/dashcam_ai/detection/ultralytics.py` | `29a68b34ed0eaba6a9fdaca0b44c30a581166fb16b2a87a8cc3ac7ca3e53100b` |
| `configs/default.yaml` | `78d321a8b5820b53e2fccf50044c32eced3819074e704c82516a43fd6676b6f2` |
| `configs/mac.yaml` | `c798834c56178f3df934d164f941b45483df0ff2582ba28522938da385631347` |
| `configs/nvidia.yaml` | `957570e2afff931126b49127a3e626f1c98d5ecccd72164db1bdd37c9cc84fa5` |
| `tests/fixtures/ego_vehicle_review.json` | `d81d1b548c17c5b8e2ace96b0a27044cc452560e1bd4b96393653fae5288b1a5` |
| `scripts/check_ego_vehicle_review.py` | `6612cc5f0c5bb53c43c0f5ec4c4768a2bcad653108e4383de6f40c8eb227382c` |
| `output/ego-exclusion-mps-v3/manifest.json` | `92c8494623eb410800ba4abd3ec06006a9dd11d59d09fca0a9f4c344eefc7d83` |
| `output/ego-exclusion-mps-v3/review.json` | `a24fe2262e8bb17630fe9cbd5fe9ca370275f00f51c2e6a7354224194afa8a62` |
| `sample_1/annotated.mp4` | `ed8f591da5f2b295f67350414cfdae0ba2a4396b3f419fbe6dc2b3de7a15052f` |
| `sample_2/annotated.mp4` | `f6eea211246bbe33692866b525a4efbfe450f072226dcb9da98d8455389bf235` |
| `yolo26m.pt` | `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7` |

所有來源／原片／模型與各 JSONL、tracks、metadata 的完整雜湊見 manifest。原始影片及模型不加入 Git；review.json 的 gate 只屬於本次固定攝影機的自車診斷及 11 個控制。

本次新追蹤框會改變白線的車框排除輸入，先前白線／原型紀錄不能視為新設定下的品質證明。M2 全片白線品質與 clean 來源的 macOS MPS、Linux CUDA 正式驗收仍需另行重跑。沒有修改其他平台的機器報告。
