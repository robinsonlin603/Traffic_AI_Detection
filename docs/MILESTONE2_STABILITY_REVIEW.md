# Milestone 2 曲線穩定性驗證（v7）

2026-10-02。本輪完成固定案例及白線後處理修正；**不是 Milestone 2 正式驗收，也沒有開始 Milestone 3**。基準 HEAD 為 `b0cf8be22d735aeed9d7ce9e7d49cbe8ea441996`，工作樹包含先前及本輪未提交修改，不能把 HEAD 當成本輪執行來源。

## 修正與原因

1. **4.27 秒的錯誤彎線（frame 128）**：中央短線 `(611,404)→(678,421)` 與右側候選 `(858,457)→(1042,545)` 被合併。舊版沿用約 102 px 的時間配對容許值作為片段連接門檻，約 98 px 的連接誤差仍被接受；合併後又繼承已確認身分。現在先檢查個別片段的道路／箭頭上下文，再以雙向、垂直於曲線的連接誤差確認是否可合併。
2. **跨幀變形**：配對除了共同高度內的位置距離，也檢查方向差異，拒絕不相容曲線繼承舊身分。平滑的水平位移限制在當前候選標線半寬內，新增端點不被舊端點拉偏。正常延長與遮擋縮短的配對測試仍通過。
3. **43.5 秒左線（frame 1305）**：模型候選前半段落在柏油，後半段有連續白漆；整段低道路支持度讓可見真線一起被刪除。現在僅在這種淘汰原因下，嘗試裁去缺乏漆面支持的端部，再重新檢查。沒有插補被遮擋或模型未偵測到的線。
4. **短斜線與箭頭**：低漆面支持的短斜線，還須結合附近已辨識的圖案才能拒絕；幾乎完全沒有白漆支持時也拒絕。試作曾因單用明暗支持比例而漏掉第二支影片的暗色左線，已撤回該過度篩選方式，加入控制案例。近垂直真車道線仍保留。

沒有修改車輛偵測／追蹤器、延長漏線保留時間、增加新模型或實作換道事件。

## 固定案例

座標來自 1280×720 原片，frame_id 從 0 開始，30 fps；標註點與輸出曲線最近距離在 10 px 內算命中。只比較人工確認的可見位置，避開車框遮擋；不是逐像素 precision/recall，也不評估整段車道是否連續。

`tests/fixtures/lane_stability_review.json`：sample_2 的 13 個案例。

| 時間／frame | 目標 | v6 命中點 | v7 命中點 |
|---|---|---:|---:|
| 1.00 s / 30 | 保留左線，2 點 | 1 | 2 |
| 4.27 s / 128 | 拒絕錯誤合併線，3 點 | 3 | 0 |
| 14.00 s / 420 | 拒絕停止線，3 點 | 0 | 0 |
| 20.00 s / 600 | 拒絕左箭頭，2 點 | 0 | 0 |
| 21.00 s / 630 | 保留近垂直車道線，2 點 | 2 | 2 |
| 28.00 s / 840 | 拒絕黃路緣，2 點 | 0 | 0 |
| 40.00、41.00 s / 1200、1230 | 保留左線，各 1 點 | 1、1 | 1、1 |
| 43.50 s / 1305 | 保留可見左線尾段，2 點 | 0 | 2 |
| 45.00、51.00、54.00 s / 1350、1530、1620 | 保留左線，各 1 點 | 1、1、1 | 1、1、1 |
| 55.00 s / 1650 | 拒絕分離箭頭，2 點 | 0 | 0 |

短片段重播結果為 v6 10/13、v7 13/13。完整重播驗證結果另記於本文件末尾。

42–44 秒的原因並不相同：frame 1260 原片沒有清楚可見左線，不能計為召回失敗；1275 有真左線候選；1290 左側可見候選很短、白漆與道路支持不足，沒有足夠證據補線；1305 是本輪修復的錯誤拒絕；1320 有正常左線候選。修正不代表這兩秒內每幀都能畫出左線。

`tests/fixtures/lane_stability_control.json`：sample_1 的 frame 100、110、112 暗色左線控制案例，3/3 通過。這些案例是在試作暴露退步後加入，且 sample_1 過去已使用過，不能當作未見驗收集。

## 執行與檢查

- sample_2 使用相同原片與 v5 既有車框，重新執行白線辨識；**未重新執行 YOLO 車輛偵測或 BoT-SORT**。
- sample_1 比較 90–119、600–629、1200–1229 共 90 幀；兩版都沒有車框，使用相同模型輸出。前後皆為 29 valid、61 unknown；人工抽查及 3 個點位控制確認左線保留。valid 數量相同不代表品質等同。路面文字及停車格仍有誤畫，這是剩餘限制。
- 40 項白線單元測試通過；完整 pytest **121 passed**；`ruff check .` 通過；`mypy src` 通過（40 個來源檔）；`git diff --check` 通過。
- 7 項新回歸／控制測試用凍結 v6 實作重跑，6 個錯誤案例失敗、正常延長／遮擋控制通過；修正版全部通過。對舊平滑函式只適配參數數量，未改算法；其位移為 26 px，超出測試的 8 px 上限。
- 點位檢查器在 v6 得到 10/13 並回傳非零狀態，在修正版固定案例通過。缺少影格與輸入雜湊不符會拒絕驗證。

本機重現（需既有原片、模型、v5 車框與 v6 產物；大檔皆不提交）：

```sh
PYTHONPATH=src .venv/bin/python output/lane-review-v7/replay.py --out output/lane-review-v7/reproduce
.venv/bin/python scripts/check_lane_review.py \
  --input samples/sample_2.mp4 \
  --cases tests/fixtures/lane_stability_review.json \
  --baseline output/lane-review-v6/lane-lines.jsonl \
  --records output/lane-review-v7/reproduce/lane-lines.jsonl \
  --output output/lane-review-v7/reproduce/verification.json
```

診斷及重播腳本位於 ignored 的 `output/lane-review-v7`，不包含於乾淨 checkout；固定標註及點位比較器保留於專案。`final` 是已棄用的過度篩選試作；最終完整影片使用 `final2`，第二支控制影片使用 `sample1-final`。

## 來源與平台證據

| 項目 | SHA-256 |
|---|---|
| v6 segmentation.py | `6ccb88504db410e5f78256e6fb3601cc423741bcc0b547e36f074e22830df106` |
| v7 segmentation.py | `3fcc6b2e0fc6eeba9a0ac1bfb0443d954617320a4d33d165f01a54bb18f69a44` |
| sample_2.mp4 | `48f0e3010d7acffebf95b8bd55e1193589785a3d8835ec528040f0298edbf4ba` |
| sample_1.mp4 | `220611d110329da7a5ad89405c475ed4104c4c5c8eae6eb0ec8f34e0d6f0ea2d` |
| YOLOP ONNX | `cd66a3e0087a7258ae07768cc02cb742eed93865727ae4c9baf969b8fa190696` |
| v5 車框 frames.jsonl | `276d0524a0609797b8086c3c5e8015ee158e23bf82c107761ba8cacf30165667` |

本輪來源未提交；上述程式雜湊僅識別本機修正版本，不取代乾淨 source_commit。

| 階段／平台 | 狀態 | 證據 |
|---|---|---|
| M1 macOS MPS | stale | 現存報告來源 `16bedd00158f53e6329260e61119859a1e7eced1` 與 HEAD 不同，原判定為 blocked |
| M1 Linux CUDA | missing | 沒有目前 M1 平台報告 |
| M2 macOS MPS | blocked；正式報告 missing | 本輪 devices 顯示 MPS 不可用；validate --milestone 2 回報目前僅支援 milestone 1 |
| M2 Linux CUDA | missing | 未在 Linux 執行；macOS 結果不能轉移 |
| 本機 CPU 白線檢查 | 限定案例已通過 | OpenCV DNN CPU、dirty 工作樹、重用車框；不是正式平台驗收 |

舊 M2 歷史報告來源 macOS `cef1205`、Linux `c7d77e3` 均 stale，且屬舊階段定義。本輪未改寫任何平台的既有機器報告。

進入 M3 前仍需要完整標註的品質評估，以及相同乾淨來源的 macOS MPS／Linux CUDA 整合驗證。本輪固定案例只證明列出的錯誤已改善，不能推出所有箭頭、路緣及左線問題均已消失。

## 完整重播結果

完整 sample_2 重播 **1800/1800 幀**；新版與左右對照影片都已逐幀解碼，皆為 30 fps、60 秒。sample_1 對照影片也完整解碼 90/90 幀。完整重播的固定案例仍為 **13/13 通過**，第二支影片控制案例 **3/3 通過**。

sample_2 状態分布：v6 valid/degraded/unknown = 402/569/829；v7 = 413/573/814。這些是輸出狀態統計，不能當作準確率。

本輪程式／設定／測試來源集合雜湊：`08fab337872e33677fefac9ea128c70214a0b3101ad80d46705d7313fbefa0e6`；逐檔來源、輸入與輸出雜湊見本機 `output/lane-review-v7/final2/manifest.json`。

| 本機產物 | SHA-256 |
|---|---|
| `output/lane-review-v7/final2/annotated.mp4` | `c9cc6efa7579f5463580179c959eaeabcff7c2a55e7f5462e33c4ac0cf3508ab` |
| `output/lane-review-v7/final2/comparison.mp4` | `766be1dd4b54c015926dddde85504417a8acc309ebb6a25e7331afc1070e71d6` |
| `output/lane-review-v7/sample1-final/comparison.mp4` | `75d460f934c0121cbe7444e08bd1a6a1d941e28cce69aa471049578566f18b50` |
