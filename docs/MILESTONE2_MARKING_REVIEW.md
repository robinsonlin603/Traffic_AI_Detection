# Milestone 2：標線修正 review

2026-10-04。四類問題已有局部改善；本輪仍需通過連續影格品質目標，
不能以單元測試通過宣稱 Milestone 2 完成。

## 比較條件與範圍

HEAD 為 `b0cf8be22d735aeed9d7ce9e7d49cbe8ea441996`，工作樹 dirty。
修改前 detector 保存於 ignored `output/lane-review-v8/baseline/segmentation.py`，
SHA-256 為 `3fcc6b2e0fc6eeba9a0ac1bfb0443d954617320a4d33d165f01a54bb18f69a44`。
最後 detector SHA-256：
`8cc370af35596a9252cc998bcd161ed42edea188d5afa6a8ef5cb6b6aed9c246`。

兩支影片各 1280×720、30 fps、1800 幀，YOLOP ONNX 沿用本機既有模型，
設定不變；修改前／後共享每幀相同模型輸出與各自完整 v7 的固定車框。
CPU 白線重播不重新偵測車輛，不證明 YOLO/BoT-SORT 或 MPS/CUDA 品質。

原片 SHA-256：

| 輸入 | SHA-256 |
|---|---|
| sample_1 | `220611d110329da7a5ad89405c475ed4104c4c5c8eae6eb0ec8f34e0d6f0ea2d` |
| sample_2 | `48f0e3010d7acffebf95b8bd55e1193589785a3d8835ec528040f0298edbf4ba` |
| YOLOP ONNX | `cd66a3e0087a7258ae07768cc02cb742eed93865727ae4c9baf969b8fa190696` |
| configs/default.yaml | `67d92221459a41762119453498ba4ebef3af6915ca181f8157ec0e6c3d1bc4ce` |
| 連續標註 fixture | `fb9690e1bd7119c4443c35ad695c4f612e3eb835c94765ab5949e862116bbbcb` |

完整來源、車框與產物雜湊見兩份 replay manifest / summary；HEAD 不能替代
dirty 來源的雜湊。原片、權重、大型影片與中間診斷留於 ignored output。

## 實作

辨認相連的停車格轉角、附近分離格線及成組寬行穿線，在合併前排除
候選，合併後重新檢查。當幀圖案排除證據也會阻止舊曲線短暫延續。
遠處小箭頭需有寬度展開證據，不能把所有小漆片都當圖案。

對語意道路分數偏低的真線，要求沿線有窄白漆、兩側鄰近柏油支持，
再保留淺斜或短片段，局部亮斑比較鄰近背景。暖色白漆的路緣排除例外
在原有黃色路緣控制造成退步，已收回；保留原有彩色路緣規則。
未改模型、設定、缺失壽命或確認幀數。

新增相容診斷欄位，記錄上下文、合併、長度、時間確認及延續的淘汰原因。
沒有修改車輛偵測／追蹤或其他 milestone 行為。

## 量測

55 個案例為人工確認的可見白漆折線或非車道圖案區域，使用局部影像
運動輔助對齊並逐個區段檢查。每組含 5 個相鄰影格；20 個正例、35 個
負例。保留區段與開發區段分開，保留區段未參與參數調整。它們仍來自
已用過的兩支影片；停車格保留區段是同一格線的不同時刻，獨立性有限，
不能稱為未見資料集 precision / recall。

正例折線每約 2 px 採樣，距離實際可見輸出中心線 10 px 內算命中，
每案例目標至少 90%。人工折線只含可見白漆；模型車框遮罩只裁切輸出，
不刪除人工真值分母。負例計算實際 5 px 白筆畫含抗鋸齒邊緣與區域的
交集，目標 0 像素。最長漏判時間只涵蓋已標註的 5 幀區段。

開發標註校正：sample_1 第 100 幀白漆實際端點為 (281,415)，原先
(330,402) 位於柏油；已依原片校正。保留區段沒有更改。

## 最後結果：品質 failed，M2 未完成

| 量測項目 | 修改前 | 最後版 | 結論 |
|---|---:|---:|---|
| sample_1 可見左線長度覆蓋 | 81.08% | 93.69% | 每個案例仍須至少 90%，尚未全部通過 |
| sample_2 可見左線長度覆蓋 | 29.32% | 56.71% | 每個案例仍須至少 90%，尚未全部通過 |
| 箭頭誤畫幀 | 3/10 | 0/10 | 通過已標註區段 |
| 停車格誤畫幀 | 9/10 | 5/10 | 仍失敗 |
| 行穿線誤畫幀 | 2/10 | 0/10 | 通過已標註區段 |
| 額外路面文字控制誤畫幀 | 4/5 | 0/5 | 通過已標註區段 |

兩片合計 55 個連續案例，修改前 21/55、最後版 39/55 通過。這不是整片影片的
準確率；目前仍有 16 個案例失敗。原有單點控制 16/16 通過，不能取代整段驗收。

| 分區 | sample_1 修改前 → 後 | sample_2 修改前 → 後 |
|---|---:|---:|
| 開發區段 | 8/20 → 19/20 | 3/15 → 10/15 |
| 保留區段 | 5/10 → 5/10 | 5/10 → 5/10 |

已知失敗與診斷：

- sample_1 第 93..97 幀（3.10..3.23 秒）：停車格仍誤畫，最後區域交集
  為 2968、2935、2916、3148、3268 像素，後兩幀甚至較基準多。原片的
  格線與附近文字形成 22 個簡化輪廓頂點，現有單一簡單轉角假設未辨認。
  此為保留區段，未根據它調整參數。
- sample_1 第 102 幀（3.40 秒）：左線覆蓋 73.02%，低於 90%；同組長度
  加權平均 93.69% 不能讓這個失敗影格通過。
- sample_2 第 58..62 幀（1.93..2.07 秒）：可見左線仍為 0% 覆蓋。58..61
  幀有相符模型候選，但被 colored_curb_context 拒絕；第 62 幀的車框遮罩
  完全覆蓋已標註可見白漆。暖白漆例外曾使真黃色路緣誤畫，已收回。
- sample_2 第 1228..1232 幀（40.93..41.07 秒）：保留左線區段覆蓋依序為
  83.61%、55.74%、85.00%、52.46%、69.84%，平均 69.28%，仍未改善。
  車框遮罩覆蓋部分實際可見白漆，整段量測會計為漏畫。

最長連續低於目標的已標註區段：sample_1 為 1 幀（0.033 秒），
sample_2 為 5 幀（0.167 秒）。這不是整片最長漏判時間。

兩支最後版本各有 1800 幀 lane 記錄，annotated.mp4 與 comparison.mp4
共四支影片都逐幀解碼到 1800 幀；manifest.complete=true，執行期間來源未變。

| 本機產物 | SHA-256 |
|---|---|
| sample1-final-v6/annotated.mp4 | `7da29fef6a6210cb4225323b3b9d71bc666968544af8018ea97ecdb48694ceb8` |
| sample1-final-v6/comparison.mp4 | `5aaa70b866aa7704163446f9a53b3e8042efe616bf11e63911b0cd6f77de9688` |
| sample1-final-v6/lane-lines.jsonl | `30cb220cddb2310819f5a7878643f5c452a7123eccd610330bd0b7ce64f150d1` |
| sample2-final-v6/annotated.mp4 | `d2e675675699ceea5f6fe5aeb4150c62f49dbe828c8d276b88cd6e47f93209aa` |
| sample2-final-v6/comparison.mp4 | `4eb1897ea3bc1a4bf671555418b7316bcc557a12bb6191377564196b500c430d` |
| sample2-final-v6/lane-lines.jsonl | `1f6b548f1842e45999a129e5341fe1a1b01586f903803f56b5663f2731ed51b9` |

完整明細：`output/lane-review-v8/sample1-v6-all.json`、
`sample2-v6-all.json`，開發與保留區段另有各自報告。
最後畫面每 30 幀輸出 snapshot；視覺檢查不能外推到未標註區域的 0 誤判。

## 一般與整合檢查

- pytest：141 passed。
- Ruff：passed。
- Mypy：passed，40 個來源檔案。
- diff whitespace 檢查：passed。
- 新增 8 個標線問題回歸案例在凍結 v7 failed、修正版 passed，包含鏡像
  行穿線、分離停車格、舊曲線延續、小箭頭及淺斜真線；另有左右鏡像的
  窄黃色路緣負例控制。
- 量測工具另測試整段覆蓋、錯誤遮罩、白筆畫邊緣、缺幀、重複幀、
  輸入雜湊不符與品質失敗非零返回。
- 正式 analyze 入口 CPU smoke：sample_1 原片 98..127 幀重新編碼的
  30 幀片段，YOLO imgsz=1280，重新跑 YOLO/BoT-SORT，不重用車框。
  30 幀完成、影片逐幀解碼、frame/lane 記錄各 30 幀、新診斷可解析。
  最後來源執行約 44.293 秒；這不是原片完整整合品質或 GPU 驗收。

## 平台與尚未達成項目

目前執行環境 MPS unavailable；validate 入口只支援 M1，M2 命令 exit 2。
macOS MPS 為 blocked、正式 JSON missing；Linux CUDA missing。
M1 macOS 16bedd0 與 legacy M2 cef1205/c7d77e3 均 stale，不能套用目前來源。
詳見 [本機阻塞紀錄](../validation/milestone-2/macos-mps.md)。

完整正式 analyze 品質驗收須待標線品質通過；本輪 smoke 只證明入口相容。
來源未提交、工作樹未 clean，正式雙平台驗收不能完成。沒有 commit/push，
也沒有改寫其他平台機器報告。

## 重現

從 repository root 執行，使用既有 .venv、原片、ONNX 與 v7 車框。
重播輸出目錄必須不存在；要再次執行請更換輸出目錄名稱。

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_lane_review.py \
  --input samples/sample_1.mp4 \
  --boxes output/sample_1_milestone2_yolop_v7/frames.jsonl \
  --baseline-source output/lane-review-v8/baseline/segmentation.py \
  --output output/lane-review-v8/sample1-final-v6

PYTHONPATH=src .venv/bin/python scripts/replay_lane_review.py \
  --input samples/sample_2.mp4 \
  --boxes output/sample_2_milestone2_yolop_v7/frames.jsonl \
  --baseline-source output/lane-review-v8/baseline/segmentation.py \
  --output output/lane-review-v8/sample2-final-v6

.venv/bin/python scripts/check_lane_review.py \
  --input samples/sample_1.mp4 \
  --cases tests/fixtures/lane_markings_review.json \
  --baseline output/lane-review-v8/sample1-final-v6/before.jsonl \
  --records output/lane-review-v8/sample1-final-v6/lane-lines.jsonl \
  --output output/lane-review-v8/sample1-v6-all.json

.venv/bin/python scripts/check_lane_review.py \
  --input samples/sample_2.mp4 \
  --cases tests/fixtures/lane_markings_review.json \
  --baseline output/lane-review-v8/sample2-final-v6/before.jsonl \
  --records output/lane-review-v8/sample2-final-v6/lane-lines.jsonl \
  --output output/lane-review-v8/sample2-v6-all.json
```

以 `--split development` 或 `--split held_out` 分開量測，並更換 report
輸出名稱。品質不合格會先寫出明細，再回傳 exit 1；不可忽略狀態。
既有點位控制以同一命令改用 lane_stability_control.json（sample_1）或
lane_stability_review.json（sample_2），範圍仍僅是原有人工點位。
這個相容模式只測曲線中心線距離，不能證明整段可見輸出；其 fixture
文字描述的是較早的 review。本輪點位重測同樣使用最新 v7 車框，不能
直接沿用較早未使用車框／使用 v5 車框的統計。

基準程式與 v7 車框需自行保留／提供；它們不是權重或樣本下載流程。
未提供基準時，replay 可產生新版本單獨影片；不能用空 before.jsonl 做
修改前／後品質比較。
