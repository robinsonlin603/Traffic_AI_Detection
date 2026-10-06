# Milestone 2 macOS MPS：blocked

2026-10-04 手動阻塞紀錄；不是 `dashcam-ai validate` 產生的機器報告。

本輪評估 HEAD：`b0cf8be22d735aeed9d7ce9e7d49cbe8ea441996`。
工作樹 dirty，包含原有尚未提交的 M2 實作及本輪修正。HEAD 本身不能代表
本輪來源；精確程式／設定／模型雜湊見白線 review 的 manifest。

前輪於沙箱內執行 `.venv/bin/dashcam-ai devices`，CPU available，MPS unavailable，
CUDA unavailable。這是目前執行環境觀察，不是其他環境的硬體能力結論。

`.venv/bin/dashcam-ai validate --milestone 2 --platform macos-mps` 回傳 exit 2：
`currently supported milestone: 1`。正式 M2 macOS JSON 報告 missing。
沒有執行 MPS acceptance run；不得以 CPU 結果替代。

前輪標線修正一般檢查：141 個 pytest passed、`ruff check .` passed、`mypy src`
passed（40 個來源檔案）、`git diff --check` passed。8 個新增標線問題案例
在凍結 v7 detector 下 failed，在修正版下 passed；2 個路緣負例控制通過，
另有量測工具回歸測試。

CPU 分析入口 smoke：原片 sample_1 的 98..127 幀重新編碼為 30 幀片段；
重新執行 YOLO、BoT-SORT 與白線辨識，30 幀完成並全部解碼，診斷資料可
重新解析。這不是原片完整品質或 GPU 驗收。

兩支原片的 CPU 品質比較與未達成項目見
[標線修正 review](../../docs/MILESTONE2_MARKING_REVIEW.md)。

平台狀態：

| 評估項目 | 狀態 | 原因 |
|---|---|---|
| 目前 M2 macOS MPS | blocked | 驗證工具不支援 M2、dirty 來源；正式 JSON missing（本輪自車專項已觀察 MPS） |
| 目前 M2 Linux CUDA | missing | 尚無對應目前來源的 CUDA 報告；本機 CPU 結果不可替代 |
| M1 macOS MPS，16bedd0 | stale | 與目前 HEAD 不同，該舊報告原 verdict 為 blocked |
| M1 Linux CUDA | missing | 無目前適用報告 |
| legacy M2 macOS MPS，cef1205 | stale | 舊實作且來源不同，不能套用新 M2 |
| legacy M2 Linux CUDA，c7d77e3 | stale | 舊實作且來源不同，不能套用新 M2 |

後續需要在品質通過、來源提交且工作樹 clean、M2 驗證入口可用後，分別
執行 macOS MPS 與 Linux CUDA，確認實際 accelerator 並產生各自平台證據。
本輪未更動其他平台的機器報告。

## 2026-10-04 後續 CPU 原型評估

同一 HEAD、dirty 來源下，完成標線候選分類器與車身像素遮罩的四條件
固定窗口比較。163 個 pytest passed、`ruff check .` passed、`mypy src`
passed（41 個來源檔案）、`git diff --check` passed。491 個原片影格的兩支
比較影片全部解碼，來源／輸入／輸出雜湊一致；沒有執行兩片完整 analyze。

組合原型已知案例 51/55、額外案例 49/50、原有控制 15/16；校正後人工
車身輪廓只有 1/4 達標。採納品質 failed，正式後端未切換、M2 未完成。
具體結果及 dirty 來源雜湊見
[原型品質 review](../../docs/MILESTONE2_SEMANTIC_PROTOTYPE_REVIEW.md)。

該原型輪次未觀察到 MPS/CUDA，M2 macOS MPS 仍 blocked、正式 JSON missing，
Linux CUDA 仍 missing。上述 CPU 原型與測試不能替代 clean 對應來源的
任一 GPU 平台驗收；其他平台的機器報告未更動。

## 2026-10-04 固定自車 ID 排除：MPS 專項通過，正式驗收仍 blocked

沙箱外本機 devices 確認 MPS available，CUDA unavailable；先前 MPS unavailable
只描述沙箱內觀察。最終兩個 YOLO26m 模型實際 device 均為 mps。

同一 HEAD、dirty 來源下，兩支原片各 1800 幀全長重播及解碼通過；逐幀
自車形狀框 0，11/11 旁車控制通過，來源／輸入／輸出雜湊一致。182 個
pytest、Ruff、Mypy（42 個來源檔）、diff 檢查通過。預設 CLI 含白線另完成
20 幀整合／解碼檢查。兩支全長專項停用白線以檢查 ID，不能宣稱全片標線
品質通過。來源與 gate 範圍見
[自車排除 review](../../docs/EGO_VEHICLE_EXCLUSION_REVIEW.md)。

正式 M2 macOS MPS 仍 blocked：來源 dirty、validate 只支持 M1（M2 命令
exit 2），正式 JSON missing。Linux CUDA missing，需對新來源另外重跑。
M1 macOS 16bedd0、legacy M2 macOS cef1205／CUDA c7d77e3 全 stale；
M1 CUDA missing。上述 MPS 專項不替代 clean 的任何正式 milestone 驗收。
CPU／灰色候選／型別修正前的中斷輸出均保留 complete=false，不列成功證據。
本輪未修改其他平台機器報告、commit 或 push。

## 2026-10-04 v8 人工回報修正：完整 CPU 比較完成，品質仍 failed

HEAD b0cf8be22d735aeed9d7ce9e7d49cbe8ea441996、dirty 來源；白線檔 SHA256
`a41de794acd7ba5122f5aa0f10d9adc3a5ec64fdafded130fbf83afaff8d642a`，完整來源清單與產物雜湊見
[人工回報 review](../../docs/MILESTONE2_HUMAN_REVIEW.md)。201 個 pytest、Ruff、
Mypy（42 個來源檔）及 diff 檢查 passed。兩支各 1800 幀成對 CPU 白線重播、
四支產出影片完整解碼、來源／輸入／輸出核對 passed。沿用 v8 的車框，沒有
重新執行完整 GPU 追蹤；11 個旁車控制及自車形狀框 0 只驗證沿用觀察。

直線 25/27，固定右上斑馬線區域誤畫 130→0；原有真線 15/16、圖案 45/55、
額外 49/50，沒有新增已標註案例退步。公車格、左線及其他斑馬線仍 failed，
不能用測試或淨改善宣稱 M2 完成。

最終來源另完成預設 CLI 20 幀整合／全數解碼，requested=auto、resolved=mps，
自車形狀框 0；白線 ONNX 仍 CPU。這個 0.667 秒重新編碼片段不能替代完整
GPU 品質或正式平台驗收。精確雜湊見 ignored
`output/human-review-v1/default-cli-smoke-v7-review.json`。

正式 M2 macOS MPS 仍 blocked（dirty、validate 只支持 M1、正式 JSON missing）；
M2 Linux CUDA missing。M1 macOS 16bedd0 與 legacy M2 macOS cef1205／CUDA c7d77e3
stale；M1 CUDA missing。品質通過、clean 固定來源及 M2 驗證入口可用後，
兩個 GPU 平台須分別重跑。本輪只更新本機手動紀錄，沒有修改其他平台的
機器報告、commit 或 push。

## 2026-10-05 左白漆局部修正：完整 CPU 比較通過回歸，品質仍 failed

HEAD b0cf8be22d735aeed9d7ce9e7d49cbe8ea441996、dirty；白線來源 SHA256
`91ed098dfd5f76142280fbd24dce6769e5ede1ecbf111fb22c865373623fd727`。
[左白漆 review](../../docs/MILESTONE2_LEFT_PAINT_REVIEW.md) 記錄完整來源及範圍。

212 個 pytest、Ruff、Mypy（42 檔）通過；兩支各 1800 幀成對 CPU 白線重播、
四支影片完整解碼、43 個來源／重播工具與輸入／輸出核對通過。圈選 615／660
在完整序列中 100% 覆蓋，630／680 左線仍 0%。直線 25/27、原控制 15/16、
圖案 45/55、額外 49/50，固定斑馬線區域誤畫 0/1131，無新增已標註退步。
沿用 v8 車框，自車形狀框 0、11/11 旁車控制只是錄製觀察核對，不是新全片追蹤。

最終來源 20 幀 CLI 整合／全數解碼通過，requested=auto、resolved=mps，
自車形狀框 0；白線 ONNX 為 CPU。ignored
output/left-review-v1/default-cli-smoke-v4-review.json 保存來源與產物雜湊。
短片與 CPU 回歸不能替代任何正式 GPU 平台驗收，整體品質仍 failed、M2 未完成。

正式 M2 macOS MPS blocked（dirty、validate 只支援 M1、正式 JSON missing）；
M2 Linux CUDA missing。M1 macOS 16bedd0、legacy M2 macOS cef1205／CUDA c7d77e3
stale，M1 CUDA missing；前輪 a41de794 的人工證據對目前來源 stale。品質通過、
clean 固定來源與 M2 驗證入口可用後，兩個 GPU 平台須各自重跑。只更新 macOS
手動紀錄，沒有修改其他平台機器報告、commit 或 push。

## 2026-10-06 共用公車格排除：本輪完成，正式驗收仍 blocked

HEAD b0cf8be22d735aeed9d7ce9e7d49cbe8ea441996、dirty；最終白線 SHA256
`1a52da559c4b7bee62a0825e7226ee514bb3dbcbca65565dd68359927ac485e1`。前輪白線局部證據對此來源 stale。
[公車格 review](../../docs/MILESTONE2_BUS_BAY_REVIEW.md) 記錄範圍與固定雜湊。

249 pytest、Ruff、Mypy（42 檔）、git diff --check 通過。642 幀聚焦及兩片
各 1800 幀連續 AFTER CPU 重播完成，BEFORE 複用已完整核對的凍結紀錄與自身
遮罩，來源／輸入／推論函式一致性及方法記錄於 manifest。四片影片完整解碼，
44 個 runtime／重播來源、固定輸入／輸出核對通過；新公車格及真線
控制 136/136，1410–1414 真線恢復，無新增已標註退步，兩段 127 幀再次逐幀
檢視通過。615／660 白漆仍 100%、630／680 仍漏，整體品質 failed。

沿用 v8 車框，自車形狀框 0、11/11 旁車控制只核對錄製觀察，不是新的全片
追蹤。同來源 20 幀預設 CLI requested=mps／resolved=mps 實際觀察、完整解碼、
來源／輸入／產物核對通過；車輛模型 MPS、白線 ONNX CPU。ignored
output/bus-review-v1/default-cli-smoke-v28-review.json 保存精確來源；0.667 秒
重新編碼短片不能取代全片或正式平台驗收。

正式 M2 macOS MPS blocked（dirty、validate 只支援 M1、正式 JSON missing）；
M2 Linux CUDA missing。M1 macOS 16bedd0、legacy M2 macOS cef1205／CUDA c7d77e3
stale，M1 CUDA missing。品質通過、clean 固定來源與 M2 驗證入口可用後，兩個
GPU 平台須各自重跑。只更新本機手動紀錄，未修改其他平台機器報告，未 commit
或 push。
