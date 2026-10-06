# Milestone 2：左側可見白漆恢復

2026-10-05 使用者批准的共用規則修正，範圍見 [ExecPlan](EXEC_PLAN_MILESTONE2_LEFT_PAINT.md)。區段、聚焦及兩片完整比較完成；兩處圈選白漆恢復、沒有新增已標註退步，整體品質仍 failed，不能宣稱整段左線或 M2 通過。

## 修正與來源

車框額外擴張會蓋住機車之間的可見白漆；真縱向白線在透視下也可能呈淺斜角，被道路邊緣規則排除。本輪只改共用白線判斷，沒有影片名稱、幀號或固定道路座標例外。

只有至少一半取樣點被額外遮罩蓋住、方向收斂到場景、窄漆及亮度對比支持率均達 90% 的候選，才能解除額外遮罩。原車框內不新增解除；既有大型角落車框道路證據例外保留。恢復候選仍須通過原有箭頭、停車格、斑馬線、長度與確認規則。一般幾何擬合最小跨度維持影像寬的 2.5%；外圍恢復可對 2% 跨度的實際白漆作共識擬合，不延伸端點、不新增短線救回資格。

淺斜角道路邊緣例外要求窄漆與亮峰支持均達 90%，方向收斂到場景中央。紅黃路緣、亮度階梯及圖案排除保留。完整圖案分類沿用本幀原擴張遮罩判斷車輛污染，候選與渲染使用恢復後的有效遮罩；兩者用途不同，不借用上一幀。分類後必定恢復有效遮罩，reset 清除兩種遮罩。

配對驗證工具原本對 before／after 使用同一遮罩；本輪修正為各用自己的有效遮罩，新增測試確認差異。人工可見真值的分母保持不變，誤遮仍算漏線。

分支 `codex/restore-milestone1`，HEAD `b0cf8be22d735aeed9d7ce9e7d49cbe8ea441996`，工作樹 dirty；HEAD 不能代表實測來源。白線檔 SHA256：

| 版本 | SHA256 |
|---|---|
| 本輪凍結 baseline，前輪 final-v7 | `a41de794acd7ba5122f5aa0f10d9adc3a5ec64fdafded130fbf83afaff8d642a` |
| 本輪修正版 | `91ed098dfd5f76142280fbd24dce6769e5ede1ecbf111fb22c865373623fd727` |

42 個 runtime 來源檔中只有 segmentation.py 改變；設定、原片、模型、v8 車框輸入雜湊均未變。沒有 commit、push、模型訓練、依賴安裝、追蹤／自車 ID 變更或原型後端切換。既有未提交修改保留。

## 兩張截圖與人工真值

原圖以畫面匹配對齊 sample2 frame_id 615（20.50 秒，767 個 homography inliers）與 660（22.00 秒，520 個）。本文幀號全部零基；暫按一基 F 換算為 616F／661F。兩支原片均 1280×720、30fps、1800 幀。

`tests/fixtures/lane_left_paint_review.json` 保存截圖雜湊、匹配幀號及人眼可見漆面端點，私人截圖不進 Git。615 只標圈內約 34px 的可見短漆；660 及舊 630／680 的原有人工折線不變。容許距離 10px、至少 90% 覆蓋才通過。

137 幀前後緩衝區段（599..735）的結果：

| 原片 frame_id | 修改前覆蓋 | 修正後覆蓋 | 結果 |
|---|---:|---:|---|
| 615，第一張圈選短漆 | 0% | 100% | passed |
| 630，舊左線控制 | 0% | 0% | failed |
| 660，第二張圈選白漆 | 0% | 100% | passed |
| 680，舊左線控制 | 0% | 0% | failed |

630、680 在人工折線的 50 個取樣點附近 ±3px 均沒有達門檻的模型訊號；本輪解除遮罩沒有可恢復候選。原車框分別覆蓋 26/50、4/50 個點；兩種問題不能混為單純車輛遮擋。615／660 周圍模型支持為 50/50，原框覆蓋為 3/50、0/50。這是局部診斷，不是全片模型召回率。

不能從四個代表幀推算 615–710F 全部 96 幀正確或最大漏線時間；完全遮住的線不補畫。下一輪需優先評估 630／680 的候選召回與車身像素證據，不能再全面放寬當前圖案保護或套用 failed 原型。

## 驗證與未完成範圍

212 個 pytest、Ruff、Mypy（42 個來源檔）通過。四個新增左右鏡像正例在凍結 baseline 失敗、修正後通過；弱漆與寬漆四個負例維持通過。另有原框保護、上一幀遮罩獨立性、已可見白線不延伸、圖案分類不受外圍恢復影響、個別版本遮罩量測控制。

571 個聚焦幀使用同一模型機率及車框比較；可用的舊錨點、27 個直線、真線、圖案與額外案例沒有新增已標註退步。這次聚焦只有舊控制 6/16、圖案 30/55、額外 5/50 在快取內，不能當作全部控制通過。

最終來源另完成預設 CLI 20 幀整合／完整解碼，requested=auto、resolved=mps，自車形狀框 0；白線 ONNX 仍 CPU。這是 0.667 秒短片，不是完整 GPU 品質驗收。

完整兩支 1800 幀成對 CPU 重播、全數解碼、全部控制及 1131 幀固定斑馬線區域的來源核對與回歸完成，結果見下表。沿用錄製 v8 車框，不代表新跑完整 GPU 追蹤。

## 最終完整比較（final-v4）

兩支各 1800 幀成對 CPU 重播完成；before／after records 完整為 0..1799，四支影片全部解碼，43 個來源／重播工具檔及輸入／輸出雜湊一致。前後使用同一幀的模型機率、相同設定與錄製 v8 車框；各用自己的有效遮罩評估。

| 已標註範圍 | 凍結 baseline | 修正版 | 新增退步 |
|---|---:|---:|---:|
| 16 個人工錨點 | 10/16 | 11/16 | 0 |
| 27 個連續短白線 | 25/27 | 25/27 | 0 |
| 原有 16 個控制 | 15/16 | 15/16 | 0 |
| 原有 55 個圖案／真線案例 | 45/55 | 45/55 | 0 |
| 50 個額外案例 | 49/50 | 49/50 | 0 |
| 本輪 4 個左線真值 | 0/4 | 2/4 | 0 |

sample1 指定固定右上斑馬線區域的 1131 幀誤畫 0→0，沒有新增已標註退步。兩處圈選在完整時間序列中仍為 100% 覆蓋；630 原本正確的中央白線保住，該幀左線與 680 左線仍 0%。四個左線真值與其他表格重疊，不能合計為獨立樣本或推算整段召回率。

原有 sample2 30 真線、631 額外真線仍未達標；公車格、其他斑馬線與最前兩幀直線仍有既有失敗。整體品質 verdict **failed**，本輪局部修正與保留既有通過案例的回歸閘門通過，M2 未完成。沿用車框的自車形狀框為 0、旁車控制 11/11，這只是錄製觀察的核對，沒有重新跑全片 GPU 追蹤。

## 撤回試驗

focus-v1 中斷，complete=false。focus-v2 在 sample1 748／749 增加固定斑馬線區域誤畫，並讓 sample2 151／152 真線退步；拒絕採用，改為只救大部分被額外遮罩蓋住的候選。focus-v3 又使 sample2 630 的中央真線被 crosswalk_context 排除；拒絕採用，改為穩定本幀圖案污染判斷。失敗與舊來源的短片 MPS 檢查保留，不能套用最終來源。

focus-v4 的 571 幀及 window-v4 的 137 幀完成；完整來源採 final-v4 新目錄，不覆蓋歷史證據。

## 證據位置

ignored `output/left-review-v1` 保存 baseline-manifest.json、baseline-test-proof.json、source-checks-v4.json、focus-v4/quality-audit.json、sample_2-window-v4/left-review.json、default-cli-smoke-v4-review.json、left-input-evidence.json、final-audit-v4.json 及兩支 final-v4 的 manifest／summary／各項 review。窗口兩片完整解碼，來源／輸入／輸出雜湊記於 manifest／summary。CPU paired replay 的相同機率輸入僅證明已標註回歸，不證明未標註場景都正確。

人工複查：[sample2 左線區段比較](../output/left-review-v1/sample_2-window-v4/comparison.mp4)，左側為凍結 baseline，右側為修正版。

完整產物雜湊：

| 產物（output/left-review-v1 下） | SHA256 |
|---|---|
| sample_1-final-v4/lane-lines.jsonl | `1ae6386496a1fbd106a171a56a4af90662fece593edc3df858df7eefd0b83ba3` |
| sample_1-final-v4/annotated.mp4 | `e26195793380c9d36c46233fab8a7f62645d768e47e25bbabf4d53b14e138401` |
| sample_1-final-v4/before.jsonl | `f75cf5109dffdd818414ed6a9e6c13a953e8f4e3896ba55c3e657a34a8f065eb` |
| sample_1-final-v4/comparison.mp4 | `5a3336b0b08833811f914885a1c1a566e484e8a62205783c9f6b174059c1a7ad` |
| sample_2-final-v4/lane-lines.jsonl | `80623353eabdaca2bc5d53f365b8b011c36507a75625d9cad93e9cee9758ac0a` |
| sample_2-final-v4/annotated.mp4 | `3e91f422161ccb6e4f27db48be22cac40bbbdede8fed52500443a0e05c884de9` |
| sample_2-final-v4/before.jsonl | `d1e96982c3d3f268480db181587629a833116455489f8a85b16feb5299da54ae` |
| sample_2-final-v4/comparison.mp4 | `6b63d0b1a85ddcee15ad446b71bfef5adcd92533fe424b7c8af1f0ccf8df9a76` |

完整對照：[sample1](../output/left-review-v1/sample_1-final-v4/comparison.mp4)、[sample2](../output/left-review-v1/sample_2-final-v4/comparison.mp4)。原片／模型／設定雜湊與前輪一致，另見本輪 manifest。

## 平台狀態

| 必要平台／紀錄 | 狀態 | 原因 |
|---|---|---|
| 目前正式 M2 macOS MPS | blocked | dirty、validate 只支援 M1、正式 JSON missing；短片 MPS 不等於驗收 |
| 目前 M2 Linux CUDA | missing | 沒有對應本輪來源的 CUDA 報告，須另機重跑 |
| M1 macOS 16bedd0 | stale | 舊 commit，原 verdict blocked |
| M1 Linux CUDA | missing | 無目前適用報告 |
| legacy M2 macOS cef1205／CUDA c7d77e3 | stale | 舊來源與階段 |

正式 GPU 驗收仍需品質達標、clean 固定來源與 M2 驗證入口可用後，分別重跑；CPU 結果與任一 GPU 平台不可互相替代。本輪只更新 macOS 手動紀錄，其他平台機器報告不變。前輪 a41de794 的證據適用歷史來源，對目前來源已 stale。
