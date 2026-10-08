# Milestone 2：白線召回與非車道標線排除


本文件依全域 PLANS.md 維護，為 2026-10-03 使用者批准的 ExecPlan。Progress、Surprises & Discoveries、Decision Log 與 Outcomes & Retrospective 隨實作更新。

## Purpose / Big Picture


保留可見道路縱向白線，包括左側、陰影及部分遮擋標線；排除轉彎箭頭、停車格及行人穿越線。使用相同原片比較真線覆蓋及錯線輸出，不以曲線數或 valid 幀數代表品質。

## Progress


- [x] 2026-10-03：使用者批准；完成工作樹、計劃、報告唯讀檢查。
- [x] 2026-10-04：保存 v7 基準；標註 55 個連續影格案例，含開發與保留區段。
- [x] 2026-10-04：圖案群組、短真線、遠處箭頭與局部亮斑白漆支持；新增候選淘汰診斷。
- [x] 2026-10-04：新增 8 個標線問題回歸案例及 2 個路緣負例控制；8 個問題在凍結 v7 失敗、新版通過；路緣負例通過。
- [x] 2026-10-04：完成可重現重播與可見折線／誤畫區域量測工具。
- [x] 2026-10-04：最後 v6 兩支 1800 幀重播與四支輸出影片逐幀解碼；保留區段與分析入口 smoke 已彙整。
- [x] 2026-10-04：141 個 pytest、Ruff、Mypy（40 個來源檔案）及 diff 檢查通過。
- [ ] 所有品質目標及正式 M2 雙平台驗收通過；目前尚未達成。

## Context and Orientation


repository root 為目前工作目錄；分支 codex/restore-milestone1、HEAD b0cf8be22d735aeed9d7ce9e7d49cbe8ea441996，已有大量未提交 M1/M2 修改，全部保留。src/dashcam_ai/lane/segmentation.py 的 YoloPLaneLineDetector.detect 使用 YOLOP 車道線與道路分數，再經白漆支持、車框遮罩、曲線擬合、上下文拒絕、片段合併及時間確認。候選指模型提出但尚未通過這些檢查的線。src/dashcam_ai/visualization/annotator.py 合併白線、車框及黃色車輛歷史軌跡；軌跡不能當白線誤判。

本機已有 samples/sample_1.mp4、samples/sample_2.mp4、models/yolop-640-640.onnx。兩支原片都是 1280×720、30 fps、1800 幀。實作時另找到更新的完整 v7 產物 output/sample_1_milestone2_yolop_v7 與 output/sample_2_milestone2_yolop_v7；本輪兩支影片均使用各自 v7/frames.jsonl 的固定車框，修改前／後共享相同模型輸出與車框。先前 output/lane-review-v7/final2 曾用 v5 車框，不能混用其量測結果。整合 smoke 另外重跑 YOLO 與 BoT-SORT。原片與模型皆不提交。

## Plan of Work and Milestones


第一階段保存修改前 segmentation.py、輸入與設定雜湊，寫到獨立 ignored output/lane-review-v8/baseline。從兩支原片找四類問題的連續區段，標註可见真線折線、非車道圖案區域及不確定／遮擋處。固定資料存 tests/fixtures/lane_markings_review.json；保留區段事先指定、不參與調參。診斷模型、漆面、遮罩、元件、上下文、合併、確認及繪圖階段的流失；本階段驗收是同輸入可重現原錯誤並區分漏判原因。

第二階段在 segmentation.py 先依完整圖案關係排除非車道片段。箭頭檢查箭身與尖端／分叉，包括分離及部分遮擋；停車格檢查重複格線、連接轉角，不固定遮掉兩側道路；行穿線檢查成組寬條紋、間隔與透視，不單靠畫面方向。合併後再檢查不跨越非標線路面，當幀被排除的圖案不可藉舊軌跡延續。修復模型已有候選但因陰影、遮罩或短片段規則流失的白線，左右規則相同，不延長缺失壽命、不無證據外推。本階段以舊程式失敗、新版通過的錯誤案例與正常控制證明改善。

第三階段建立 scripts/replay_lane_review.py，使乾淨 checkout 可使用自備原片、既有模型及選擇性固定車框重現；輸出 annotated.mp4、comparison.mp4、lane-lines.jsonl、summary.json、manifest.json 與來源／輸入雜湊。擴充 scripts/check_lane_review.py 比較連續線段覆蓋與圖案區域誤畫，兼容既有點位檢查。完整重播兩支影片並逐幀解碼；品質通過後才做完整正式分析驗收。30 幀 CPU analyze smoke 只檢查入口與輸出相容，不取代完整驗收，也不取代固定車框的品質比較。

## Concrete Steps


從 repository root 執行，使用既有 .venv，不安裝依賴：

    .venv/bin/python -m pytest tests/unit/test_lane_segmentation.py
    .venv/bin/python -m pytest
    .venv/bin/python -m ruff check .
    .venv/bin/python -m mypy src
    git diff --check
    PYTHONPATH=src .venv/bin/python scripts/replay_lane_review.py --help
    .venv/bin/dashcam-ai devices
    .venv/bin/dashcam-ai validate --milestone 2 --platform macos-mps

重播與品質檢查完整命令在工具完成後補入本計劃及 docs/MILESTONE2_MARKING_REVIEW.md。檢查缺影格、來源雜湊不符及品質失敗須回傳非零狀態。

## Validation and Acceptance


四類問題需在原版重現並在修改後的連續標註區段通過。真線覆蓋以人工確認的可見折線每 2 px 採樣、距離輸出曲線 10 px 內計算，目標至少 90%；遮擋與不確定處不列分母。原有 13 個固定案例及 3 個暗線控制保留。非車道標註區域不應有白線疊圖；另記各類錯線影格數、每案例覆蓋及最長漏判時間。圖案與真線交會時僅標註不含真線的負例區域。

測試、Ruff、Mypy 與 diff 檢查通過，完整產物逐幀可解碼。保留區段不得用來調參；若失敗如實報告，不重新命名為開發案例以宣稱通過。兩支影片均已曾使用，只是現有影片驗收，不是未見資料集精確率／召回率。

若模型完全沒有候選且品質目標無法達成，單獨報告頻率與實例；模型替換／訓練需提出更新計劃及重新批准。平台驗證工具擴充也不在本輪。M2 MPS 上次條件 blocked、正式報告 missing，Linux CUDA missing；M1 MPS 來源 16bedd0 stale（原 blocked）、Linux missing；legacy M2 cef1205/c7d77e3 均 stale 且不適用。CPU dirty 重播不能完成正式雙平台驗收。

## Idempotence and Recovery


新輸出在 output/lane-review-v8/ 的獨立子目錄，禁止覆蓋原片、權重及舊產物。基準程式保留以方便同輸入重跑。中斷產物標為 incomplete；重播工具拒絕覆蓋已有輸出。不 commit/push，不改寫其他平台報告。只改核准範圍。

## Interfaces and Dependencies


沿用 NumPy/OpenCV、YoloPLaneLineDetector、LaneCurve、LaneLineFrame；必要的診斷新增相容欄位，為每候選記錄淘汰階段與原因。設定變更僅在實拍收益確認後加入，不建立額外模型架構。Annotator 僅在證明繪圖造成流失時修改。

## Surprises & Discoveries


v7 13 個點位案例及 sample_1 3 個控制曾通過，但仍留下停車格／文字誤畫。單點與狀態統計不代表整段品質。原程式雜湊為 3fcc6b2e0fc6eeba9a0ac1bfb0443d954617320a4d33d165f01a54bb18f69a44，與 v7 紀錄一致。

2026-10-04：完整 v7 車框產物比先前 review 新；因此改用它們做成對重播，不能沿用舊 review 的量測結論。JPEG 壓縮會改變白漆元件拓撲，像素診斷一律使用原片解碼；JPEG 只供人眼檢查。

2026-10-04：淺斜白線的道路分數可降到約 0.29；原有道路上下文及短片段檢查會刪掉真線。整片白漆做填洞會合併無關漆面並造成控制退步，已棄用；只對停車格及行穿線的局部圖案做關係判斷。遠處小箭頭低於原有最小圖案面積，需以寬度展開證據辨認，而非把所有小漆片當箭頭。

2026-10-04：暖色日照白漆會被原有黃色路緣規則判為彩色，曾嘗試排除候選白漆自身的顏色；這個例外在實拍退步，已收回。局部白漆寬度比較鄰近柏油的修正仍保留，不能以較遠暗處將整片亮斑當成寬白漆。

2026-10-04：上述暖色白漆／路緣例外在完整 v3 的原有第 840 幀黃色路緣控制造成退步。局部修補仍無法保留兩者，已完整收回路緣例外並保留原有規則，新增左右鏡像的窄黃色路緣負例控制。第 832..848 幀重播已確認黃色路緣控制恢復；最後完整重播改用 v6，不能把 v3 產物當最後來源。

2026-10-04：effective 車框遮罩會覆蓋人工確認可見的白線（例如 sample_2 第 62 幀）。它只能用來重現實際繪圖的裁切，不能刪除人工真值的分母；量測工具已修正，遮掉可見白線會計為漏畫。sample_1 第 100 幀標註端點由 (330,402) 更正為原片白漆實際端點 (281,415)，避免把柏油算成白線；這是開發標註校正，不修改保留區段。

2026-10-04：第一輪完整比較仍有保留區段停車格與左線失敗。保留區段維持固定，後續變更僅依據開發區段與成對測試，不能把失敗區段改名或當新驗收資料。

## Decision Log


2026-10-03 / Codex：使用者批准四類修正。先建連續案例與圖案關係，再處理有證據的白線流失；不全面調整模型門檻。理由是提高／降低門檻會在誤判與漏判間互相惡化。

2026-10-04 / Codex：保持模型、設定、最大缺失壽命與時間確認不變；修改圖案證據與窄白漆支持。所有參數調整只依據開發區段，完整重播前凍結最後來源。人工折線只含可見白漆；輸出遮罩不能改寫真值。理由是避免錯誤遮罩使召回指標虛高。

2026-10-04 / Codex：收回未通過原有路緣控制的暖白漆例外，保留局部背景寬度判斷、小箭頭及圖案群組修正。理由是不可用新的道路邊線誤畫交換左線召回改善。強光失敗如實保留，不刪除案例。

## Artifacts and Notes


精簡標註、重播工具及報告留於專案；影片、模型與中間影像留在 ignored output。每份比較必須記錄原片、模型、設定、車框及實作雜湊；HEAD 不代表 dirty 實作來源。

## Follow-up / Approved 2026-10-04

目前尚未達成的品質項目不能直接勾選完成。保留區段失敗已公開，後續只能作為回歸資料，不能再宣稱它是未使用驗收資料。

方案 A 是繼續拆解白漆線段與轉角關係，不更換模型。成本較低，但圖案與真線在遮擋、暖色強光下可能仍共用相同局部證據，車框遮罩也仍會刪掉可見道路。

建議先評估方案 B 的原型，而不是立即替換現有模型：

1. 補齊失敗案例的標線類別與實際車身輪廓真值，分開車道線、箭頭、停車格、行穿線、路緣與車身；沿用本輪折線與誤畫指標，另固定新驗收區段／新影片。現有影片的已知案例只用於開發與回歸。
2. 評估能區分標線種類的語意模型／候選分類器，以及能保留道路的車身像素遮罩；比較既有 YOLOP，記錄來源、權重、輸入雜湊及速度。沒有證明改進前，不切換預設後端。
3. 只有在可見真線每案例至少 90%、負例誤畫 0、原有 16 個控制通過後，才整合正式 analyze 並跑兩支原片；來源提交且 clean 後，分別補 MPS/CUDA 平台證據。

這涉及額外模型、權重取得與小型訓練，超出上一輪保留現有模型範圍。使用者於 2026-10-04 另行批准方案 B 原型評估；詳細範圍與後續進度見 [原型 ExecPlan](EXEC_PLAN_MILESTONE2_SEMANTIC_PROTOTYPE.md)。原有後端只有在品質證明後才能整合，commit/push 與正式平台工具擴充仍沒有授權。

## Outcomes & Retrospective


2026-10-04：本輪實作、回歸測試、最後 v6 兩支完整 CPU 重播、逐幀解碼及 30 幀正式入口 smoke 已完成。141 個測試、Ruff、Mypy、diff 檢查與原有 16 個控制通過；8 個問題案例在原版失敗、新版通過。

55 個連續案例由 21/55 改善到 39/55 通過。已標註箭頭誤畫 3→0 幀、行穿線 2→0 幀、停車格 9→5 幀；sample_1 左線長度覆蓋 81.08%→93.69%、sample_2 29.32%→56.71%。保留停車格仍 5/5 幀失敗，保留左線平均仍 69.28%；開發區段也仍有單幀／強光漏畫。品質為 failed，正式 M2 macOS MPS blocked、Linux CUDA missing，未勾選整體驗收完成。

最終量測、來源、重現指令與未解決影格見 docs/MILESTONE2_MARKING_REVIEW.md；本機平台阻塞見 validation/milestone-2/macos-mps.md。上述結果是原型評估前一輪的紀錄；2026-10-04 使用者已另行批准原型，詳見獨立 ExecPlan。沒有 commit/push。

修訂原因：保存已批准方案，補充實作證據、車框來源修正、真值量測修正及尚未達成的驗收項目。
