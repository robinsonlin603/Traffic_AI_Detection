# Milestone 2：曲線穩定性與固定案例驗收

本文件依全域 PLANS.md 維護，2026-10-02 使用者在評估四步進階條件後要求「開始修改專案」。本輪執行已提出的前兩步及其回歸驗證：固定案例、修正曲線突變與可見左線流失。車道歸屬、換道事件、模型重訓、追蹤器修改及平台驗證工具擴充不在本輪。

## Purpose / Big Picture

避免約 4.27 秒的錯誤線段繼承已確認曲線身分並被畫成彎線；查明 42–44 秒可見左線在哪個步驟流失。使用者可比較同一影片的 v6 與本輪影片，查看每個固定案例的保留／拒絕結果。不能把曲線數量增加或 valid 狀態當成品質通過。

## Progress

- [x] 2026-10-02 使用者批准開始，檢查現有工作樹、計畫及平台報告。
- [x] 固定問題及正常控制案例，保存修改前診斷及 tests/fixtures/lane_stability_review.json。
- [x] 實作先過濾再合併、雙向連接檢查、方向配對、平滑位移上限、左線有效尾段及成對回歸測試。
- [x] 完整重播 1800 幀並解碼；13 個固定案例、第二影片 3 個控制案例通過；121 項測試、Ruff、Mypy、diff 檢查通過。正式平台驗證仍 blocked/missing/stale，見驗證報告。

## Context and Orientation

從 repository root 執行命令。分支 codex/restore-milestone1，HEAD b0cf8be22d735aeed9d7ce9e7d49cbe8ea441996，工作樹包含先前未提交的 M1/M2 修改，須保留。src/dashcam_ai/lane/segmentation.py 的 detect 依序做模型分割、像素遮罩、元件擬合、片段合併、道路／漆面上下文及時間追蹤。_curve_distance 現在只檢查共同高度內的位置差異，_smooth 對共同高度取加權平均。可能把小段重合、整體不同的線當成同一條線，必須先用原片驗證。

本機 samples/sample_2.mp4 是 1280×720、30 fps、1800 幀。models/yolop-640-640.onnx 為既有模型。output/sample_2_milestone2_yolop_v5/frames.jsonl 為既有車框。output/lane-review-v6 保存前輪完整白線重播；其中程式 SHA-256 為 6ccb88504db410e5f78256e6fb3601cc423741bcc0b547e36f074e22830df106。新資料寫入 ignored output/lane-review-v7，不覆蓋舊影片或模型。samples/sample_1.mp4 可供獨立片段檢查，但它亦曾用於過往調整，不能宣稱是完全未見測試集。

## Plan of Work

第一段保存 frame 120–139、1260–1320 的候選與各步驟結果，抽查原始畫面。固定真線控制包含 frame 30、630、1200、1230、1350、1530、1620；負例包含 128 的錯線、420 停止線、600 箭頭、840 路緣、1650 箭頭。以可見漆面標註點位／線段，不把被遮擋部分當漏判，保存精簡標註與輸入雜湊供重現。

第二段只在已確認的流失／變形階段修改 segmentation.py。跨幀配對須拒絕方向／形狀不相容的曲線，讓它重新確認；保留正常遮擋縮短及透視延長。若合併或擬合造成錯線，檢查其與當幀像素證據的一致性。平滑不能使結果偏離當前真標線。左線依實際淘汰原因修復，不以固定左右位置或影格編號決定輸出，不全面降低信心門檻、不延長缺失線壽命。

第三段在 tests/unit/test_lane_segmentation.py 加入能重現錯誤及保留正常行為的測試，先證明錯誤案例在舊程式失敗。重用固定車框重跑白線，明確不宣稱重跑 YOLO/BoT-SORT。至少完整重播 sample_2 並逐幀解碼輸出，以獨立 sample_1 片段檢查是否有明顯退步。對照片及診斷保持本機，大檔不提交。

## Concrete Steps

在 repository root 執行：

    .venv/bin/python -m pytest tests/unit/test_lane_segmentation.py
    .venv/bin/python -m pytest
    .venv/bin/python -m ruff check .
    .venv/bin/python -m mypy src
    git diff --check

問題診斷與重播腳本保留在 output/lane-review-v7；最終報告記錄可重現命令、輸入／來源／產物雜湊及確切影格。以本機 CLI 檢查裝置及 M2 驗證是否可執行，不冒用 M1 報告。

## Validation and Acceptance

4 秒錯誤線不得繼承確認狀態後跨越無標線路面；正常縮短、伸長、近垂直、左右斜向標線仍可保持身分。對固定可見真線比較修改前後位置覆蓋，對負例確認沒有疊線，另列不確定案例。42–44 秒須逐項說明模型缺候選、遮罩移除、規則拒絕或時間未確認，不能全部籠統歸因於遮擋。若原片缺乏足夠證據則保留 unknown，不虛構標線。

完整測試、Ruff、Mypy 須通過，影片逐幀數等於輸入。固定點位不是完整 precision/recall，也不代表正式 milestone 通過。當前 M1 macOS 報告來源 16bedd0 為 stale（原 blocked），M1 Linux missing；新 M2 macOS 上次執行 blocked、報告 missing，Linux missing。歷史 M2 macOS cef1205、Linux c7d77e3 均 stale 且不適用。正式通過仍需品質標註與相同乾淨來源雙平台整合驗證。

## Idempotence and Recovery

保留所有既有未提交修改；不 commit/push、不下載依賴。修改前保存 segmentation.py 至新的 ignored 產物目錄，試跑輸出與最終輸出分開。中斷可重跑同一固定輸入，報告必須標明完整或未完成。只更新目前平台可產生的機器證據。

## Surprises & Discoveries

原片確認 frame 128 中央短橫線與右側候選在 _merge_fragments 被接成長彎線；97 px 水平連接誤差仍小於時間配對的 102 px 容許值，合併後又繼承原有身分。改用雙向垂直於曲線的連接誤差。frame 1305 左線候選前半段落在柏油，後半段有連續白漆，整段道路支持度低被拒絕；改檢查能否裁掉不受白漆支持的端部。frame 1260 原片沒有清楚可見左線，不屬可見標線召回失敗。既有紀錄 frame 127–129 的同一 boundary_id 橫向跨度由約 114→426→80 px，但尚不能單憑輸出區分元件、合併與平滑原因。

初次短斜線篩選只看白漆支持比例，導致 sample_1 的暗色左線流失；改為結合附近圖案證據，新增 3 個第二影片控制點位。這些是開發回歸案例，不是未見驗收集。

## Decision Log

2026-10-02：本輪先完成白線穩定性與固定案例，避免把下游換道邏輯加入尚不穩定的輸入。使用現有 NumPy/OpenCV，不增加模型或演算法框架。

## Outcomes & Retrospective

本輪修正及限定案例驗證已完成，詳見 [v7 穩定性驗證](MILESTONE2_STABILITY_REVIEW.md)。本機產物為 output/lane-review-v7/final2，sample_2 的 13 案例通過；sample_1 的 3 個控制案例通過。所有影片完整解碼，121 個測試通過。仍有模型缺候選、短線確認、路面文字／停車格誤畫等限制；不將這次 CPU 白線重播當作正式 M2 品質／雙平台驗收，也未開始 M3。

修訂原因：第 4 秒新案例揭露跨幀不穩定；依使用者核准把進階條件的前兩步轉成可追蹤工作。
