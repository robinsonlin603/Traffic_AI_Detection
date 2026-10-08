# Milestone 2：v5 截圖修正與完整白線重播

2026-09-30。使用者核准修正路緣／人行道、箭頭誤畫與左側漏線；「垂直」經確認指橫跨行車方向的停止線／行穿線，不是畫面中上下延伸的真車道線。這是單一實拍影片的修正紀錄，不是 Milestone 2 正式驗收。

## 來源與實作

分支 codex/restore-milestone1，HEAD b0cf8be22d735aeed9d7ce9e7d49cbe8ea441996，工作樹 dirty。保留原有未提交內容，未 commit/push。v5 的 metadata 未記錄白線程式與完整設定雜湊，因此比較對象是既有實際影片與 JSON，不宣稱它等於某個乾淨提交。

本輪主要修改 src/dashcam_ai/lane/segmentation.py，並在 tests/unit/test_lane_segmentation.py 加入回歸案例。候選需通過形狀、道路上下文、長度及兩幀確認；偵測與繪圖的車身像素遮罩仍共用，沒有移除車輛框。

淺斜候選向道路 ROI 上緣延伸後若遠離影像範圍，標記 transverse_marking。這是寬鬆透視檢查，保留近垂直線與左右外側斜線，不是完整的道路方向估計。曲線兩側道路支持度不對稱時，白漆例外改要求至少 80% 採樣具有 35 灰階的雙側對比；有持續紅／黃色路緣證據時額外拒絕 colored_curb_context。

漆面形狀改於車框裁切前辨識，避免把短暫可見白線切成不規則塊。元件超過半數落於遮擋區時，不把它當作可靠道路符號形狀，避免亮車身污染白線。箭頭新增內部相鄰區段寬度突變判斷，補足粗箭身使整體寬度比例不明顯的案例；寬度膨脹改比較第 90 與第 25 百分位，辨識與箭身分離的三角形箭頭。通過上下文的短曲線必須有端點緊鄰遮擋遮罩，且可見長度足夠，才可低於原本跨度門檻，避免孤立小圖案被當成短白線；時間確認仍為兩幀、最多延續一幀。

## 驗證方法

在 repository root 執行：

    .venv/bin/python -m pytest
    .venv/bin/python -m ruff check .
    .venv/bin/python -m mypy src
    git diff --check

結果為 114 tests passed、Ruff passed、strict Mypy passed（40 source files）。回歸涵蓋鏡像橫向停止線與淺斜真線、近垂直真線保留、弱亮路緣與強白漆的區別、車框裁切的形狀污染、部分遮擋箭頭、短白線與小亮點、彩色路緣及粗箭身寬度突變、孤立箭頭三角形及無遮擋短線不放行。

本機重播使用 samples/sample_2.mp4 的全部 1800 幀（1280×720、30 fps、60 秒），重用 output/sample_2_milestone2_yolop_v5/frames.jsonl 車框，只重跑 YOLOP 白線與繪圖，不重新執行 YOLO／BoT-SORT。output/lane-review-v6/replay.py 保存本機重播步驟；annotated.mp4 為新版，comparison.mp4 左 v5、右新版。原始輸入、權重與 v5 輸出均未覆蓋。trial1／trial2 為未完成修正的試跑，不作最終驗收證據。

本機產物與大檔不進 Git；manifest.json 記錄來源雜湊，summary.json 記錄候選淘汰數，verification.json 記錄人工確認位置的曲線覆蓋與逐幀解碼。位置檢查只代表列出的點位，不代表完整精確率／召回率；曲線觀測數亦不能當成辨識正確率。

## 來源雜湊

輸入影片 SHA-256：48f0e3010d7acffebf95b8bd55e1193589785a3d8835ec528040f0298edbf4ba。

YOLOP ONNX：cd66a3e0087a7258ae07768cc02cb742eed93865727ae4c9baf969b8fa190696。

v5 車框 JSONL：276d0524a0609797b8086c3c5e8015ee158e23bf82c107761ba8cacf30165667。

本輪 segmentation.py：6ccb88504db410e5f78256e6fb3601cc423741bcc0b547e36f074e22830df106。

## 平台狀態與限制

本輪白線推論為 OpenCV DNN CPU，最終重播使用四個 OpenCV 執行緒。本機裝置檢查 MPS／CUDA unavailable；validate --milestone 2 --platform macos-mps 仍回覆只支援 milestone 1。新 Milestone 2 MPS 執行 blocked、報告 missing；Linux CUDA missing。Milestone 1 macOS 的來源 16bedd0、legacy macOS cef1205、legacy Linux c7d77e3 均不等於本次 HEAD，屬 stale，且 legacy 實作不適用。本輪未更新任何平台機器報告。

完全遮擋或模型沒有提出候選時，不補畫假線。高度遮擋的箭頭、與亮車身連接的白線、逆光及道路模型分界錯誤仍可能造成誤判或缺線；有限方向規則無法代表所有彎道與相機角度。相同乾淨提交的雙平台驗證與更完整人工品質標註仍未完成。

## 最終影片結果

最終重播完成 1800 幀，耗時 510.77 秒；新版與左右對照影片均逐幀解碼確認 1800 幀，尺寸分別為 1280×720、2560×720。新版狀態為 valid 402、degraded 569、unknown 829；這些是輸出可用性狀態，不是人工正確率。

抽查 1 秒與 40 秒可見左白線，由 v5 無覆蓋變成有覆蓋；20 秒左箭頭、28 秒黃路緣的指定點位由有覆蓋變成無覆蓋；21 秒近垂直真車道線保留。14 秒橫向停止線及 55 秒孤立箭頭在新版未畫線。停止線指定點位在 v5 JSON 中亦未命中，因此不能用該點位統計量化舊影片中可見的誤畫改善。所有點位結果見本機 verification.json。

左半部新鮮曲線觀測每十秒分段，v5 為 7、65、6、43、201、295，新版為 9、58、16、54、203、265。數量沒有全面上升，也不能區分真線與誤畫。42–44 秒等片段仍有左側缺線；本輪只確認特定案例改善，不能宣稱已解決所有漏線或完成品質驗收。

新版影片 SHA-256：e5f75a75a2753f9bc5a6792fffc6c1067298988c4a8578edfa0c4ceaf50f8287。

對照影片 SHA-256：6740d8bd67c589dcfab50cc020fe233925f9df6b1819251e756768da598d86e8。
