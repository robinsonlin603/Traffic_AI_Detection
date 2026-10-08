# Milestone 2 白線聚焦修正紀錄

2026-09-29。本輪由使用者核准遮罩、箭頭／路緣上下文與成對驗證。屬局部品質改善，不代表完整 Milestone 2 驗收。

## 來源與限制

來源分支 `codex/restore-milestone1`，HEAD `b0cf8be22d735aeed9d7ce9e7d49cbe8ea441996`，工作樹 dirty；保留本輪開始前未提交的 Milestone 2 內容。未 commit/push。

輸入 `samples/sample_2.mp4`，1280×720、30 fps，SHA-256 `48f0e3010d7acffebf95b8bd55e1193589785a3d8835ec528040f0298edbf4ba`。YOLOP ONNX SHA-256 `cd66a3e0087a7258ae07768cc02cb742eed93865727ae4c9baf969b8fa190696`。

重用 `output/sample_2_milestone2_yolop_v3/frames.jsonl` 的車輛框，只重跑白線模型及標註，未重新執行 YOLO／BoT-SORT。對照基準是該目錄既有的 `lane-lines.jsonl`，其 metadata 沒有足夠來源版本證據，不能視為乾淨提交的正式基線。

## 修正

偵測與繪圖共用有效像素遮罩。大型貼底角落框只釋放有高分數道路支持的區域，保留非道路車身與其他車輛遮擋，左右對稱；綠色自車誤框本身仍屬既有車輛偵測結果，未在此刪除。

ONNX 已輸出 Sigmoid 分數，移除二次 Sigmoid，改以兩類分數總和正規化。道路門檻由 0.2 改為 0.5；這些分數不等同校準過的正確率。

箭頭判斷回看模型候選外的完整可見漆面，以主方向上的形狀與寬度膨脹拒絕箭頭。過小亮點不作整個箭頭的形狀證據。獨立分叉計數試驗會誤刪光影交界的白線，沒有納入最終版本。

路緣判斷檢查曲線兩側的道路支持。真實標線可能也位於模型道路分界，故加入中央亮、兩側暗的持續漆面證據與方向限制，避免直接刪除所有道路邊界。短片段先合併再檢查輸出跨度。平滑只作用於共同高度，新增端點維持當幀位置，避免不自然彎折。

## 驗證方式

在 repository root 執行：

    .venv/bin/python -m pytest
    .venv/bin/python -m ruff check .
    .venv/bin/python -m mypy src
    git diff --check

結果：103 tests passed；Ruff 通過；strict Mypy 通過（40 source files）；diff whitespace 檢查通過。新增回歸涵蓋 Sigmoid 分數、左右角落大框、保留真車遮擋、箭頭箭身與真白線成對辨識、道路邊界白漆例外、短片段合併、平滑端點、單像素漆面及繪圖共用遮罩。

原片重播選取 frame 0–239、525–554、1275–1304、1440–1679，共 540 幀。每段重設白線追蹤，段首確認狀態與 v3 完整影片不同，因此不能把曲線數差異直接解讀為召回率或精確率。

本機產物位於 `output/lane-review-v4-final/`：`annotated.mp4` 為新結果；`comparison.mp4` 左為既有 v3、右為新結果；`lane-lines.jsonl` 與 `summary.json` 為本輪診斷。兩支影片串接所選片段，並非完整 60 秒影片。影片與大檔不進 Git。

最終重播為 210 幀 valid、141 幀 degraded、189 幀 unknown；左半部新鮮曲線觀測數由基準 211 增至 336。這是診斷計數，不是人工標註的召回率。抽查 frame 31 恢復大框內可見左線；frame 1500 的箭頭未被畫線，實際道路白線保留；frame 1620 的道路線保留、路緣未被畫線；控制 frame 531、540 保留真白線，frame 1278 保留兩條真白線。

左半部新鮮曲線觀測分段為：0–8 秒 9→7，17.5–18.5 秒 30→23，42.5–43.5 秒 9→13，48–56 秒 163→293。改善主要集中於後段，不能宣稱每段召回都改善；前 8 秒仍不穩定。段首重設、誤線排除與真線漏判均可能影響這些計數，尚無完整人工標註可拆分。最終兩支輸出影片均已逐幀解碼成功，共 540 幀；新結果尺寸 1280×720，左右對照尺寸 2560×720。

最終來源檔 SHA-256：`lane/segmentation.py` 為 `d4f20653a9611f2a152ba1f9d7671bd66f8b429f5e121bf7e468c0efdca049fe`，`visualization/annotator.py` 為 `dd7a90b1abf85899a0eaebda5455284996b2a230278053157d1fec047e3c6721`，`application/analyzer.py` 為 `9bd15d5ca71c4810ec8a3092a2e90c3d47baf808a2e3c786220864acd3dc0a02`；皆相對於 `src/dashcam_ai/`。檔案雜湊用於辨認 dirty working tree，不能取代乾淨提交的平台驗收。

## 剩餘限制與平台狀態

逆光、真實遮擋、模型沒有提出候選，以及短片段確認仍可造成間歇缺線。路緣與真正道路邊線在其他鏡頭角度仍可能混淆；本輪是單一影片的聚焦驗證，不能承諾所有箭頭或路緣均不再誤判。`valid` 僅代表有至少兩條確認曲線，不是正確率標籤。

本機 macOS 裝置檢查：CPU 可用，MPS／CUDA 不可用。`dashcam-ai validate --milestone 2 --platform macos-mps` 回覆只支援 milestone 1，故未產生或冒用平台報告。新 Milestone 2 的 macOS MPS 執行條件為 blocked、報告 missing；Linux CUDA 報告 missing；整體正式驗收 blocked。現有 Milestone 1 MPS 報告來源 `16bedd0` 與 HEAD 不符，屬 stale（原 verdict blocked）。兩份 legacy Milestone 2 報告亦為其他提交，且對應已移除實作，不適用本輪。

仍需完整實拍品質驗收，以及相同乾淨來源版本的 MPS／CUDA 驗證。此紀錄不是機器生成的跨平台驗證報告。
