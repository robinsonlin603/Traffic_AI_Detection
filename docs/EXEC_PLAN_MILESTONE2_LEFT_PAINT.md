# 恢復被車框與道路邊緣判定排除的可見白漆

本 ExecPlan 依全域 PLANS.md 維護。Progress、Surprises & Discoveries、Decision Log、Outcomes & Retrospective 隨實作更新。使用者於 2026-10-05 批准共用規則的局部修正，sample2 615–710F 作驗收案例；不得新增影片／幀號／固定道路座標例外。

## Purpose / Big Picture

讓機車之間確實可見的左側白漆恢復疊圖。車框擴張可能遮到道路，透視也會讓縱向白線呈淺斜角；修正須以原片白漆與道路證據作判斷，保留真正車身及完全遮蔽區域。框內無法可靠區分的漆面如實列為限制，不依使用者圈選位置直接畫線。

## Progress

- [x] 2026-10-05：使用者批准，檢查 HEAD、dirty worktree、既有 M2 計畫及平台紀錄。
- [x] 保存 baseline 與來源／輸入雜湊；來源白線檔 a41de794acd7ba5122f5aa0f10d9adc3a5ec64fdafded130fbf83afaff8d642a。
- [x] 對齊兩張截圖到原片 frame_id 615（20.50 秒）及 660（22.00 秒）。前者圈附近的模型訊號被遮罩排除；後者部分可見候選被 road_edge_context 排除。
- [x] 補入兩張截圖與既有 630／680 真值；加入左右鏡像、窄／弱／寬漆、車框內保護、重複幀、已可見白線及道路邊緣控制。
- [x] 實作共用車框外圍白漆恢復及淺斜線判斷；原框內保護與圖案排除保留。失敗試驗保留。
- [x] 137 幀區段及 571 幀聚焦比較：兩張新截圖 100% 覆蓋，既有可用真值無新增退步；212 pytest、Ruff、Mypy 42 檔與 20 幀 MPS 整合通過。
- [x] 兩支完整 1800 幀比較、四支影片完整解碼、全案例與 1131 幀固定區域回歸、43 個來源／工具檔及輸入／輸出雜湊核對。沒有新增已標註退步；整體品質仍 failed。

## Context and Orientation

分支 codex/restore-milestone1，HEAD b0cf8be22d735aeed9d7ce9e7d49cbe8ea441996，已有大量未提交修改，全部保留。src/dashcam_ai/lane/segmentation.py 的 _candidate_mask 用模型與白漆訊號產生候選，再以擴張車框遮蔽；occlusion_mask 同時供渲染使用。_context_rejection 判斷道路邊緣、箭頭與斑馬線。_narrow_stripe_support 用垂直於曲線的像素取樣判斷狹窄白漆及兩側暗路面。修正這兩個位置；模型、設定、追蹤、自車 ID 與原型後端保持既有來源。

兩支原片均 1280×720、30fps、1800 幀。F 暫按一基（frame_id=F−1），本輪截圖匹配以原片幀號為準。既有 615–710F 回報相當於 frame_id 614..709；重播使用 599..735 覆蓋前後緩衝及恢復位置。圖1的小圈只作可見局部正例，不能要求補整條被車遮住的線。

## Plan of Work and Milestones

第一階段建立 tests/fixtures/lane_left_paint_review.json，由原片確認 615、660 的可見白漆端點，保留既有 630、680 控制。保存截圖檔案雜湊及匹配幀號，不保存原圖到 Git。ignored output/left-review-v1/baseline 保存修改前 segmentation.py、設定與 manifest；不覆蓋歷史輸出。

第二階段優先解決額外車框邊界的誤遮。只有模型、狹窄白漆及道路方向都支持時才減少額外遮罩；原車框內保護不任意取消。對淺斜角候選，要求足夠窄漆、合理延伸方向及原有圖案排除通過才允許道路邊緣例外；不放寬全域模型或亮度門檻。若需對框內車身作像素辨識，先證明能區分車身與白漆；無可靠證據則保留限制，本輪不引入新模型或訓練。

第三階段凍結來源，以 scripts/replay_lane_review.py 的相同模型訊號／v8 車框比較 sample2 599..735。先完成聚焦回歸再做完整兩片對照；檢查既有 27 直線、16 真線、55 圖案、50 額外及 sample1 固定斑馬線區域 669..1799。已通過的案例有新增退步則拒絕來源，保留 failed／incomplete 證據並用新目錄重跑。新真值不能依偵測遮罩縮小分母。

## Concrete Steps

在 repository root 使用既有 .venv，不安裝依賴。

    .venv/bin/python -m pytest tests/unit/test_lane_segmentation.py tests/unit/test_lane_review.py
    .venv/bin/python -m pytest
    .venv/bin/python -m ruff check .
    .venv/bin/python -m mypy src
    git diff --check
    PYTHONPATH=src .venv/bin/python scripts/replay_lane_review.py --input samples/sample_2.mp4 --boxes output/sample_2_milestone2_yolop_v8/frames.jsonl --baseline-source output/left-review-v1/baseline/segmentation.py --start 599 --stop 736 --output output/left-review-v1/sample_2-window

本輪最終命令（重播時換不存在的新輸出目錄）：

    OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. .venv/bin/python output/left-review-v1/focus.py focus-v4
    OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. .venv/bin/python scripts/replay_lane_review.py --input samples/sample_2.mp4 --boxes output/sample_2_milestone2_yolop_v8/frames.jsonl --baseline-source output/left-review-v1/baseline/segmentation.py --start 599 --stop 736 --output output/left-review-v1/sample_2-window-v4
    OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. .venv/bin/python scripts/replay_lane_review.py --input samples/sample_1.mp4 --boxes output/sample_1_milestone2_yolop_v8/frames.jsonl --baseline-source output/left-review-v1/baseline/segmentation.py --output output/left-review-v1/sample_1-final-v4
    OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. .venv/bin/python scripts/replay_lane_review.py --input samples/sample_2.mp4 --boxes output/sample_2_milestone2_yolop_v8/frames.jsonl --baseline-source output/left-review-v1/baseline/segmentation.py --output output/left-review-v1/sample_2-final-v4
    OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. .venv/bin/python output/left-review-v1/final_audit_v4.py

每次重跑選不存在的新目錄。scripts/check_lane_review.py 用相同真值、前後 records 與實際渲染遮罩作比較；原有測試門檻不降低。補入實際命令、雜湊、通過／失敗結果後才能宣布改善。

## Validation and Acceptance

可見人工白漆在 10px 容許範圍內至少 90% 覆蓋；被車完全遮住的部分不補畫。弱漆、車身白色物件、圖案及道路邊緣負例需保留。恢復白漆的新正例測試必須在凍結 baseline 失敗、修正後通過；舊 baseline 通過案例不得新增退步。影片須全數解碼、來源／輸入／輸出雜湊一致。

目前正式 M2 macOS MPS blocked（dirty、驗證入口只支援 M1、正式 JSON missing），Linux CUDA missing。M1 macOS 16bedd0、legacy M2 macOS cef1205／CUDA c7d77e3 stale，M1 CUDA missing。先前 201 個測試及短片 MPS 只適用原來源；本輪須重跑本機檢查。CPU 白線對照沿用錄製車框，不代表重新驗證 GPU 追蹤。只更新 macOS 手動紀錄；兩個正式 GPU 平台仍須各自在 clean 固定來源與可用入口重新驗收。

## Idempotence and Recovery

修改限定白線、必要真值／測試及文件。大型影片／權重／私人截圖保持 ignored；Git 只保存雜湊與緊湊量測。原片、既有 v8／final-v7 輸出及前輪失敗資料不覆蓋。沒有 commit、push、依賴變更或追蹤修改授權。

## Interfaces and Dependencies

沿用 NumPy、OpenCV、LaneCurve、LaneLineFrame、BBox 及既有 replay/checker。以現有私有輔助判斷承載漆面證據，不增加公開設定。渲染與候選必須使用同一有效遮罩；不能只讓候選通過但最後仍被擦掉。

## Surprises & Discoveries

試驗 focus-v2 在 sample1 748／749 增加斑馬線區域誤畫，並使 sample2 151／152 原真線退步，拒絕採用。診斷顯示解除原本已可見候選的少量遮罩會改變幾何或圖案分類；收緊為至少一半取樣點原本被額外遮罩蓋住。615 可見白漆約 30px，margin 恢復內部的共識擬合最小跨度改為影像寬的 2%；一般幾何擬合仍維持 2.5%，沒有降低候選接受長度或延伸端點。

focus-v3 又使 sample2 630 原本中央真線退步，原因是新暴露的小漆面改變全局 crosswalk_context；目前以該幀車框的原擴張遮罩維持完整圖案／車輛污染分類，而候選與渲染仍使用恢復白漆後的有效遮罩。這不是沿用上一幀的遮罩，reset 會清除，分類完成必定恢復有效遮罩。

配對量測工具原本將新遮罩同時套在 before／after；本輪修正為每一版本使用各自的有效遮罩，並加入回歸測試。

615 的圈附近模型訊號仍高於門檻，不能將本例誤說成模型沒有找到線。660 的未遮蔽候選距既有人工真值約 1.79px、窄漆支持率 100%，但被 road_edge_context 排除；此前源頭診斷須區分影格與排除階段。

## Decision Log

2026-10-05 / Codex：先處理車框額外邊界及透視斜線，維持原框內保護；直接解除全部遮罩會把車身白色零件當線，超出本輪可接受風險。固定道路座標只用人工驗收，不放入偵測程式。

## Outcomes & Retrospective

本輪已批准的共用白漆局部修正與驗證完成；兩張圈選 0%→100%，舊 630／680 左線仍 0%，沒有新增已標註退步。完整回歸結果與精確來源見 MILESTONE2_LEFT_PAINT_REVIEW.md。模型缺候選未在本輪擴大為訓練或後端替換；整段左線、M2 品質與正式雙平台驗收仍未通過。

修訂說明：2026-10-05，記錄使用者新批准的左白漆局部修正與可觀察驗收條件。
