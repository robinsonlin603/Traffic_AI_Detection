# 排除公車停靠格，保留相鄰真車道線

本計畫依全域 PLANS.md 維護；Progress、Surprises & Discoveries、Decision Log、Outcomes & Retrospective 隨工作更新。使用者於 2026-10-05 明確批准本輪。批准範圍為共用公車格辨識、可見格線真值、必要測試、兩支影片回歸與本機紀錄；沒有提交、推送、模型訓練、追蹤修改或依賴安裝授權。

## Purpose / Big Picture


讓 sample1 19–97F 與 sample2 914–961F 中已確認的可見公車格邊界不再被畫成車道線，同時保留旁邊的真車道線及前輪修復的可見左白漆。固定座標僅供人工驗收，不能成為偵測規則。被公車遮住的部分不要求重建，也不以公車位置本身當成格線證據。交付修改前後對照影片與可重現的量測。

## Progress


- [x] 2026-10-05：使用者批准；檢查分支、HEAD、dirty worktree、既有計畫與平台紀錄。
- [x] 2026-10-05：凍結 91ed098d 白線來源與輸入雜湊；ORB 原片匹配截圖到 frame_id 29／93／911（inliers 360／339／228）。
- [x] 加入可見公車格／相鄰真線案例，現為 136 個（129 個禁畫區、7 個真線控制），以及左右鏡像、亮／暗、車身／道路外、場景切換、可靠運動與到期的測試。新 fixture 部分過寬區域經放大原片修正；原有 fixture 不變。
- [x] 2026-10-06：v26 共用修正及 571 幀聚焦回歸完成；136/136 公車格案例通過、無新增已標註退步，保留失敗／中斷試驗。
- [x] 2026-10-06：v28 完整 AFTER 3600 幀與凍結 BEFORE 比較／四片解碼、所有 fixture 稽核及來源核對通過；249 pytest、Ruff、Mypy 42 檔、20 幀 MPS 整合通過。
- [x] 2026-10-06：更新本機紀錄與結果文件，輸出半速對照短片；完整兩段 127 幀視覺核對通過，M2 既有失敗及正式平台限制保留。

## Context and Orientation


工作目錄為 repository root。分支 codex/restore-milestone1，HEAD b0cf8be22d735aeed9d7ce9e7d49cbe8ea441996，已有大量 dirty 變更，必須保留。src/dashcam_ai/lane/segmentation.py 的 YoloPLaneLineDetector 從模型概率（像素屬於白線的分數）及道路白漆提取曲線，再用 _grouped_markings 與 _context_rejection 排除箭頭、格線、斑馬線。現有格角規則要求相连轮廓及低實心比例，並只在影像寬 4% 的鄰域收回分離長邊。遮蔽超過一半的輪廓不作圖案證據。這些條件可能漏掉與格角分離的長邊，须以實際原片診斷確認。

兩支 samples/sample_1.mp4、samples/sample_2.mp4 均 1280×720、30fps、1800 幀。frame_id 為從零起算的內部幀號；截圖已直接比對原片，播放器 F 不硬套減一。截圖播放器顯示 29F、93F、911F；第三張作進入原回報 914–961F 前的緩衝案例。私人截圖和大型資料不存入 Git。錄製車框來自 output/sample_1_milestone2_yolop_v8/frames.jsonl 與 sample_2 對應路徑。output/human-review-v1/cache 的 npz 保存部分原幀的模型概率與道路分數，可用於快速重播；正式完整對照由 scripts/replay_lane_review.py 生成。

前輪目前白線來源 SHA256 為 91ed098dfd5f76142280fbd24dce6769e5ede1ecbf111fb22c865373623fd727。兩張左線人工錨點 frame_id 615、660 覆蓋 100%；630、680 仍 0%。既有 16 controls 15 通過、27 straight 25 通過、55 markings 45 通過、extra 50 中 49 通過、sample1 固定右上斑馬線 1131 幀零誤畫。已通過案例不得新增失敗。這些是已知開發影片，不能當成獨立泛化準確率。

## Plan of Work and Milestones


第一階段在 ignored output/bus-review-v1 保存白線 baseline、來源與資料雜湊。對齊截圖並查看公車格長邊如何形成候選、格角在哪裡遺失。新增 tests/fixtures/lane_bus_bay_review.json，只標記原片可見格線與真線，涵蓋清楚、斷漆、遮蔽及區段前後。結果須先重現 baseline 誤畫；完全不可見部分不計入人工真值。

第二階段僅修改 segmentation.py 的圖案上下文及必要私有輔助函式。以可見長邊和橫向封口／格角之間的幾何關係補足相連輪廓規則，限制在有證據的格邊，避免將真縱向白線一併刪掉。優先單幀幾何；若必須依相鄰幀，須有可驗證的運動對應與短時失效機制，不增加固定畫面遮罩或公車位置例外。加入 tests/unit/test_lane_segmentation.py 正反例：分離格角、磨損格線、車輛遮蔽、真虛線／實線、淺斜真線、箭頭、斑馬線。新修正案例須 baseline 失敗、修改後通過，負例 baseline 與修改後皆保留。

第三階段先用已保存概率的 571 幀聚焦比較，再重播 sample1 與 sample2 全片各 1800 幀，沿用同一錄製車框。前後版各用自己的有效遮罩評分。所有現存 fixture 與新人工案例核對，觀看兩段和緩衝區。生成完整可解碼的 annotated.mp4 與 comparison.mp4。更新 docs/MILESTONE2_BUS_BAY_REVIEW.md、ROADMAP 及 validation/milestone-2/macos-mps.md；只更新本機手動紀錄，不編造平台 JSON。

## Concrete Steps


在 repository root 使用既有虛擬環境，不安裝依賴：

    .venv/bin/python -m pytest tests/unit/test_lane_segmentation.py tests/unit/test_lane_review.py
    .venv/bin/python -m pytest
    .venv/bin/python -m ruff check .
    .venv/bin/python -m mypy src
    git diff --check
    OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. .venv/bin/python scripts/replay_lane_review.py --input samples/sample_1.mp4 --boxes output/sample_1_milestone2_yolop_v8/frames.jsonl --baseline-source output/bus-review-v1/baseline/segmentation.py --output output/bus-review-v1/sample_1-final-v28-reproduced
    OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. .venv/bin/python scripts/replay_lane_review.py --input samples/sample_2.mp4 --boxes output/sample_2_milestone2_yolop_v8/frames.jsonl --baseline-source output/bus-review-v1/baseline/segmentation.py --output output/bus-review-v1/sample_2-final-v28-reproduced

上面的原成對工具用新的輸出目錄重現目前 v28 來源，不能覆蓋既有證據。本輪實際完整驗證使用 ignored output/bus-review-v1/replay-frozen-before-v28.py，額外指定 --frozen-before output/bus-review-v1/sample_1-final-v26 或 sample_2 對應基準；方法及一致性檢查見下方修訂紀錄。

聚焦與稽核輔助程式留在 ignored output/bus-review-v1，具體命令與結果在執行後补入。每次試驗選新的輸出目錄，不覆蓋基準或歷史證據。最終完整來源必須與聚焦、測試及本機整合一致。

## Validation and Acceptance


可見人工公車格禁畫區疊圖像素為零；真白漆人工線在既有 10px 容許範圍內至少 90% 覆蓋，直線幾何亦須通過。不得降低門檻或用新車框遮罩消除真值分母。稀疏錨點通過不等於整段通過；只對真正檢查的幀／露出白漆提出品質結論，額外標註與保留案例分開報告。完整重播須全部幀完成、影片完整解碼、來源／輸入／輸出雜湊一致。

正式 M2 macOS MPS blocked（dirty、validate 只支援 M1、正式 JSON missing）、M2 Linux CUDA missing。M1 macOS 16bedd0、legacy M2 macOS cef1205／CUDA c7d77e3 均 stale，相對目前 HEAD 不適用；M1 CUDA missing。CPU 白線重播及沿用車框不能證明新的 MPS/CUDA 全片行為。本機短片整合可驗證實際 MPS 被觀察，但不得取代正式 clean 固定來源驗收。兩個 GPU 平台均須各自重跑；不修改別的平台報告。

## Idempotence and Recovery


來源基準與資料只增加新版本，不覆蓋之前結果。失敗試驗清楚標 incomplete/failed，不當成完成證據。回退只恢復本輪修改，保留全部使用者與前輪 dirty 變更。任何未通過品質案例列出原片幀號與原因，不標 M2 完成。不提交或推送。

## Interfaces and Dependencies


沿用 NumPy、OpenCV、LaneCurve 與現有重播／評分工具，不新增公開設定或後端。不改檢測器公開介面；新私有函式接收該幀白漆及遮罩，以同尺寸二值圖回傳有證據的格線上下文。遮蔽區不得提供虛構白漆，原本車身及自車保護保持。

## Surprises & Discoveries


初始原始碼及既有報告顯示 sample1 frame_id 50、sample2 930 的公車格仍誤畫；sample1 95 已排除。第三張截圖 UI 是 911F，不在回報 914–961F 裡；需要以原片對齊而不是硬套 F−1。

## Decision Log


2026-10-05 / Codex：修正共用格線上下文，不對樣片或幀號寫偵測例外；固定標註座標只供量測。優先利用可見的封口與格角，避免以靠右或有公車就刪白线。原片和模型不變，可直接比較修正效果。

## Outcomes & Retrospective


試驗輸出全部保留，incomplete 不列成功證據。v18 原 23 錨點通過，但逐幀 contact review 發現 sample1 24/32/51/54/58/66 的格邊及 sample2 931..936 的標字仍誤畫，不能當作整段通過。新增 fixture 改為逐幀檢查兩段 127 幀的格邊與格內字樣，加 2 個進入前案例及 7 個真線控制，合計 136；左界依原片人工檢查、保留相鄰真線，舊人工 fixture 不改。最初新 fixture 過窄的部分禁畫區會漏計格邊，已修訂並保留修訂前雜湊與輸出。v20 新案例 sample2 54/54、sample1 76/82；v23 為 54/54、77/82 且新增真線退步，拒絕採用。v24 恢復一般窄線的原本寬度／側面量測，公車格上下文單獨處理寬漆、只在窄漆斷續而同場景時追隨目前白漆（不延長年齡），每塊消失證據獨立失效；新增單側格角／封口及 T 形停止線反例。局部測試通過，区段評分待確認，完整兩片尚未开始；整體 M2 仍 failed、正式平台狀態不變。v18 來源已更動，其 MPS smoke 與一般檢查對目前來源 stale，須重跑。

修訂說明：2026-10-05，記錄使用者批准的共用公車格修正與驗收範圍，建立實作前計畫。

修訂說明：2026-10-05，加入三張原片匹配與新增真值。公車格旁另有真虛線，新增的 911／914／960／961 禁畫區改為格邊範圍，並加入右虛線正例；舊 930 公車格標註仍正確、完全保留。試驗發現單幀筆畫會隨光影断開，加入可見白漆與道路光流的有期限延續。

修訂說明：2026-10-06，保留 v11 已觀察退步及 v12 不完整來源紀錄；v14 針對原因修正，未宣稱完整回歸通過。

修訂說明：2026-10-06，v18 對近文字筆畫保留包圍盒及實心輪廓保護，遠側格邊以實際空白區輪廓投影判斷，避免相連孔蓋污染；區段 23 案例通過後擴大回歸。

修訂說明：2026-10-06，逐幀檢查修正稀疏標註盲點，新增公車格內標字與格角；保持批准的共用圖案上下文範圍，不在 runtime 使用任何新真值座標。

修訂說明：2026-10-06，v24 區段 sample1 80/82、sample2 54/54，新增 sample1 50 真線退步，拒絕。v25 提高格邊連續長度證據後真線恢复，81/82、54/54；28 的磨損格邊頭仍漏。v26 只讓目前可見、窄白漆且與已確認長格邊共線的短片段在有限距離內繼承上下文，不畫缺漆空隙；加入平行／不同方向／遠距反例。區段重播及一般檢查、本機短片整合執行中，完整回歸尚未開始。

修訂說明：2026-10-06，v26 區段 136/136（sample1 82/82、sample2 54/54），既有已測案例無新增退步；已檢視全部 127 幀圖像。243 pytest、Ruff、Mypy 42 檔、diff 與 20 幀本機 requested/resolved=mps 整合／解碼及來源核對通過。白線來源 a196fca4f1e3bf15f74203c5f27e69c47b7c9ad2a6e627505ea1568c7c9055f1。571 幀聚焦回歸仍執行中，全片未開始，不能將区段結果當作完整品質通過。

修訂說明：2026-10-06，使用者要求接續中斷進度。來源／設定／測試雜湊皆與 v26 一致；原 571 幀程序已中斷，sample1 完成 292 幀、sample2 27 幀。保留 incomplete，新的 focus-v26-resumed 核對並複用完整 sample1 紀錄，重新跑全部 279 個 sample2 聚焦幀。合併來源核對與品質檢查通過後才開始兩片完整重播。既有 v26 一般檢查與 MPS 短片來源未變，仍為本機局部證據。

修訂說明：2026-10-06，focus-v26-resumed 完成 571 幀來源核對及所有可用舊 fixture 回歸，無新增已標註退步；新案例 136/136、615／660 左線保留，630／680 仍漏。已開始 sample_1-final-v26、sample_2-final-v26 兩片各 1800 幀成對 CPU 重播，來源凍結不修改。

修訂說明：2026-10-06，完整連續序列已跑過兩段公車格，提前快照核對 sample1 82/82、sample2 54/54；sample2 615／660 控制仍 100%，630／680 仍 0%。快照只驗證已完成區段，完整兩片重播與所有 fixture 稽核仍待完成，不能列為全片通過。

修訂說明：2026-10-06，v26 完整 3600 幀重播、解碼及來源核對完成；公車格 136/136，但 sample2 額外保留真線 1410–1414 共 5 幀退步（24/25 → 19/25），拒絕採納。這個區段不在原 571 幀 cache，聚焦不足以證明全片回歸。保留完整失敗輸出，加入同場景幾何診斷与回歸後再修正共用條件；其餘舊案例未退步。

修訂說明：2026-10-06，接續 v26 全片失敗。sample2 1410 的新上下文來自路口網格母輪廓，實際漆佔填滿輪廓僅 43.3%；其空格不能當作密集字樣。v27 僅讓實際漆佔母輪廓至少 55% 的複雜空腔提供字樣證據，格角規則不變。新增左右鏡像白色／低飽和黃色路口網格旁真線整合反例，四例在 v26 均失敗、v27 通過，保留證明。原有真值及門檻不變；增加 sample2 1360–1430 的 71 幀連續 cache，聚焦共 642 幀。

修訂說明：2026-10-06，完整驗證複用 v26 已完整解碼／核對的 frozen BEFORE 紀錄及其各幀自身遮罩，AFTER 每片由 0 起連續重新推論 1800 幀。先核對完整基準、紀錄雜湊、原片／模型／設定／車框／基準來源一致、其餘 runtime 及原重播來源一致，以及兩版推論函式 AST 一致；拒絕來源／輸入／順序／遮罩不符。不宣稱新一輪兩版同時共用記憶體中的概率，AFTER 仍重新計算。此為相同凍結基準的驗證效率調整，未修改 runtime 後端或擴大批准範圍。方法／基準 manifest／summary／before.jsonl 雜湊記錄於新 manifest；ignored replay-frozen-before-v27.py 是本輪驗證輔助，原 paired replay 仍可重現。v26 檢查及 MPS smoke 對 v27 來源已 stale，須重跑。

修訂說明：2026-10-06，v27 擴大聚焦 642 幀完成，公車格 136/136，但 1411–1414 真線仍退步，拒絕；未啟動全片。白漆內小孔洞仍可形成近似字樣，實際位於候選前方而非格邊旁。v28 將實際字樣輪廓與可見長邊的沿線間距由寬度 8% 收緊為 2.5%，避免遠處圖案影響另一段真線，另加入兩個鏡像遠處字樣反例。原 GT 及評分門檻不變；先重播公車格／路口連續窗口，再擴大聚焦與全片。v27 的一般／MPS 檢查對 v28 stale。聚焦亦可複用上一輪同概率／相同連續緩衝序列的已完成 BEFORE 紀錄與自身遮罩，核對雜湊並保存來源。

修訂說明：2026-10-06，v28 公車格／路口連續窗口 236 幀通過，新公車格 136/136、既有路口控制 10/10，1410–1414 全部恢復，沒有新增已標註退步。249 pytest、Ruff、Mypy 42 檔通過，四個格邊正例在 frozen BEFORE 失敗、兩個遠处字樣負例在 v27 失敗；最終來源皆通過。642 幀擴大聚焦與最終來源 MPS smoke 執行中，尚未以窗口結果宣稱整片通過。

修訂說明：2026-10-06，v28 642 幀擴大聚焦完成，所有可用舊案例無新增退步，公車格 136/136、路口舊真線恢復、615／660 保留 100%，630／680 仍漏。最終來源 20 幀 requested/resolved=mps 整合／解碼與来源核對通過，249 tests、Ruff、Mypy 42 檔均通過。已開始兩片各 1800 幀完整 AFTER 連續 CPU 重播，BEFORE 複用 v26 完成的凍結紀錄及自身遮罩；來源／輸入／工具凍結。完整回歸完成前不交付成功結論。

修訂說明：2026-10-06，v28 642 幀聚焦及完整 AFTER 3600 幀連續重播完成，BEFORE 複用通過完整性／來源／輸入核對的凍結紀錄與自身遮罩，before.jsonl 完全相同。公車格 136/136、1410–1414 真線恢復，所有已標註舊案例沒有新增退步，完整兩段 127 幀再次檢視通過。249 pytest、Ruff、Mypy 42 檔、同來源 20 幀 MPS 整合通過；半速短片已輸出並解碼。本輪完成，630／680 等既有失敗保留，正式 MPS blocked／CUDA missing，M2 未完成；未 commit／push。
