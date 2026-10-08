# Milestone 2：人工 v8 回報與連續白線修正

本 ExecPlan 依全域 PLANS.md 維護。Progress、Surprises & Discoveries、Decision Log、Outcomes & Retrospective 隨實作更新。使用者於 2026-10-04 明確批准前一則提出的三階段修正範圍。

## Purpose / Big Picture

讓兩支現有行車影片中的可見直白線不被畫彎，排除公車停靠格、箭頭、斑馬線、機車停等區與指定道路邊緣，恢復因後處理流失的可見左白線。交付完整前後比較影片與逐案例品質記錄；不以程式測試或曲線數代替人眼品質。無法看見或被完全遮蔽的線不補畫。

## Progress

- [x] 2026-10-04：使用者批准；檢查 branch、HEAD、dirty worktree、active docs 與平台報告。
- [x] 2026-10-04：找到兩支 v8 1800 幀影片及逐幀資料；抽查開頭／結尾變彎和左線缺失。
- [x] 保存修改前來源、設定、輸入、車框與 v8 輸出雜湊；固定 16 個人工代表案例、sample_1 1131 幀固定斑馬線區域稽核與原有 16 個控制。
- [x] 完成已證實原因的局部修正：白漆對比、直線對齊、重複圖案與複雜格角；201 個測試、Ruff、Mypy 通過。明亮路面／格角在 baseline 失敗，寬漆在 trial5 失敗，配對折點／長條斑馬線在 final-v2 失敗，修正後通過。
- [x] final-v2 兩片完整 1800 幀與解碼完成，品質 failed：固定斑馬線區域新增 75 幀誤畫，原有圖案案例新增 3 例退步；保留產物，拒絕採納。
- [x] 修正 final-v2 退步。實際短線 frame_id 148 恢復 100% 覆蓋；斑馬線 1674..1678 五幀全無誤畫；預設來源 20 幀 MPS CLI／解碼／自車診斷 passed。
- [x] final-v3 的 669..799 固定區域仍新增 18 個誤畫影格；中斷兩片並保留 complete=false，未列完成證據。
- [x] 修正短碎片因近車框而略過長度門檻的問題：必須有持續可見窄白漆才能救回。655..799 成對重播／解碼完整，669..799 的 131 例全無誤畫；trial8 的 16 錨點與 27 直線沒有新增退步，原有機車短線／左線量測仍保留。
- [x] 晚段單幀反證揭露窄漆條件仍能救回斑馬線；中斷 final-v4（complete=false），未列全片證據。
- [x] 保留對比切割前的原始模型連通區作短碎片救回證據；原始形狀與漆面都須支持被車框截斷，對齊不能新增此證據。815..879 同機率比較，final-v4 5 幀誤畫→目前 0；65 幀／影片解碼完整。trial10 開頭 11/13、結尾 14/14，16 錨點 10/16，沒有新增退步。保留整塊模型像素的 trial9 只 3/13 開頭直線，拒絕採納；最後仍用對比過濾像素作幾何。
- [x] final-v5 前 800 幀新增額外真線 frame_id 633 退步（100%→51.14%），中斷兩片、保留 complete=false。白漆過寬使嚴格對比像素分成兩段；下段模型小折點放大端點切線，使原有合併拒絕。
- [x] 修正可見寬漆合併：只在合併後達通常垂直跨度、間隔不被遮、間隔中心至少 90% 有對比白漆時，使用兩段的主要方向檢查。615..635 同機率比較 final-v5／目前版本，633 覆蓋 51.14%→100%，沒有新增退步。空白／車框遮蔽／旁邊白漆控制通過；旁邊白漆控制在 trial13 失敗。
- [x] final-v6 兩片完整重播／解碼，但固定斑馬線區域仍新增一個 frame_id 1509 誤畫，品質 failed，未採納。
- [x] 修正候選跨度不等於正常輸出長度：通常跨度以下須有窄漆或合格車遮斷證據，既有短漆尾段控制保留；36／44px 回歸在 final-v6 failed、修正後 passed；1488..1519 成對重播／解碼，誤畫 1→0。
- [x] final-v7 兩片各 1800 幀完整重播／解碼、來源／輸入／輸出核對及全部列定控制完成；固定區域誤畫 130→0，沒有新增案例退步。
- [x] 發布 review 與平台狀態；品質仍 failed，M2 未完成。

## Context and Orientation

工作目錄為 repository root。分支 codex/restore-milestone1；HEAD b0cf8be22d735aeed9d7ce9e7d49cbe8ea441996，原有大量尚未提交的修改全部保留。src/dashcam_ai/lane/segmentation.py 的 YoloPLaneLineDetector 以既有 models/yolop-640-640.onnx 產生白線／道路機率，取可見白漆候選，經車框遮罩、圖案判斷、擬合、合併與跨幀確認，輸出 domain/lane_lines.py 的曲線。visualization/annotator.py 畫白色曲線與綠色車框、黃色軌跡。src/dashcam_ai/lane/prototype.py 是前輪未通過採納的實驗，不能直接轉成預設。

samples/sample_1.mp4 和 sample_2.mp4 均為 1280×720、30fps、1800 幀。output/sample_1_milestone2_yolop_v8 與 sample_2 的對應目錄有 annotated.mp4、frames.jsonl、lane-lines.jsonl，後兩者幀號為 0..1799。其 metadata 沒有完整白線來源／設定 manifest，名稱 v8 不能證明來源等於目前程式。先保留並雜湊現有輸出，再以冻结目前程式與相同車框作成對重播；分開呈現歷史 v8 與可重現 baseline，不把它們混為同一來源。

使用者提供 13 筆回報，1..1800 暫以一基計數轉成 0..1799；4..7 與 1..13 重疊，合併追查。sample_1：1..13 右線變彎；19..97 公車格；110..135 與 215..223 直行箭頭；158..164 疑似光影，未確認不能當負例；670..1800 右上斑馬線間歇誤畫。sample_2：146..159 機車停等區；433..435 指定右側邊緣；539..541 行穿線；615..710 左線缺失，711 附近恢復；914..961 公車格；1787..1800 右線變彎。保留原始使用者範圍以便修正幀號解釋，先在邊界前後一幀比對，避免差一幀。

## Plan of Work and Milestones

第一階段：在新的 ignored output/human-review-v1 保存 baseline/segmentation.py 與相關設定／輸出雜湊，不覆蓋 v8、原片或前輪證據。將全部使用者區段記入 tests/fixtures/lane_human_review.json。由原片先確認真白漆可見部分與非車道圖案區域，再看模型輸出，為代表幀建立折線／多邊形；必要的鄰幀標註以局部運動協助，但需看原圖確認。疑似光影維持 uncertain 直到可確定；指定右側邊緣只排除該物件。此階段驗收是可定位所有回報，清楚區分已量測影格與尚未逐幀標註區段，不以抽查推算全片正確率。

第二階段：在 segmentation.py 修正由證據指出的錯誤。針對可見直白漆，比較直線與二次曲線擬合，只有資料支持彎曲才保留曲線；檢查合併與平滑是否把遠近不相關片段拉成彎線，保留真正彎道控制。對公車格、停等框、箭頭、斑馬線，使用完整局部圖案、白漆寬度與關係排除，不使用影片名稱、幀號、固定道路區域或一律畫直線的例外。對左線，分開診斷模型候選、車框裁切、圖案分類、道路上下文、長度與確認失敗；只恢復有可見漆面證據的候選，不能取消所有遮罩或延長空白補線。若既有模型根本沒有候選，記為限制，模型更換／訓練需另行計畫。tests/unit/test_lane_segmentation.py 加入能在保存 baseline 失敗、修正版通過且保留正常控制的測試。

第三階段：沿用 scripts/replay_lane_review.py 的相同模型輸出、相同最新車框成對重播兩片各 1800 幀，保存各自完整來源與輸入雜湊，逐幀解碼 annotated 與 comparison。沿用 scripts/check_lane_review.py 的可見折線覆蓋與禁止區域筆畫交集評估固定人工幀，另檢查直線形狀，防止單靠部分命中通過。列出已標註幀的誤畫數與最長連續漏畫，未標註區段明確標為人眼審查範圍。用相同新車框驗證原有白線控制；車輛輸入雜湊不變只證明沿用自車排除結果，不宣稱這個 CPU 白線重播重新驗證 GPU 追蹤。執行預設分析入口短片整合檢查，檢查自車／旁車控制範圍。

## Concrete Steps

全部在 repository root、既有 .venv 執行，不安裝依賴。先建立唯一新輸出路徑，若已有檔案換新路徑。

    .venv/bin/python -m pytest tests/unit/test_lane_segmentation.py tests/unit/test_lane_review.py
    .venv/bin/python -m pytest
    .venv/bin/python -m ruff check .
    .venv/bin/python -m mypy src
    git diff --check
    PYTHONPATH=src .venv/bin/python scripts/replay_lane_review.py --input samples/sample_1.mp4 --boxes output/sample_1_milestone2_yolop_v8/frames.jsonl --baseline-source output/human-review-v1/baseline/segmentation.py --output output/human-review-v1/sample_1-final-v7
    PYTHONPATH=src .venv/bin/python scripts/replay_lane_review.py --input samples/sample_2.mp4 --boxes output/sample_2_milestone2_yolop_v8/frames.jsonl --baseline-source output/human-review-v1/baseline/segmentation.py --output output/human-review-v1/sample_2-final-v7

本輪 checker 命令（sample2 換 input／目錄與控制 fixture）：

    PYTHONPATH=src .venv/bin/python scripts/check_lane_review.py --input samples/sample_1.mp4 --cases tests/fixtures/lane_human_review.json --baseline output/human-review-v1/sample_1-final-v7/before.jsonl --records output/human-review-v1/sample_1-final-v7/lane-lines.jsonl --output output/human-review-v1/sample_1-final-v7/human-review.json

直線另用 lane_human_straight_review.json、圖案用 lane_markings_review.json，原有控制用 lane_stability_control.json／lane_stability_review.json。16 原始錨點另從 human fixture 的 anchor_cases 選取；50 額外案例只選 lane_prototype_review.json 原有 split=extra，標籤不變。source_hashes 記錄各份案例與 records 的精確雜湊，結果見 MILESTONE2_HUMAN_REVIEW.md。完整重播期間來源凍結；需要修正時保留舊 incomplete 或 failed 產物，以新路徑重跑。

## Validation and Acceptance

可見真白線每個已標註案例中心線 10px 內覆蓋至少 90%；直線案例亦不得出現超出人工白漆容許範圍的錯誤彎段，並記錄雙向距離，防止錯誤延長仍算合格。禁止區域不應有白色疊圖，真線與圖案交會位置不納入禁止區域。分母依人眼可見白漆，不依模型遮罩縮小。原有 16 個白線控制不得退步。疑似光影若無法確定，維持未判定，不算通過或失敗。記錄全部人工抽查點、兩支完整比較影片以及尚未覆蓋的時間範圍，不宣稱現有兩片的結果代表新道路泛化。

pytest、Ruff、Mypy 和 diff 檢查須通過；兩片各 1800 幀記錄與影片完整解碼，來源／輸入／輸出雜湊一致。現行 macOS MPS 正式 M2 blocked（來源 dirty、validate 只支持 M1、正式 JSON missing）；Linux CUDA missing。M1 macOS 16bedd0 與 legacy M2 macOS cef1205／CUDA c7d77e3 均 stale；M1 CUDA missing。当前 CPU 白線結果只驗證白線；前輪 MPS 自車專項 passed 只驗證其記錄範圍。本輪更新 macOS 的手動紀錄，絕不編造 CUDA 機器報告，正式雙平台留待乾淨來源和可用驗證入口。

## Idempotence and Recovery

原片、模型、v8、既有自車遮罩與追蹤實作保留。本輪只修改白線、必要量測工具、對應測試、人工案例及報告。所有大型輸出 ignored，不提交私人路徑或影片。新 replay 路徑不可覆蓋，來源变化會使 manifest 拒絕完成。未授權 commit、push、模型訓練、更換預設後端、平台工具擴充或其他 milestone。

## Interfaces and Dependencies

沿用 OpenCV、NumPy、Pydantic、YoloPLaneLineDetector、LaneCurve、LaneLineFrame 與既有 replay/checker；盡量維持公開設定／介面。新 shape 檢查放在同一白線模組，以語意明確的私有函式實現。人工 fixture 使用原片 SHA256、尺寸、fps、原始回報範圍與零基測試幀、expected_line、polyline/polygon、category 和可見性狀態。若 checker 新增直線檢查，其門檻固定在實作調整之前，無法達標如實報告。

## Surprises & Discoveries

2026-10-04：v8 已有最新自車排除設定的 metadata 和完整 frame/lane 記錄，但無白線來源 manifest，因此不假定它就是現在的 baseline。原片與 v8 第 6 幀抽查顯示直白漆被畫成彎線；sample_2 第 630 幀左側被多車遮蔽，必須分離可見真值與完全遮蔽。sample_1 158..164 回報僅為可能光影，不能未核對即把真正白線標為負例。

2026-10-04：開頭假彎線的模型候選連到了明亮柏油；只用絕對亮度無法排除。加入局部亮度對比及可見白漆共線對齊後，frame_id 6 的局部真值覆蓋由 68.97% 提升至 100%，形狀最大偏差由 19.95px 降至 1.77px。這是代表幀結果，尚非全部 1..13F 通過證明。

2026-10-04：sample_2 frame_id 630/680 可見左漆仍無模型候選，660 只恢復約 62.75%；機車框也裁掉部分可見漆。sample_1 50 和 sample_2 930 的公車格長邊與一般縱向白漆外觀相近；不以指定座標或全域放寬門檻強行移除。sample_2 540 的局部條紋群不足，擴大條紋群凸包沒有改善，試驗已撤回。

2026-10-04：補標 27 個連續直線影格後，trial5 顯示結尾 9 幀真線截短，拒絕採納並中斷首輪全片（manifest complete=false）。根因是對齊步驟把近處較寬白漆當成無效觀測；保留近處寬漆後 sample_2 聚焦尾段 14/14 通過，sample_1 11/13（開頭兩幀仍待確認，不虛報全段通過）。另加 widening-paint 回歸，在 trial5 失敗、修正後通過；重新以 final-v2 唯一路徑跑全片。

2026-10-04：額外合成彎漆圖案在新舊 grouped-markings 都被 parking 誤判（同為 7129 個遮罩像素）；幾何對齊的彎漆單元控制通過僅證明該步驟沒有強制畫直，不代表整個圖案分類已驗收彎道。最終報告限制相應敘述，未將既有失敗計成新退步。

2026-10-04：final-v2 全片量測發現 sample1 固定區域誤畫總數雖從 130 降到 85，卻增加 75 個 baseline 正確的影格；sample2 原有 55 圖案控制中有 frame_id 148、1676、1677 三例退步。不能只用淨改善宣稱可採納。斑馬線退步原因是新增遠處長寬比條件替換了原有近處長条規則；恢復舊規則、以新規則補充。道路邊緣例外不是這些退步的主要原因，反證試驗保留。

2026-10-04：frame_id 147→148 的真短線整體配對 RMS 8.93px，中位方向差 4.38°，但兩個局部平滑折點造成方向差 77°／123°；90 百分位門檻錯誤重設 ID，保留舊短線才使可見覆蓋降到 75%。改以方向差中位數拒絕整體不一致，原有整條曲線距离／重疊檢查保留。實際曲線存成 fixture；新測試在 final-v2 失敗、目前版本通過。

2026-10-04：舊 points 控制只量原始曲線，可能把渲染遮罩完全擦掉的線算通過。改為有有效遮罩紀錄時量實際可見中心線；無遮罩的歷史記錄維持原幾何量測。以整行真值被遮掉的回歸證明舊工具錯誤通過、新工具正確失敗。未調低門檻，這也不能替代逐幀原片人工真值。

2026-10-04：final-v3 中固定區域 669..799 的誤畫 38→19，但新增 18 個 baseline 正確影格。剩餘退步的短碎片只因端點距車框 8px 內而被救回；亮度對比切開舊模型連通區後，這種近鄰不是「車遮掉了真車道線」的證據。於短碎片救回時再要求窄白漆支持率至少 90%，不改變可見白漆長度、車框遮罩或真值分母。實際重播同區域 131/131，短線／左線錨點未退步；重新以 final-v4 凍結來源完整驗證。另作保留連通區的對比試驗，僅為 ignored 探索，沒有採納或切換來源。

2026-10-04：晚段斑馬線本身也有窄直漆面，幾何對齊後支持率達 100% 仍不代表車道線。保留原始模型连通區作救回資格：只有原始父區域能通過形狀、上下文、窄漆與車框截斷檢查，對比切割的碎片才可救回；對齊前後都必須有證據。父區域只在有待救回短線時分析一次，避免在所有影格重複擬合。撤回新增道路邊緣寬放；左線沒有因此宣稱恢復。較保守的條件可能拒絕弱／破損短線，另檢查前輪 50 個保留案例。原始父區域不渲染、不加入候選、不改模型或車框遮罩。

2026-10-04：額外真線 633 是畫面中央正常長漆帶，與左線缺失錨點不同。對比過濾使 530..610 的寬漆中心缺少模型候選；下段端點切線噪聲讓 80px 間隔不能合併。只有可見漆面充分支持的長漆帶可改用主要方向；沒有漆、被遮、短橫紋仍用原判斷。間隔必須檢查中心像素，不能只取旁邊亮峰。原始模型連通區可能是空心或有多段，不能要求整段中心模型像素存在；新增的物理漆面檢查處理這個限制，模型沒有被替換。

2026-10-04：final-v6 完整驗證仍新增一個 frame_id 1509 誤畫（806px）。候選剛達 5% 提取跨度，但端點長度不足通常白線，原長度閘門沒有檢查它。通常跨度以下仍需可見窄白漆或合格車輛截斷證據；保留既有短漆尾段控制。36／44px 回歸在 final-v6 failed、目前 passed；1488..1519 同機率比較 1→0，32 幀完整解碼，最終改用 final-v7。

## Decision Log

2026-10-04 / Codex：保留 user F 與零基 frame_id，暫按一基解釋並查區段邊界。理由是使用者列到 1800，而輸出最大 frame_id 為 1799。合併重疊的開頭回報，不重複計為兩個獨立事件。

2026-10-04 / Codex：使用目前正式後處理作 baseline，不直接採納前輪 failed 的候選分類／像素遮罩原型。理由是該原型已有真線退步，改善新案例不得犧牲原有控制。使用者批准連續品質修正，但未批准新模型訓練或平台工具擴充。

2026-10-04 / Codex：sample_2 146..159F 與前輪「機車左侧短白線是真車道線」標註可能矛盾，已請使用者確認具體線段；等待期間保留舊正例，未把停等區整段一律標負。sample_1 光影報告同樣維持 uncertain。

2026-10-04 / Codex：補足直線幾何最大偏差檢查，不只依平均覆蓋判定。571 個聚焦幀中前後獨立記錄的車框有效遮罩完全相同；這證明本輪改動未替換該範圍的遮罩，不能替代 GPU 車輛全片重驗。

## Outcomes & Retrospective

本輪局部修正、人工證據與驗證已交付，完整品質仍 failed，M2 未完成。兩片各 1800 幀成對重播／解碼與全部來源核對完成；201 個測試、Ruff、Mypy 通過。直線 16/27→25/27，指定固定斑馬線區域誤畫 130→0；原有可見真線控制 14/16→15/16，圖案 39/55→45/55，額外案例 49/50，沒有新增已標註退步。公車格、sample2 540 斑馬線、630／660／680 左線與開頭兩幀仍未達標；停等框／疑似光影待確認。這些限制需後續另訂語意、候選召回與車身像素標註計畫，不擅自替換模型。本輪預設 20 幀 CLI 實際 resolved=MPS，但正式 macOS MPS blocked、Linux CUDA missing；沒有 commit 或 push。詳細量測見 MILESTONE2_HUMAN_REVIEW.md。

修訂說明：2026-10-04 新增本輪獲批准的人工回報修正計畫，完整保存範圍、量測與驗收限制。
