# Milestone 2：標線分類與車身像素遮罩原型

本 ExecPlan 依全域 PLANS.md 維護。2026-10-04 使用者批准前輪提出的方案 B 原型評估；本文件在原型程式實作前具體化核准範圍。Progress、Surprises & Discoveries、Decision Log、Outcomes & Retrospective 隨工作更新。沒有 commit/push 授權。

## Purpose / Big Picture


讓使用者看到相同道路影格在既有 YOLOP、車身像素遮罩、標線候選分類器與兩者組合下的白線差異。目標是保留實際可見左線，排除箭頭、停車格與行穿線；沒有品質證據前不切換正式 analyze 後端。像素遮罩只遮實際車身，而分類器從影像與候選線形狀預測標線種類，不能在推論時查人工標註。

## Progress


- [x] 2026-10-04：使用者批准方案 B；確認分支、HEAD、dirty 工作樹、計畫與各平台紀錄。
- [x] 2026-10-04：實作前建立詳細 ExecPlan；確定比較四個原型條件及品質閘門。
- [x] 2026-10-04：固定 50 個額外區段案例、85 個訓練區域與 4 個人工車身輪廓，保存修改前來源及輸入雜湊。
- [x] 2026-10-04：完成獨立候選分類器及車身遮罩適配器，取得官方分割權重；最終 22 個針對性測試通過。
- [x] 2026-10-04：完成分組訓練、來源與模型凍結、491 幀四條件比較及 16 個控制；組合品質閘門 failed。
- [x] 2026-10-04：163 個測試、Ruff、Mypy、diff 檢查通過；比較影片完整解碼、來源／輸出雜湊一致，可重現命令與品質報告完成。
- [ ] 條件未滿足：品質達標後才整合正式入口與完整兩片驗收；本輪原型已結束，未開始整合。clean 來源與 MPS/CUDA 平台仍另需具備。

## Context and Orientation


工作目錄是 repository root。分支 codex/restore-milestone1，HEAD b0cf8be22d735aeed9d7ce9e7d49cbe8ea441996。原有大量未提交實作全部保留。src/dashcam_ai/lane/segmentation.py 的 YoloPLaneLineDetector 使用 models/yolop-640-640.onnx 提出道路與白線分數，再以白漆、車框、圖案與時間規則篩選。當前來源 SHA256 是 8cc370af35596a9252cc998bcd161ed42edea188d5afa6a8ef5cb6b6aed9c246。兩支 samples/sample_1.mp4 與 sample_2.mp4 是 1280×720、30fps、1800 幀；輸入 SHA256 分別為 220611d110329da7a5ad89405c475ed4104c4c5c8eae6eb0ec8f34e0d6f0ea2d、48f0e3010d7acffebf95b8bd55e1193589785a3d8835ec528040f0298edbf4ba。前輪 output/lane-review-v8/sample1-final-v6 及 sample2-final-v6 保存當前基準；車框使用 output/sample_1_milestone2_yolop_v7/frames.jsonl、sample_2 的對應檔。

tests/fixtures/lane_markings_review.json 的 55 個案例全為已公開回歸資料，包括舊 held_out；不可當成未使用資料。scripts/check_lane_review.py 每 2px 採樣人工可見真線，以 10px 距離量覆蓋，並量 5px 含抗鋸齒白線與負例區域交集；模型遮罩不得刪除真值分母。原有 16 點位控制另在 tests/fixtures 下。

本環境 CPU available、MPS/CUDA unavailable。當前 M2 macOS MPS blocked（工具只支持 M1、dirty 來源、未觀察 MPS；JSON missing），Linux CUDA missing。M1 macOS 16bedd0 與 legacy M2 macOS cef1205、CUDA c7d77e3 全 stale；M1 CUDA missing。CPU 原型不能替代正式雙平台驗收。

## Plan of Work


第一階段在 output/lane-prototype-v1 保存凍結來源，並在 tests/fixtures/lane_prototype_review.json 補人工類別、車身輪廓與新區段。先固定額外影格，再看模型輸出或訓練；既有案例只做訓練／回歸。新區段與訓練影格留時間間隔，分類器不能取得影格 id、影片名稱、座標位置等洩漏特徵。新區段仍來自同兩支曾使用影片，只能稱額外區段，不能稱獨立新影片／未見道路資料。若沒有新影片，如實保留外部資料驗證缺口。

第二階段建立 src/dashcam_ai/lane/prototype.py，提供獨立 ExperimentalLaneDetector、MarkingClassifier 與 pixel_occlusion。預設後端與設定保持不動。使用既有 OpenCV RTrees（隨機森林：多個決策樹投票）建立便宜的候選分類原型，輸入局部影像顏色、梯度、白漆拓撲與線形狀；分類為 lane、arrow、parking、crosswalk、curb、text、background。以區域標註匹配候選供訓練，未標註候選不當負例。這是候選分類可行性原型，不是完整影像語意分割，也不主張少量片段已足夠訓練可泛化模型。只允許分類器明確預測 lane 時對彩色路緣上下文作例外；非車道預測可拒絕，unknown 保留原規則。

車身原型使用現有 ultralytics，取得官方 COCO 預訓練 yolo26n-seg.pt，逐像素保留 car、truck、bus、motorcycle、bicycle 與相應 rider/person。與固定車框關聯成功才以分割替換該框；未匹配車框繼續原遮罩，不能把分割漏檢解讀為車身不存在。分割影像尺寸與信心固定於開發前，保存權重 URL、SHA256、套件版本與實測速度。輪廓標註必須人工看原片，不用模型遮罩當真值。

第三階段建立 scripts/evaluate_lane_prototype.py，使用同一 YOLOP 輸出比較 baseline、pixel_mask、classifier、combined。推論不能讀取標註；標註只在訓練／評分使用。每組連續区段至少提前 15 幀暖機，同條件保持獨立時間狀態。記錄每條件遮罩 RLE（連續非零像素的起迄）、候選預測、速度、來源／權重／設定／輸入雜湊，輸出四欄影片與 jsonl。量測每条件自己实际渲染遮罩；原 checker 共用遮罩不適合直接比較新遮罩，新的比較工具分開呼叫現有度量。額外區段只在分類器及原型程式凍結後評分；其失敗不可再調參宣稱通過。

最後完成原有控制、單元測試、全套檢查、影像目視與解碼，在 docs/MILESTONE2_SEMANTIC_PROTOTYPE_REVIEW.md 報告是否值得整合。品質閘門失敗則以 failed 結束本輪原型評估，保留正式整合待辦；不延伸為無界限模型訓練。沒有承諾新模型必勝，也不可用分類器訓練分數宣稱道路品質通過。

## Concrete Steps


所有命令於 repository root，使用現有 .venv；不安裝額外依賴。官方權重已經由環境核准機制取得。重現輸出目錄／檔案必須不存在；下列命令不覆蓋本輪產物：

    PYTHONPATH=src .venv/bin/python scripts/evaluate_lane_prototype.py train --output output/lane-prototype-reproduce-training
    PYTHONPATH=src .venv/bin/python scripts/evaluate_lane_prototype.py evaluate --classifier output/lane-prototype-reproduce-training/classifier.xml --output output/lane-prototype-reproduce-evaluation
    PYTHONPATH=src .venv/bin/python scripts/check_lane_prototype_masks.py --records-dir output/lane-prototype-reproduce-evaluation --output output/lane-prototype-reproduce-body-review.json

    .venv/bin/python -m pytest tests/unit/test_lane_prototype.py
    .venv/bin/python -m pytest
    .venv/bin/python -m ruff check .
    .venv/bin/python -m mypy src
    git diff --check
    PYTHONPATH=src .venv/bin/python scripts/evaluate_lane_prototype.py --help
    .venv/bin/dashcam-ai devices

評估工具輸出不存在的新目錄；manifest 初始 complete=false，來源守衛、輸出解碼及量測完成才改 true。品質失敗回傳非零狀態，但保留結果。

## Validation and Acceptance


可見真線每案例覆蓋至少 90%，每個負例區域白線誤畫 0，16 個原控制通過。另量人工車身輪廓 precision/recall/IoU 與人工可見白線被遮罩覆蓋比例；4 個輪廓各需 precision/recall 至少 90%、IoU 至少 85%。空預測不能得到完美車身分數。未標車身像素不可以當成整張影像背景，所以車身輪廓量測限定人工完整標註的車輛／區域。訓練／回歸／額外區段分開報告，記錄候選分類混淆與速度。不能用平均達 90% 掩蓋單案例失敗。

測試需確認未匹配車框的安全回退、遮罩形狀／座標、分類器不以真值推論、負例抑制與例外限制、訓練測試群組隔離、不同條件遮罩量測與不完整輸出。pytest、Ruff、Mypy、diff 檢查通過，影片全部解碼。只有品質達標才進行完整正式 analyze，不改平台驗證工具、不提交來源，因此正式平台完成不屬本輪可保證產出。

## Idempotence and Recovery


所有權重、訓練資料影像與輸出留 ignored output 或 models，不提交影片／權重／大量紀錄。保存 baseline copy 便於重播。既有資料不覆蓋，重新評估選新目錄。任何原型問題只影響明確 opt-in 的評估工具；正式後端未改。保留原有未提交工作，不 reset、commit 或 push。

## Interfaces and Dependencies


沿用 NumPy/OpenCV、Pydantic、ultralytics、LaneCurve、LaneLineFrame。MarkingClassifier.train(samples, labels) 及 predict(frame, curve) 不接受影片／影格識別；序列化權重與特徵版本相容檢查。ExperimentalLaneDetector 是 YoloPLaneLineDetector 子類，僅供評估。輸入 pixel mask 必須原片大小、二維 uint8；所有未知／不可靠分類仍使用原规则。分析入口与默认配置不接此原型。

## Surprises & Discoveries


2026-10-04：現有環境已具備 torch 2.13.0、OpenCV 4.14.0.94、ultralytics 8.4.127，無需另安裝。公開 CeyMo 提供箭頭、行穿線等 11 類，但未提供車道縱線與停車格完整目標集合，不能直接聲稱覆蓋本需求。官方 YOLO26 分割以 COCO 訓練，可供車身原型，不能辨識道路標線類別。

2026-10-04：85 個人工訓練區域只有 71 個合格候選（lane 21、parking 19、crosswalk 14、text 9、curb 5、arrow 3）；背景類別沒有訓練樣本。相鄰影格高度相關，71 不是 71 個獨立道路場景。未提出或未匹配候選不能偷偷當成背景資料。

2026-10-04：機車騎士的 person 分割不能單獨取代 motorcycle 車框，否則可能漏遮機車；已加入對應測試與安全回退。分類器本幀拒絕的非車道候選也需在時間延續時抑制，否則仍可能畫出上一幀的停車格線；新增延續回歸測試。

2026-10-04：新增區段中的機車停等框與車身邊緣不是路邊停車格／路緣，已在推論前更正類別；sample_2 1413、1414 稀疏光流端點漂移，依原片人工更正。新增區段仍缺獨立的強光左線與路邊停車格場景；不可宣稱完整新場景泛化驗證。

2026-10-04：組合原型恢復已知強光左線、排除停車格，但像素遮罩恢復的漆面與現有圖案／道路邊緣規則衝突，sample_2 第 152、1231、1232 幀退步。原有第 30 幀控制也因新候選尚待確認而失敗；遮罩變精細不等於線段品質會全面改善。

2026-10-04：車身量測的第 1230 幀開發 audit_box 包含原標註漏列的自車外殼及後方車輛。人工補齊後另存 lane_prototype_body_review.json，以凍結 RLE 記錄和獨立 checker 重量測；原推論、分類器、線段真值與額外輪廓不變。校正後只有 1/4 輪廓達標。目視另發現第 1412 幀未標註的黃色網狀線誤畫，不可用標註區域零誤畫宣稱全畫面正確。

## Decision Log


2026-10-04 / Codex：先以既有 OpenCV 分類器與官方輕量車身分割做可重現原型。理由是直接檢验兩個主要問題的收益與退步，避免因缺乏充分標註而立即替換整套模型。已核准原型涉及權重取得與小型訓練；沒有另增正式後端或部署。

2026-10-04 / Codex：以前輪已知 held_out 作回歸，不重用其未見身份；額外區段仍明確稱同影片資料。保留來源雜湊作 dirty CPU 證據，不自動提交以取得 clean。

2026-10-04 / Codex：訓練、來源與額外區段均已凍結，開始四條件比較後不再依額外結果調參。訓練與評估使用同一完整 source_hashes。人工輪廓只在 4 個 audit_box 內量測，粗多邊形邊界誤差也是本輪量測限制。

2026-10-04 / Codex：拒絕將目前原型整合為預設後端。理由是每案例品質、原有控制及車身輪廓閘門均未全面通過；保留已知退步與額外案例失敗，不擴大分類器例外或重用額外結果調參。正式入口／兩片完整 analyze 與平台驗收的條件未成立。

## Outcomes & Retrospective


本輪已批准原型評估完成，採納結論為 failed，M2 產品驗收未完成。組合原型的已知案例由 39/55 改善至 51/55，額外案例仍 49/50；原有控制由 16/16 降至 15/16，校正後車身輪廓僅 1/4 達標。491 幀固定窗口比較及全部解碼完成，163 個 pytest、Ruff、Mypy（41 個來源檔）、diff 檢查通過。這些是 dirty CPU 原型證據，不是雙平台正式驗收。

最終訓練為 output/lane-prototype-v1/training-frozen，推論為 output/lane-prototype-v1/evaluation，校正後車身量測為 output/lane-prototype-v1/body-review-final.json。原 detector 與正式設定未改；沒有 commit/push。完整結果、雜湊、重現命令與後續資料缺口見 [品質報告](MILESTONE2_SEMANTIC_PROTOTYPE_REVIEW.md)。M2 macOS MPS blocked、Linux CUDA missing；新一輪模型／候選修正與正式整合沒有在本輪展開。

修訂說明：將使用者已批准的方案 B 展開為可執行原型計畫，定義資料隔離、模型選擇、遮罩安全回退與品質閘門。

結案修訂：補上實測結果、車身標註校正、重現步驟與拒絕採納依據；完成本輪原型工作，不把未滿足的正式整合條件勾選為完成。
