# M2 標線分類／車身遮罩原型評估

2026-10-04 完成已批准的方案 B 原型評估。**原型採納閘門 failed，M2 未完成，正式後端未切換。** 組合原型在已知資料改善停車格與強光左線，但其他左線退步、原有控制只通過 15/16、額外區段仍漏線，車身輪廓也尚未全面達標。

本輪新增 `src/dashcam_ai/lane/prototype.py`、訓練／比較工具、車身量測工具及人工標註。正式 `YoloPLaneLineDetector`、設定、時間確認／延續參數保持原來內容。沒有安裝額外依賴、commit、push 或改写其他平台報告。批准範圍與進度見 [ExecPlan](EXEC_PLAN_MILESTONE2_SEMANTIC_PROTOTYPE.md)。

## 比較結果

四種條件使用同一原片影格、YOLOP 分數與 v7 車框，各自保持時間狀態。每個區段提前 15 幀暖機。比較影片以左上 baseline、右上 pixel_mask、左下 classifier、右下 combined 排列。

| 條件 | 已知案例通過 | 額外案例通過 | 原有控制 | sample_1 左線覆蓋 | sample_2 左線覆蓋 | 停車格誤畫幀 |
|---|---:|---:|---:|---:|---:|---:|
| baseline：目前 YOLOP／規則 | 39/55 | 49/50 | 16/16 | 93.69% | 56.71% | 5/10 |
| pixel_mask：車身像素遮罩 | 41/55 | 49/50 | 15/16 | 93.69% | 46.61% | 5/10 |
| classifier：候選分類器 | 42/55 | 49/50 | 16/16 | 93.69% | 56.71% | 2/10 |
| combined：兩者組合 | 51/55 | 49/50 | 15/16 | 93.69% | 76.90% | 0/10 |

覆蓋是人工可見線段每 2px 採樣的長度加權結果，每案例仍須至少 90%，不能以平均達標替代。四條件的已知箭頭 10 幀、行穿線 20 幀、道路文字 5 幀均無標註區域誤畫。額外 40 個負例區域全部通過，額外 10 個正例仍有 1 個未達 90%。這些是已標註區域的結果，並非整張影像／所有道路標線的精確率或召回率。

已知 55 案例全為前輪公開回歸資料，包含曾經的 held_out，也參與本輪候選監督。這些改善不能當作新場景泛化成效。額外 50 案例在本輪模型推論／訓練前固定，全部來自同兩支曾使用影片，沒有獨立新影片；額外資料沒有改善。sample_1 額外資料只有負例，sample_2 額外正例測的是另外兩段可見白線，並未提供新的強光左線或路邊停車格場景。

## 具體改善與退步

sample_1 第 93..97 幀的停車格仍為 baseline 的 5 個誤畫；classifier 剩第 96、97 幀，combined 全部排除。單用像素遮罩沒有排除停車格。第 102 幀真左線仍只有 73.02% 覆蓋，四條件都失敗，沒有被遮罩覆蓋。

sample_2 第 58..62 幀的強光左線在前三條件皆為 0%，combined 全部 100%。像素遮罩先恢復第 62 幀被車框完整遮掉的可見漆面；分類器再對明確 lane 的彩色路緣判定作有限例外。單獨加入任何一者都沒有完成這個區段。

但 sample_2 第 148..152 幀，baseline 覆蓋 100%，pixel_mask／combined 降到 79.80%，第 152 幀變成 0%。新的漆面連通與圖案判斷造成 `arrow_like_fit`／`arrow_paint_context` 問題，舊線也被本幀圖案證據抑制。不能為改善强光而忽略這個退步。

sample_2 第 1228..1232 幀，人工可見線被遮罩覆盖的幀平均比例由 47.95% 降至 7.51%，但輸出覆蓋卻由 69.28% 降至 58.50%。第 1228..1230 幀有 95.08%、100%、100% 覆蓋，第 1231、1232 幀皆 0%。診斷指出 `road_edge_context` 拒絕了新候選；1231 分類器仍以 85.9% 投票認為 lane，但原型只允許彩色路緣例外，沒有全面取消道路邊緣規則。**恢復像素並不保證現有後處理會留下真線。**

原有控制在 pixel_mask／combined 的 sample_2 第 30 幀失敗，原因包含新候選尚在 `awaiting_confirmation`，無可輸出的已確認白線。classifier 單獨保留 16/16。額外正例 sample_2 第 631 幀為 87.14%，四條件一致失敗；額外區段及模型已凍結，沒有再據此調參。

目視另發現 sample_2 第 1412 幀黃色網狀格線仍有白色疊圖，baseline 也存在。該區域不在本輪事先固定的負例多邊形內，不加進額外 40 區域統計；但它證明「標註負例零誤畫」不能宣稱整張道路無誤畫。快照與診斷記錄已保存，後續需新增回歸標註。

## 車身輪廓量測

車身量測以人工多邊形在指定 audit_box 內的全部可見車身為真值；不是把未標車身當背景，也不是以模型遮罩生成真值。指標 precision 量預測像素中真正車身比例、recall 量人工車身被遮住比例、IoU 量兩者交集與聯集比例。

| sample_2 幀 | 資料用途 | baseline IoU | pixel_mask precision | pixel_mask recall | pixel_mask IoU | 本案例達標 |
|---|---|---:|---:|---:|---:|---|
| 60 | 開發輪廓 | 48.02% | 94.48% | 85.06% | 81.04% | 否 |
| 1230 | 開發輪廓，校正後 | 72.31% | 91.01% | 96.24% | 87.89% | 是 |
| 632 | 額外輪廓 | 39.71% | 85.49% | 93.92% | 81.01% | 否 |
| 1412 | 額外輪廓 | 52.73% | 87.26% | 89.80% | 79.39% | 否 |

門檻是每案例 precision/recall ≥90%、IoU ≥85%；目前只 1/4 達標。輪廓是粗多邊形，邊界／機車孔隙有標註誤差，不能視為稠密資料集結果。第 1230 幀的 baseline recall 為 100%，像素遮罩降至 96.24%，因此不能宣稱所有車身召回都改善。

第一份 `evaluation/report.json` 的第 1230 幀車身標註視窗漏列自車外殼及後方車輛，其 preliminary precision／IoU **不適用最終車身判斷**。已人工補齊該開發區域，用相同已凍結的 RLE 推論記錄重新量測；未更改分類器、像素推論、可見白線標註或額外輪廓。正式本輪車身依據是 `tests/fixtures/lane_prototype_body_review.json` 與 `output/lane-prototype-v1/body-review-final.json`。原始回歸／額外資料保留供來源追溯。

## 模型、資料與來源

候選分類器採既有 OpenCV RTrees，使用局部白漆、梯度、顏色及相對線形，沒有影格編號、影片名稱或絕對位置特徵。85 個人工區域只取得 71 個有效候選：lane 21、parking 19、crosswalk 14、text 9、curb 5、arrow 3，background 沒有樣本。相鄰影格高度相關，並非 71 個独立場景。分類器低投票信心保持 unknown、沿用現有規則；無法救回在候選形成之前已刪掉的線。

車身採 [Ultralytics 官方 YOLO26 instance segmentation](https://docs.ultralytics.com/tasks/segment/)，COCO 預訓練 `yolo26n-seg.pt`，CPU、imgsz 640、conf 0.25、retina_masks=True，保留 person、bicycle、car、motorcycle、bus、truck。原有車框只有在匹配到足夠大的車輛實例時才換成像素遮罩；無匹配仍沿用車框。person 單獨不能替換 motorcycle 框，避免只遮騎士而漏遮車體。分類器新增非車道證據也會阻止舊曲線延續。

查閱的 [CeyMo 原始資料集](https://github.com/oshadajay/CeyMo) 有箭頭、行穿線等類別，但未覆蓋本需求的完整縱向白線與停車格集合，因此沒有拿其公開分數宣稱可解決本專案，也沒有下載／訓練該資料集。

HEAD 為 `b0cf8be22d735aeed9d7ce9e7d49cbe8ea441996`，分支 `codex/restore-milestone1`、worktree dirty。以下雜湊與 manifest 才識別實際來源：

| 內容 | SHA256 |
|---|---|
| 原 detector，未修改 | `8cc370af35596a9252cc998bcd161ed42edea188d5afa6a8ef5cb6b6aed9c246` |
| prototype.py | `b60a082b28ef0f888a28fe3274e58d1d87ffea480018f14bfff14d78c64256c8` |
| evaluate_lane_prototype.py | `f2f64251169e9cb03175456c17f04babd91488ec6acd8ada71da83a9110b55f9` |
| 最終車身 checker | `ce90708a0f44de3e0bd6abb3fa14f4e39c74cdc6aa9b8ed90a800ff0ce1a2a1a` |
| 回歸／額外／訓練 fixture | `8c9e131b58cf02b8f04155aada0a725906359efdddcfba817323b89d59cae128` |
| 校正後車身 fixture | `89da7c887702f54b5529a160bafc834c61992675cde1284c38ab35807e3e425b` |
| classifier.xml | `c975d53e0f1cb05f49274005916ed21ad54f164f465897c4081d011cc9f27090` |
| yolo26n-seg.pt | `361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f` |
| YOLOP ONNX | `cd66a3e0087a7258ae07768cc02cb742eed93865727ae4c9baf969b8fa190696` |
| configs/default.yaml | `67d92221459a41762119453498ba4ebef3af6915ca181f8157ec0e6c3d1bc4ce` |
| sample_1 原片 | `220611d110329da7a5ad89405c475ed4104c4c5c8eae6eb0ec8f34e0d6f0ea2d` |
| sample_2 原片 | `48f0e3010d7acffebf95b8bd55e1193589785a3d8835ec528040f0298edbf4ba` |
| sample_1 v7 車框 | `03038423a91269e57051702c289fbd6d0a914672dba08367b4cb1c5f6700df20` |
| sample_2 v7 車框 | `276d0524a0609797b8086c3c5e8015ee158e23bf82c107761ba8cacf30165667` |

權重 URL：`https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt`。既有環境為 Python 3.12.11、torch 2.13.0、OpenCV 4.14.0.94、ultralytics 8.4.127。實際版本以 manifest.environment 為準。

凍結訓練產物在 `output/lane-prototype-v1/training-frozen/`，訓練與推論的完整來源雜湊一致。多次同資料、同種子訓練得到相同 XML SHA；早期同內容訓練資料夾僅為中間產物，不引用為最後來源。模型、原片與影像皆 ignored。

## 驗證與產物

163 個 pytest、Ruff、Mypy（41 個來源檔）及 diff 檢查通過。22 個新增測試覆蓋未匹配車框回退、只偵測騎士、額外車身、分類器例外限制、舊線延續、序列化、資料隔離、各条件遮罩獨立量測，以及人工多車身聯集量測。

四條件共重播相同 491 個原片影格，sample_1 為 135、sample_2 為 356。兩支四格比較影片均完整解碼到預期幀數；8 份 JSONL、輸出雜湊、來源與輸入守衛完整。這是固定連續窗口比較，**不是兩支各 1800 幀完整 analyze**。依已批准計畫，品質失敗後不整合正式入口／重跑全片 tracker。

總執行約 538.472 秒，含四條件、車框回退遮罩、編碼與驗證。CPU 實測每幀中位數：YOLOP 300.6ms、車身分割 49.4ms；後處理 baseline 111.8ms、pixel_mask 111.8ms、classifier 119.7ms、combined 119.9ms。分項未含全部車框回退建立與輸出，因此不能相加宣稱正式分析端到端 FPS；也不是 MPS/CUDA 速度。

最終產物：

- `output/lane-prototype-v1/evaluation/manifest.json`：complete=true，原始來源／輸入／輸出雜湊。
- `evaluation/report.json`：線段品質、控制、初步車身量測、逐案例診斷；SHA `3c46803b7fa390cfdb988d068d644e2ea9b2abf46915433a5e86c5933ccb9a77`。
- `body-review-final.json`：校正後車身量測、資料／工具／記錄雜湊；SHA `d0fa1edd545493850cc611c3e804e0df76180fc62a1b2bd8d1e0a572736c3f03`。
- `evaluation/sample_1-comparison.mp4`：SHA `06bcf889bee1f9ad2ea4f944cddf87de218123b7f8cda6463a83cf8ad47ca121`。
- `evaluation/sample_2-comparison.mp4`：SHA `cc40fe6a60207dfc635200093b76988c78a568be9ddd008c2b8b0c6da6aae901`。
- 各條件 `sample_N-{variant}.jsonl`：實際曲線、各自遮罩 RLE、分類預測、替換車框數；全部雜湊見 manifest。

所有原始推論来源與輸出雜湊在最終人工 review 後再次確認一致。車身 checker 是之後新增的獨立量測工具，自己的來源／fixture／記錄守衛在 body-review-final.json，沒有改寫原始推論產物。

## 重現

由 repository root 使用現有 .venv、自備同雜湊原片／v7 車框／YOLOP，官方車身模型放到 models/yolo26n-seg.pt。工具不自動下載、覆蓋或切換正式後端。`--output` 必須不存在，已有產物時改用新名稱。

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_lane_prototype.py train \
  --output output/lane-prototype-reproduce-training

PYTHONPATH=src .venv/bin/python scripts/evaluate_lane_prototype.py evaluate \
  --classifier output/lane-prototype-reproduce-training/classifier.xml \
  --output output/lane-prototype-reproduce-evaluation

PYTHONPATH=src .venv/bin/python scripts/check_lane_prototype_masks.py \
  --records-dir output/lane-prototype-reproduce-evaluation \
  --output output/lane-prototype-reproduce-body-review.json

.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src
git diff --check
```

evaluate 與校正後車身 checker 均先寫結果，再因品質失敗回傳 exit 1；這是正確拒絕採納，不能忽略。若第三個命令重用現有凍結記錄，改 `--records-dir output/lane-prototype-v1/evaluation` 即可，不須重做推論。

## 採納決定與平台

本輪原型評估完成，決定 **不採納目前組合為預設後端**。已證明「分類器 + 更精確車身遮罩」對已知強光／停車格有收益，也證明單換遮罩會與現有圖案、道路邊緣及時間確認規則衝突。下一步宜先處理這些候選／上下文退步，補足背景及更多道路類別資料，取得獨立新影片與新的保留區段，再評估更完整標線語意模型。這是後續方向，沒有在本輪開始擴大訓練或繼續依額外區段調參。

目前執行環境 CPU available、MPS/CUDA unavailable。M2 macOS MPS blocked（驗證工具只支持 M1、來源 dirty、無 MPS；JSON missing），Linux CUDA missing。M1 macOS 16bedd0 与 legacy M2 macOS cef1205、CUDA c7d77e3 stale；M1 CUDA missing。來源未提交且工作樹 dirty，無正式雙平台驗收；[本機紀錄](../validation/milestone-2/macos-mps.md) 只為手動阻塞說明，沒有偽造機器報告。
