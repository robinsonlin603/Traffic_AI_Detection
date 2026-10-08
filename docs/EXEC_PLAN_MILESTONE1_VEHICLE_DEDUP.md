# Milestone 1 有動力車輛統一與追蹤前去重

## Purpose / Big Picture

Milestone 1 只關注 car、truck、bus、motorcycle，統一輸出為 vehicle，畫面只顯示 #ID。停用尺寸篩選，並在 BoT-SORT 分配 ID 前移除同一幀高度重疊的車輛框，降低同一物件產生多個 ID 的情況。

## Progress

- [x] 2026-09-15 使用者核准範圍；實拍影片重跑及人工審查由使用者執行。
- [x] 統一有動力車輛類別並移除 bicycle。
- [x] 將標籤改為只顯示 #ID。
- [x] 停用尺寸篩選並新增追蹤前跨類別去重。
- [x] 完成第一輪 pytest、Ruff、Mypy。
- [x] 2026-09-16 使用者人工確認仍有巢狀重複框與其造成的 ID 切換。
- [x] 新增包含比例與中心距離去重，並完成第二輪自動驗證。
- [x] 2026-09-16 核准追蹤輸出去重與單一漏幀 ID 銜接。
- [x] 完成第三輪自動驗證。
- [x] 完成缺少一至兩幀銜接與第四輪自動驗證。
- [x] 新增同幀 canonical ID 碰撞防護並完成第五輪自動驗證。

## Context and Orientation

configs/*.yaml 與 config/models.py 定義偵測類別；detection/ultralytics.py 在 Ultralytics 後處理完成、BoT-SORT callback 執行前處理偵測；visualization/annotator.py 產生標籤。frames.jsonl 與 tracks.json 由應用層接收 TrackedObject 後寫出。tracking/identity.py 在 BoT-SORT 後處理巢狀追蹤框，並只對中間缺少一至兩幀且端點框高度重疊的新 ID 接回舊 ID。

## Plan of Work

設定只保留 car、truck、bus、motorcycle，minimum_vehicle_area_ratio 預設設為 0。YOLO 仍以原始 COCO 類別偵測；在追蹤前對有動力車輛執行信心優先的 class-agnostic IoU 去重，預設門檻 0.85，再將保留框映射到單一 vehicle 類別送入 BoT-SORT。輸出 class_name 為 vehicle，標註只顯示 #ID。追蹤輸出再次移除巢狀框，較低信心 ID 映射至保留 ID；新 ID 只有在中間缺少一至兩幀，且端點 IoU 至少 0.45，或符合原有嚴格包含與中心距離條件時才銜接。相鄰影格直接替換不銜接。

## Validation and Acceptance

新增 vehicle 映射、跨原始類別去重、純 #ID 標籤與設定測試。執行 pytest、ruff check .、mypy src、git diff --check。本計畫不執行實拍影片分析、不更新平台機器報告；使用者自行重跑與人工審查。

## Scope Limits

不加入 ReID、不銜接超過兩個漏幀的軌跡、不實作車種統計、白線、車道或換道判斷。IoU 去重只處理同一幀；互相遮擋且 IoU 達門檻的不同車輛仍有誤合併風險。

## Decision Log

2026-09-15：使用者要求標籤只顯示 #ID，只關注有動力車輛，自動驗證止於 pytest、Ruff、Mypy；實拍影片由使用者重跑。


## Outcomes

完成有動力車輛統一、0.85 IoU 信心優先去重、純 #ID 標籤、停用預設尺寸篩選，以及設定與 runtime metadata。63 項 pytest、Ruff、strict Mypy（36 來源檔）及 git diff --check 通過。依核准範圍未執行實拍影片或平台驗證，未 commit/push。


2026-09-16 決策：人工審查確認 #44/#74、#587/#579 與 #566/#596/#598 為巢狀重複框。使用者核准加入包含比例與中心距離規則；不降低 IoU 門檻、不加入 ReID，實拍仍由使用者重跑。


第二輪結果：新增巢狀跨類別框、中心距離保護、設定驗證與 CLI 傳遞測試。完整 71 項 pytest、Ruff、strict Mypy（36 來源檔）與 git diff --check 通過。依核准範圍未執行實拍影片或平台驗證，未 commit/push。


2026-09-16 決策：人工確認 #8/#9 是機車本體與含後箱範圍的同車重複框；#508→#523 是中間缺一幀的 ID 切換。#177/#197 與 #275/#294 是不同車輛。採追蹤輸出去重及僅限單一漏幀的保守銜接，避免相鄰影格的 #275/#294 被誤合併。

第三輪結果：新增 BoT-SORT 輸出後巢狀框消除與單一漏幀 ID 銜接。既有 innovv-test-1 frames.jsonl 重播確認 #8/#9 合併、#508/#523 銜接，且 #177/#197、#275/#294 保持分離。完整 76 項 pytest、Ruff、strict Mypy（37 來源檔）與 git diff --check 通過。未重跑影片、未更新平台報告，未 commit/push。


2026-09-16 決策：第二次人工審查確認重複框已排除，並確認七組短中斷 ID 切換。將銜接範圍擴至缺少一至兩幀，端點 IoU 門檻為 0.45；相鄰影格直接替換仍不銜接。最新 innovv-test-1 frames.jsonl 重播確認七組正例均銜接，#211/#224 與 #275/#294 維持分離。

第四輪結果：新增缺少兩幀與較大端點位移測試。完整 78 項 pytest、Ruff、strict Mypy（37 來源檔）及 git diff --check 通過。未重跑影片、未更新平台報告，未 commit/push。


2026-09-16 決策：第三次人工審查發現兩台不同車在八個影格同時輸出 #275。同一 canonical ID 若對應到兩個非重複框，保留真正持有原始 ID 的軌跡，解除另一條軌跡的 alias 並恢復其原始 ID。

第五輪結果：新增 canonical ID 碰撞與信心排序回歸測試。完整 79 項 pytest、Ruff、strict Mypy（37 來源檔）及 git diff --check 通過。未重跑影片、未更新平台報告，未 commit/push。
