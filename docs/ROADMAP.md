# 專案 Roadmap

2026-09-03 起重新劃分階段。目前 Milestone 1 已恢復，新的 Milestone 2 已實作並完成兩支 60 秒 CPU 白線重播，標線品質仍有失敗、雙平台仍待驗收；後續階段尚未實作。舊 Milestone 2 的完成勾選不能套用到新編號。

| Milestone | 目標 | 狀態 |
|---|---|---|
| 1 | YOLO 偵測、BoT-SORT 追蹤、簡單標註、結構化輸出 | 已恢復，驗證見 MILESTONE_1.md |
| 2 | 動態辨識並畫出道路白線 | 兩片 CPU 重播完成，部分左線／停車格品質 failed，雙平台待驗收 |
| 3 | 車道、運動估計、一般換道事件 | 規劃中，未實作 |
| 4 | 方向燈辨識與換道事件融合 | 未開始 |
| 5 | FastAPI、GPS、事件影片片段 | 未開始 |
| 6 | VLM／LLM 進階語意分析 | 未開始 |

Milestone 3 只問「是否有車從一個車道換到另一個車道」。不要求進入自車車道，不判斷 cut-in、距離、TTC 或責任。

新計畫：[動態白線](EXEC_PLAN_MILESTONE2_DYNAMIC_WHITE_LINES.md)、[一般換道](EXEC_PLAN_MILESTONE3_GENERAL_LANE_CHANGE.md)。舊紀錄：[歷史封存](history/legacy-milestone-2/README.md)。

2026-10-04 標線修正：[ExecPlan](EXEC_PLAN_MILESTONE2_MARKING_FILTERS.md)、[品質 review](MILESTONE2_MARKING_REVIEW.md)。141 個測試與原有 16 個控制通過，但連續案例只有 39/55 通過，M2 仍未完成。

2026-10-04 後續原型評估：[ExecPlan](EXEC_PLAN_MILESTONE2_SEMANTIC_PROTOTYPE.md)、[品質 review](MILESTONE2_SEMANTIC_PROTOTYPE_REVIEW.md)。標線候選分類器與車身像素遮罩組合使已知案例達 51/55，但額外案例仍 49/50、原有控制降至 15/16，車身輪廓只 1/4 達標；採納閘門 failed，未切換正式後端。163 個測試、Ruff、Mypy 通過，完成 491 幀 CPU 窗口比較；這不是兩片完整 analyze 或 GPU 驗收，M2 仍未完成。

2026-10-04 固定自車 ID 排除：[驗證 review](EGO_VEHICLE_EXCLUSION_REVIEW.md)。兩支各 1800 幀 MPS 專項重播／解碼、自車形狀框 0、11/11 旁車控制、182 個測試通過。沙箱外已確認 MPS 可用；正式 M2 仍因 dirty 來源及驗證入口不支援而 blocked，CUDA missing。這輪全片只驗證車輛 ID，白線全片品質需在新追蹤輸入下另行驗收。

2026-10-04 v8 人工回報修正：[ExecPlan](EXEC_PLAN_MILESTONE2_HUMAN_REVIEW.md)、[驗證 review](MILESTONE2_HUMAN_REVIEW.md)。201 個測試、Ruff、Mypy；兩支各 1800 幀 CPU 成對重播／解碼與來源核對完成。直線 25/27，指定固定斑馬線區域誤畫 130→0；原有可見真線 15/16、圖案 45/55、額外 49/50，無新增已標註退步。公車格、左線與其他斑馬線仍 failed，M2 未完成；20 幀預設整合已觀察 MPS，正式 macOS MPS blocked、CUDA missing。

2026-10-05 左白漆修正：[ExecPlan](EXEC_PLAN_MILESTONE2_LEFT_PAINT.md)、[驗證 review](MILESTONE2_LEFT_PAINT_REVIEW.md)。兩張圈選白漆恢復 100%，舊 630／680 左線仍漏；兩支各 1800 幀 CPU 成對比較／解碼及來源核對完成，直線 25/27、原控制 15/16、圖案 45/55、額外 49/50，固定斑馬線區域 1131 幀誤畫 0，無新增已標註退步。212 個測試、Ruff、Mypy 與 20 幀 MPS 整合通過；整體品質 failed、正式 MPS blocked、CUDA missing，M2 未完成。

2026-10-06 公車格排除：[ExecPlan](EXEC_PLAN_MILESTONE2_BUS_BAYS.md)、[驗證 review](MILESTONE2_BUS_BAY_REVIEW.md)。v28 兩片各 1800 幀 AFTER CPU 重播與凍結 BEFORE 比較／四片完整解碼、來源與所有 fixture 稽核完成，公車格及真線控制 136/136、1410–1414 真線退步恢復，無新增已標註退步；615／660 左漆保留 100%，630／680 仍漏。249 tests、Ruff、Mypy 42 檔、20 幀 MPS 整合通過。整體品質 failed、正式 MPS blocked、CUDA missing，M2 未完成。

2026-10-07 可見左線／斑馬線：[ExecPlan](EXEC_PLAN_MILESTONE2_LEFT_CROSSWALK.md)、[驗證 review](MILESTONE2_LEFT_CROSSWALK_REVIEW.md)。trial-27 指定六處左漆達 90% 覆蓋門檻（1222 為 95.45%，其餘 100%），540 斑馬線誤畫 1361→0；兩片各 1800 幀連續 AFTER、四片全數解碼、44 個來源／固定輸入／產物及全 fixture 稽核完成，固定斑馬線區 1131 幀零誤畫、公車格及控制 136/136，無新增已標註退步。294 tests、Ruff、Mypy 42 檔及 20 幀 MPS 整合通過，三段半速短片待新人工確認。1228–1232 近處左線仍 failed，新案例 10/15；整體品質 failed、正式 MPS blocked、CUDA missing，M2 未完成。失敗 trial-23 的 21 筆退步與其餘試驗證據保留。
