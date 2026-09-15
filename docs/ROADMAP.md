# 專案 Roadmap

2026-09-03 起重新劃分階段。現行程式僅提供 Milestone 1；後續階段全部尚未實作。舊 Milestone 2 的完成勾選不能套用到新編號。

| Milestone | 目標 | 狀態 |
|---|---|---|
| 1 | YOLO 偵測、BoT-SORT 追蹤、簡單標註、結構化輸出 | 已恢復，驗證見 MILESTONE_1.md |
| 2 | 動態辨識並畫出道路白線 | 規劃中，未實作 |
| 3 | 車道、運動估計、一般換道事件 | 規劃中，未實作 |
| 4 | 方向燈辨識與換道事件融合 | 未開始 |
| 5 | FastAPI、GPS、事件影片片段 | 未開始 |
| 6 | VLM／LLM 進階語意分析 | 未開始 |

Milestone 3 只問「是否有車從一個車道換到另一個車道」。不要求進入自車車道，不判斷 cut-in、距離、TTC 或責任。

新計畫：[動態白線](EXEC_PLAN_MILESTONE2_DYNAMIC_WHITE_LINES.md)、[一般換道](EXEC_PLAN_MILESTONE3_GENERAL_LANE_CHANGE.md)。舊紀錄：[歷史封存](history/legacy-milestone-2/README.md)。
