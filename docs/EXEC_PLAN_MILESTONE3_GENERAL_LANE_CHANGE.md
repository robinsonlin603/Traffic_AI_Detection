# Milestone 3：一般換道

## Purpose / Big Picture

以已通過 Milestone 2 驗收的白線證據建立車道，辨識任何受追蹤車輛從一個車道換到相鄰車道。左右換道皆成立，不要求進入自車車道，不實作 cut-in。

## Progress

- [x] 2026-09-03 核准階段拆分與目標。
- [ ] 開始實作前完成具體方案與分段核准。
- [ ] 實作、合成測試、真實影片驗收。
- [ ] 同一乾淨來源版本的 MPS／CUDA 驗證。

## Context and Orientation

本文件為尚未實作的階段計畫。現行 src/dashcam_ai 僅有 Milestone 1 感知流程；cli.py 建立 Analyzer，application/analyzer.py 處理影片，visualization/annotator.py 畫精簡物件標註。舊 Milestone 2 程式已移除，不得將歷史報告當成此階段證據。

## Plan of Work

先建立車道區域與相鄰關係；再加入背景運動估計及自車運動補償；接著以跨幀證據判定車道歸屬、跨線與目的車道穩定停留；最後輸出一般換道事件與精簡事件標註。每段需包含合成正反例再驗收實片。

## Concrete Steps

在專案根目錄檢查 Git、docs/ROADMAP.md 及 validation/。開始實作前補齊選定方案、檔案與命令介面並取得核准。本次只交付計畫，不建立未經測試的功能入口。

驗證既有基線：.venv/bin/pytest、.venv/bin/ruff check .、.venv/bin/mypy src。後續依功能加入真正檢驗辨識品質的測試；實片結果放新的 output/ 子目錄。

## Validation and Acceptance

事件包含 track_id、來源／目的車道、左右方向、開始／跨線／完成時間及信心依據。直行、邊界抖動、道路拓撲改變與未知車道不得誤確認；真實左換右、右換左正例皆需驗收。不實作 cut-in、TTC、碰撞責任、方向燈或跨 ID 重識別。 保留 Milestone 1 所有輸出與装置行為。最終需 macOS MPS、Linux CUDA 的同一 clean source_commit 報告；缺少平台即 blocked。

## Idempotence and Recovery

不覆蓋原始影片、權重或先前輸出。不自動下載模型、安裝新套件或提交 Git；選定依賴後先更新計畫供核准。

## Interfaces and Dependencies

延續原始影像座標、PerceptionBackend 與串流 Analyzer。模型和具體新介面尚未選定，不能把舊 YOLOP 試驗視為已採用方案。套件與權重版本、雜湊及執行成本在選型階段記錄。

## Surprises & Discoveries

歷史試驗顯示路旁結構可能被誤認為車道線；測試全數通過不能代替真實影像品質驗收。

## Decision Log

2026-09-03 使用者要求動態畫白線與車道／運動／換道分成獨立階段；換道目標為任何車輛的一般換道，不是 cut-in。

## Outcomes & Retrospective

僅完成範圍規劃，未實作，無本階段通過證據。
