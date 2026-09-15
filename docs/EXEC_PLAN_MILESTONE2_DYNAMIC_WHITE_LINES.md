# Milestone 2：動態白線

## Purpose / Big Picture

從影片辨識真實道路白線，使用白色曲線疊圖跟隨直路、彎道與鏡頭變化。此階段不建立車道歸屬、運動估計或換道事件。

## Progress

- [x] 2026-09-03 核准階段拆分與目標。
- [ ] 開始實作前完成具體方案與分段核准。
- [ ] 實作、合成測試、真實影片驗收。
- [ ] 同一乾淨來源版本的 MPS／CUDA 驗證。

## Context and Orientation

本文件為尚未實作的階段計畫。現行 src/dashcam_ai 僅有 Milestone 1 感知流程；cli.py 建立 Analyzer，application/analyzer.py 處理影片，visualization/annotator.py 畫精簡物件標註。舊 Milestone 2 程式已移除，不得將歷史報告當成此階段證據。

## Plan of Work

先以本機代表性影片建立首中尾及彎道、虛線、實線、遮擋、無白線的人工標記樣本；再比較可用白線辨識方案；之後加入曲線跨幀配對與平滑；最後整合可單獨檢視的白線影片輸出。每段先以合成影像測試，再抽查真實影片。

## Concrete Steps

在專案根目錄檢查 Git、docs/ROADMAP.md 及 validation/。開始實作前補齊選定方案、檔案與命令介面並取得核准。本次只交付計畫，不建立未經測試的功能入口。

驗證既有基線：.venv/bin/pytest、.venv/bin/ruff check .、.venv/bin/mypy src。後續依功能加入真正檢驗辨識品質的測試；實片結果放新的 output/ 子目錄。

## Validation and Acceptance

白線須沿實際標線，不能把路緣、分隔島、陰影或天空當可信白線；短暫缺失只容許有界延續，超時停止繪製並記錄 unknown。偵測品質與覆蓋率分開記錄，全部不畫線不能算通過。 保留 Milestone 1 所有輸出與装置行為。最終需 macOS MPS、Linux CUDA 的同一 clean source_commit 報告；缺少平台即 blocked。

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
