# 恢復 Milestone 1 並重編 roadmap

## Purpose / Big Picture

恢復純 YOLO/BoT-SORT 分析，保留目前 #ID 類別縮寫、深色底板、標籤避讓及軌跡。移除舊車道、運動、換道程式；新 Milestone 2 動態白線與 Milestone 3 一般換道僅規劃。遵循全域 ExecPlan 規格。

## Progress

- [x] 2026-09-03 使用者核准完整回退計畫；基準 5fdf4e6，工作目錄乾淨。
- [x] 移除場景、設定、CLI 與測試的舊 Milestone 2 依賴。
- [x] 保留簡單標註並封存歷史、建立新 roadmap；歷史文件與報告逐位元組比對一致。
- [x] pytest、Ruff、Mypy、完整 CPU 影片及首中尾抽查；執行本機平台驗證並記錄 blocked。

## Context and Orientation

cli.py 建立 Analyzer；application/analyzer.py 串流處理影片；storage/artifacts.py 保存結果；visualization/annotator.py 原本混合物件與場景標註。config/models.py 與 configs/*.yaml 提供設定。以目前版本做手術式刪改，保留 minimum_track_length、裝置選擇與進度修正。

## Plan of Work

先解除 Analyzer、FrameRecord、ArtifactStore、CLI 與 Annotator 的場景依賴，events.json 保持空陣列，移除逐幀 analysis 欄位。再移除 lane/、motion/、events/、application/scene.py、application/lane_overlay.py 和專屬領域模型／測試。保留共用幾何、感知與驗證工具。驗證入口改用 milestone-1。

將舊計畫與原始報告封存於 docs/history/legacy-milestone-2/ 及 validation/history/legacy-milestone-2/，不更改機器報告內容。README 與 MILESTONE_1 說明目前功能，ROADMAP 與新計畫定義白線、一般換道，再依序方向燈、API/GPS、LLM/VLM。

## Concrete Steps

在專案根目錄執行 .venv/bin/pytest、.venv/bin/ruff check .、.venv/bin/mypy src、git diff --check。以本機 samples/test1.mp4 與 yolo26m.pt 執行 analyze，輸出到新的 output/milestone1-restored-review* 目錄。執行 validate --milestone 1 --platform macos-mps，僅寫本機報告。

## Validation and Acceptance

保留偵測追蹤、最短追蹤長度、CLI 與影片整合測試；確認 lane-overlay 不存在。實片輸出完整可讀、events.json 為 []、逐幀僅感知資料，首中尾僅物件框與簡單標註。現有兩平台報告分別測 cef1205 與 c7d77e3，對基準均 stale。修改後 dirty 報告必須 blocked；不能假稱最終 clean commit 驗證。Linux CUDA 本次缺少新報告，待該平台重跑。

## Idempotence and Recovery

不 reset、不 commit/push、不動影片權重或既有輸出。Git 保留刪除前內容。輸出目录存在則另選目錄。歷史報告原樣搬移。

## Interfaces and Dependencies

保留 PerceptionBackend.process 與 OpenCVAnnotator.annotate(frame, objects)、原始影像座標及現有依賴；不安裝套件、不下載模型。

## Surprises & Discoveries

簡單標註與場景標註共用類別，需保留標籤定位演算法再刪場景部分。

## Decision Log

使用者 2026-09-03 核准：本次只恢復 M1，新 M2/M3 僅計畫；一般車輛換道不要求進入自車車道，不做 cut-in。歷史文件與報告封存而非抹除歷史。

## Outcomes & Retrospective

實作完成。55 項 pytest、Ruff、Mypy（36 source files）通過。macOS MPS 報告 blocked（dirty worktree、MPS unavailable）；Linux CUDA missing。CPU 真實影片完整驗證完成：625 幀完整解碼、88 tracks、events=[]、逐幀無 analysis。首中尾（0/312/624）標註符合要求；已知模型仍誤認儀表部位為 car，本次不改偵測。正式 clean-commit MPS／CUDA 驗收仍待有相應環境時重跑。本次未提交或推送。
