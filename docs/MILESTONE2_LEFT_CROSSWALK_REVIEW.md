# 可見左線與斑馬線修正 review

2026-10-07，trial-27 完成本輪指定左線與 540 斑馬線修正、完整兩片回歸與對照交付，沒有新增已通過案例失敗。新影片仍待使用者人工確認。整體品質仍 failed，Milestone 2 未完成。

## 來源與修正

HEAD `38e937ffd8ab043a7d9038756af119d26948e712`、dirty；本輪白線 SHA256 `32a8fa3028199bc40ad1256f768f8b2ca503d9f99e819cda3a731eeaaabdc070`。凍結 BEFORE 是前輪公車格 v28 的 `1a52da559c4b7bee62a0825e7226ee514bb3dbcbca65565dd68359927ac485e1`。兩份前輪公車格人工確認更新保留，沒有 commit 或 push。

共用流程不使用影片名稱、幀號或人工座標特例。恢復車框 padding 附近的淺斜短白漆，須有當下模型、狹窄漆面、合理方向與足夠遮罩交疊；深車身內部仍受保護。新恢復曲線獨立擬合，一般元件抽取、後續精修與父元件支持保持原有遮罩，只對當下已驗證漆面解除繪圖遮罩。恢復線跨幀更新限原本由此流程建立的線，必須仍有當下漆面與模型證據；消失時停止延續。

斑馬線利用重複橫向條紋的共同範圍補足排除區，保留強模型與窄漆支持的真正縱向線。新恢復漆面若沒有強道路語義支持，額外要求寬度隨透視縮小、曲線靠近漆面中心，避免遠處條紋誤晉升；一般候選寬度檢查不變。沒有改車輛追蹤、公開設定、模型或後端；prototype.py 只轉交新增私有參數，未啟用原型。

固定原片 SHA256：sample1 `220611d110329da7a5ad89405c475ed4104c4c5c8eae6eb0ec8f34e0d6f0ea2d`、sample2 `48f0e3010d7acffebf95b8bd55e1193589785a3d8835ec528040f0298edbf4ba`；設定 `78d321a8b5820b53e2fccf50044c32eced3819074e704c82516a43fd6676b6f2`。量測工具 SHA256 `254881cb12d536da0e404e16f9de4f5f1272e7a5b76380b3c9f3a8a805ff3c02` 與既有 fixture 未改。新 fixture SHA256 `0fcd9bb28be3bb425c92d85794bcc085f56367d57c0f6960b8bd7dc6db3d29ad`。完整來源、固定輸入與產物雜湊保存在 ignored `output/left-crosswalk-review-v1/`，大型資料不進 Git。

## 人工圈選驗收

原片零基 frame_id，10px 容許距離、至少 90% 覆蓋，負例要求零疊圖像素。所有人工可見漆面分母固定，不因偵測遮罩縮小。新 fixture 15 案例；630／680 新實際漆面與舊偏移標註分開，1222／1244 遠處圈選與 1228–1232 近處線也分開，舊標註保留。

| frame_id | 可見左漆 BEFORE | AFTER | 結果 |
|---|---:|---:|---|
| 630 | 0% | 100% | passed |
| 657 | 100% | 100% | passed |
| 680 | 0% | 100% | passed |
| 697 | 0% | 100% | passed |
| 1222 | 0% | 95.45% | passed |
| 1244 | 0% | 100% | passed |

540 斑馬線禁畫區 1361→0 像素，相鄰真線 100%；615／660 保留 100%。1228–1232 近處線約 80–84%，未達 90%，和 BEFORE 相同；1230 保留 81.67%。新案例 4→10/15；不得沿用失敗 trial-23 的 11/15 宣稱本輪結果。540 截圖與原片 535..545 的匹配最佳為 540（197 inliers、median reprojection error 1.27px），證據 `match-540.json`。

## 完整回歸

755 個不同影格聚焦完成，舊通過案例保留。sample2 v8／v10 車框完整 SHA256 一致，核對後保存別名，不算獨立證據。聚焦 BEFORE 複用 trial-12 相同影格、間斷 reset、車框與模型快取；AFTER 用相同已核對快取，各自使用實際有效遮罩。

兩片各 1800 幀 AFTER 連續重新模型推論。BEFORE 複用前輪 v28 完整 AFTER，先核對其精確來源、原片、模型、設定、車框、推論 AST、產物雜湊、順序與自身遮罩。這是已驗證 BEFORE 加本輪新 AFTER，不是本輪雙推論。每片 44 個 runtime／重播來源、全部固定輸入及輸出雜湊核對通過；四支完整影片各 1800 幀全部解碼。`final-audit-trial-27.json` 保存全 fixture 稽核，包含完整固定斑馬線區 1131 幀，均零誤畫；無新增已通過案例失敗。各家族結果如下，這是已知案例回歸，不是新場景準確率。

| 影片 | 案例家族 | BEFORE 通過 | AFTER 通過／總數 | 新增退步 |
|---|---|---:|---:|---:|
| sample1 | anchors | 9 | 9/9 | 0 |
| sample1 | human | 1136 | 1136/1136 | 0 |
| sample1 | straight | 11 | 11/13 | 0 |
| sample1 | controls | 3 | 3/3 | 0 |
| sample1 | markings | 30 | 30/30 | 0 |
| sample1 | extra | 25 | 25/25 | 0 |
| sample1 | bus | 82 | 82/82 | 0 |
| sample2 | anchors | 4 | 5/7 | 0 |
| sample2 | human | 4 | 5/7 | 0 |
| sample2 | straight | 14 | 14/14 | 0 |
| sample2 | controls | 12 | 12/13 | 0 |
| sample2 | markings | 15 | 15/25 | 0 |
| sample2 | extra | 24 | 24/25 | 0 |
| sample2 | bus | 54 | 54/54 | 0 |
| sample2 | left-old | 2 | 2/4 | 0 |
| sample2 | new | 4 | 10/15 | 0 |

公車格及真線控制合計 136/136。沿用錄製車輛觀察的自車形狀框 0、旁車控制 11/11，只證明沿用觀察保留，沒有重新執行完整 GPU 車輛追蹤。各家族仍有原有失敗，`quality_passed=false`、`regressions_detected=false`；不得以改善抵銷失敗或宣稱 M2 完成。

## 程式與平台檢查

同來源 294 pytest、Ruff、Mypy 42 檔及 diff 檢查 passed，見 `source-checks-trial-27.json`。正反例與左右鏡像包含車框 padding、白色車身、弱候選、近水平白漆、消失白漆、普通候選遮罩及透視寬度。透視測試對 trial-24：兩個遠寬條紋負例失敗、四個遠窄／近寬正例通過；本輪六個均通過。普通候選遮罩鏡像測試在 trial-23 失敗，本輪通過。相關證據 `before-test-proof-perspective-trial-27.json`、`before-test-proof-normal-mask-trial-27.json`。

同來源 20 幀預設 CLI 實際 requested=mps／resolved=mps、全數解碼、42 個來源與輸入／產物核對 passed，見 `default-cli-smoke-trial-27-review.json`；輸入是 sample2 900..919 重新編碼片段。車輛模型 MPS、白線 ONNX CPU，只是局部整合，不能代替全片 GPU 或正式平台驗收。

正式 M2 macOS MPS **blocked**（dirty、validate 只支援 M1、正式 JSON missing）；M2 Linux CUDA **missing**。M1 macOS 16bedd0 與 legacy M2 macOS cef1205／CUDA c7d77e3 **stale**，M1 CUDA **missing**。品質通過、clean 固定來源與 M2 入口可用後，兩個 GPU 平台須各自重跑。只更新 macOS 手動紀錄，不改其他平台機器報告。

## 人工對照交付

三段 H264 半速短片（15fps），左 BEFORE、右 AFTER，畫面直接標原片 frame_id。全部 249 幀解碼、fps 與來源／產物雜湊通過；編碼抽樣 PSNR 最低 32.67dB。直接從已完成完整比較擷取，沒有重新畫驗收曲線。`clips-trial-27-review.json` 保存方法與雜湊。

- `sample_2-crosswalk-before-after-trial-27.mp4`：520–565；優先看 539–541，確認斑馬線不畫、相鄰真線保留。
- `sample_2-left-near-before-after-trial-27.mp4`：599–735；優先看 630、657、680、697，確認畫在可見漆面，沒有接入車身或遮蔽區。
- `sample_2-left-far-before-after-trial-27.mp4`：1200–1265；優先看 1222、1244 遠處白漆。1228–1232 近處漏線仍列 failed。

八個原片幀 540、630、657、680、697、1222、1244、1230 的 RAW／BEFORE／AFTER 局部，從完整影片直接解碼擷取並檢視，兩張 `full-key-crops-trial-27-*.jpg` 與 `key-crops-trial-27-review.json` 保存證據。新來源仍待使用者人工確認；不沿用前輪「通過」作本輪確認。

## 拒絕試驗與限制

trial-23 聚焦通過但完整稽核新增 21 筆退步，拒絕；trial-24 遮罩隔離後仍 19 筆失敗。trial-25 透視寬度作用於所有恢復候選，雖修復退步窗口但強道路正例失敗，拒絕；trial-26 容許額外 1px 寬度仍有 5 筆退步，拒絕。成組條紋診斷會誤刪真左線，未加入 runtime。trial-27 只在弱道路支持時採嚴格透視寬度，79 幀窗口重現並修復 21/21，完整稽核也確認全部修復。各失敗來源、報告與中斷 complete=false 保留，未覆寫。聚焦成功不能代替完整固定斑馬線區或全片稽核；本輪只完成指定修正，其他標線失敗與正式雙平台驗收仍待處理。
