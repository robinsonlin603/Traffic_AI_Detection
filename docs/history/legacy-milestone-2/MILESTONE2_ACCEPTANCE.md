# Milestone 2 人工驗收表

本表只驗收 configured lanes 之間的一般換道，不驗收 ego-lane、cut-in、距離、TTC、
碰撞風險或方向燈。每個案例必須使用與該影片鏡頭和道路相符的 lane configuration。

## 執行方式

```bash
dashcam-ai analyze \
  --input ./samples/test1.mp4 \
  --output ./output/test1-general-lane-change \
  --config ./configs/mac.yaml
```

同時檢查 `annotated.mp4`、`events.json` 與必要的 `frames.jsonl`。事件時間以影片時間為準，
截圖只能輔助說明，不能單獨證明完整換道時間線。

## 必要案例

至少各提供一個案例：未換道、向左換道、向右換道、證據不足。若測試影片沒有某一類，
請標記 `missing`，不要用其他行為代替。

```markdown
### Track #<ID>

- 影片時間範圍：
- 人工結果：未換道 / 左換道 / 右換道 / 證據不足
- 來源車道：
- 目的車道：
- 開始接近邊界時間：
- 跨越邊界時間：
- 完成換道時間：
- 系統狀態：未產生 / candidate / confirmed / rejected / unknown
- 系統方向：left / right / unknown
- Track ID 是否穩定：是 / 否 / 無法確定
- Lane geometry 是否貼合：是 / 否
- Ego-motion 是否出現 unknown：是 / 否
- 截圖或補充說明：
```

## 判定規則

- `confirmed` 必須有穩定來源車道、共享邊界、跨線、目的車道停留與有效運動證據。
- 車輛只靠近邊界後返回來源車道，應為 `rejected` 或不產生事件。
- 非相鄰 lane jump、相機晃動、固定 geometry 因彎道漂移、ID switch 或遮擋造成的證據不足，
  不得被人工標成正確 confirmed。
- `started_at <= lane_crossed_at <= completed_at`；candidate、rejected、unknown 可缺少後段時間。
- `events.json` 只能包含 `lane_change`，不得包含 cut-in 或 forward corridor 欄位。

## 平台證據

macOS MPS 與 Linux CUDA 必須在相同最終 source commit、乾淨 worktree 上各自執行驗證。
任一平台的成功不能替代另一平台；CPU 結果只算補充證據。

## Test5 configured geometry baseline（2026-09-01）

`samples/test5.mp4` 是 82.45 秒、2471 幀、1920×1080 的實際兩條同向車道路段。
`output/test5` 使用 `configs/mac.yaml` 的固定三車道 geometry 在 Apple MPS 產生，僅作為
動態車道幾何導入前的失敗基準；大型輸出不納入 Git。

`events.json` 有 75 個 lane-change lifecycle 結果：70 rejected、5 unknown、0 confirmed。
方向分布是 61 left、14 right；其中 55 個是 `lane_center -> lane_left`。Ego-motion 在
2470/2471 幀有效，唯一 unknown 是沒有前一幀的 frame 0。因此大量候選主要不是
ego-motion 失敗，而是固定三車道 geometry 與實際兩車道路面不一致。

人工檢查確認兩條綠色 shared boundaries 沒有沿著真實車道線，並朝錯誤的天空區域
收斂；左側黃色 `lane_left` polygon 已覆蓋中央分隔區及對向車道。`00:15–00:20` 的
候選皆位於對向車道。`00:55–01:00` 的 `#697`、`#770`、`#793`、`#797`、`#798`、
`#809`、`#869` 也是對向車；`#792` 與 `#804` 是同向停放車。其他人工結論如下：

- `#1` 確實向左轉，但轉向或進入岔路不可直接當成跨越相鄰車道的 lane change。
- `#20` 全程是同一輛 truck，沒有向左移動；其事件是幾何誤判。
- `#355` 確實由右車道換到左車道，但遮擋後 Track ID 變成 `#407`。
- `#405` 是停放車，遮擋後 Track ID 變成 `#431`。
- `#1038` 是停在對向側的車。
- `#1097` 與 `#1137` 是停在同向側的車。
- `#1223` 是停在對向側的車，並非片尾未完成換道。

此 baseline 證明事件 lifecycle 能保守避免 confirmed，但不能證明換道辨識準確。動態
幾何完成後，至少必須觀察到真實兩車道拓撲、邊界沿著道路而非天空、`#20` 不建立
候選，以及道路／路口線索不足時 geometry 回傳 degraded 或 unknown。`#355 -> #407`
屬於獨立的跨 ID continuity 問題，不能把 dynamic geometry 無法解決此案例誤判為
geometry regression。

## Dynamic Slice 5 驗收稽核（2026-09-02）

目前程式碼以 CPU 重跑 `samples/test5.mp4` 的 configured comparison，共 2471 幀、358
個 Track、75 個 lifecycle event，耗時 1377.943 秒（1.793 FPS）。結果仍是 70
rejected、5 unknown、0 confirmed；transition 分布為 `lane_center -> lane_left` 55、
`lane_center -> lane_right` 13、`lane_right -> lane_center` 6、`lane_left -> lane_center` 1。
Track `#20` 仍產生 `lane_right -> lane_center` rejected event，因此 configured geometry
只保留為可重現的失敗 baseline，不能通過真實車道驗收。

YOLOP temporal diagnostic 在 test5 前 120 幀得到 0 valid、69 degraded、51 unknown。
其中 50 幀雖形成 `topology-1` 三區域，但人工疊圖仍可看到中央分隔設施或路緣被當成
邊界。安全閘門會阻止這些 geometry 產生 membership 或事件，因此沒有誤報；但「全部
降級」也不能證明系統成功辨識兩條同向車道。Dynamic acceptance 目前為 **blocked**，
必須先加入同向 drivable-area／road-edge 證據並排除 median、roadside component，之後
重新執行完整 test5 dynamic pipeline。

本機執行環境在本次稽核無法使用 MPS，因此 CPU comparison 不能替代 macOS MPS 平台
報告。現有 macOS MPS 與 Linux CUDA 報告都對應舊 source commit，狀態均為 stale；
必須在最終 commit 與乾淨 worktree 上分別重跑。
