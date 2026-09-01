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
