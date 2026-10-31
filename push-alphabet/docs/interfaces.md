# ノード間インターフェース

シミュ（Gazebo）と実機で共通にする約束です。座標・寸法はすべて [`scene.yaml`](../ros2_ws/src/pa_core/pa_core/scene.yaml) の world 座標（台の中心が原点、+y が文字の「上」、単位 m・rad）。

## トピック

| トピック | 型 | 出す側 → 受ける側 | 内容 |
| --- | --- | --- | --- |
| `/overhead_camera/image_raw` | `sensor_msgs/Image` (bgr8) | Gazebo（実機は usb_cam）→ pa_perception | 真上カメラ。画像右 = world +x、画像上 = world +y。画角・解像度は `scene.yaml` の `camera` |
| `/letters` | `pa_interfaces/LetterStateArray` | pa_perception → pa_planner | 台上（`workspace` 内）に見えている文字の種類と姿勢 |
| `/active_letter` | `std_msgs/String`（transient_local） | pa_gazebo `scene_manager` → pa_planner | いま直す対象の文字（`sequence` 順）。全部終わったら `""` |
| `/letter_done` | `std_msgs/String` | pa_planner → `scene_manager` | 対象の文字がそろった。`scene_manager` が展示エリアへ移し、次の文字を出す |
| `/push_cmd` | `pa_interfaces/PushCommand` | pa_planner → `push_executor`（実機は Arduino ブリッジ） | ヘラを `angle_deg` まで振って戻す |
| `/push_done` | `pa_interfaces/PushCommand` | `push_executor` → pa_planner | 実行し終えた指令をそのまま返す（`id` で対応） |
| `/point_push_cmd` | `pa_interfaces/PointPushCommand` | pa_planner → `point_push_executor` | 点の押し棒（`pusher:=point`）: `(x0, y0)` に降ろして `(x1, y1)` までまっすぐ押す |
| `/point_push_done` | `pa_interfaces/PointPushCommand` | `point_push_executor` → pa_planner | 実行し終えた指令をそのまま返す（`id` で対応） |
| `/hand/state` | `pa_interfaces/HandState` | `host/hand_bridge.py`（rosbridge 経由）→ `hand_actor`, pa_planner | 手の有無と掌中心（world 座標） |
| `/planner/status` | `std_msgs/String` | pa_planner → 表示用 | 状態機械の状態名（`IDLE` / `WAIT_HAND` / `RECOVER` / `DONE` …） |

## ヘラ

- `side` は `scene.yaml` の `pushers.list` の順番（0=south, 1=east, 2=north, 3=west）。
- Gazebo では revolute ジョイント `pusher_<name>_joint`（z 軸回り、0 rad で `rest_heading`、正の角度で反時計回りに作業領域へ入る）。ピボット位置・腕の寸法は `scene.yaml` の値を使う。
- 速度は `pushers.speed_deg_s`、最大角で `hold_s` 止めてから戻す（2D モデル `pa_core.pushsim` と同じ動き）。

## 補足

- 台の上面が z = 0。文字モデルの原点は底面上の面積重心。
- `/push_cmd` の `angle_deg` は 0..`max_angle` に丸める。`side` が範囲外なら動かさずにすぐ `/push_done` を返す。
- `/active_letter` は文字を出し終えてから送る。対象の文字が最初からそろっていても、崩されて直すまで `/letter_done` は出ない。
- 真値は Gazebo の `/model_states`（モデル名 `letter_<X>`）で取れる（評価・データ生成用）。
- カメラ画像は 640×640（`scene.yaml`）。

## ポート

- rosbridge: `ws://localhost:9090`、noVNC: `http://localhost:6080`
