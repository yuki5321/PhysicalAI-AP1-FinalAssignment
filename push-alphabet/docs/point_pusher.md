# 点の押し棒（Push-T 型）への切り替え

4 本のヘラの代わりに、Push-T と同じ「点で押す」押し棒を使います。押し棒は細い円柱で、文字の手前に降ろし、まっすぐ押して、持ち上げて待機位置に戻ります（実機は XY 方向に動く装置やアームを想定）。寸法と速度は `scene.yaml` の `point_pusher`、どちらを使うかは `pusher_mode`。

## 約束

- 指令: `/point_push_cmd`（`pa_interfaces/PointPushCommand`: `id, x0, y0, x1, y1, speed`）。実行し終えたら同じメッセージを `/point_push_done` に返す（`id` で対応）。ヘラの `/push_cmd` / `/push_done` は残す。
- 2D モデル: `pa_core.pointpush.PointPushSim`（`set_pose(x, y, theta)`、`push(x0, y0, x1, y1, speed=None) -> (x, y, theta)`、`pose()`）。壁・床摩擦・世界の作り直しは `pa_core.pushsim` と同じ。
- プランナー: `pa_core.pointplanner.PointPushPlanner(letter)`、`plan(pose, explore=0.0) -> PointPlan(x0, y0, x1, y1, predicted, cost)`。費用は `pa_core.planner.cost` と同じ。
- 状態機械: `PushCmd` に押し棒用の `stroke=(x0, y0, x1, y1)` を追加（ヘラのときは `None`）。`planner_node` はパラメータ `pusher`（既定は `scene.yaml` の `pusher_mode`）で指令の出し先を選ぶ。
- Gazebo: 押し棒は SDF モデル `point_pusher`（オレンジ、知覚では「隠すもの」として扱われる色）。`point_push_executor` が `/set_entity_state` で sim 時間に合わせて動かす（降ろす → 押す → 持ち上げる → 待機位置）。`pusher:=point` のときヘラは URDF に入れない。
