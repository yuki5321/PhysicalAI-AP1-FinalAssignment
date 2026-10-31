# 崩し→復帰の成功率評価（`tools/eval_recovery.py`）

`docs/plan.md` の検証方法「各文字 20 回ランダムに崩し、成功率（目標 I・U 90% 以上、Y・K 70% 以上）と平均押し回数を CSV に記録」を行うための約束です。

## 1 試行の流れ

1. 文字を目標姿勢（`scene.yaml` の `target`）に置く。
2. ランダムに崩す（`--disturb`）。`teleport`: 作業領域内のランダムな姿勢へ移す（`is_disturbed` を満たすまで引き直す）。`hand`: Gazebo の手モデルと同じ形（手のひら＋指）・同じ軌道（`hand_logic.step_hand`、最大 0.15 m/s）で文字を払う。崩れなければ最大 4 回やり直し、だめなら teleport。`disc`: 旧方式（直径 15 mm の円で払う、sim2d のみ）。
   - `hand` では、払っている間は状態機械に `on_hand(True)` を送り、手が隠した文字は見えない（画像の知覚では手を合成画像に描く）。手が去ってから `hand_clear_s` 待って復帰が始まることを確かめる。
   - `--rehand-prob p`（sim2d のみ）: 確率 p で、復帰中の 1〜3 回目の押しのあとに手がもう一度来る（一時停止→再開の確認）。
3. 判断は本番と同じ `pa_planner.state_machine.PlannerStateMachine` を使う（時刻はバックエンドの時計）。観測はバックエンドから受け取る。
4. 終了条件: `LetterDone`（成功）、`FAILED` 状態、押し回数が `--max-pushes`（既定 30）に到達、または時間切れ（`--timeout`）。
5. 成功の判定は **真値** の姿勢で `pa_core.letters.is_aligned` を確かめる（知覚が「そろった」と言っても真値でずれていれば失敗、`fail_reason=false_done`）。

## バックエンド

`tools/eval_recovery.py --backend sim2d|gazebo`。どちらも次の形を持つ:

```python
class Backend:
    def reset(self, letter: str, pose: tuple, rng) -> None   # 文字を pose (x, y, theta) に置く
    def disturb(self, mode: str, rng) -> tuple               # 崩した後の真値姿勢
    def observe(self) -> list                                # [(letter, x, y, theta, confidence), ...] 知覚の出力
    def push(self, side: int, angle_deg: float) -> None      # 1 回振って戻るまで（ブロッキング）
    def true_pose(self) -> tuple                             # 真値 (x, y, theta)
    def now(self) -> float                                   # バックエンドの時計 [s]
    def close(self) -> None
```

- `sim2d`（ROS 不要）: 世界 = `pa_core.pushsim.PushSim`、観測 = `--perception truth|classic|cnn|hybrid`。`truth` は真値＋任意のノイズ、それ以外は `pa_perception.synth` で真値姿勢から画像を描き（ヘラは静止位置）、その検出器を通す。時計は押し 1 回ぶんの時間（`sweep_profile`）＋静定時間で進める。
- `gazebo`（ROS 2 + Gazebo 起動済み）: 世界 = Gazebo、真値 = `/model_states`、置く・崩す = `/set_entity_state`、観測 = `/letters`（`letter_detector` ノード）、押す = `/push_cmd` → `/push_done`。判断はこのツール内の状態機械で行うので `planner_node` は止めておく。

## 出力

`--csv PATH`: 1 行 1 試行。列は

```
trial, backend, perception, letter, seed, disturb, start_x, start_y, start_theta,
success, fail_reason, pushes, sim_time_s, wall_time_s, final_dp_mm, final_da_deg
```

`fail_reason` は `""`（成功）、`max_pushes`、`failed`（状態機械が FAILED）、`timeout`、`lost`（知覚が見失ったまま）、`out_of_workspace`、`false_done`、`pushed_with_hand`（手がある間、または手が消えてから `hand_clear_s` 未満で押した）。

上の 16 列のあとに、手による崩し用の列を追加で出す: `disturb_tries`（崩しに要したストローク数、5 は teleport に切り替えた）、`disturb_dp_mm` / `disturb_da_deg`（崩した直後の目標からのずれ）、`wait_hand_s`（手が最後に見えてから復帰を始めるまでの待ち時間の合計）、`rehand_at_push`（復帰中に手が戻ってきた押し回数、無ければ空）、`hand_violations`（手がある間に押した回数）。

標準出力には文字ごとの表（成功数 / 試行数、成功率、成功時の平均押し回数、平均所要時間、失敗理由の内訳）と、目標（I・U 90%、Y・K 70%）に届いたかを出す。`--markdown` で同じ表を Markdown で出す。

## Gazebo バックエンドの起動

```bash
ros2 launch pa_gazebo eval.launch.py            # gui:=false 既定、method:=classic|cnn|hybrid、hand:=true
python3 tools/eval_recovery.py --backend gazebo --disturb teleport|hand --trials 20 --csv ...
```

`eval.launch.py` = `sim.launch.py`（`scene_manager:=false hand_actor:=false`）＋ `letter_detector` ＋ 手モデルの生成（退避位置）。`planner_node` は起動しない。

- 置く: 対象以外の `letter_*` は机の上（視野外）へ退避し、`/set_entity_state` で置いてから真値が `settle_s`（既定 0.4 s、sim 時間）静止するまで待つ。
- `hand`: バックエンドが `hand` モデルを `/set_entity_state` で直接動かす（`hand_logic.step_hand` と同じ動き、30 Hz、`/hand/state` は使わない）。上から文字の手前に降ろし、ランダムな向き・ずれ・行き過ぎ量で文字を払って上げ、退避する。崩れていない・作業領域外・倒れた場合は置き直して最大 4 回、それでも駄目なら teleport に切り替える（ログに出す）。
- 観測: 最後の置く／崩す／押すが静止した時刻以降に撮られた画像（`header.stamp`）の `/letters` のうち最新のもの。3 s 来なければ `[]`。
- 押す: `/push_cmd`（id を毎回増やす）→ 同じ id の `/push_done` を待ち（30 s）、静止を待つ。
- 時計は `/clock`（sim 時間）。真値が台から落ちた（z < −1 cm）・倒れた（傾き > 0.35 rad）・台の外のときは `out_of_workspace()` が真。
- ROS 2 / Gazebo 上ではまだ動かしていない（ROS なしの単体テストと、偽の rclpy を使った動作確認のみ）。
